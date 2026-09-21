"""Time-windowed metrics + incremental CSV writing for the vLLM benchmark.

The benchmark driver hands completed `RequestRecord`s to `MetricsLogger`,
which:

  - records per-request outcomes (hit/miss, latency, session, etc.)
  - aggregates them into time-windowed buckets (e.g. every 30 sim seconds)
  - computes per-session hit rates
  - writes results incrementally to CSV so that a Kaggle/Colab disconnect
    mid-run does not lose completed data (per the spec's Execution
    Environment notes).
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

@dataclass
class RequestRecord:
    """One completed request's outcome."""

    request_id: str
    session_id: str
    turn_index: int
    submit_t: float
    complete_t: float
    latency_ms: float
    n_prompt_tokens: int
    n_output_tokens: int
    hit: bool  # True if a hit was observed for this request
    shared: bool  # True if the request's content was detected as shared
    policy_name: str
    capacity_setting: str
    in_flight_at_submit: int  # number of other requests in flight at submit time
    # --- ground-truth prefix-cache accounting -----------------------------
    # Prompt tokens vLLM served from an existing KV block (-1 = engine did
    # not report it). This, not latency, defines `hit`.
    num_cached_tokens: int = -1
    # "cached_tokens" (ground truth) | "latency_proxy" (fallback only).
    hit_basis: str = "cached_tokens"
    # What the old latency threshold *would* have said, for comparison.
    proxy_hit: bool = False
    # True when this request's session history was trimmed to the context
    # window, which invalidates that session's cached prefix. Tracked
    # because it is a legitimate cause of misses unrelated to the policy.
    context_truncated: bool = False
    # End-to-end latency from when the user's turn became available, i.e.
    # INCLUDING time spent waiting in the dispatch queue. `latency_ms` above
    # excludes that wait, so it flatters policies that starve requests.
    # Goodput is computed on this one.
    e2e_latency_ms: float = 0.0
    queue_wait_ms: float = 0.0
    # The engine raised instead of serving this request. Such a record has
    # no prompt tokens, no cache counter and a ~0 ms latency, so it must
    # never reach the hit/latency statistics: round 2's `fifo_generous_s2`
    # was 82% failed requests that were scored as instant proxy "hits".
    error: bool = False

    @property
    def cached_fraction(self) -> float:
        if self.num_cached_tokens <= 0 or self.n_prompt_tokens <= 0:
            return 0.0
        return min(1.0, self.num_cached_tokens / self.n_prompt_tokens)


@dataclass
class MetricsConfig:
    window_s: float = 30.0
    sla_latency_ms: float = 2000.0
    # Legacy latency threshold. Only used to emit the `proxy_hit_rate`
    # diagnostic column; `hit` itself now comes from vLLM's per-request
    # num_cached_tokens. See bench.py's module docstring.
    hit_latency_threshold_ms: float = 300.0
    output_dir: str = "results/csv"
    run_label: str = "run"
    # Leading windows excluded from the run summary as engine warmup.
    # They are still written to the time series (with a `warmup` flag) so
    # nothing is hidden -- they are just not allowed to set the headline
    # P99, which is how run order came to dominate the first results.
    discard_warmup_windows: int = 1
    # A run is only `saturated` if its final-window median dispatch wait also
    # exceeds this (wall ms). The launcher sets it to half a session's mean
    # idle gap in wall time: past that, requests are served after their
    # session would normally have moved on, so their prefix may be evicted
    # -- which is the actual harm. 0 keeps the old test.
    saturation_floor_ms: float = 0.0
    # Whether `saturated` makes the run unusable. True only for FIFO: the
    # FIFO run of each seed certifies that the LOAD is servable. A
    # reordering policy can then build a long wait for itself by starving
    # low-priority requests (none of them age), which is an outcome of that
    # policy -- visible in its e2e/queue-wait tail -- not a broken run.
    # Excluding it would drop exactly the policies under test.
    saturation_blocks_usable: bool = True


def percentile(xs: Sequence[float], pct: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


import pandas as pd


class MetricsLogger:
    """Incremental, windowed metrics for one policy x capacity run.

    Each call to `record()` appends a `RequestRecord`; `flush_window()`
    emits a CSV row for the given window. The logger keeps:

      - an in-memory list of all per-request records (for final per-session
        aggregation),
      - a per-window aggregator that is flushed to disk on demand,
      - an "open" time-series CSV file that we write to row-by-row.

    Use `finalize()` at the end of a run to flush remaining windows and
    write a summary row.
    """

    def __init__(self, cfg: MetricsConfig) -> None:
        self.cfg = cfg
        self.records: List[RequestRecord] = []
        # Requests the engine failed. Kept apart from `records` so every
        # downstream statistic sees served requests only.
        self.failed: List[RequestRecord] = []
        os.makedirs(cfg.output_dir, exist_ok=True)
        # Per-window in-memory accumulators
        self._win_states: Dict[int, Dict] = {}
        # Path to the time-series CSV. We may rewrite it at finalize
        # time once we have complete data per window.
        self.ts_path = os.path.join(
            cfg.output_dir, f"time_series_{cfg.run_label}.csv"
        )
        self._ts_fh = None
        self._ts_writer = None
        self._finalized = False

    def record(self, r: RequestRecord) -> None:
        if r.error:
            self.failed.append(r)
            return
        self.records.append(r)
        # Classify into a window based on submit time
        w = int(r.submit_t // self.cfg.window_s)
        st = self._win_states.setdefault(w, self._new_win_state())
        st["lookups"] += 1
        if r.hit:
            st["hits"] += 1
            if r.shared:
                st["shared_hits"] += 1
            else:
                st["unique_hits"] += 1
        else:
            st["misses"] += 1
        # Token-level ground truth: the headline metric. Summing tokens
        # (rather than averaging per-request fractions) weights long
        # contexts correctly and matches how vLLM computes its own
        # gpu_prefix_cache_hit_rate.
        st["prompt_tokens"] += max(0, r.n_prompt_tokens)
        if r.num_cached_tokens >= 0:
            st["cached_tokens"] += r.num_cached_tokens
            st["gt_lookups"] += 1
            if r.shared:
                st["shared_prompt_tokens"] += max(0, r.n_prompt_tokens)
                st["shared_cached_tokens"] += r.num_cached_tokens
        if r.proxy_hit:
            st["proxy_hits"] += 1
        if r.context_truncated:
            st["truncated"] += 1
        st["latencies"].append(r.latency_ms)
        st["e2e_latencies"].append(r.e2e_latency_ms or r.latency_ms)
        st["queue_waits"].append(r.queue_wait_ms)
        # SLA is a user-facing promise, so it is judged on end-to-end latency.
        if (r.e2e_latency_ms or r.latency_ms) <= self.cfg.sla_latency_ms:
            st["good"] += 1
        st["in_flight_sum"] += r.in_flight_at_submit
        if r.in_flight_at_submit > st["in_flight_max"]:
            st["in_flight_max"] = r.in_flight_at_submit

    def _saturation_diagnostics(self, warm_w: int) -> Dict:
        """Did the dispatch backlog grow without bound during this run?

        Offered load above engine capacity makes p50 queue wait climb
        monotonically window over window. That run is measuring queue
        depth, not cache behaviour: requests are dispatched so long after
        their turn that the session's prefix has already been evicted, so
        the hit rate decays with the backlog. We compare the first and
        last non-warmup windows that actually carry requests.

        `saturated` requires a large growth ratio AND an absolute wait that
        is itself meaningful, so a run that merely rises from 20 ms to
        80 ms is not flagged.

        Growth is measured against at least 1 s. With few engine slots the
        first steady window often has a median wait of exactly 0, which
        made the ratio infinite and flagged any run whose end-heavy last
        window queued for >10 s -- even when the backlog drained and no
        request waited anywhere near a session's idle gap. Round 2's
        runaway (32 s -> 545 s) is still flagged either way.
        """
        wins = sorted(w for w in self._win_states if w >= warm_w
                      and self._win_states[w]["lookups"] > 0)
        if len(wins) < 2:
            return {"first_ms": 0.0, "last_ms": 0.0, "growth": 0.0,
                    "saturated": False}
        first = percentile(self._win_states[wins[0]]["queue_waits"], 0.50)
        last = percentile(self._win_states[wins[-1]]["queue_waits"], 0.50)
        growth = last / max(first, 1000.0)
        saturated = bool(last > self.cfg.sla_latency_ms * 4 and growth > 2.0
                         and last > self.cfg.saturation_floor_ms)
        return {"first_ms": first, "last_ms": last, "growth": growth,
                "saturated": saturated}

    def _new_win_state(self) -> Dict:
        return {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "shared_hits": 0,
            "unique_hits": 0,
            "latencies": [],
            "e2e_latencies": [],
            "queue_waits": [],
            "good": 0,
            "in_flight_sum": 0,
            "in_flight_max": 0,
            # ground-truth token accounting
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "gt_lookups": 0,
            "shared_prompt_tokens": 0,
            "shared_cached_tokens": 0,
            # legacy latency-proxy diagnostic
            "proxy_hits": 0,
            "truncated": 0,
        }

    def flush_window(self, w: int) -> None:
        """Mark window `w` as having been seen. No-op for I/O.

        We previously wrote the row to disk here, but that caused
        "double-row" issues when late stragglers for the same window
        arrived after we'd already flushed. Now we keep the data in
        memory and rewrite the time-series CSV atomically at finalize.
        The in-memory aggregation continues regardless of how many
        times `flush_window` is called.

        The "durability for survive-disconnects" property is provided
        by the per-session CSV and the summary CSV, which are written
        at finalize — small enough that the loss window on a disconnect
        is at most a few seconds of one in-flight run.
        """
        # Touch the window in the dict so that finalize emits a row
        # even if it had no data.
        self._win_states.setdefault(w, self._new_win_state())

    def _format_window_row(self, w: int, st: Dict) -> List:
        total = st["hits"] + st["misses"]
        hr = (st["hits"] / total) if total else 0.0
        shr = (st["shared_hits"] / (st["shared_hits"] + st["misses"])) if (st["shared_hits"] + st["misses"]) else 0.0
        uhr = (st["unique_hits"] / (st["unique_hits"] + st["misses"])) if (st["unique_hits"] + st["misses"]) else 0.0
        goodput = (st["good"] / st["lookups"]) if st["lookups"] else 0.0
        mean_inf = (st["in_flight_sum"] / st["lookups"]) if st["lookups"] else 0.0
        # Headline: token-level prefix-cache hit rate from vLLM's own counter.
        ctr = (st["cached_tokens"] / st["prompt_tokens"]) if st["prompt_tokens"] else 0.0
        sctr = (
            st["shared_cached_tokens"] / st["shared_prompt_tokens"]
        ) if st["shared_prompt_tokens"] else 0.0
        proxy_hr = (st["proxy_hits"] / st["lookups"]) if st["lookups"] else 0.0
        return [
            w,
            w * self.cfg.window_s,
            (w + 1) * self.cfg.window_s,
            1 if w < self.cfg.discard_warmup_windows else 0,
            st["lookups"],
            st["hits"],
            st["misses"],
            f"{hr:.6f}",
            st["shared_hits"],
            st["unique_hits"],
            f"{shr:.6f}",
            f"{uhr:.6f}",
            st["prompt_tokens"],
            st["cached_tokens"],
            f"{ctr:.6f}",
            f"{sctr:.6f}",
            f"{percentile(st['latencies'], 0.50):.2f}",
            f"{percentile(st['latencies'], 0.99):.2f}",
            f"{percentile(st['e2e_latencies'], 0.50):.2f}",
            f"{percentile(st['e2e_latencies'], 0.99):.2f}",
            f"{percentile(st['queue_waits'], 0.50):.2f}",
            f"{goodput:.6f}",
            f"{mean_inf:.2f}",
            st["in_flight_max"],
            f"{proxy_hr:.6f}",
        ]


    def finalize(self, sim_window_s: float) -> Dict[str, float]:
        """Aggregate everything in memory and write the per-run CSVs.

        Writes the time-series CSV atomically (replacing any partial
        rows that were tentatively written earlier) plus the per-session
        and summary CSVs. Returns the summary dict.
        """
        self._finalized = True

        # Determine the full window range we should emit
        max_win = 0
        if self._win_states:
            max_win = max(self._win_states.keys())
        # Always include windows covering [0, sim_window_s)
        end_win = max(max_win, int(math.ceil(sim_window_s / self.cfg.window_s)) - 1)
        # Write the time-series CSV atomically
        with open(self.ts_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "window_idx", "t_start", "t_end", "is_warmup",
                    "lookups", "hits", "misses", "hit_rate",
                    "shared_hits", "unique_hits",
                    "shared_hit_rate", "unique_hit_rate",
                    "prompt_tokens", "cached_tokens",
                    "cached_token_rate", "shared_cached_token_rate",
                    "p50_latency_ms", "p99_latency_ms",
                    "p50_e2e_latency_ms", "p99_e2e_latency_ms",
                    "p50_queue_wait_ms",
                    "goodput", "mean_in_flight", "max_in_flight",
                    "proxy_hit_rate",
                ]
            )
            for w in range(end_win + 1):
                st = self._win_states.get(w, self._new_win_state())
                writer.writerow(self._format_window_row(w, st))

        # Per-session DataFrame (steady state only, same as the summary)
        steady_for_sessions = [
            r for r in self.records
            if int(r.submit_t // self.cfg.window_s)
            >= self.cfg.discard_warmup_windows
        ]
        sess_rows: Dict[str, Dict] = {}
        for r in steady_for_sessions:
            row = sess_rows.setdefault(
                r.session_id,
                {"session_id": r.session_id, "lookups": 0, "hits": 0,
                 "misses": 0, "shared_hits": 0, "p99_max": 0.0,
                 "prompt_tokens": 0, "cached_tokens": 0},
            )
            row["lookups"] += 1
            if r.hit:
                row["hits"] += 1
                if r.shared:
                    row["shared_hits"] += 1
            else:
                row["misses"] += 1
            row["prompt_tokens"] += max(0, r.n_prompt_tokens)
            if r.num_cached_tokens >= 0:
                row["cached_tokens"] += r.num_cached_tokens
            if r.latency_ms > row["p99_max"]:
                row["p99_max"] = r.latency_ms
        for row in sess_rows.values():
            tot = row["hits"] + row["misses"]
            row["hit_rate"] = (row["hits"] / tot) if tot else 0.0
            row["shared_hit_rate"] = (row["shared_hits"] / row["lookups"]) if row["lookups"] else 0.0
            # Fairness is judged on this: the share of each session's own
            # prompt tokens that it got to reuse.
            row["cached_token_rate"] = (
                row["cached_tokens"] / row["prompt_tokens"]
            ) if row["prompt_tokens"] else 0.0

        ps_df = pd.DataFrame(list(sess_rows.values()))
        ps_path = os.path.join(self.cfg.output_dir, f"per_session_{self.cfg.run_label}.csv")
        ps_df.to_csv(ps_path, index=False)

        # ---------------------------------------------------------------
        # Summary. Computed over STEADY-STATE records only: the first
        # `discard_warmup_windows` windows carry one-time engine warmup
        # (model load, CUDA graph capture) that is a property of when the
        # run happened in the matrix, not of the policy. Leaving them in
        # is what previously made FIFO -- which simply ran first -- look
        # like it had an 8x worse P99.
        # ---------------------------------------------------------------
        warm_w = self.cfg.discard_warmup_windows
        steady = [
            r for r in self.records
            if int(r.submit_t // self.cfg.window_s) >= warm_w
        ]
        n_all = len(self.records)
        n = len(steady)
        n_failed = sum(
            1 for r in self.failed
            if int(r.submit_t // self.cfg.window_s) >= warm_w
        )
        error_rate = n_failed / (n + n_failed) if (n + n_failed) else 0.0
        if n == 0:
            summary = {
                "policy": self.cfg.run_label,
                "lookups": 0, "hits": 0, "misses": 0,
                "shared_hits": 0, "hit_rate": 0.0, "shared_hit_rate": 0.0,
                "prompt_tokens": 0, "cached_tokens": 0,
                "cached_token_rate": 0.0, "shared_cached_token_rate": 0.0,
                "hit_basis": "none", "cache_ground_truth_coverage": 0.0,
                "p50_latency_ms": 0.0, "p99_latency_ms": 0.0,
                "p50_e2e_latency_ms": 0.0, "p99_e2e_latency_ms": 0.0,
                "p50_queue_wait_ms": 0.0, "p99_queue_wait_ms": 0.0,
                "dispatch_queued_fraction": 0.0,
                "goodput": 0.0, "proxy_hit_rate": 0.0,
                "n_records_all": n_all, "n_warmup_windows_discarded": warm_w,
                "context_truncated_rate": 0.0,
                "queue_wait_first_window_ms": 0.0,
                "queue_wait_last_window_ms": 0.0,
                "queue_wait_growth_ratio": 0.0,
                "saturated": 0, "usable": 0,
                "n_failed": n_failed, "error_rate": error_rate,
            }
        else:
            total_hits = sum(1 for r in steady if r.hit)
            total_misses = n - total_hits
            shared_hits = sum(1 for r in steady if r.hit and r.shared)
            lats = [r.latency_ms for r in steady]
            e2e = [(r.e2e_latency_ms or r.latency_ms) for r in steady]
            qw = [r.queue_wait_ms for r in steady]
            good = sum(1 for r in steady
                       if (r.e2e_latency_ms or r.latency_ms) <= self.cfg.sla_latency_ms)
            # Token-level ground truth -- the headline number.
            gt = [r for r in steady if r.num_cached_tokens >= 0]
            prompt_tok = sum(max(0, r.n_prompt_tokens) for r in gt)
            cached_tok = sum(r.num_cached_tokens for r in gt)
            sh_prompt_tok = sum(max(0, r.n_prompt_tokens) for r in gt if r.shared)
            sh_cached_tok = sum(r.num_cached_tokens for r in gt if r.shared)
            bases = {r.hit_basis for r in steady}
            basis = bases.pop() if len(bases) == 1 else "mixed:" + "+".join(sorted(bases))
            sat = self._saturation_diagnostics(warm_w)
            coverage = len(gt) / n if n else 0.0
            summary = {
                "policy": self.cfg.run_label,
                "lookups": n,
                "hits": total_hits,
                "misses": total_misses,
                "shared_hits": shared_hits,
                "hit_rate": total_hits / n if n else 0.0,
                "shared_hit_rate": (shared_hits / (shared_hits + total_misses)) if (shared_hits + total_misses) else 0.0,
                "prompt_tokens": prompt_tok,
                "cached_tokens": cached_tok,
                "cached_token_rate": (cached_tok / prompt_tok) if prompt_tok else 0.0,
                "shared_cached_token_rate": (sh_cached_tok / sh_prompt_tok) if sh_prompt_tok else 0.0,
                "hit_basis": basis,
                "cache_ground_truth_coverage": coverage,
                "p50_latency_ms": percentile(lats, 0.50),
                "p99_latency_ms": percentile(lats, 0.99),
                # End-to-end, including dispatch-queue wait. This is the
                # user-visible number and the one goodput is judged on.
                "p50_e2e_latency_ms": percentile(e2e, 0.50),
                "p99_e2e_latency_ms": percentile(e2e, 0.99),
                "p50_queue_wait_ms": percentile(qw, 0.50),
                "p99_queue_wait_ms": percentile(qw, 0.99),
                # Share of requests that waited in OUR dispatch queue, i.e.
                # the only requests a policy could reorder. Round 3 had
                # ~14%, 1-2 deep: every policy dispatched the same order.
                # (250 ms is well above the 50 ms dispatch-loop tick.)
                "dispatch_queued_fraction": sum(1 for w in qw if w > 250.0) / n,
                "goodput": good / n,
                # What the discredited latency threshold would have reported,
                # on the same requests. Published side by side on purpose.
                "proxy_hit_rate": sum(1 for r in steady if r.proxy_hit) / n,
                "n_records_all": n_all,
                "n_warmup_windows_discarded": warm_w,
                # Share of requests whose prefix was invalidated by hitting
                # the context window. A miss cause that is not the policy's
                # fault; report it alongside the hit rate.
                "context_truncated_rate": sum(
                    1 for r in steady if r.context_truncated
                ) / n,
                # --- Saturation: is this run measuring cache or backlog? ---
                # When offered load exceeds engine capacity the dispatch
                # backlog grows without bound, requests are served long
                # after their turn (so their prefix is gone), and the run's
                # hit rate decays with queue depth rather than reflecting
                # the policy. These make that visible per run instead of
                # requiring someone to eyeball the time series.
                "queue_wait_first_window_ms": sat["first_ms"],
                "queue_wait_last_window_ms": sat["last_ms"],
                "queue_wait_growth_ratio": sat["growth"],
                "saturated": int(sat["saturated"]),
                # Requests the engine failed rather than served. Excluded
                # from every number above; a run with more than a stray one
                # did not serve the workload and is not a data point.
                "n_failed": n_failed,
                "error_rate": error_rate,
                # One flag the analysis layer can filter on: a run is usable
                # as a data point only if hit/miss came from the engine for
                # essentially every request, the engine actually served the
                # workload, AND the run reached steady state.
                "usable": int(
                    basis == "cached_tokens"
                    and coverage >= 0.99
                    and error_rate <= 0.01
                    and not (sat["saturated"] and self.cfg.saturation_blocks_usable)
                ),
                "saturation_blocks_usable": int(self.cfg.saturation_blocks_usable),
            }
        summary_path = os.path.join(self.cfg.output_dir, f"summary_{self.cfg.run_label}.csv")
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        return summary

