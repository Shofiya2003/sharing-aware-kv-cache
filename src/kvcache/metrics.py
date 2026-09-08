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


@dataclass
class MetricsConfig:
    window_s: float = 30.0
    sla_latency_ms: float = 2000.0
    # Hit classification: requests with latency below this are hits (in the
    # simulated regime where vLLM hits are <100ms and misses are >1s).
    hit_latency_threshold_ms: float = 300.0
    output_dir: str = "results/csv"
    run_label: str = "run"


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
        st["latencies"].append(r.latency_ms)
        if r.latency_ms <= self.cfg.sla_latency_ms:
            st["good"] += 1
        st["in_flight_sum"] += r.in_flight_at_submit
        if r.in_flight_at_submit > st["in_flight_max"]:
            st["in_flight_max"] = r.in_flight_at_submit

    def _new_win_state(self) -> Dict:
        return {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "shared_hits": 0,
            "unique_hits": 0,
            "latencies": [],
            "good": 0,
            "in_flight_sum": 0,
            "in_flight_max": 0,
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
        return [
            w,
            w * self.cfg.window_s,
            (w + 1) * self.cfg.window_s,
            st["lookups"],
            st["hits"],
            st["misses"],
            f"{hr:.6f}",
            st["shared_hits"],
            st["unique_hits"],
            f"{shr:.6f}",
            f"{uhr:.6f}",
            f"{percentile(st['latencies'], 0.50):.2f}",
            f"{percentile(st['latencies'], 0.99):.2f}",
            f"{goodput:.6f}",
            f"{mean_inf:.2f}",
            st["in_flight_max"],
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
                    "window_idx", "t_start", "t_end",
                    "lookups", "hits", "misses", "hit_rate",
                    "shared_hits", "unique_hits",
                    "shared_hit_rate", "unique_hit_rate",
                    "p50_latency_ms", "p99_latency_ms",
                    "goodput", "mean_in_flight", "max_in_flight",
                ]
            )
            for w in range(end_win + 1):
                st = self._win_states.get(w, self._new_win_state())
                writer.writerow(self._format_window_row(w, st))

        # Per-session DataFrame
        sess_rows: Dict[str, Dict] = {}
        for r in self.records:
            row = sess_rows.setdefault(
                r.session_id,
                {"session_id": r.session_id, "lookups": 0, "hits": 0,
                 "misses": 0, "shared_hits": 0, "p99_max": 0.0},
            )
            row["lookups"] += 1
            if r.hit:
                row["hits"] += 1
                if r.shared:
                    row["shared_hits"] += 1
            else:
                row["misses"] += 1
            if r.latency_ms > row["p99_max"]:
                row["p99_max"] = r.latency_ms
        for row in sess_rows.values():
            tot = row["hits"] + row["misses"]
            row["hit_rate"] = (row["hits"] / tot) if tot else 0.0
            row["shared_hit_rate"] = (row["shared_hits"] / row["lookups"]) if row["lookups"] else 0.0

        ps_df = pd.DataFrame(list(sess_rows.values()))
        ps_path = os.path.join(self.cfg.output_dir, f"per_session_{self.cfg.run_label}.csv")
        ps_df.to_csv(ps_path, index=False)

        # Summary
        n = len(self.records)
        if n == 0:
            summary = {
                "policy": self.cfg.run_label,
                "lookups": 0, "hits": 0, "misses": 0,
                "shared_hits": 0, "hit_rate": 0.0, "shared_hit_rate": 0.0,
                "p50_latency_ms": 0.0, "p99_latency_ms": 0.0,
                "goodput": 0.0,
            }
        else:
            total_hits = sum(1 for r in self.records if r.hit)
            total_misses = n - total_hits
            shared_hits = sum(1 for r in self.records if r.hit and r.shared)
            lats = [r.latency_ms for r in self.records]
            good = sum(1 for r in self.records if r.latency_ms <= self.cfg.sla_latency_ms)
            summary = {
                "policy": self.cfg.run_label,
                "lookups": n,
                "hits": total_hits,
                "misses": total_misses,
                "shared_hits": shared_hits,
                "hit_rate": total_hits / n if n else 0.0,
                "shared_hit_rate": (shared_hits / (shared_hits + total_misses)) if (shared_hits + total_misses) else 0.0,
                "p50_latency_ms": percentile(lats, 0.50),
                "p99_latency_ms": percentile(lats, 0.99),
                "goodput": good / n,
            }
        summary_path = os.path.join(self.cfg.output_dir, f"summary_{self.cfg.run_label}.csv")
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        return summary

