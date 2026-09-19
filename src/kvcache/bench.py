"""Benchmark harness: drives a workload through a dispatch policy + vLLM.

This is the orchestrator. It is the only module in the project that knows
about both the workload stream and the live vLLM backend.

Per-workload-event flow:

  1. A TurnEvent becomes "available" at its `t` (simulation time).
  2. The driver pushes a `QueuedRequest` for it into the priority queue.
  3. Before each dispatch, the driver asks the policy to score the queue.
  4. The driver submits the head of the queue to vLLM, up to
     `max_num_seqs - in_flight`.
  5. As requests complete, the driver:
       - reads vLLM's ground-truth `num_cached_tokens` for the request
       - records a `RequestRecord` into the `MetricsLogger`
       - touches the overlap index with the request's tokens so future
         sharing-aware decisions can see them
       - flushes any completed time-window

Hit/miss classification
-----------------------
We use vLLM's own per-request prefix-cache counter, `num_cached_tokens`
on `RequestOutput`: the number of prompt tokens served from an existing
KV block instead of being prefilled. This is ground truth, not a proxy.

  token-level hit rate  = sum(num_cached_tokens) / sum(n_prompt_tokens)
  per-request hit       = cached_fraction >= cfg.hit_cached_fraction

The token-level rate is the headline metric (it is the same quantity
vLLM reports as `gpu_prefix_cache_hit_rate`, and `VLLMBackend.
prefix_cache_hit_rate()` is logged alongside it as a cross-check). The
per-request binary is only used where a per-request outcome is needed:
the per-session fairness ECDF.

An earlier version of this harness thresholded end-to-end *latency* to
guess hits. That was wrong and the results it produced are not
comparable: with `max_new_tokens=24` the latency distribution is
decode-dominated and spans ~660-1100ms as a single mode, so a threshold
placed inside it (797ms was used) measured batch-queueing jitter, not
cache reuse. A 160ms shift in median latency swung the reported "hit
rate" from 1.00 to 0.19. The latency proxy is still computed, but only
as a labelled diagnostic column (`proxy_hit_rate`) so the two can be
compared directly in the writeup.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .metrics import MetricsConfig, MetricsLogger, RequestRecord
from .overlap import OverlapIndex, ngrams, ngram_id
from .policies import (
    CombinedPolicy,
    DispatchPolicy,
    FIFOPolicy,
    QueuedRequest,
    SessionAwarePolicy,
    SharingAwarePolicy,
)
from .session import Session
from .vllm_backend import BackendConfig, RequestResult, VLLMBackend, tokens_to_text
from .workload import Workload


class OverloadedRun(RuntimeError):
    """Raised when the dispatch backlog proves offered load exceeds capacity.

    Not a crash: it means the configuration, not the code, is wrong. The
    run is abandoned deliberately because its numbers would describe the
    queue rather than the cache.
    """


class EngineFailure(RuntimeError):
    """Raised when the engine stops serving: every request comes back failed.

    Round 2's `fifo_generous_s2` is what this guards against. The engine
    died early, every later request failed instantly, and the run went on
    to record ~450 "completions" at ~0 ms that the latency proxy scored as
    hits. Retrying on a fresh engine is the right response, so the caller
    gets a distinct exit code instead of a finished-looking CSV.
    """


# Consecutive failed requests after which the engine is presumed dead.
# A healthy engine does not fail requests at all; five in a row is not
# a transient.
ENGINE_DEAD_AFTER = 5


@dataclass
class BenchConfig:
    """Knobs for one benchmark run."""

    policy_name: str = "fifo"
    combined_alpha: float = 0.5

    backend: BackendConfig = field(default_factory=BackendConfig)
    capacity_setting: str = "constrained"

    max_new_tokens: int = 24
    speed_factor: float = 1.0  # sim seconds per real second (1.0 = real time)

    sla_latency_ms: float = 2000.0
    # Legacy latency proxy. Retained ONLY to emit the `proxy_*` diagnostic
    # columns for comparison against ground truth; it no longer drives any
    # headline metric. See the module docstring.
    hit_latency_threshold_ms: float = 300.0
    # A request counts as a cache hit when at least this fraction of its
    # prompt tokens were served from an existing KV block. 0.10 keeps
    # trivial block-boundary reuse from registering as a hit.
    hit_cached_fraction: float = 0.10

    output_dir: str = "results/csv"
    run_label: str = "run"
    # Windows to drop from the summary as engine warmup. The first window
    # of a run carries one-time model-load / CUDA-graph-capture cost,
    # which is what made run order (not policy) dominate the old P99
    # numbers: FIFO ran first in the matrix and absorbed it.
    discard_warmup_windows: int = 1

    # Abort the run when the oldest request has been sitting in OUR dispatch
    # queue longer than this (wall seconds). 0 disables the watchdog.
    #
    # Purpose: when offered load exceeds engine capacity the backlog grows
    # without bound and the run stops measuring cache behaviour -- requests
    # get dispatched so late that the session prefix is long evicted, so the
    # hit rate tracks queue depth instead of the policy. Left unattended a
    # 24-run matrix will happily spend six GPU-hours producing that. Failing
    # the first run loudly is cheaper than discovering it in the CSVs.
    abort_on_backlog_s: float = 0.0


def make_policy(name: str, alpha: float = 0.5) -> DispatchPolicy:
    if name == "fifo":
        return FIFOPolicy()
    if name == "session-aware":
        return SessionAwarePolicy()
    if name == "sharing-aware":
        return SharingAwarePolicy()
    if name == "combined":
        return CombinedPolicy(alpha=alpha)
    raise ValueError(f"unknown policy: {name}")


def _classify_hit(res: RequestResult, cfg: BenchConfig) -> tuple:
    """Classify one completed request. Returns `(hit, basis)`.

    Ground truth first: a request is a hit when vLLM reports that at least
    `cfg.hit_cached_fraction` of its prompt tokens came from an existing KV
    block. Only when the engine reports no counter at all do we fall back
    to the legacy latency threshold, and we label the basis so the output
    CSV says plainly which definition produced the number.
    """
    if res.has_cache_ground_truth:
        return res.cached_fraction >= cfg.hit_cached_fraction, "cached_tokens"
    return (res.latency_ms <= cfg.hit_latency_threshold_ms), "latency_proxy"


def _classify_hit_latency_proxy(latency_ms: float, threshold_ms: float) -> bool:
    """The old, discredited definition. Kept to emit `proxy_*` diagnostics."""
    return latency_ms <= threshold_ms


def _record_request(
    qr: QueuedRequest,
    res: RequestResult,
    cfg: BenchConfig,
    metrics: MetricsLogger,
    overlap: OverlapIndex,
    in_flight_at_submit: int,
    sim_complete_t: float,
    arrival_wall_t: Optional[float] = None,
) -> None:
    # Two different latencies, and the distinction matters.
    #
    # `latency_ms` (engine latency) runs from the moment we handed the
    # request to vLLM to completion. It does NOT include time the request
    # spent waiting in OUR dispatch queue.
    #
    # `e2e_latency_ms` runs from the moment the user's turn became
    # available (`qr.event.t`, mapped to wall clock) to completion. That is
    # what a user actually experiences.
    #
    # Reporting only engine latency systematically flatters any policy that
    # reorders aggressively: a request the policy deprioritises waits in the
    # dispatch queue, and that wait is invisible. Starvation would show up
    # as *better* latency. Goodput is therefore computed on e2e.
    wall_latency_ms = (res.complete_t - res.submit_wall_t) * 1000.0
    latency_ms = wall_latency_ms
    if arrival_wall_t is not None:
        e2e_latency_ms = (res.complete_t - arrival_wall_t) * 1000.0
        queue_wait_ms = max(0.0, (res.submit_wall_t - arrival_wall_t) * 1000.0)
    else:
        e2e_latency_ms = latency_ms
        queue_wait_ms = 0.0
    hit, hit_basis = _classify_hit(res, cfg)
    proxy_hit = _classify_hit_latency_proxy(
        latency_ms, cfg.hit_latency_threshold_ms
    )
    other_refs = set()
    for gram in ngrams(qr.event.prompt_tokens, overlap.n):
        gid = ngram_id(gram)
        for sid in overlap._refs.get(gid, ()):  # noqa: SLF001
            if sid != qr.event.session_id:
                other_refs.add(sid)
    shared = len(other_refs) > 0
    # Bucket by the event's request time (the moment the user wanted
    # the turn), not the completion time. This way a request's row
    # belongs to the window its prompt was issued in, which is the
    # semantically meaningful "when in the simulation did this happen?"
    metrics.record(
        RequestRecord(
            request_id=res.request_id,
            session_id=qr.event.session_id,
            turn_index=qr.event.turn_index,
            submit_t=qr.event.t,
            complete_t=res.complete_t,
            latency_ms=latency_ms,
            n_prompt_tokens=res.n_prompt_tokens,
            n_output_tokens=res.n_output_tokens,
            hit=hit,
            shared=shared,
            policy_name=cfg.policy_name,
            capacity_setting=cfg.capacity_setting,
            in_flight_at_submit=in_flight_at_submit,
            num_cached_tokens=res.num_cached_tokens,
            hit_basis=hit_basis,
            proxy_hit=proxy_hit,
            context_truncated=getattr(qr.event, "context_truncated", False),
            e2e_latency_ms=e2e_latency_ms,
            queue_wait_ms=queue_wait_ms,
            error=res.error is not None,
        )
    )
    overlap.touch_session(qr.event.session_id, qr.event.prompt_tokens)


def _set_aside_aborted(cfg: BenchConfig, reason: str) -> None:
    """Move an aborted run's CSVs out of the results directory.

    An abandoned run still gets finalized (so its partial numbers can be
    inspected), but it must not sit in `output_dir`: the launcher treats
    any `summary_<label>.csv` there as finished and would never re-run it,
    and the analysis would pool it.
    """
    dest = os.path.join(cfg.output_dir, "_aborted")
    os.makedirs(dest, exist_ok=True)
    for kind in ("summary", "time_series", "per_session"):
        src = os.path.join(cfg.output_dir, f"{kind}_{cfg.run_label}.csv")
        if os.path.exists(src):
            os.replace(src, os.path.join(dest, os.path.basename(src)))
    print(f"[bench:{cfg.run_label}] aborted ({reason}); CSVs moved to {dest}/ "
          f"so the label is re-run, not counted as done.")


@dataclass
class RunSummary:
    config: BenchConfig
    summary: Dict[str, float]
    n_records: int


async def run_benchmark(
    workload: Workload,
    cfg: BenchConfig,
    backend: Optional[object] = None,
) -> RunSummary:
    """Run the benchmark end-to-end.

    If `backend` is None, a real `VLLMBackend` is created from
    `cfg.backend` and started. If `backend` is provided (e.g. a
    `MockVLLMBackend`), it is used as-is. Either way, the backend must
    be started *before* this call (or `start_backend=True` style flag,
    but here we just expect it to be live by the time we use it).
    """
    if backend is None:
        backend = VLLMBackend(cfg.backend)
        await backend.start()
        should_stop = True
    else:
        should_stop = False

    sessions: Dict[str, Session] = {s.session_id: s for s in workload.sessions}
    policy = make_policy(cfg.policy_name, alpha=cfg.combined_alpha)
    overlap = OverlapIndex(n=8)

    metrics = MetricsLogger(
        MetricsConfig(
            window_s=30.0,
            sla_latency_ms=cfg.sla_latency_ms,
            hit_latency_threshold_ms=cfg.hit_latency_threshold_ms,
            output_dir=cfg.output_dir,
            run_label=cfg.run_label,
            discard_warmup_windows=cfg.discard_warmup_windows,
        )
    )

    queue: List[QueuedRequest] = []
    in_flight: Dict[str, QueuedRequest] = {}
    in_flight_results: Dict[str, RequestResult] = {}
    seq = 0

    sim_window = workload.config.sim_window_s
    speed = max(1e-3, cfg.speed_factor)
    sim_start_wall = time.monotonic()
    last_window_flushed = -1
    n_dispatched = 0
    n_completed = 0
    n_failed = 0
    consecutive_failures = 0
    n_events = 0
    aborted: Optional[BaseException] = None
    sim_now = 0.0

    async def _reap(rid: str, qr: QueuedRequest) -> None:
        """Collect one finished request and record it, failed or not."""
        nonlocal n_completed, n_failed, consecutive_failures
        try:
            res = await backend.wait(rid)
        except Exception as e:  # noqa: BLE001
            now = time.monotonic()
            res = RequestResult(
                request_id=rid,
                text="",
                submit_t=qr.event.t,
                submit_wall_t=now,
                first_token_t=0.0,
                complete_t=now,
                n_output_tokens=0,
                n_prompt_tokens=0,
                error=repr(e),
            )
        _record_request(
            qr, res, cfg, metrics, overlap, len(in_flight) + 1, sim_now,
            arrival_wall_t=sim_start_wall + qr.event.t / speed,
        )
        n_completed += 1
        if res.error is not None:
            n_failed += 1
            consecutive_failures += 1
            if n_failed <= 3:
                print(f"[bench:{cfg.run_label}] request {rid} FAILED: "
                      f"{res.error[:300]}", flush=True)
            if consecutive_failures >= ENGINE_DEAD_AFTER:
                raise EngineFailure(
                    f"[bench:{cfg.run_label}] ABORT: {consecutive_failures} "
                    f"consecutive requests failed ({n_failed} total); the "
                    f"engine is no longer serving. Last error: "
                    f"{res.error[:300]}"
                )
        else:
            consecutive_failures = 0
        if n_completed == 1 or n_completed % 50 == 0:
            print(f"[bench:{cfg.run_label}] progress: "
                  f"completed={n_completed} failed={n_failed} "
                  f"dispatched={n_dispatched} "
                  f"enqueued={ei}/{n_events} in_flight={len(in_flight)} "
                  f"queue={len(queue)} sim_t={sim_now:.1f}s")
    print(f"[bench:{cfg.run_label}] start: policy={cfg.policy_name} "
          f"capacity={cfg.capacity_setting} events={len(workload.events)} "
          f"sessions={len(workload.sessions)} sim_window={sim_window:.0f}s "
          f"speed x{speed} max_new_tokens={cfg.max_new_tokens} "
          f"sla={cfg.sla_latency_ms:.0f}ms "
          f"hit=cached_fraction>={cfg.hit_cached_fraction:.2f} (ground truth) "
          f"discard_warmup_windows={cfg.discard_warmup_windows} "
          f"[latency proxy {cfg.hit_latency_threshold_ms:.0f}ms kept as diagnostic only]")

    try:
        ei = 0
        n_events = len(workload.events)
        while ei < n_events or in_flight or queue:
            # 1. Advance simulation time
            target_sim_t = workload.events[ei].t if ei < n_events else sim_window
            target_wall = sim_start_wall + target_sim_t / speed
            now_wall = time.monotonic()
            if now_wall < target_wall:
                await asyncio.sleep(min(0.05, target_wall - now_wall))
            sim_now = (time.monotonic() - sim_start_wall) * speed
            # Cap sim_now at sim_window: once we've passed the workload's
            # last event, the simulation is "done" — additional wall time
            # spent waiting for in-flight requests should not advance the
            # simulation clock. The remaining latency / completion is
            # recorded against the final window.
            sim_now = min(sim_now, sim_window)

            # 2. Enqueue all newly-available events
            while ei < n_events and workload.events[ei].t <= sim_now:
                ev = workload.events[ei]
                queue.append(QueuedRequest(event=ev, arrival_t=ev.t, enqueue_seq=seq))
                seq += 1
                ei += 1

            # 2b. Backlog watchdog. `queue` is OUR dispatch queue, so the
            # oldest entry's wall-clock wait is the clearest signal that
            # offered load has outrun the engine. Checked before scoring so
            # an unrecoverable run dies in minutes, not in twenty.
            if cfg.abort_on_backlog_s > 0 and queue:
                oldest_arrival_wall = sim_start_wall + min(
                    q.arrival_t for q in queue) / speed
                backlog_s = time.monotonic() - oldest_arrival_wall
                if backlog_s > cfg.abort_on_backlog_s:
                    raise OverloadedRun(
                        f"[bench:{cfg.run_label}] ABORT: oldest queued request "
                        f"has waited {backlog_s:.0f}s (limit "
                        f"{cfg.abort_on_backlog_s:.0f}s). Offered load exceeds "
                        f"engine capacity, so this run would measure queue "
                        f"depth rather than cache reuse. Lower --speed-factor "
                        f"or --num-sessions and re-run; "
                        f"experiments/calibrate_load.py picks a safe value."
                    )

            # 3. Re-score the queue with the active policy
            decision = policy.score_queue(queue, sessions, overlap, sim_now)
            queue = list(decision.ordered)

            # 4. Dispatch up to vLLM's concurrency limit
            max_in_flight = cfg.backend.max_num_seqs
            while queue and len(in_flight) < max_in_flight:
                head = queue.pop(0)
                in_flight_count = len(in_flight)
                # Submit the full accumulated conversation, not just this
                # turn's delta -- see TurnEvent.context_tokens.
                prompt = tokens_to_text(head.event.prompt_tokens, 4000)
                rid = await backend.submit(
                    prompt=prompt,
                    session_id=head.event.session_id,
                    turn_index=head.event.turn_index,
                    submit_t=head.event.t,
                    max_new_tokens=cfg.max_new_tokens,
                )
                in_flight[rid] = head
                n_dispatched += 1
                if len(in_flight) >= max_in_flight:
                    break

            # 5. Collect completions. Two sub-steps:
            #   (a) reap futures that are ALREADY done (no waiting). This
            #       must come first: a future that finished between dispatch
            #       and this scan would otherwise be skipped by the pending
            #       filter below and sit stale in `in_flight` forever,
            #       wedging the driver in a sleep(0) spin.
            #   (b) if anything is still in flight, wait for the next
            #       completion. Drain ALL done futures (no `break`): with
            #       concurrent backends several requests routinely finish in
            #       the same wait window.
            if in_flight:
                for rid in list(in_flight.keys()):
                    fut = getattr(backend, "_pending", {}).get(rid)
                    if fut is None or not fut.done():
                        continue
                    qr = in_flight.pop(rid, None)
                    if qr is None:
                        continue
                    await _reap(rid, qr)

                wait_tasks = []
                for rid in list(in_flight.keys()):
                    fut = getattr(backend, "_pending", {}).get(rid)
                    if fut is not None and not fut.done():
                        wait_tasks.append((rid, fut))
                if wait_tasks:
                    done, _ = await asyncio.wait(
                        [t for _, t in wait_tasks],
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=5.0,
                    )
                    for fut in done:
                        for rid, f in wait_tasks:
                            if f is fut:
                                qr = in_flight.pop(rid, None)
                                if qr is None:
                                    break
                                await _reap(rid, qr)
                                break
                    if not done:
                        await asyncio.sleep(0.0)
                else:
                    # In-flight rids with no pending future in the backend
                    # (should not happen) — yield to avoid a hot spin.
                    await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.0)

            # 6. Flush completed windows based on simulated time.
            # We flush window W only when sim_now has crossed (W+1) * window_s,
            # which means W is fully past. Window 0 is always flushed by
            # `finalize` so we don't have to worry about partial-row races.
            # NOTE: flush_window() is a metadata touch only (no I/O); the
            # time-series CSV is written atomically at finalize, so late
            # stragglers for the same window are still counted.
            current_window = int(sim_now // metrics.cfg.window_s)
            while last_window_flushed < current_window - 1:
                last_window_flushed += 1
                metrics.flush_window(last_window_flushed)
                if last_window_flushed % 5 == 0:
                    print(f"[bench:{cfg.run_label}] window {last_window_flushed} closed "
                          f"(sim_t={sim_now:.1f}s completed={n_completed})")
    except (OverloadedRun, EngineFailure) as e:
        aborted = e
        raise
    finally:
        print(f"[bench:{cfg.run_label}] finalizing: dispatched={n_dispatched} "
              f"completed={n_completed} failed={n_failed} events={n_events}")
        # Cross-check our summed per-request counters against the engine's
        # own cumulative prefix-cache accounting, BEFORE stopping it. If
        # these two disagree materially, our per-request reads are wrong
        # and the run should not be reported.
        engine_hr = None
        try:
            getter = getattr(backend, "prefix_cache_hit_rate", None)
            engine_hr = getter() if callable(getter) else None
        except Exception:  # noqa: BLE001
            engine_hr = None
        summary = metrics.finalize(sim_window_s=sim_window)
        summary["engine_prefix_cache_hit_rate"] = (
            float(engine_hr) if engine_hr is not None else -1.0
        )
        basis = summary.get("hit_basis", "?")
        cov = summary.get("cache_ground_truth_coverage", 0.0)
        print(f"[bench:{cfg.run_label}] hit_basis={basis} "
              f"ground_truth_coverage={cov:.1%} "
              f"cached_token_rate={summary.get('cached_token_rate', 0):.4f} "
              f"(engine says {engine_hr if engine_hr is not None else 'n/a'}) "
              f"| legacy latency proxy would report "
              f"{summary.get('proxy_hit_rate', 0):.4f}")
        if basis != "cached_tokens":
            print(f"[bench:{cfg.run_label}] *** WARNING: hit rate is NOT "
                  f"ground truth (basis={basis}). Do not report these "
                  f"numbers as cache hit rates. ***")
        # Re-write the summary CSV now that the engine cross-check is in it.
        import pandas as _pd
        _pd.DataFrame([summary]).to_csv(
            os.path.join(cfg.output_dir, f"summary_{cfg.run_label}.csv"),
            index=False,
        )
        if aborted is not None:
            _set_aside_aborted(cfg, type(aborted).__name__)
        if should_stop:
            await backend.stop()

    return RunSummary(config=cfg, summary=summary, n_records=len(metrics.records))


# ---------------------------------------------------------------------------
# Mock backend (for unit tests + CPU-only dev)
# ---------------------------------------------------------------------------


class MockVLLMBackend:
    """A drop-in replacement for VLLMBackend with a simulated prefix cache.

    This models the one thing the experiment actually measures: an LRU
    pool of KV blocks keyed by prompt prefix, with a finite capacity
    derived from `gpu_memory_utilization`. On each submit it walks the
    prompt block by block, counts the leading blocks already resident as
    `num_cached_tokens`, then inserts the whole prompt and evicts LRU
    blocks past capacity.

    That makes the mock exercise the real code path: ground-truth cached
    tokens respond to capacity pressure and to submission order, so the
    policies and metrics can be validated on CPU. It is still a model,
    not vLLM -- notably it is prefix-aligned, where vLLM's hashing is
    too, but our workload deliberately places shared content mid-context.

    Used by:
      - unit tests in `tests/`
      - any local CPU-only development
      - the optional `--mock` flag in `experiments/run_experiment.py`
    """

    BLOCK = 16  # tokens per KV block, matching vLLM's default

    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg
        self._started = False
        self._pending: Dict[str, asyncio.Future] = {}
        # Simulated KV block pool: block-hash -> insertion counter (LRU).
        self._blocks: Dict[int, int] = {}
        self._clock = 0
        # Capacity in blocks, scaled off gpu_memory_utilization so that
        # "constrained" vs "generous" is a real, observable difference.
        self.capacity_blocks = max(
            8, int(4000 * self.cfg.gpu_memory_utilization)
        )
        self.n_evictions = 0

    def _lookup_and_insert(self, prompt: str) -> tuple:
        """Return (num_cached_tokens, n_prompt_tokens) and update the pool."""
        toks = prompt.split()
        n = len(toks)
        n_blocks = n // self.BLOCK
        # Walk the prefix: vLLM can only reuse a contiguous leading run.
        cached_blocks = 0
        h = 0
        hashes = []
        for b in range(n_blocks):
            chunk = " ".join(toks[b * self.BLOCK:(b + 1) * self.BLOCK])
            h = hash((h, chunk))  # chained hash, like vLLM's block hashing
            hashes.append(h)
        for bh in hashes:
            if bh in self._blocks:
                cached_blocks += 1
            else:
                break
        # Touch/insert every block of this prompt.
        for bh in hashes:
            self._clock += 1
            self._blocks[bh] = self._clock
        # Evict LRU past capacity.
        while len(self._blocks) > self.capacity_blocks:
            victim = min(self._blocks, key=self._blocks.get)
            del self._blocks[victim]
            self.n_evictions += 1
        return cached_blocks * self.BLOCK, n

    def prefix_cache_hit_rate(self):  # parity with VLLMBackend
        return None

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        self._started = False

    @property
    def is_started(self) -> bool:
        return self._started

    def num_in_flight(self) -> int:
        return sum(0 if t.done() else 1 for t in self._pending.values())

    async def submit(
        self,
        prompt: str,
        session_id: str,
        turn_index: int,
        submit_t: float,
        max_new_tokens: int = 32,
        sampling_params: Optional[Dict] = None,
    ) -> str:
        if not self._started:
            raise RuntimeError("MockVLLMBackend.start() must be called first")
        import uuid
        import random as _r

        rid = f"{session_id}__t{turn_index}__{uuid.uuid4().hex[:6]}"
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[rid] = fut

        submit_wall = time.monotonic()
        n_cached, n_prompt_tokens = self._lookup_and_insert(prompt)
        n_prompt_tokens = max(1, n_prompt_tokens)
        # Latency model driven by the tokens that actually had to be
        # prefilled, plus a fixed decode cost. This is only here so the
        # driver has something to wait on -- latency no longer defines
        # hit/miss, `n_cached` does.
        to_prefill = max(0, n_prompt_tokens - n_cached)
        latency = 0.02 + 0.0004 * to_prefill
        latency *= 1.0 + _r.uniform(-0.15, 0.25)

        async def _finish():
            await asyncio.sleep(latency)
            complete_wall = time.monotonic()
            res = RequestResult(
                request_id=rid,
                text="ok",
                submit_t=submit_t,
                submit_wall_t=submit_wall,
                first_token_t=complete_wall - latency + 0.01,
                complete_t=complete_wall,
                n_output_tokens=min(max_new_tokens, 24),
                n_prompt_tokens=n_prompt_tokens,
                num_cached_tokens=n_cached,
                cached_tokens_source="MockVLLMBackend.simulated_prefix_cache",
            )
            if not fut.done():
                fut.set_result(res)

        asyncio.create_task(_finish())
        return rid

    async def wait(self, request_id: str, timeout: Optional[float] = None) -> RequestResult:
        fut = self._pending[request_id]
        if timeout is None:
            return await fut
        return await asyncio.wait_for(fut, timeout=timeout)

