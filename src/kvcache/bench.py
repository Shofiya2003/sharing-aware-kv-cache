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
       - classifies hit/miss based on observed latency
       - records a `RequestRecord` into the `MetricsLogger`
       - touches the overlap index with the request's tokens so future
         sharing-aware decisions can see them
       - flushes any completed time-window

Note on hit/miss classification: vLLM does not currently expose per-
request cache hit/miss counters in a stable, queryable way. We use
*latency* as a proxy. Threshold is calibrated in Phase 1.
"""

from __future__ import annotations

import asyncio
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
    hit_latency_threshold_ms: float = 300.0

    output_dir: str = "results/csv"
    run_label: str = "run"


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


def _classify_hit(latency_ms: float, threshold_ms: float) -> bool:
    return latency_ms <= threshold_ms


def _record_request(
    qr: QueuedRequest,
    res: RequestResult,
    cfg: BenchConfig,
    metrics: MetricsLogger,
    overlap: OverlapIndex,
    in_flight_at_submit: int,
    sim_complete_t: float,
) -> None:
    # Use wall-clock times for latency; res.submit_t is the simulation time
    # supplied by the driver, not a wall clock.
    wall_latency_ms = (res.complete_t - res.submit_wall_t) * 1000.0
    latency_ms = wall_latency_ms
    hit = _classify_hit(latency_ms, cfg.hit_latency_threshold_ms)
    other_refs = set()
    for gram in ngrams(qr.event.tokens, overlap.n):
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
        )
    )
    overlap.touch_session(qr.event.session_id, qr.event.tokens)


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

            # 3. Re-score the queue with the active policy
            decision = policy.score_queue(queue, sessions, overlap, sim_now)
            queue = list(decision.ordered)

            # 4. Dispatch up to vLLM's concurrency limit
            max_in_flight = cfg.backend.max_num_seqs
            while queue and len(in_flight) < max_in_flight:
                head = queue.pop(0)
                in_flight_count = len(in_flight)
                prompt = tokens_to_text(head.event.tokens, 4000)
                rid = await backend.submit(
                    prompt=prompt,
                    session_id=head.event.session_id,
                    turn_index=head.event.turn_index,
                    submit_t=head.event.t,
                    max_new_tokens=cfg.max_new_tokens,
                )
                in_flight[rid] = head
                if len(in_flight) >= max_in_flight:
                    break

            # 5. Wait for at least one in-flight request to complete
            if in_flight:
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
                                try:
                                    res = await backend.wait(rid)
                                except Exception as e:  # noqa: BLE001
                                    res = RequestResult(
                                        request_id=rid,
                                        text="",
                                        submit_t=qr.event.t,
                                        submit_wall_t=time.monotonic(),
                                        first_token_t=0.0,
                                        complete_t=time.monotonic(),
                                        n_output_tokens=0,
                                        n_prompt_tokens=0,
                                        error=repr(e),
                                    )
                                _record_request(
                                    qr, res, cfg, metrics, overlap, len(in_flight) + 1, sim_now
                                )
                                break
                else:
                    await asyncio.sleep(0.0)
            else:
                await asyncio.sleep(0.0)

            # 6. Flush completed windows based on simulated time.
            # We flush window W only when sim_now has crossed (W+1) * window_s,
            # which means W is fully past. Window 0 is always flushed by
            # `finalize` so we don't have to worry about partial-row races.
            current_window = int(sim_now // metrics.cfg.window_s)
            if last_window_flushed >= 0:
                while (last_window_flushed + 1) * metrics.cfg.window_s < sim_now:
                    last_window_flushed += 1
                    metrics.flush_window(last_window_flushed)
    finally:
        summary = metrics.finalize(sim_window_s=sim_window)
        if should_stop:
            await backend.stop()

    return RunSummary(config=cfg, summary=summary, n_records=len(metrics.records))


# ---------------------------------------------------------------------------
# Mock backend (for unit tests + CPU-only dev)
# ---------------------------------------------------------------------------


class MockVLLMBackend:
    """A drop-in replacement for VLLMBackend that simulates latency.

    Latency depends on prompt length: short prompts get a fast latency
    (a "hit"), long prompts get a slow latency (a "miss"). This produces
    qualitatively similar behavior to a real vLLM under memory pressure
    for the purpose of validating the driver + policies + metrics.

    Used by:
      - unit tests in `tests/`
      - any local CPU-only development
      - the optional `--mock` flag in `experiments/run_experiment.py`
    """

    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg
        self._started = False
        self._pending: Dict[str, asyncio.Future] = {}

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
        n_prompt_tokens = max(1, len(prompt.split()))
        # Crude latency model: short => hit; long => miss
        if n_prompt_tokens < 80:
            latency = 0.005
        elif n_prompt_tokens < 200:
            latency = 0.05
        else:
            latency = 0.2
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

