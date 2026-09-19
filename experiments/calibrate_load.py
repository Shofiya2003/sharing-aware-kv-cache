"""Measure sustained engine capacity, then pick a safe arrival rate.

Why this exists
---------------
Round 2's matrix ran with `--speed-factor 1` on the reasoning that demand
(~3500 prefill tok/s) was about half a T4's throughput. The CSVs say
otherwise: p50 dispatch-queue wait climbed monotonically window over
window (32s -> 148s -> 545s in `fifo_constrained`) and in-flight sat
pinned at `max_num_seqs` the whole run. Offered load was roughly 2.8x
what the engine could actually serve.

That matters beyond latency. Once the backlog exceeds a session's idle
gap, requests are dispatched so long after their turn that the session's
prefix has already been evicted, so the measured hit rate decays with
queue depth. `cached_token_rate` fell 0.90 -> 0.49 across windows in
every run. Part of what the round-2 numbers describe is the queue.

The paper-throughput estimate was wrong because it ignored preemption:
at `gpu_memory_utilization=0.3` the KV budget cannot hold 16 concurrent
sequences of ~1600 tokens plus any reusable history, so vLLM preempts
and recomputes, and effective prefill throughput collapses well below
the number a microbenchmark would show.

So measure it instead of estimating it. This script replays a closed
loop at the real concurrency and prompt length and reports the sustained
request rate, then prints the `--speed-factor` that puts offered load at
a target utilization (default 0.6).

Usage
-----
    python experiments/calibrate_load.py --gpu-memory 0.3 --max-num-seqs 8

Run it for the CONSTRAINED capacity: that arm is the bottleneck, and the
matrix must use one arrival rate for both arms or the capacity
comparison is confounded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics as st
import sys
import time

from kvcache.vllm_backend import BackendConfig, VLLMBackend, tokens_to_text
from kvcache.workload import WorkloadConfig, generate_workload


async def measure_capacity(
    gpu_memory: float,
    max_num_seqs: int,
    max_model_len: int,
    max_new_tokens: int,
    prompts: list,
    duration_s: float,
) -> dict:
    """Closed-loop: keep exactly `max_num_seqs` requests in flight.

    Closed loop, not open loop, on purpose: we want the engine's service
    rate at saturation, which is precisely the quantity the arrival rate
    has to stay below. An open-loop probe would just reproduce the
    unbounded queue we are trying to avoid measuring.
    """
    backend = VLLMBackend(BackendConfig(
        gpu_memory_utilization=gpu_memory,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    ))
    await backend.start()
    try:
        # Untimed warmup so kernel compilation is not counted as service time.
        for i in range(2):
            rid = await backend.submit(prompt="warmup probe request",
                                       session_id=f"__warm{i}__", turn_index=0,
                                       submit_t=0.0, max_new_tokens=4)
            await backend.wait(rid)
        backend.snapshot_prefix_cache_baseline()

        completed = 0
        prompt_tokens = 0
        latencies = []
        pending = set()
        idx = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration_s:
            while len(pending) < max_num_seqs:
                p = prompts[idx % len(prompts)]
                idx += 1
                rid = await backend.submit(prompt=p, session_id=f"cal{idx}",
                                           turn_index=0, submit_t=0.0,
                                           max_new_tokens=max_new_tokens)
                pending.add(asyncio.ensure_future(backend.wait(rid)))
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED)
            for fut in done:
                res = fut.result()
                completed += 1
                prompt_tokens += max(0, res.n_prompt_tokens)
                latencies.append(res.latency_ms)
        elapsed = time.monotonic() - t0
        for fut in pending:
            fut.cancel()
        engine_rate = backend.prefix_cache_hit_rate()
    finally:
        await backend.stop()

    return {
        "elapsed_s": elapsed,
        "completed": completed,
        "req_per_s": completed / elapsed if elapsed else 0.0,
        "prompt_tok_per_s": prompt_tokens / elapsed if elapsed else 0.0,
        "mean_prompt_tokens": prompt_tokens / completed if completed else 0.0,
        "p50_latency_ms": st.median(latencies) if latencies else 0.0,
        "engine_prefix_cache_hit_rate": engine_rate,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-memory", type=float, default=0.3,
                   help="Use the CONSTRAINED value: it is the binding arm.")
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--duration-s", type=float, default=90.0)
    p.add_argument("--target-utilization", type=float, default=0.6,
                   help="Offered load as a fraction of measured capacity. "
                        "0.6 leaves headroom for the burstiness the workload "
                        "generator introduces; above ~0.8 the queue grows "
                        "during bursts and never fully drains.")
    # The workload the matrix will actually run, so prompt lengths match.
    p.add_argument("--num-sessions", type=int, default=12)
    p.add_argument("--sim-window", type=float, default=600.0)
    p.add_argument("--max-context-tokens", type=int, default=3072)
    p.add_argument("--turn-min-tokens", type=int, default=32)
    p.add_argument("--turn-max-tokens", type=int, default=96)
    p.add_argument("--mean-active-burst-turns", type=float, default=4.0)
    p.add_argument("--mean-idle-gap-s", type=float, default=25.0)
    p.add_argument("--json-out", default="",
                   help="Also write the measurement here. The launcher passes "
                        "`arrival_rate_req_s` to run_experiment.py as "
                        "--target-arrival-rate, which derives the speed "
                        "factor per workload.")
    args = p.parse_args()

    wl = generate_workload(WorkloadConfig(
        num_sessions=args.num_sessions,
        sim_window_s=args.sim_window,
        seed=0,
        max_context_tokens=args.max_context_tokens,
        turn_min_tokens=args.turn_min_tokens,
        turn_max_tokens=args.turn_max_tokens,
        mean_active_burst_turns=args.mean_active_burst_turns,
        mean_idle_gap_s=args.mean_idle_gap_s,
    ))
    # Sample representative prompts from the real workload, biased to the
    # back of each session where contexts are longest -- capacity at the
    # mean prompt length is what the matrix will experience.
    prompts = [tokens_to_text(e.prompt_tokens, 4000)
               for e in wl.events[len(wl.events) // 4:]][:64]
    if not prompts:
        print("[calibrate] workload produced no events; check the knobs")
        return 1

    print(f"[calibrate] workload: {len(wl.sessions)} sessions, "
          f"{len(wl.events)} events over {args.sim_window:.0f} sim-s")
    print(f"[calibrate] probing capacity at gpu_mem={args.gpu_memory} "
          f"max_num_seqs={args.max_num_seqs} for {args.duration_s:.0f}s ...")
    m = asyncio.run(measure_capacity(
        args.gpu_memory, args.max_num_seqs, args.max_model_len,
        args.max_new_tokens, prompts, args.duration_s))

    print()
    print("=" * 70)
    print("MEASURED SUSTAINED CAPACITY (closed loop, at saturation)")
    print("=" * 70)
    print(f"  completed             : {m['completed']} in {m['elapsed_s']:.0f}s")
    print(f"  service rate          : {m['req_per_s']:.3f} req/s")
    print(f"  prefill throughput    : {m['prompt_tok_per_s']:.0f} prompt tok/s")
    print(f"  mean prompt tokens    : {m['mean_prompt_tokens']:.0f}")
    print(f"  p50 engine latency    : {m['p50_latency_ms']:.0f} ms")
    print(f"  engine prefix hit rate: {m['engine_prefix_cache_hit_rate']}")

    mu = m["req_per_s"]
    if mu <= 0:
        print("[calibrate] no throughput measured; not recommending a rate")
        return 1
    target_lambda = mu * args.target_utilization
    # events / (sim_window / speed) = target_lambda  =>  speed = ...
    speed = target_lambda * args.sim_window / len(wl.events)
    wall_s = args.sim_window / speed

    print()
    print("=" * 70)
    print("RECOMMENDED ARRIVAL RATE")
    print("=" * 70)
    print(f"  target utilization    : {args.target_utilization:.2f}")
    print(f"  target arrival rate   : {target_lambda:.3f} req/s")
    print(f"  --speed-factor        : {speed:.3f}")
    print(f"  wall time per run     : {wall_s:.0f}s "
          f"(+ ~60s engine start) => ~{(wall_s + 60) * 24 / 3600:.1f}h "
          f"for a 4x2x3 matrix")
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump({
                # The quantity that transfers across workloads.
                "arrival_rate_req_s": target_lambda,
                "service_rate_req_s": mu,
                "target_utilization": args.target_utilization,
                "gpu_memory": args.gpu_memory,
                "max_num_seqs": args.max_num_seqs,
                "mean_prompt_tokens": m["mean_prompt_tokens"],
                # Valid ONLY for the workload knobs above; kept for reference.
                "speed_factor": speed,
            }, fh, indent=2)
        print(f"  wrote {args.json_out}")

    print()
    print("  Round 2 ran at ~2.8x capacity; its hit rates decayed with queue")
    print("  depth. If the wall-time estimate above is too long for one")
    print("  session, cut --num-sessions or --sim-window rather than raising")
    print("  --speed-factor: shortening the run is honest, overloading is not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
