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
prompt-token throughput.

Round 4: load is set against the PEAK, not the mean
---------------------------------------------------
Contexts grow through a run, so the busiest 30 s window carries 2-2.5x the
mean prompt-token demand. A rate set at 85% of average capacity runs the
last third of every run at >100%: in simulation every run was flagged
saturated and most hit the backlog watchdog. So capacity is measured on
end-of-run prompts, in prompt tokens/s, and the offered load is set so the
busiest window runs at `--target-peak-utilization` of it. run_experiment.py
takes that as `--target-peak-prompt-tok-s` and derives each seed's speed
factor from its own peak window.

It also picks the workload size: the first of `--candidates` whose 12-run
matrix fits `--budget-h`, so the session cannot run out of time.

Usage
-----
    python experiments/calibrate_load.py --gpu-memory 0.3 --max-num-seqs 2 --budget-h 8

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

PEAK_WINDOW_S = 30.0   # must match run_experiment.PEAK_WINDOW_S
RUN_OVERHEAD_S = 150   # engine start + warmup + drain (launcher_utils)


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


def _workload(ns: int, window: float, seed: int, args):
    return generate_workload(WorkloadConfig(
        num_sessions=ns, sim_window_s=window, seed=seed,
        max_context_tokens=args.max_context_tokens,
        turn_min_tokens=args.turn_min_tokens,
        turn_max_tokens=args.turn_max_tokens,
        mean_active_burst_turns=args.mean_active_burst_turns,
        mean_idle_gap_s=args.mean_idle_gap_s,
    ))


def _peak_tokens(wl, window_s: float = PEAK_WINDOW_S) -> int:
    per = {}
    for e in wl.events:
        w = int(e.t // window_s)
        per[w] = per.get(w, 0) + len(e.prompt_tokens)
    return max(per.values()) if per else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-memory", type=float, default=0.3,
                   help="Use the CONSTRAINED value: it is the binding arm.")
    p.add_argument("--max-num-seqs", type=int, default=2)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--duration-s", type=float, default=180.0)
    p.add_argument("--target-peak-utilization", type=float, default=0.9,
                   help="Offered prompt-token load in the workload's busiest "
                        "30 s window, as a fraction of measured capacity.")
    p.add_argument("--candidates", default="8x240,6x200,6x150",
                   help="Workload sizes to try, largest first, as "
                        "SESSIONSxSIM_WINDOW. The first whose matrix fits "
                        "--budget-h is chosen.")
    p.add_argument("--max-context-tokens", type=int, default=3072)
    p.add_argument("--turn-min-tokens", type=int, default=16)
    p.add_argument("--turn-max-tokens", type=int, default=48)
    p.add_argument("--mean-active-burst-turns", type=float, default=4.0)
    p.add_argument("--mean-idle-gap-s", type=float, default=25.0)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--runs-per-seed", type=int, default=4,
                   help="Runs per seed in the matrix (policies x capacities).")
    p.add_argument("--budget-h", type=float, default=8.0,
                   help="Wall hours available for the matrix. Exit code 4 "
                        "if even the smallest candidate exceeds it.")
    p.add_argument("--run-tag", default="",
                   help="Stored in the JSON so a later session can tell it "
                        "belongs to the round it is resuming.")
    p.add_argument("--json-out", default="")
    args = p.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cands = [tuple(int(v) for v in c.split("x"))
             for c in args.candidates.split(",") if c.strip()]

    # Probe with the END of the largest candidate's seed-0 run, in order:
    # the busiest windows are what the peak rate has to survive, and in
    # order keeps the natural within-session prefix reuse.
    wl0 = _workload(cands[0][0], cands[0][1], 0, args)
    tail = wl0.events[int(len(wl0.events) * 0.6):]
    prompts = [tokens_to_text(e.prompt_tokens, 4000) for e in tail]
    if not prompts:
        print("[calibrate] workload produced no events; check the knobs")
        return 1
    print(f"[calibrate] probing capacity at gpu_mem={args.gpu_memory} "
          f"max_num_seqs={args.max_num_seqs} for {args.duration_s:.0f}s on "
          f"{len(prompts)} end-of-run prompts ...")
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
    cap = m["prompt_tok_per_s"]
    if cap <= 0 or m["completed"] < 5:
        print("[calibrate] too few completions to trust; not recommending a rate")
        return 1
    peak_rate = cap * args.target_peak_utilization

    print()
    print("=" * 70)
    print(f"LOAD: busiest 30 s window at {args.target_peak_utilization:.0%} "
          f"of capacity = {peak_rate:.0f} prompt tok/s")
    print("=" * 70)
    chosen = None
    for ns, window in cands:
        per_seed, total = {}, 0.0
        for sd in seeds:
            wl = _workload(ns, window, sd, args)
            speed = peak_rate * PEAK_WINDOW_S / _peak_tokens(wl)
            est = window / speed + RUN_OVERHEAD_S
            per_seed[sd] = {"events": len(wl.events), "speed_factor": speed,
                            "est_run_s": est}
            total += args.runs_per_seed * est
        fits = total <= args.budget_h * 3600
        print(f"  {ns} sessions x {window} s: "
              + ", ".join(f"seed {sd} {v['events']} req ~{v['est_run_s'] / 60:.0f} min"
                          for sd, v in per_seed.items())
              + f"  => matrix {total / 3600:.1f} h "
              + ("FITS" if fits else f"> budget {args.budget_h:.1f} h"))
        chosen = (ns, window, per_seed, total)
        if fits:
            break
    ns, window, per_seed, total = chosen
    over = total > args.budget_h * 3600
    print(f"\n  CHOSEN: --num-sessions {ns} --sim-window {window}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump({
                "run_tag": args.run_tag,
                # What run_experiment.py --target-peak-prompt-tok-s takes.
                "peak_prompt_tok_s": peak_rate,
                "capacity_prompt_tok_s": cap,
                "service_rate_req_s": m["req_per_s"],
                "target_peak_utilization": args.target_peak_utilization,
                "gpu_memory": args.gpu_memory,
                "max_num_seqs": args.max_num_seqs,
                "mean_prompt_tokens": m["mean_prompt_tokens"],
                "num_sessions": ns,
                "sim_window_s": window,
                "per_seed": {str(k): v for k, v in per_seed.items()},
                "matrix_estimate_s": total,
                "budget_s": args.budget_h * 3600,
            }, fh, indent=2)
        print(f"  wrote {args.json_out}")
    if over:
        print(f"\n  *** OVER BUDGET even at the smallest candidate: "
              f"{total / 3600:.1f} h > {args.budget_h:.1f} h.")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
