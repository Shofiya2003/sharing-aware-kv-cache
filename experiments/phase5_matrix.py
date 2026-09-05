"""Phase 5 — Run the 8-run matrix (4 policies x 2 capacities) end-to-end.

Each (policy, capacity) is one run. After each run, its CSV row(s) are
flushed to disk immediately (incremental write) so that a Kaggle/Colab
session disconnect mid-matrix does not lose completed runs.

Usage:
    python experiments/phase5_matrix.py
    python experiments/phase5_matrix.py --mock
    python experiments/phase5_matrix.py --num-sessions 20 --gpu-memory 0.3
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from kvcache.bench import BenchConfig, MockVLLMBackend, run_benchmark
from kvcache.vllm_backend import BackendConfig, VLLMBackend
from kvcache.workload import WorkloadConfig, generate_workload


POLICIES = ["fifo", "session-aware", "sharing-aware", "combined"]


async def run_one(
    policy: str,
    capacity: str,
    workload,
    cfg_template: BenchConfig,
    mock: bool,
    output_dir: str,
) -> dict:
    label = f"{policy}_{capacity}"
    print(f"[phase5] >>> starting run: {label}")
    cfg = BenchConfig(
        policy_name=policy,
        combined_alpha=cfg_template.combined_alpha,
        backend=cfg_template.backend,
        capacity_setting=capacity,
        max_new_tokens=cfg_template.max_new_tokens,
        speed_factor=cfg_template.speed_factor,
        sla_latency_ms=cfg_template.sla_latency_ms,
        hit_latency_threshold_ms=cfg_template.hit_latency_threshold_ms,
        output_dir=output_dir,
        run_label=label,
    )
    if mock:
        backend = MockVLLMBackend(cfg.backend)
        await backend.start()
    else:
        backend = VLLMBackend(cfg.backend)
        await backend.start()

    t0 = time.monotonic()
    try:
        r = await run_benchmark(workload, cfg, backend=backend)
    finally:
        if mock:
            await backend.stop()
        else:
            await backend.stop()
    elapsed = time.monotonic() - t0
    print(
        f"[phase5] <<< finished {label} in {elapsed:.1f}s | "
        f"n={r.n_records} hit_rate={r.summary['hit_rate']:.3f} "
        f"p99={r.summary['p99_latency_ms']:.0f}ms "
        f"goodput={r.summary['goodput']:.2f}"
    )
    return r.summary


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--num-sessions", type=int, default=15)
    p.add_argument("--sim-window", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument(
        "--generous-gpu-mem",
        type=float,
        default=0.7,
        help="gpu_memory_utilization for the generous-capacity runs",
    )
    p.add_argument(
        "--constrained-gpu-mem",
        type=float,
        default=0.3,
        help="gpu_memory_utilization for the constrained-capacity runs",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
    )
    p.add_argument("--speed-factor", type=float, default=10.0)
    p.add_argument("--sla-latency-ms", type=float, default=2500.0)
    p.add_argument("--hit-latency-threshold-ms", type=float, default=300.0)
    p.add_argument("--output-dir", default="results/csv")
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use MockVLLMBackend instead of real vLLM (CPU-only test).",
    )
    p.add_argument(
        "--max-model-len", type=int, default=4096,
    )
    p.add_argument(
        "--only", nargs="*", default=None,
        help="If set, only run these (policy, capacity) pairs. E.g. 'combined' or 'constrained'.",
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    workload = generate_workload(
        WorkloadConfig(
            num_sessions=args.num_sessions,
            sim_window_s=args.sim_window,
            seed=args.seed,
        )
    )
    print(
        f"[phase5] workload: {len(workload.sessions)} sessions, "
        f"{len(workload.events)} events"
    )

    base_backend = BackendConfig(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    base_cfg = BenchConfig(
        policy_name="fifo",
        backend=base_backend,
        max_new_tokens=args.max_new_tokens,
        speed_factor=args.speed_factor,
        sla_latency_ms=args.sla_latency_ms,
        hit_latency_threshold_ms=args.hit_latency_threshold_ms,
    )

    pairs = []
    for policy in POLICIES:
        for cap_name, gpu_mem in [
            ("generous", args.generous_gpu_mem),
            ("constrained", args.constrained_gpu_mem),
        ]:
            if args.only:
                if not any(o in (policy, cap_name) for o in args.only):
                    continue
            cfg = base_cfg
            cfg.backend.gpu_memory_utilization = gpu_mem
            pairs.append((policy, cap_name, cfg))

    results = []
    for policy, cap, cfg in pairs:
        s = await run_one(policy, cap, workload, cfg, args.mock, args.output_dir)
        results.append({"policy": policy, "capacity": cap, **s})

    # Print a small summary
    print()
    print("[phase5] matrix summary")
    print(f"{'policy':14s}  {'cap':12s}  {'hit':>6s}  {'p50':>6s}  {'p99':>6s}  {'goodput':>8s}")
    for r in results:
        print(
            f"{r['policy']:14s}  {r['capacity']:12s}  "
            f"{r['hit_rate']:6.3f}  {r['p50_latency_ms']:6.0f}  "
            f"{r['p99_latency_ms']:6.0f}  {r['goodput']:8.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

