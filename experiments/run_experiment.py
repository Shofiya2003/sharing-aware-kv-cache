"""Top-level CLI entry point.

Examples
--------
# Real vLLM on a Kaggle/Colab GPU:
python experiments/run_experiment.py --policy combined --capacity constrained

# CPU-only smoke (no GPU required):
python experiments/run_experiment.py --policy combined --capacity constrained --mock

# Just one policy / one capacity (used for incremental re-runs):
python experiments/run_experiment.py --policy fifo --capacity generous

# Just produce charts from existing CSVs:
python experiments/run_experiment.py --analyze-only
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from kvcache.bench import BenchConfig, MockVLLMBackend, run_benchmark
from kvcache.analysis import run_analysis, ensure_base_results
from kvcache.vllm_backend import BackendConfig, VLLMBackend
from kvcache.workload import WorkloadConfig, generate_workload


async def run_single(
    policy: str,
    capacity: str,
    workload,
    gpu_mem: float,
    args,
) -> dict:
    label = f"{policy}_{capacity}"
    if args.label_suffix:
        suffix = args.label_suffix
        if not suffix.startswith("_"):
            suffix = "_" + suffix
        label += suffix
    print(f"[run] >>> {label}: policy={policy} capacity={capacity} gpu_mem={gpu_mem} "
          f"mock={args.mock} sessions={len(workload.sessions)} events={len(workload.events)} "
          f"sla={args.sla_latency_ms:.0f}ms hit_thr={args.hit_latency_threshold_ms:.0f}ms "
          f"max_seqs={args.max_num_seqs} max_new_tokens={args.max_new_tokens} "
          f"speed x{args.speed_factor}", flush=True)
    cfg = BenchConfig(
        policy_name=policy,
        combined_alpha=args.combined_alpha,
        backend=BackendConfig(
            model=args.model,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=gpu_mem,
        ),
        capacity_setting=capacity,
        max_new_tokens=args.max_new_tokens,
        speed_factor=args.speed_factor,
        sla_latency_ms=args.sla_latency_ms,
        hit_latency_threshold_ms=args.hit_latency_threshold_ms,
        output_dir=args.csv_dir,
        run_label=label,
    )
    if args.mock:
        backend = MockVLLMBackend(cfg.backend)
    else:
        backend = VLLMBackend(cfg.backend)
    await backend.start()
    # Warmup (untimed, excluded from metrics): a fresh engine pays one-time
    # kernel-compilation cost on its first requests. Without this, the first
    # workload events absorb ~30s of compilation and pollute P99 + early
    # windows. Both backends expose submit()/wait().
    print(f"[run] {label}: warmup (2 untimed requests) ...", flush=True)
    for i in range(2):
        rid = await backend.submit(
            prompt="warmup probe request",
            session_id=f"__warm{i}__",
            turn_index=0,
            submit_t=0.0,
            max_new_tokens=4,
        )
        await backend.wait(rid)
    print(f"[run] {label}: warmup done", flush=True)
    t0 = time.monotonic()
    try:
        r = await run_benchmark(workload, cfg, backend=backend)
    finally:
        await backend.stop()
    elapsed = time.monotonic() - t0
    print(
        f"[run] <<< {label} done in {elapsed:.1f}s | "
        f"hit_rate={r.summary['hit_rate']:.3f} "
        f"p99={r.summary['p99_latency_ms']:.0f}ms "
        f"goodput={r.summary['goodput']:.2f}"
    )
    return r.summary


async def amain(args) -> int:
    if args.analyze_only:
        run_analysis(args.csv_dir, args.fig_dir)
        return 0

    os.makedirs(args.csv_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    ensure_base_results(args.csv_dir)

    workload = generate_workload(
        WorkloadConfig(
            num_sessions=args.num_sessions,
            sim_window_s=args.sim_window,
            seed=args.seed,
        )
    )
    print(
        f"[run] workload: {len(workload.sessions)} sessions, "
        f"{len(workload.events)} events"
    )

    pairs = []
    if args.policy and args.capacity:
        gpu_mem = args.constrained_gpu_mem if args.capacity == "constrained" else args.generous_gpu_mem
        pairs.append((args.policy, args.capacity, gpu_mem))
    else:
        for policy in ["fifo", "session-aware", "sharing-aware", "combined"]:
            for cap, gpu_mem in [
                ("generous", args.generous_gpu_mem),
                ("constrained", args.constrained_gpu_mem),
            ]:
                if args.policy and policy != args.policy:
                    continue
                if args.capacity and cap != args.capacity:
                    continue
                pairs.append((policy, cap, gpu_mem))

    for policy, cap, gpu_mem in pairs:
        await run_single(policy, cap, workload, gpu_mem, args)

    run_analysis(args.csv_dir, args.fig_dir)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default=None, help="fifo|session-aware|sharing-aware|combined")
    p.add_argument("--capacity", default=None, help="generous|constrained")
    p.add_argument("--combined-alpha", type=float, default=0.5)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--num-sessions", type=int, default=15)
    p.add_argument("--sim-window", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--generous-gpu-mem", type=float, default=0.7)
    p.add_argument("--constrained-gpu-mem", type=float, default=0.3)
    p.add_argument("--speed-factor", type=float, default=10.0)
    p.add_argument("--sla-latency-ms", type=float, default=2500.0)
    p.add_argument("--hit-latency-threshold-ms", type=float, default=300.0)
    p.add_argument("--csv-dir", default="results/csv")
    p.add_argument("--fig-dir", default="results/figures")
    p.add_argument("--mock", action="store_true", help="Use the mock backend (no GPU).")
    p.add_argument("--analyze-only", action="store_true", help="Skip runs; only regenerate charts from existing CSVs.")
    p.add_argument("--label-suffix", default="",
                   help="Appended to the run label (CSV/chart names), e.g. '_s1' "
                        "for a reseed repeat or '_a025' for combined_alpha=0.25. "
                        "Lets follow-up runs coexist with the base matrix instead "
                        "of overwriting it. Analysis plots variants as dashed lines.")
    args = p.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())

