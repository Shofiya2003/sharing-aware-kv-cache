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
import copy
import os
import sys
import time

from kvcache.bench import BenchConfig, MockVLLMBackend, run_benchmark
from kvcache.analysis import ensure_base_results
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
    label_suffix: str = "",
) -> dict:
    label = f"{policy}_{capacity}"
    if label_suffix:
        if not label_suffix.startswith("_"):
            label_suffix = "_" + label_suffix
        label += label_suffix
    # NOTE: cfg_template.backend must be deep-copied per run. All pairs
    # previously shared one BackendConfig object, so the last-assigned
    # gpu_memory_utilization (constrained) silently applied to every run,
    # invalidating the generous-vs-constrained comparison.
    print(f"[phase5] >>> starting run: {label} "
          f"(gpu_mem={cfg_template.backend.gpu_memory_utilization}, mock={mock})")
    cfg = BenchConfig(
        policy_name=policy,
        combined_alpha=cfg_template.combined_alpha,
        backend=copy.deepcopy(cfg_template.backend),
        capacity_setting=capacity,
        max_new_tokens=cfg_template.max_new_tokens,
        speed_factor=cfg_template.speed_factor,
        sla_latency_ms=cfg_template.sla_latency_ms,
        hit_latency_threshold_ms=cfg_template.hit_latency_threshold_ms,
        hit_cached_fraction=cfg_template.hit_cached_fraction,
        discard_warmup_windows=cfg_template.discard_warmup_windows,
        output_dir=output_dir,
        run_label=label,
    )
    if mock:
        backend = MockVLLMBackend(cfg.backend)
        await backend.start()
    else:
        backend = VLLMBackend(cfg.backend)
        await backend.start()

    # Warmup (untimed, excluded from metrics): see run_experiment.py.
    print(f"[phase5] {label}: warmup (2 untimed requests) ...", flush=True)
    for i in range(2):
        rid = await backend.submit(
            prompt="warmup probe request",
            session_id=f"__warm{i}__",
            turn_index=0,
            submit_t=0.0,
            max_new_tokens=4,
        )
        await backend.wait(rid)
    print(f"[phase5] {label}: warmup done", flush=True)

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
    p.add_argument("--combined-alpha", type=float, default=0.5,
                   help="Weight on the session signal in the combined policy; "
                        "(1-alpha) goes to the sharing signal.")
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
    p.add_argument("--hit-latency-threshold-ms", type=float, default=300.0,
                   help="LEGACY latency proxy. Only emits the proxy_hit_rate "
                        "diagnostic column; does not define hit/miss.")
    p.add_argument("--hit-cached-fraction", type=float, default=0.10,
                   help="A request is a hit when at least this fraction of its "
                        "prompt tokens were served from cache (ground truth).")
    p.add_argument("--discard-warmup-windows", type=int, default=1,
                   help="Leading time windows excluded from the run summary as "
                        "engine warmup. Keeps run order out of the headline P99.")
    p.add_argument("--shared-attach-position", default="random",
                   choices=["session_preamble", "session_mid", "prefix", "mid", "random"],
                   help="Where cross-session shared content lands. vLLM reuses "
                        "only a contiguous prefix from token 0, so "
                        "'session_preamble' (doc opens the session, hence sits "
                        "at position 0 of every prompt it issues) is the ONLY "
                        "mode that yields cross-session reuse. 'prefix' opens "
                        "the individual turn, which is mid-prompt once context "
                        "accumulates. 'random'/'mid' are detectable but not "
                        "reusable.")
    p.add_argument("--overlap-fraction", type=float, default=0.6,
                   help="Fraction of sessions that reference shared documents.")
    p.add_argument("--num-shared-docs", type=int, default=4)
    p.add_argument("--shared-doc-min-tokens", type=int, default=64,
                   help="Min size of a shared document. With session_preamble "
                        "placement this is the cross-session reusable prefix, "
                        "so it must be large relative to the conversation for "
                        "the sharing signal to have any headroom.")
    p.add_argument("--shared-doc-max-tokens", type=int, default=192)
    p.add_argument("--turn-min-tokens", type=int, default=32)
    p.add_argument("--turn-max-tokens", type=int, default=96)
    p.add_argument("--max-context-tokens", type=int, default=3072,
                   help="Per-session context window. Requests carry the whole "
                        "conversation so far, capped here.")
    p.add_argument("--no-accumulate-context", action="store_true",
                   help="Submit only each turn's delta instead of the full "
                        "conversation. Reproduces the old (broken) behavior "
                        "where there was no prefix for the cache to reuse.")
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
    p.add_argument(
        "--label-suffix", default="",
        help="Appended to every run label, e.g. '_s1' for a reseed repeat. "
             "Lets follow-up matrixes coexist with earlier runs.",
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ensure_base_results(args.output_dir)

    workload = generate_workload(
        WorkloadConfig(
            num_sessions=args.num_sessions,
            sim_window_s=args.sim_window,
            seed=args.seed,
            accumulate_context=not args.no_accumulate_context,
            max_context_tokens=args.max_context_tokens,
            shared_attach_position=args.shared_attach_position,
            overlap_fraction=args.overlap_fraction,
            num_shared_docs=args.num_shared_docs,
            shared_doc_min_tokens=args.shared_doc_min_tokens,
            shared_doc_max_tokens=args.shared_doc_max_tokens,
            turn_min_tokens=args.turn_min_tokens,
            turn_max_tokens=args.turn_max_tokens,
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
        hit_cached_fraction=args.hit_cached_fraction,
        discard_warmup_windows=args.discard_warmup_windows,
        combined_alpha=args.combined_alpha,
    )

    pairs = []
    for policy in POLICIES:
        for cap_name, gpu_mem in [
            ("generous", args.generous_gpu_mem),
            ("constrained", args.constrained_gpu_mem),
        ]:
            if args.only:
                # AND semantics: every token must match this pair, where a
                # token matches if it equals the policy, the capacity, or
                # the "policy_capacity" label. E.g. `--only fifo
                # constrained` runs just fifo_constrained; `--only
                # combined` runs both combined_*; `--only constrained`
                # runs all *_constrained.
                label = f"{policy}_{cap_name}"
                if not all(
                    o in (policy, cap_name, label) for o in args.only
                ):
                    continue
            # Fresh backend per pair: never mutate a shared object.
            backend = BackendConfig(
                model=base_backend.model,
                gpu_memory_utilization=gpu_mem,
                max_model_len=base_backend.max_model_len,
                max_num_seqs=base_backend.max_num_seqs,
            )
            cfg = BenchConfig(
                policy_name="fifo",
                backend=backend,
                max_new_tokens=base_cfg.max_new_tokens,
                speed_factor=base_cfg.speed_factor,
                sla_latency_ms=base_cfg.sla_latency_ms,
                hit_latency_threshold_ms=base_cfg.hit_latency_threshold_ms,
                hit_cached_fraction=base_cfg.hit_cached_fraction,
                discard_warmup_windows=base_cfg.discard_warmup_windows,
            )
            pairs.append((policy, cap_name, cfg))
            print(f"[phase5] queued run: {policy}_{cap_name} (gpu_mem={gpu_mem})")

    results = []
    for policy, cap, cfg in pairs:
        s = await run_one(policy, cap, workload, cfg, args.mock, args.output_dir,
                          label_suffix=args.label_suffix)
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

