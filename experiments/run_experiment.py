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

from kvcache.bench import (
    BenchConfig,
    EngineFailure,
    MockVLLMBackend,
    OverloadedRun,
    run_benchmark,
)
from kvcache.analysis import run_analysis, ensure_base_results
from kvcache.vllm_backend import BackendConfig, VLLMBackend
from kvcache.workload import WorkloadConfig, generate_workload


PEAK_WINDOW_S = 30.0  # = the metrics window


def peak_window_prompt_tokens(workload, window_s: float = PEAK_WINDOW_S) -> int:
    """Prompt tokens submitted in the busiest `window_s` of simulated time."""
    per = {}
    for e in workload.events:
        w = int(e.t // window_s)
        per[w] = per.get(w, 0) + len(e.prompt_tokens)
    return max(per.values()) if per else 0


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
            enforce_eager=args.enforce_eager,
        ),
        capacity_setting=capacity,
        max_new_tokens=args.max_new_tokens,
        speed_factor=args.speed_factor,
        sla_latency_ms=args.sla_latency_ms,
        hit_latency_threshold_ms=args.hit_latency_threshold_ms,
        hit_cached_fraction=args.hit_cached_fraction,
        discard_warmup_windows=args.discard_warmup_windows,
        abort_on_backlog_s=args.abort_on_backlog_s,
        # Half a session's mean idle gap, in wall ms: a queue shorter than
        # that cannot be what evicted a session's prefix.
        saturation_floor_ms=0.5 * args.mean_idle_gap_s / args.speed_factor * 1000.0,
        extra_summary={
            "run_tag": args.run_tag,
            "max_num_seqs": args.max_num_seqs,
            "n_events": len(workload.events),
            "num_sessions": args.num_sessions,
            "sim_window_s": args.sim_window,
            "arrival_rate_req_s": len(workload.events) * args.speed_factor
                                  / args.sim_window,
            "speed_factor": args.speed_factor,
            "target_peak_prompt_tok_s": args.target_peak_prompt_tok_s,
        },
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
    # Baseline the engine's cumulative block counters here, so the
    # cross-check rate covers the workload only and not these probes.
    if hasattr(backend, "snapshot_prefix_cache_baseline"):
        backend.snapshot_prefix_cache_baseline()
    print(f"[run] {label}: warmup done", flush=True)
    t0 = time.monotonic()
    try:
        r = await run_benchmark(workload, cfg, backend=backend)
    finally:
        await backend.stop()
    elapsed = time.monotonic() - t0
    print(
        f"[run] <<< {label} done in {elapsed:.1f}s | "
        f"cached_token_rate={r.summary['cached_token_rate']:.3f} "
        f"p50_e2e={r.summary['p50_e2e_latency_ms']:.0f}ms "
        f"p99_e2e={r.summary['p99_e2e_latency_ms']:.0f}ms "
        f"goodput={r.summary['goodput']:.2f} "
        f"failed={r.summary.get('n_failed', 0)} "
        f"usable={r.summary.get('usable', '?')}"
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
            accumulate_context=not args.no_accumulate_context,
            max_context_tokens=args.max_context_tokens,
            shared_attach_position=args.shared_attach_position,
            overlap_fraction=args.overlap_fraction,
            num_shared_docs=args.num_shared_docs,
            shared_doc_min_tokens=args.shared_doc_min_tokens,
            shared_doc_max_tokens=args.shared_doc_max_tokens,
            turn_min_tokens=args.turn_min_tokens,
            turn_max_tokens=args.turn_max_tokens,
            mean_active_burst_turns=args.mean_active_burst_turns,
            mean_idle_gap_s=args.mean_idle_gap_s,
        )
    )
    print(
        f"[run] workload: {len(workload.sessions)} sessions, "
        f"{len(workload.events)} events"
    )
    if args.target_peak_prompt_tok_s > 0:
        # Pace by the PEAK 30 s window of prompt-token demand, not the mean.
        # Contexts grow through a run, so the last windows carry ~2x the
        # mean demand; a rate set from the average overloads the second
        # half of every run.
        if not workload.events:
            print("[run] workload has no events; cannot derive a speed factor")
            return 1
        peak = peak_window_prompt_tokens(workload)
        args.speed_factor = args.target_peak_prompt_tok_s * PEAK_WINDOW_S / peak
        print(f"[run] --target-peak-prompt-tok-s {args.target_peak_prompt_tok_s:.0f} "
              f"(peak window {peak} prompt tokens / {PEAK_WINDOW_S:.0f} sim-s) "
              f"-> --speed-factor {args.speed_factor:.4f} "
              f"(~{args.sim_window / args.speed_factor / 60:.1f} min wall per run)")
    elif args.target_arrival_rate > 0:
        # A speed factor is only meaningful for the workload it was computed
        # on: the same factor over a denser workload is a higher arrival
        # rate. Arm 2 (20 sessions, 300 s) at arm 1's factor (12 sessions,
        # 600 s) would have been offered ~2x the load. Derive it here from
        # the measured request rate instead.
        if not workload.events:
            print("[run] workload has no events; cannot derive a speed factor")
            return 1
        args.speed_factor = (args.target_arrival_rate * args.sim_window
                             / len(workload.events))
        print(f"[run] --target-arrival-rate {args.target_arrival_rate:.3f} req/s "
              f"-> --speed-factor {args.speed_factor:.3f} "
              f"(~{args.sim_window / args.speed_factor / 60:.1f} min wall per run)")

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
        try:
            await run_single(policy, cap, workload, gpu_mem, args)
        except OverloadedRun as e:
            # Configuration error, not a bug. Exit non-zero so an unattended
            # matrix stops here instead of producing 23 more runs of the
            # same unusable data.
            print(f"\n{e}\n", flush=True)
            return 2
        except EngineFailure as e:
            # The engine died mid-run. Usually transient (GPU memory not
            # yet released by a previous engine); the launcher retries.
            print(f"\n{e}\n", flush=True)
            return 3

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
    p.add_argument("--enforce-eager", action="store_true",
                   help="Disable CUDA graphs. Must match the calibration "
                        "(load_calibration.json enforce_eager), or the "
                        "offered load was measured on a different engine.")
    p.add_argument("--generous-gpu-mem", type=float, default=0.7)
    p.add_argument("--constrained-gpu-mem", type=float, default=0.3)
    p.add_argument("--speed-factor", type=float, default=10.0)
    p.add_argument("--target-arrival-rate", type=float, default=0.0,
                   help="Offered load in requests per wall second. When set, "
                        "overrides --speed-factor with the value that gives "
                        "THIS workload that rate. Use the rate from "
                        "experiments/calibrate_load.py.")
    p.add_argument("--target-peak-prompt-tok-s", type=float, default=0.0,
                   help="Offered load as prompt tokens per wall second in "
                        "the workload's busiest 30 s window. Overrides "
                        "--target-arrival-rate and --speed-factor. Use "
                        "`peak_prompt_tok_s` from calibrate_load.py.")
    p.add_argument("--sla-latency-ms", type=float, default=2500.0)
    p.add_argument("--hit-latency-threshold-ms", type=float, default=300.0,
                   help="LEGACY latency proxy. Only emits the proxy_hit_rate "
                        "diagnostic column; does not define hit/miss.")
    p.add_argument("--hit-cached-fraction", type=float, default=0.10,
                   help="A request is a hit when at least this fraction of its "
                        "prompt tokens were served from cache (ground truth).")
    p.add_argument("--abort-on-backlog-s", type=float, default=90.0,
                   help="Abort the run if the oldest request has waited this "
                        "long in the DISPATCH queue. Guards against the "
                        "failure mode where offered load exceeds engine "
                        "capacity and the run measures queue depth rather "
                        "than cache reuse. 0 disables.")
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
    p.add_argument("--mean-active-burst-turns", type=float, default=4.0,
                   help="Mean turns per active burst. Lower values grow "
                        "contexts more slowly and so reduce context-window "
                        "truncation, which is a confound on reuse.")
    p.add_argument("--mean-idle-gap-s", type=float, default=25.0,
                   help="Mean idle gap between a session's bursts.")
    p.add_argument("--max-context-tokens", type=int, default=3072,
                   help="Per-session context window. Requests carry the whole "
                        "conversation so far, capped here.")
    p.add_argument("--no-accumulate-context", action="store_true",
                   help="Submit only each turn's delta instead of the full "
                        "conversation. Reproduces the old (broken) behavior "
                        "where there was no prefix for the cache to reuse.")
    p.add_argument("--run-tag", default="",
                   help="Written to the summary CSV's run_tag column. The "
                        "launcher only counts a summary as done when its tag "
                        "matches the current round, so a previous round's "
                        "CSVs (different load, different max_num_seqs) are "
                        "never mistaken for finished runs.")
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

