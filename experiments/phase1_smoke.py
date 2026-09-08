"""Phase 1 — Confirm vLLM serves requests and can be put under memory pressure.

Usage:
    python experiments/phase1_smoke.py
    python experiments/phase1_smoke.py --gpu-memory 0.3 --burst 30

What it does:
    1. Starts vLLM with the configured model and `gpu_memory_utilization`.
    2. Sends a baseline single request and records the latency.
    3. Sends a burst of N long-prompt concurrent requests and records
       the resulting latencies. If the burst exceeds the configured
       capacity, you'll see clear latency inflation (the proxy for
       eviction pressure / cache misses).
    4. Prints a report and exits. Suggests the SLA and hit threshold
       to use in the main experiment.

This is a quick smoke test, not a full experiment. Use it on the GPU
host to confirm Phase 1 checkpoint before running the full matrix.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time

from kvcache.vllm_backend import BackendConfig, VLLMBackend, VLLMUnavailable


LONG_PROMPT = (
    "the of and to in a is that for on with as it was by an be this are not "
    "from at or have but his they she which we one all there their what when "
    "your can said about would been if more her than them no time only do "
    "could so my some these other into make them then like over also our who "
    "very long way years use work first well water than ever little place "
    "after thing just great world life still find here something take why "
    "help put different again kind hand high mean keep never much"
) * 6  # ~600 words => ~600 prompt tokens


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--gpu-memory", type=float, default=0.5)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--burst", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--warmup", type=int, default=2,
                   help="Untimed warmup requests before the timed baseline. "
                        "The first requests pay one-time costs (Triton kernel "
                        "compilation, CUDA-graph capture); without warmup the "
                        "baseline is inflated 10-50x and the recommended "
                        "SLA/hit-threshold are garbage.")
    args = p.parse_args()

    cfg = BackendConfig(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
    )
    backend = VLLMBackend(cfg)
    print(f"[phase1] [1/4] starting vLLM: model={args.model} gpu_mem={args.gpu_memory} "
          f"max_len={args.max_model_len} max_seqs={args.max_num_seqs} "
          f"enforce_eager={args.enforce_eager} burst={args.burst} "
          f"max_new_tokens={args.max_new_tokens} ...", flush=True)
    try:
        await backend.start()
    except VLLMUnavailable as e:
        print(f"[phase1] vLLM unavailable: {e}")
        print("[phase1] (this is expected on CPU-only dev hosts)")
        print("[phase1] Kaggle hints: enable GPU (Settings -> Accelerator -> GPU T4), "
              "enable Internet (needed for model download), re-run launcher cell 2 first.")
        return 1
    print("[phase1] [1/4] engine start OK", flush=True)

    try:
        # Warmup (untimed): first requests compile kernels / capture graphs.
        if args.warmup > 0:
            print(f"[phase1] [1b/4] warmup: {args.warmup} untimed requests ...", flush=True)
            for i in range(args.warmup):
                t0 = time.monotonic()
                await backend.submit_and_wait(
                    prompt=LONG_PROMPT[:200],
                    session_id=f"warm{i}",
                    turn_index=0,
                    submit_t=0.0,
                    max_new_tokens=4,
                )
                print(f"[phase1]   warmup {i+1}/{args.warmup}: "
                      f"{(time.monotonic() - t0) * 1000.0:.0f} ms "
                      f"(first one is normally huge — that's compilation, ignore it)",
                      flush=True)
            print("[phase1] [1b/4] warmup done — steady-state timings follow", flush=True)

        # Baseline single request
        print("[phase1] [2/4] baseline: sending 1 short probe request ...", flush=True)
        t0 = time.monotonic()
        res = await backend.submit_and_wait(
            prompt=LONG_PROMPT[:200],
            session_id="probe0",
            turn_index=0,
            submit_t=0.0,
            max_new_tokens=args.max_new_tokens,
        )
        baseline_ms = (time.monotonic() - t0) * 1000.0
        print(f"[phase1] [2/4] baseline single-request latency: {baseline_ms:.0f} ms "
              f"(prompt tokens: {res.n_prompt_tokens}, output: {res.n_output_tokens})", flush=True)

        # Burst
        print(f"[phase1] [3/4] burst: firing {args.burst} concurrent long-prompt "
              f"requests (max_seqs={args.max_num_seqs}) ...", flush=True)
        submit_t = time.monotonic()
        rids = []
        for i in range(args.burst):
            rid = await backend.submit(
                prompt=LONG_PROMPT,
                session_id=f"burst{i}",
                turn_index=0,
                submit_t=0.0,
                max_new_tokens=args.max_new_tokens,
            )
            rids.append(rid)
        lats = []
        for j, rid in enumerate(rids):
            r = await backend.wait(rid)
            lats.append(r.latency_ms)
            print(f"[phase1]   burst req {j+1}/{len(rids)} done: {r.latency_ms:.0f}ms "
                  f"(prompt={r.n_prompt_tokens}tok)", flush=True)
        total_ms = (time.monotonic() - submit_t) * 1000.0
        print(f"[phase1] [3/4] burst complete in {total_ms:.0f} ms wall", flush=True)
        if not lats:
            print("[phase1] ERROR: burst returned no latencies; engine produced no results.")
            return 1
        print(f"[phase1]   per-request latency: min={min(lats):.0f}ms "
              f"median={statistics.median(lats):.0f}ms "
              f"max={max(lats):.0f}ms "
              f"mean={statistics.mean(lats):.0f}ms")

        # Recommendation. hit_thr must sit BETWEEN fast (cache-hit-like)
        # and slow (contended full-prefill) latencies: use the geometric
        # mean, which stays sane even when the two differ by 100x (an
        # arithmetic mean would be dominated by the slow end). SLA must sit
        # above hit_thr, else every hit is also an SLA violation.
        print()
        print("[phase1] [4/4] recommended Phase 5 settings:", flush=True)
        import math
        if min(lats) < baseline_ms * 2:
            hit_thr = int(min(lats) * 1.3)
        else:
            hit_thr = int(math.sqrt(baseline_ms * min(lats)))
        sla = int(max(baseline_ms * 8, 2000))
        if sla <= hit_thr:
            sla = int(hit_thr * 4)
            print(f"[phase1] note: baseline*8 SLA fell below hit_thr; "
                  f"raised SLA to 4x hit_thr.", flush=True)
        print(f"  sla_latency_ms       = {sla}")
        print(f"  hit_latency_threshold= {hit_thr}")
        print(f"  generous gpu_memory  = 0.7 (or higher if VRAM allows)")
        print(f"  constrained gpu_mem  = {args.gpu_memory}")
    finally:
        await backend.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

