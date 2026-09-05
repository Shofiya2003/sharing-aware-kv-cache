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
    args = p.parse_args()

    cfg = BackendConfig(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
    )
    backend = VLLMBackend(cfg)
    print(f"[phase1] starting vLLM with model={args.model} gpu_mem={args.gpu_memory} ...")
    try:
        await backend.start()
    except VLLMUnavailable as e:
        print(f"[phase1] vLLM unavailable: {e}")
        print("[phase1] (this is expected on CPU-only dev hosts)")
        return 1

    try:
        # Baseline single request
        t0 = time.monotonic()
        res = await backend.submit_and_wait(
            prompt=LONG_PROMPT[:200],
            session_id="probe0",
            turn_index=0,
            submit_t=0.0,
            max_new_tokens=args.max_new_tokens,
        )
        baseline_ms = (time.monotonic() - t0) * 1000.0
        print(f"[phase1] baseline single-request latency: {baseline_ms:.0f} ms")
        print(f"[phase1]   (prompt tokens: {res.n_prompt_tokens})")

        # Burst
        print(f"[phase1] firing burst of {args.burst} concurrent long-prompt requests ...")
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
        for rid in rids:
            r = await backend.wait(rid)
            lats.append(r.latency_ms)
        total_ms = (time.monotonic() - submit_t) * 1000.0
        print(f"[phase1] burst complete in {total_ms:.0f} ms wall")
        print(f"[phase1]   per-request latency: min={min(lats):.0f}ms "
              f"median={statistics.median(lats):.0f}ms "
              f"max={max(lats):.0f}ms "
              f"mean={statistics.mean(lats):.0f}ms")

        # Recommendation
        print()
        print("[phase1] recommended Phase 5 settings:")
        print(f"  sla_latency_ms       = {int(max(baseline_ms * 8, 2000))}")
        if min(lats) < baseline_ms * 2:
            hit_thr = int(min(lats) * 1.3)
        else:
            hit_thr = int((baseline_ms + min(lats)) / 2)
        print(f"  hit_latency_threshold= {hit_thr}")
        print(f"  generous gpu_memory  = 0.7 (or higher if VRAM allows)")
        print(f"  constrained gpu_mem  = {args.gpu_memory}")
    finally:
        await backend.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

