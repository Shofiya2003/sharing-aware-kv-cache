"""Phase 2 — One session, multiple real turns against vLLM.

Usage:
    python experiments/phase2_session.py

What it does:
    1. Starts vLLM (real).
    2. Creates one session with 5 turns of growing context.
    3. Submits each turn and reports per-turn latency.
    4. Confirms the session's context grows correctly turn over turn.

For Phase 2 the spec calls for a *real* session, not a mock. Run this
on the GPU host.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from kvcache.session import Session, Turn
from kvcache.vllm_backend import BackendConfig, VLLMBackend, VLLMUnavailable, tokens_to_text


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--gpu-memory", type=float, default=0.5)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--turns", type=int, default=5)
    args = p.parse_args()

    backend = VLLMBackend(
        BackendConfig(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory,
            max_num_seqs=2,
        )
    )
    try:
        await backend.start()
    except VLLMUnavailable as e:
        print(f"[phase2] vLLM unavailable: {e}")
        return 1

    try:
        sess = Session(session_id="s0", start_t=0.0, state="ACTIVE")
        for i in range(args.turns):
            toks = tuple(range(40 + i * 5))  # deterministic token ids
            turn = Turn(turn_index=i, t=i * 0.5, tokens=toks, role="user")
            sess.turns.append(turn)
            sess.last_turn_t = turn.t
            prompt = tokens_to_text(turn.tokens, 4000)
            t0 = time.monotonic()
            res = await backend.submit_and_wait(
                prompt=prompt,
                session_id=sess.session_id,
                turn_index=i,
                submit_t=turn.t,
                max_new_tokens=args.max_new_tokens,
            )
            dt = (time.monotonic() - t0) * 1000.0
            print(
                f"[phase2] turn {i}: ctx_len={len(sess.context_tokens)} "
                f"prompt_tokens={res.n_prompt_tokens} latency={dt:.0f}ms"
            )
        print(f"[phase2] session final context length: {len(sess.context_tokens)}")
    finally:
        await backend.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

