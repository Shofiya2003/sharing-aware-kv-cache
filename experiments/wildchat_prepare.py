"""Run the WildChat loader end to end and print what it produced.

    PYTHONPATH=src python experiments/wildchat_prepare.py
    PYTHONPATH=src python experiments/wildchat_prepare.py --replay 2000 --rate 10

Needs one WildChat shard in data/wildchat/ (see CACHE_SIMULATION.md).
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
import time

import numpy as np

from kvcache.prefix import block_hashes
from kvcache.wildchat import (build_workload, load_tokenizer, read_conversations,
                              split_by_time, system_tokens, tokenize)

DEFAULT_SHARD = "data/wildchat/train-00000-of-00014.parquet"


def describe(name, convs):
    turns = [c.n_turns for c in convs]
    gaps = [b - a for c in convs for a, b in zip(c.turn_times, c.turn_times[1:])]
    days = (convs[-1].start - convs[0].start) / 86400
    p = lambda xs, q: int(np.percentile(xs, q))
    print(f"[{name}] {len(convs)} conversations over {days:.1f} days")
    print(f"   turns per conversation: median {st.median(turns):.0f}, "
          f"p90 {p(turns, 90)}, max {max(turns)}; "
          f"single-turn {sum(t == 1 for t in turns) / len(turns):.1%}")
    print(f"   gap between turns (s): p25 {p(gaps, 25)}, median {p(gaps, 50)}, "
          f"p75 {p(gaps, 75)}, p90 {p(gaps, 90)}, p99 {p(gaps, 99)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=DEFAULT_SHARD)
    ap.add_argument("--replay", type=int, default=2000,
                    help="test conversations to tokenize and replay")
    ap.add_argument("--rate", type=float, default=10.0,
                    help="new conversations per minute in the replay")
    ap.add_argument("--max-context", type=int, default=4096,
                    help="context limit; 4096 = max_model_len in the GPU runs")
    a = ap.parse_args()

    t = time.time()
    convs, skipped = read_conversations(a.shard)
    print(f"read {len(convs)} conversations ({skipped} skipped as malformed) "
          f"in {time.time() - t:.0f}s\n")

    train, test = split_by_time(convs, train_frac=0.6)
    describe("train", train)
    describe("test ", test)

    tok = load_tokenizer()
    system = system_tokens(tok)
    replay = test[:a.replay]
    t = time.time()
    tokenize(replay, tok)
    print(f"\ntokenized {len(replay)} test conversations in {time.time() - t:.0f}s; "
          f"system prompt = {len(system)} tokens")

    events = build_workload(replay, system, a.rate, a.max_context)
    prompt = [len(e.prompt_tokens) for e in events]
    reply = [len(e.output_tokens) for e in events]
    span_h = (events[-1].t - events[0].t) / 3600
    print(f"\n[workload] {len(events)} requests from {len(replay)} conversations at "
          f"{a.rate:g}/min, spanning {span_h:.1f} h")
    print(f"   prompt tokens: median {int(st.median(prompt))}, "
          f"p90 {int(np.percentile(prompt, 90))}, max {max(prompt)}")
    print(f"   reply tokens:  median {int(st.median(reply))}, "
          f"p90 {int(np.percentile(reply, 90))}")
    print(f"   truncated to fit {a.max_context} tokens: "
          f"{sum(e.context_truncated for e in events) / len(events):.1%} of requests")

    # Show the property the cache depends on, on one multi-turn conversation.
    c = next(c for c in replay if c.n_turns >= 4)
    evs = sorted((e for e in events if e.session_id == c.conv_id), key=lambda e: e.t)
    print(f"\n[example] conversation {c.conv_id[:8]}, {c.n_turns} turns")
    prev = None
    for e in evs[:6]:
        hs = block_hashes(e.prompt_tokens)
        # Blocks this prompt shares with the previous turn's prompt + reply.
        reuse = 0
        if prev is not None:
            for x, y in zip(hs, block_hashes(prev.prompt_tokens + prev.output_tokens)):
                if x != y:
                    break
                reuse += 1
        print(f"   turn {e.turn_index}: t=+{e.t - evs[0].t:7.0f}s  prompt {len(e.prompt_tokens):5d} tok "
              f"({len(hs):3d} blocks), {reuse:3d} blocks reusable from the previous turn"
              + ("  [truncated]" if e.context_truncated else ""))
        prev = e
    return 0


if __name__ == "__main__":
    sys.exit(main())
