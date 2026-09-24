"""Eviction policies on real WildChat traffic: how much of the oracle gap
does the learned reuse predictor close?

    PYTHONPATH=src python experiments/wildchat_eviction.py          # ~15 min
    PYTHONPATH=src python experiments/wildchat_eviction.py --rates 20 --capacities 3004 --seeds 0

1. Fit the predictor on the first 60% of days (train).
2. From the last 40% (test), take `--replay` conversations per seed; each
   seed is a different, non-overlapping slice of test conversations.
3. Replay them at each load (new conversations per minute) through the
   cache model at each KV budget, under every eviction policy.
4. Write one CSV row per run and print the summary, paired by seed.

See PREDICTOR.md for the predictor and CACHE_SIMULATION.md for the model.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
import time

from kvcache.cachesim import ALL_POLICIES, simulate
from kvcache.predictor import FitModel, ReturnModel, ReusePredictor
from kvcache.wildchat import (build_workload, load_tokenizer, read_conversations,
                              split_by_time, system_tokens, tokenize)

DEFAULT_SHARD = "data/wildchat/train-00000-of-00014.parquet"
FIELDS = ["rate_per_min", "capacity_blocks", "seed", "policy", "cached_token_rate",
          "recomputed_tokens", "prompt_tokens", "n_requests", "evictions"]


def ints(s):
    return [int(x) for x in s.split(",") if x]


def fit_predictor(train, tok, max_context, sample_size=5000):
    """Fit the reuse predictor on the training days (see PREDICTOR.md)."""
    sample = train[:: max(1, len(train) // sample_size)]
    tokenize(sample, tok)
    return ReusePredictor(ReturnModel.fit(train), FitModel.fit(sample, max_context))


def sweep(test, system, tok, predictor, rates, caps, seeds, replay=2000,
          max_context=4096, warmup_s=600.0, verbose=True):
    """Replay held-out conversations under every policy. Returns CSV-ready rows.

    Each seed is a different, non-overlapping slice of test conversations.
    """
    rows = []
    for seed in seeds:
        block = test[seed * replay:(seed + 1) * replay]
        if len(block) < replay:
            if verbose:
                print(f"seed {seed}: only {len(block)} test conversations left; stopping")
            break
        tokenize(block, tok)
        for rate in rates:
            events = build_workload(block, system, rate, max_context, seed=seed)
            t0 = time.time()
            runs = [(None, "infinite")] + [(c, p) for c in caps for p in ALL_POLICIES]
            for cap, pol in runs:
                r = simulate(events, cap, pol, warmup_s=warmup_s, predictor=predictor)
                rows.append(dict(rate_per_min=rate, capacity_blocks=cap or "inf", seed=seed,
                                 policy=r.policy, cached_token_rate=round(r.cached_token_rate, 6),
                                 recomputed_tokens=r.recomputed_tokens,
                                 prompt_tokens=r.prompt_tokens, n_requests=r.n_requests,
                                 evictions=r.evictions))
            if verbose:
                print(f"seed {seed} rate {rate:g}/min: {len(events)} requests, "
                      f"{len(runs)} runs in {time.time() - t0:.0f}s", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=DEFAULT_SHARD)
    ap.add_argument("--rates", default="5,10,20,40", help="new conversations per minute")
    ap.add_argument("--capacities", default="1000,3004,6000",
                    help="KV blocks; 3004 ~ the T4 at gpu_memory_utilization=0.3")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--replay", type=int, default=2000, help="test conversations per seed")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--warmup-s", type=float, default=600.0,
                    help="leave the first 10 minutes (cold cache) out of the rates")
    ap.add_argument("--out", default="results/wildchat/eviction.csv")
    a = ap.parse_args()
    rates, caps, seeds = [float(x) for x in a.rates.split(",")], ints(a.capacities), ints(a.seeds)

    convs, _ = read_conversations(a.shard)
    train, test = split_by_time(convs, 0.6)
    tok = load_tokenizer()
    predictor = fit_predictor(train, tok, a.max_context)
    print(f"predictor fitted on {len(train)} training conversations", flush=True)
    rows = sweep(test, system_tokens(tok), tok, predictor, rates, caps, seeds,
                 a.replay, a.max_context, a.warmup_s)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {a.out}")
    summarize(rows)
    return 0


def summarize(rows):
    """Per load x budget: each policy's mean rate, its gain over LRU (paired by
    seed), and the share of the LRU -> oracle gap it closes."""
    by = {}
    for r in rows:
        by[(r["rate_per_min"], r["capacity_blocks"], r["seed"], r["policy"])] = r["cached_token_rate"]
    cells = sorted({(r["rate_per_min"], r["capacity_blocks"]) for r in rows
                    if r["capacity_blocks"] != "inf"})
    seeds = sorted({r["seed"] for r in rows})
    show = [p for p in ALL_POLICIES if p != "lru"]
    print("\ncached_token_rate (mean over seeds); per policy: gain over LRU "
          "[share of the LRU->oracle gap closed] (seeds above LRU)")
    for rate, cap in cells:
        lru = [by[(rate, cap, s, "lru")] for s in seeds]
        orc = [by[(rate, cap, s, "oracle")] for s in seeds]
        inf = st.mean(by[(rate, "inf", s, "infinite")] for s in seeds)
        print(f"\nload {rate:g}/min, {cap} blocks: LRU {st.mean(lru):.3f}, "
              f"oracle {st.mean(orc):.3f}, infinite {inf:.3f}")
        for p in show:
            v = [by[(rate, cap, s, p)] for s in seeds]
            d = [x - l for x, l in zip(v, lru)]
            gap = [o - l for o, l in zip(orc, lru)]
            closed = st.mean(dd / g for dd, g in zip(d, gap) if g > 1e-9) if any(g > 1e-9 for g in gap) else float("nan")
            print(f"   {p:15s} {st.mean(v):.3f}  {st.mean(d):+.3f}  [{closed:+.0%}]  "
                  f"({sum(x > 0 for x in d)}/{len(d)})")


if __name__ == "__main__":
    sys.exit(main())
