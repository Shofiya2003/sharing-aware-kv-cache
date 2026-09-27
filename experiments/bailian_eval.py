"""Does the reuse predictor hold up on a second, real chat trace? (Qwen-Bailian Trace A)

    PYTHONPATH=src python experiments/bailian_eval.py

Two checks, the same two as on WildChat:

1. Predictor quality on held-out sessions (Brier, AUC, calibration), for
   (a) tables refitted on this trace's training sessions,
   (b) the tables fitted on WildChat, applied unchanged (transfer),
   (c) a constant average rate.
2. Eviction in the cache simulator: cached-token rate for LRU and the other
   policies, with the predictor fitted on this trace and with WildChat's.

Sessions are split by start time (first 60% train, last 40% test). The
simulator replays the whole trace so the cache is warm; only requests from
the start of the test period on are counted.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time

import numpy as np

from kvcache.bailian import fit_model_from, read_sessions, split_sessions
from kvcache.cachesim import ALL_POLICIES, simulate
from kvcache.predictor import ReturnModel, ReusePredictor
from kvcache.wildchat import read_conversations, split_by_time
from predictor_fit import IDLES, auc, labelled_points
from wildchat_eviction import DEFAULT_SHARD


def quality(models, test, horizon):
    pts = labelled_points(test, IDLES, horizon)
    y = np.array([p[2] for p in pts])
    print(f"   {len(pts)} situations, {y.mean():.1%} returned within {horizon:.0f} s")
    base = np.full(len(y), y.mean())
    print(f"   {'model':28s} {'Brier':>7s} {'AUC':>6s}")
    print(f"   {'constant average rate':28s} {np.mean((base - y) ** 2):7.4f} {'-':>6s}")
    edges = [0, .05, .1, .2, .3, .5, .7, 1.0001]
    for name, m in models.items():
        pred = np.array([m.p_return_within(k, a, horizon) for k, a, _ in pts])
        print(f"   {name:28s} {np.mean((pred - y) ** 2):7.4f} {auc(pred, y):6.3f}")
        cal = []
        for lo, hi in zip(edges, edges[1:]):
            s = (pred >= lo) & (pred < hi)
            if s.sum():
                cal.append(f"{pred[s].mean():.2f}->{y[s].mean():.2f}")
        print(f"      calibration (predicted->observed): " + "  ".join(cal))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="data/bailian/qwen_traceA_blksz_16.jsonl")
    ap.add_argument("--wildchat-shard", default=DEFAULT_SHARD)
    ap.add_argument("--horizon", type=float, default=300.0)
    ap.add_argument("--max-context", type=int, default=32768,
                    help="context window assumed for the 'fits' term (not in the trace)")
    ap.add_argument("--subsample", type=float, default=0.1,
                    help="fraction of sessions replayed per seed (keeps the simulator fast)")
    ap.add_argument("--fractions", default="0.02,0.05,0.10",
                    help="cache size as a fraction of the replayed trace's unique blocks")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--policies", default=",".join(ALL_POLICIES))
    ap.add_argument("--out", default="results/bailian/eviction.csv")
    a = ap.parse_args()

    sessions, events = read_sessions(a.trace)
    train, test = split_sessions(sessions, 0.6)
    t_split = test[0].start
    print(f"{len(sessions)} sessions, {len(events)} requests; train {len(train)}, "
          f"test {len(test)} (from t={t_split:.0f}s)\n")

    own = ReturnModel.fit(train)
    wc_convs, _ = read_conversations(a.wildchat_shard)
    wc_train, _ = split_by_time(wc_convs, 0.6)
    wildchat = ReturnModel.fit(wc_train)
    print(f"1. Predictor quality on the held-out sessions (horizon {a.horizon:.0f} s)")
    quality({"refit on Bailian train": own, "WildChat tables (transfer)": wildchat},
            test, a.horizon)

    fits = fit_model_from(train, a.max_context)
    predictors = {"reuse": ReusePredictor(own, fits),
                  "reuse-wildchat": ReusePredictor(wildchat, fits)}

    print("\n2. Eviction in the simulator (cached-token rate on requests after the split)")
    by_session = {}
    for e in events:
        by_session.setdefault(e.session_id, []).append(e)
    sids = sorted(by_session)
    rows = []
    for seed in [int(x) for x in a.seeds.split(",")]:
        rng = random.Random(seed)
        keep = set(rng.sample(sids, int(len(sids) * a.subsample)))
        evs = [e for e in events if e.session_id in keep]
        uniq = len({b for e in evs for b in e.block_ids})
        print(f"seed {seed}: {len(evs)} requests, {uniq} unique blocks", flush=True)
        for frac in [float(x) for x in a.fractions.split(",")]:
            cap = max(1, int(uniq * frac))
            runs = [(None, "infinite", None)]
            for pol in a.policies.split(","):
                if pol == "reuse":
                    runs += [(cap, "reuse", "reuse"), (cap, "reuse", "reuse-wildchat")]
                else:
                    runs.append((cap, pol, None))
            t0 = time.time()
            for c, pol, pname in runs:
                r = simulate(evs, c, pol, warmup_s=t_split,
                             predictor=predictors.get(pname))
                name = pname or pol
                if c is None and frac != float(a.fractions.split(",")[0]):
                    continue
                rows.append(dict(seed=seed, cache_fraction=frac if c else "inf",
                                 capacity_blocks=c or "inf", policy=name,
                                 cached_token_rate=round(r.cached_token_rate, 6),
                                 recomputed_tokens=r.recomputed_tokens,
                                 prompt_tokens=r.prompt_tokens, evictions=r.evictions))
            print(f"  cache {frac:.0%} ({cap} blocks): {len(runs)} runs in "
                  f"{time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
