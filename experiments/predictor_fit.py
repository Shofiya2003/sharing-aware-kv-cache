"""Fit the reuse predictor on WildChat's training days; check it on the test days.

    PYTHONPATH=src python experiments/predictor_fit.py

Before the predictor drives any eviction, it should be a good predictor on
its own: when it says "70% likely to return within 5 minutes", about 70% of
such conversations should. This script prints what was learned and how well
it holds up on conversations from later days that it never saw.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from kvcache.predictor import TURN_GROUPS, FitModel, ReturnModel, turn_group
from kvcache.wildchat import load_tokenizer, read_conversations, split_by_time, tokenize

DEFAULT_SHARD = "data/wildchat/train-00000-of-00014.parquet"
IDLES = (0, 30, 60, 120, 300, 900, 1800, 3600)


def group_name(g):
    lo, hi = TURN_GROUPS[g]
    return str(lo) if lo == hi else (f"{lo}+" if hi == float("inf") else f"{lo}-{int(hi)}")


def labelled_points(convs, idles, horizon):
    """(turns_done, idle, returned_within_horizon) for each still-idle case.

    A point exists only if the conversation really was still idle after
    `idle` seconds (its next turn came later, or never): the same situation
    the predictor is asked about at eviction time.
    """
    pts = []
    for c in convs:
        for k in range(1, c.n_turns + 1):
            nxt = (c.turn_times[k] - c.turn_times[k - 1]) if k < c.n_turns else None
            for a in idles:
                if nxt is not None and nxt <= a:
                    continue                     # already returned by then
                pts.append((k, a, nxt is not None and nxt <= a + horizon))
    return pts


def auc(scores, labels):
    """Probability a random positive outranks a random negative (ties = 1/2)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):                         # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=DEFAULT_SHARD)
    ap.add_argument("--horizon", type=float, default=300.0,
                    help="'return soon' window for the calibration check (s)")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--length-sample", type=int, default=5000,
                    help="training conversations tokenized for message lengths")
    a = ap.parse_args()

    convs, _ = read_conversations(a.shard)
    train, test = split_by_time(convs, 0.6)
    returns = ReturnModel.fit(train)

    print(f"Fitted on {len(train)} training conversations "
          f"(first 60% of days); checked on {len(test)} test conversations.\n")
    print("1. What happens after a turn, by turns so far (training days)")
    print(f"   {'turns':>6s} {'cases':>7s} {'ended':>7s} {'median gap':>11s} {'p90 gap':>9s}")
    for g in range(len(TURN_GROUPS)):
        gs = returns.gaps[g]
        print(f"   {group_name(g):>6s} {returns.n_obs[g]:7d} {returns.p_end[g]:7.1%} "
              f"{(gs[len(gs) // 2] if gs else float('nan')):10.0f}s "
              f"{(gs[int(len(gs) * .9)] if gs else float('nan')):8.0f}s")

    print(f"\n2. P(returns within {a.horizon:.0f} s | idle so far), by turns so far")
    print("   " + f"{'turns':>6s} " + " ".join(f"{'idle ' + str(x) + 's':>10s}" for x in IDLES))
    for g in range(len(TURN_GROUPS)):
        k = TURN_GROUPS[g][0]
        print("   " + f"{group_name(g):>6s} " + " ".join(
            f"{returns.p_return_within(k, x, a.horizon):10.3f}" for x in IDLES))

    print(f"\n3. Held-out check on the test days (horizon {a.horizon:.0f} s)")
    pts = labelled_points(test, IDLES, a.horizon)
    y = np.array([p[2] for p in pts])
    pred = np.array([returns.p_return_within(k, x, a.horizon) for k, x, _ in pts])
    base = np.full(len(y), y.mean())            # "everyone returns at the average rate"
    print(f"   {len(pts)} situations, {y.mean():.1%} returned within the horizon")
    print(f"   Brier score (lower is better): predictor {np.mean((pred - y) ** 2):.4f}, "
          f"constant average {np.mean((base - y) ** 2):.4f}")
    print(f"   AUC (0.5 = coin flip, 1.0 = perfect ranking): {auc(pred, y):.3f}")
    print("   Calibration: predicted vs observed, in bins of the prediction")
    edges = [0, .05, .1, .2, .3, .5, .7, 1.0001]
    for lo, hi in zip(edges, edges[1:]):
        m = (pred >= lo) & (pred < hi)
        if m.sum():
            print(f"     predicted {lo:4.2f}-{min(hi, 1):4.2f}: n={m.sum():6d}  "
                  f"mean predicted {pred[m].mean():.3f}  observed {y[m].mean():.3f}")

    tok = load_tokenizer()
    sample = train[:: max(1, len(train) // a.length_sample)]
    tokenize(sample, tok)
    fits = FitModel.fit(sample, a.max_context)
    print(f"\n4. P(next prompt still fits {a.max_context} tokens), from "
          f"{len(fits.followup_lengths)} follow-up messages")
    for cur in (500, 1000, 2000, 3000, 3500, 3900, 4096):
        print(f"   conversation at {cur:4d} tokens -> {fits.p_fits(cur):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
