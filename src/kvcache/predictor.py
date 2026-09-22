"""Reuse predictor: how valuable is it to keep a conversation's KV cache?

    value = P(returns within H | idle so far, turns so far) x P(next prompt fits)

Both terms are empirical distributions learned from the TRAINING days of
WildChat; nothing about the conversations being scored is known beyond
what has already happened to them. See PREDICTOR.md for the reasoning.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

# Groups by turns completed so far. Conversations that have already
# continued are more likely to continue again, so each group gets its own
# end probability and gap distribution.
TURN_GROUPS: Tuple[Tuple[int, float], ...] = (
    (1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, float("inf")))


def turn_group(turns_done: int) -> int:
    for i, (lo, hi) in enumerate(TURN_GROUPS):
        if lo <= turns_done <= hi:
            return i
    return len(TURN_GROUPS) - 1


def _ecdf(sorted_xs: Sequence[float], x: float) -> float:
    """Fraction of observations <= x."""
    if not sorted_xs:
        return 0.0
    return bisect.bisect_right(sorted_xs, x) / len(sorted_xs)


@dataclass
class ReturnModel:
    """P(a conversation returns within H seconds, given it has been idle a).

    Per turn group: `p_end` = share of conversations with no further turn,
    `gaps` = sorted observed gaps to the next turn (seconds).
    """
    p_end: List[float] = field(default_factory=list)
    gaps: List[List[float]] = field(default_factory=list)
    n_obs: List[int] = field(default_factory=list)

    @classmethod
    def fit(cls, convs) -> "ReturnModel":
        ended = [0] * len(TURN_GROUPS)
        gaps: List[List[float]] = [[] for _ in TURN_GROUPS]
        for c in convs:
            for k in range(1, c.n_turns + 1):     # k = turns done so far
                g = turn_group(k)
                if k == c.n_turns:
                    ended[g] += 1
                else:
                    gaps[g].append(c.turn_times[k] - c.turn_times[k - 1])
        n = [e + len(gs) for e, gs in zip(ended, gaps)]
        return cls(p_end=[e / m if m else 1.0 for e, m in zip(ended, n)],
                   gaps=[sorted(gs) for gs in gaps], n_obs=n)

    def cdf(self, turns_done: int, x: float) -> float:
        """P(the next turn arrives within x seconds of the last one)."""
        g = turn_group(turns_done)
        return (1.0 - self.p_end[g]) * _ecdf(self.gaps[g], x)

    def p_return_within(self, turns_done: int, idle: float, horizon: float) -> float:
        """P(next turn in (idle, idle + horizon] | no turn during the first idle s)."""
        still_possible = 1.0 - self.cdf(turns_done, idle)
        if still_possible <= 1e-12:
            return 0.0
        return (self.cdf(turns_done, idle + horizon)
                - self.cdf(turns_done, idle)) / still_possible


@dataclass
class FitModel:
    """P(the next prompt still fits the context limit)."""
    followup_lengths: List[int] = field(default_factory=list)
    max_context: int = 4096

    @classmethod
    def fit(cls, convs, max_context: int) -> "FitModel":
        """`convs` must be tokenized. Uses follow-up messages (turn >= 2):
        the message that would extend an already cached conversation."""
        lens = sorted(len(u) for c in convs for u in c.user_tokens[1:])
        return cls(followup_lengths=lens, max_context=max_context)

    def p_fits(self, current_length: int) -> float:
        room = self.max_context - current_length
        if room <= 0:
            return 0.0
        return _ecdf(self.followup_lengths, room)


@dataclass
class ReusePredictor:
    returns: ReturnModel
    fits: FitModel

    def session_value(self, turns_done: int, idle: float, current_length: int,
                      horizon: float) -> float:
        """Value of keeping this conversation's cache for the next `horizon` s.

        `current_length` = tokens of its last prompt + reply (what is cached).
        """
        return (self.returns.p_return_within(turns_done, idle, horizon)
                * self.fits.p_fits(current_length))

    def describe(self) -> Dict[str, list]:
        return {"p_end": self.returns.p_end, "n_obs": self.returns.n_obs,
                "median_gap_s": [gs[len(gs) // 2] if gs else None
                                 for gs in self.returns.gaps]}
