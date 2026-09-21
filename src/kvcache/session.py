"""Phase 1 — minimal multi-turn session with growing context and lifecycle metadata.

A `Session` represents one independent user/agent conversation that lives for
the duration of the simulation. Each turn appends a block of tokens to the
session's growing context; the session alternates between ACTIVE (rapid
turns) and IDLE (long gaps) to mirror real bursty chat/agent traffic.

We intentionally do not depend on a real tokenizer or LLM: turns are
sequences of integer token ids, and "context" is the cumulative token list.
The cache manager (Phase 2) and the workload generator (Phase 3) consume
these sequences directly so that cache behavior is fully observable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class Turn:
    """A single conversational turn inside a session.

    Attributes:
        turn_index: 0-based index of this turn within the session.
        t: simulation time (seconds) at which the turn was issued.
        tokens: the tokens appended at this turn.
        role: who emitted the tokens ("user", "assistant", "tool", "doc").
    """

    turn_index: int
    t: float
    tokens: Tuple[int, ...]
    role: str
    # Pinned turns are never dropped when the context window overflows.
    # Real deployments keep the system prompt / shared document preamble at
    # position 0 and evict middle turns instead, which is what keeps the
    # cacheable prefix stable. Dropping oldest-first would delete the
    # preamble and destroy cross-session reuse entirely.
    pinned: bool = False

    @property
    def size(self) -> int:
        return len(self.tokens)


@dataclass
class Session:
    """One long-lived, multi-turn session.

    A session alternates between ACTIVE and IDLE phases. While active it
    issues turns at `active_inter_turn_dt` seconds; while idle it produces
    no turns for an exponentially distributed gap. The session is "alive"
    from `start_t` until the simulation ends.
    """

    session_id: str
    start_t: float
    state: str = "IDLE"
    turns: List[Turn] = field(default_factory=list)
    last_turn_t: Optional[float] = None
    last_active_end_t: Optional[float] = None
    total_active_intervals: int = 0
    total_idle_intervals: int = 0
    sum_idle_gap: float = 0.0
    sum_active_burst_len: int = 0

    @property
    def context_tokens(self) -> Tuple[int, ...]:
        out: List[int] = []
        for turn in self.turns:
            out.extend(turn.tokens)
        return tuple(out)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def is_alive(self) -> bool:
        return True

    def avg_idle_gap(self) -> float:
        if self.total_idle_intervals == 0:
            return float("inf")
        return self.sum_idle_gap / self.total_idle_intervals

    def recent_idle_gap(self) -> Optional[float]:
        if self.last_active_end_t is None or self.last_turn_t is None:
            return None
        return max(0.0, self.last_turn_t - self.last_active_end_t)

    def return_likelihood(self, now: float) -> float:
        """Heuristic return-likelihood in [0, 1]. Higher = more likely to return soon."""
        if self.last_turn_t is None:
            return 0.5
        since_last = max(0.0, now - self.last_turn_t)
        avg_gap = self.avg_idle_gap()
        if math.isinf(avg_gap) or avg_gap <= 0:
            recency = 0.5
        else:
            recency = math.exp(-since_last / avg_gap)
        size_prior = min(1.0, self.turn_count / 20.0)
        return 0.7 * recency + 0.3 * size_prior


def sample_idle_gap(rng: random.Random, mean: float) -> float:
    if mean <= 0:
        return 0.0
    return rng.expovariate(1.0 / mean)


def sample_active_burst(rng: random.Random, mean_burst: float) -> int:
    """Geometric with mean `mean_burst`, at least 1. We implement it manually
    because `random.Random` does not expose `.geometric` directly."""
    if mean_burst <= 0:
        return 1
    # p = 1 / (mean_burst + 1)  =>  E[X] = (1 - p) / p = mean_burst
    p = 1.0 / (mean_burst + 1.0)
    # geometric: count of trials until first success (1-indexed).
    # Sample u in (0, 1], then k = ceil(log(u) / log(1 - p))
    u = rng.random()
    if u <= 0.0:
        u = 1e-12
    k = math.ceil(math.log(u) / math.log1p(-p))
    return max(1, k)


def sample_active_inter_turn_dt(rng: random.Random, mean: float) -> float:
    if mean <= 0:
        return 0.0
    return rng.expovariate(1.0 / mean)


def sample_turn_tokens(
    rng: random.Random,
    unique_vocab: Sequence[int],
    min_size: int,
    max_size: int,
) -> Tuple[int, ...]:
    size = rng.randint(min_size, max_size)
    return tuple(rng.choice(unique_vocab) for _ in range(size))



# A gap between two of a session's turns longer than this (sim seconds)
# counts as an idle period. The generator's active inter-turn gap is
# exponential with mean 1 s (P(>5 s) < 1%) and its idle gaps average 25 s.
IDLE_GAP_THRESHOLD_S = 5.0


class LiveSessions(dict):
    """session_id -> Session, built ONLY from turns that have been served.

    The generator's `workload.sessions` is filled in for the whole run
    before it starts: its `turns`, `turn_count`, `last_turn_t` and idle-gap
    statistics describe the session's FUTURE. Policies used to read those,
    so "session-aware" amounted to "serve the sessions that will have the
    most turns in total" -- oracle knowledge, not a return prediction.

    The benchmark calls `observe()` as each request completes, so a policy
    scoring the queue at time `now` sees exactly the history a real
    serving system would have.
    """

    def __init__(self, idle_gap_threshold_s: float = IDLE_GAP_THRESHOLD_S):
        super().__init__()
        self.idle_gap_threshold_s = idle_gap_threshold_s

    def observe(self, session_id: str, turn_index: int, t: float,
                tokens: Tuple[int, ...] = (), role: str = "user") -> Session:
        sess = self.get(session_id)
        if sess is None:
            sess = Session(session_id=session_id, start_t=t, state="ACTIVE")
            self[session_id] = sess
        prev = sess.last_turn_t
        # A session can have two turns in flight and they can complete out
        # of order; only a later turn moves the clock or counts as a gap.
        if prev is None or t >= prev:
            if prev is not None and t - prev > self.idle_gap_threshold_s:
                sess.total_idle_intervals += 1
                sess.sum_idle_gap += t - prev
                sess.last_active_end_t = prev
            sess.last_turn_t = t
        sess.turns.append(Turn(turn_index=turn_index, t=t, tokens=tuple(tokens), role=role))
        return sess
