"""Dispatch / scheduling policies for the vLLM serving layer.

ARCHITECTURAL NOTE
==================
These policies operate *only* on the order in which requests are submitted
to vLLM. They do not — and must not — touch vLLM's internal KV cache, block
manager, or eviction policy. vLLM's real, built-in automatic prefix caching
runs exactly as it would in production under whatever memory pressure we
configured via `gpu_memory_utilization`.

What each policy controls is the dispatch priority of pending requests in
the queue. When the driver pulls events from the workload, it pushes them
into a priority queue. The policy reorders that queue. The driver then
submits the head of the queue to vLLM, up to vLLM's `max_num_seqs`
concurrency.

The four policies
-----------------
1. FIFO (naive):    submit in arrival order. No signal awareness.
2. Session-aware:   prioritize sessions likely to return soon and sessions
                     with expensive accumulated context.
3. Sharing-aware:   prioritize requests whose content overlaps with other
                     live sessions' content.
4. Combined:        weighted sum of session and sharing signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class QueuedRequest:
    """A pending request sitting in the dispatch queue."""

    event: TurnEvent
    arrival_t: float
    enqueue_seq: int
    score: float = 0.0


@dataclass
class DispatchDecision:
    """Result of scoring the queue. `ordered` is the recommended order
    in which to dispatch the queued requests (highest priority first)."""

    ordered: List[QueuedRequest]
    notes: Dict[str, Any] = field(default_factory=dict)


class DispatchPolicy:
    """Base class for dispatch policies."""

    name: str = "base"

    def score_queue(
        self,
        queue: List[QueuedRequest],
        sessions: Dict[str, Session],
        overlap_index: OverlapIndex,
        now: float,
    ) -> DispatchDecision:
        raise NotImplementedError

from .overlap import OverlapIndex, ngrams, ngram_id
from .session import Session
from .workload import TurnEvent


class FIFOPolicy(DispatchPolicy):
    """FIFO: submit in arrival order. The naive baseline."""

    name = "fifo"

    def score_queue(
        self,
        queue: List[QueuedRequest],
        sessions: Dict[str, Session],
        overlap_index: OverlapIndex,
        now: float,
    ) -> DispatchDecision:
        ordered = sorted(queue, key=lambda q: (q.arrival_t, q.enqueue_seq))
        return DispatchDecision(ordered=ordered, notes={"policy": self.name})


class SessionAwarePolicy(DispatchPolicy):
    """Session-aware: prioritize sessions likely to return soon and sessions
    with expensive accumulated context.

    score = 0.6 * return_likelihood + 0.4 * context_size_prior
    Higher score = higher dispatch priority.
    """

    name = "session-aware"

    def score_queue(
        self,
        queue: List[QueuedRequest],
        sessions: Dict[str, Session],
        overlap_index: OverlapIndex,
        now: float,
    ) -> DispatchDecision:
        def score(q: QueuedRequest) -> float:
            sess = sessions.get(q.event.session_id)
            if sess is None:
                return 0.0
            rl = sess.return_likelihood(now)
            ctx = min(1.0, sess.turn_count / 20.0)
            return 0.6 * rl + 0.4 * ctx

        indexed = sorted(
            list(zip(queue, [score(q) for q in queue])),
            key=lambda qs: (-qs[1], qs[0].arrival_t, qs[0].enqueue_seq),
        )
        ordered = [q for q, _ in indexed]
        return DispatchDecision(
            ordered=ordered,
            notes={"policy": self.name, "scores": [s for _, s in indexed]},
        )


def _sharing_count(q: QueuedRequest, overlap_index: OverlapIndex, cap: int = 20) -> int:
    """Count distinct other sessions whose referenced n-grams overlap with
    this request's prompt. Used by SharingAwarePolicy and CombinedPolicy.

    Scores the full accumulated prompt (`prompt_tokens`), not just this
    turn's delta, so that sharing inherited from earlier turns still counts
    -- the prompt is what occupies cache blocks.
    """
    other_sids = set()
    for gram in ngrams(q.event.prompt_tokens, overlap_index.n):
        gid = ngram_id(gram)
        for sid in overlap_index._refs.get(gid, ()):  # noqa: SLF001
            if sid != q.event.session_id:
                other_sids.add(sid)
    return min(len(other_sids), cap)


class SharingAwarePolicy(DispatchPolicy):
    """Sharing-aware: prioritize requests whose content overlaps with
    content already referenced by other live sessions."""

    name = "sharing-aware"

    def __init__(self, max_share_cap: int = 20) -> None:
        self.max_share_cap = max_share_cap

    def score_queue(
        self,
        queue: List[QueuedRequest],
        sessions: Dict[str, Session],
        overlap_index: OverlapIndex,
        now: float,
    ) -> DispatchDecision:
        counts = [_sharing_count(q, overlap_index, self.max_share_cap) for q in queue]
        max_c = max(counts) if counts else 1
        max_c = max(max_c, 1)
        scores = [c / max_c for c in counts]
        indexed = sorted(
            list(zip(queue, scores)),
            key=lambda qs: (-qs[1], qs[0].arrival_t, qs[0].enqueue_seq),
        )
        ordered = [q for q, _ in indexed]
        return DispatchDecision(
            ordered=ordered,
            notes={"policy": self.name, "overlap_counts": counts, "scores": scores},
        )


@dataclass
class CombinedPolicy(DispatchPolicy):
    """Combined session- and sharing-aware.

    score = alpha * session_score + (1 - alpha) * sharing_score
    - alpha = 0   -> identical to SharingAwarePolicy
    - alpha = 1   -> identical to SessionAwarePolicy
    - alpha = 0.5 -> equal weighting (default)
    """

    name = "combined"
    alpha: float = 0.5
    max_share_cap: int = 20

    def _session_score(self, q: QueuedRequest, sessions: Dict[str, Session], now: float) -> float:
        sess = sessions.get(q.event.session_id)
        if sess is None:
            return 0.0
        rl = sess.return_likelihood(now)
        ctx = min(1.0, sess.turn_count / 20.0)
        return 0.6 * rl + 0.4 * ctx

    def score_queue(
        self,
        queue: List[QueuedRequest],
        sessions: Dict[str, Session],
        overlap_index: OverlapIndex,
        now: float,
    ) -> DispatchDecision:
        sess_scores = [self._session_score(q, sessions, now) for q in queue]
        raw_overlap = [_sharing_count(q, overlap_index, self.max_share_cap) for q in queue]
        max_c = max(raw_overlap) if raw_overlap else 1
        max_c = max(max_c, 1)
        share_scores = [c / max_c for c in raw_overlap]
        a = self.alpha
        combined = [a * s + (1.0 - a) * sh for s, sh in zip(sess_scores, share_scores)]
        indexed = sorted(
            list(zip(queue, combined)),
            key=lambda qs: (-qs[1], qs[0].arrival_t, qs[0].enqueue_seq),
        )
        ordered = [q for q, _ in indexed]
        return DispatchDecision(
            ordered=ordered,
            notes={
                "policy": self.name,
                "alpha": a,
                "session_scores": sess_scores,
                "sharing_scores": share_scores,
                "combined": combined,
            },
        )

