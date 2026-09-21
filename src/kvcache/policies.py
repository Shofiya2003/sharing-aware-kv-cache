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
3. Sharing-aware:   prioritize requests whose prompt OPENING matches blocks
                     other sessions recently sent -- the only cross-session
                     content vLLM's prefix cache can reuse (see prefix.py).
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
        prefix_index: PrefixIndex,
        now: float,
    ) -> DispatchDecision:
        raise NotImplementedError

from .prefix import PrefixIndex
from .session import Session
from .workload import TurnEvent


class FIFOPolicy(DispatchPolicy):
    """FIFO: submit in arrival order. The naive baseline."""

    name = "fifo"

    def score_queue(
        self,
        queue: List[QueuedRequest],
        sessions: Dict[str, Session],
        prefix_index: PrefixIndex,
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
        prefix_index: PrefixIndex,
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


def _shared_prefix_tokens(q: QueuedRequest, prefix_index: PrefixIndex) -> int:
    """Leading prompt tokens this request shares with other sessions.

    Counted in whole KV blocks from token 0, the way vLLM matches them, so
    it is the cross-session reuse vLLM could actually deliver if those
    blocks are still cached. Shared text anywhere else in the prompt
    scores nothing: vLLM cannot reuse it (the old 8-gram detector counted
    it, which is why sharing-aware could never show an effect).
    """
    n, _others = prefix_index.shared_prefix(q.event.session_id, q.event.prompt_tokens)
    return n


class SharingAwarePolicy(DispatchPolicy):
    """Sharing-aware: prioritize requests whose opening blocks match
    prompts other sessions sent recently, i.e. whose prefix another
    session may have left in the cache. Serving them first reuses those
    blocks before other traffic evicts them."""

    name = "sharing-aware"

    def score_queue(
        self,
        queue: List[QueuedRequest],
        sessions: Dict[str, Session],
        prefix_index: PrefixIndex,
        now: float,
    ) -> DispatchDecision:
        counts = [_shared_prefix_tokens(q, prefix_index) for q in queue]
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
            notes={"policy": self.name, "shared_prefix_tokens": counts, "scores": scores},
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
        prefix_index: PrefixIndex,
        now: float,
    ) -> DispatchDecision:
        sess_scores = [self._session_score(q, sessions, now) for q in queue]
        raw_shared = [_shared_prefix_tokens(q, prefix_index) for q in queue]
        max_c = max(raw_shared) if raw_shared else 1
        max_c = max(max_c, 1)
        share_scores = [c / max_c for c in raw_shared]
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

