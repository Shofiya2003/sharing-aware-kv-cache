"""Multi-session workload generator with built-in cross-session content overlap.

Generates M independent sessions that:

  - start at staggered times across a simulation window,
  - alternate ACTIVE bursts (geometric bursts of turns) and IDLE gaps
    (exponentially distributed),
  - accumulate context per turn, and
  - deliberately include cross-session content overlap, both prefix-aligned
    and mid-context, by sampling shared "documents" from a small pool.

The generator is deterministic given a seed. It returns a `Workload` with:

  - a list of `Session` objects (one per session),
  - a flat time-sorted list of `TurnEvent` items, each describing a single
    turn to be submitted to vLLM,
  - a `shared_doc_pool` so the overlap detector and analyses can inspect
    what content is being shared.

The "tokens" we generate are integers; when a real run is performed these
are rendered to text via the tokenizer (see `vllm_backend.py`) and submitted
to the model. Turn text and shared docs are derived deterministically from
the integer token ids, so a given workload renders to the same text on every
machine.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .session import (
    Session,
    Turn,
    sample_active_burst,
    sample_active_inter_turn_dt,
    sample_idle_gap,
)

@dataclass
class WorkloadConfig:
    """All knobs for the workload generator."""

    num_sessions: int = 15
    sim_window_s: float = 300.0
    start_window_fraction: float = 0.5

    # Activity model
    mean_idle_gap_s: float = 25.0
    mean_active_burst_turns: float = 4.0
    mean_active_inter_turn_dt_s: float = 1.0

    # Turn sizes (in tokens)
    turn_min_tokens: int = 32
    turn_max_tokens: int = 96

    # Multi-turn context. Each request carries the session's whole history
    # so far, like a real chat API call. This is what grows the per-session
    # KV footprint over time and gives the prefix cache something to reuse.
    accumulate_context: bool = True
    # Hard cap on prompt length, mirroring a real deployment's context
    # window. When a session exceeds it we drop its oldest turns, which
    # necessarily invalidates that session's cached prefix -- the generator
    # flags those events so the analysis can account for them.
    max_context_tokens: int = 3072

    # Cross-session content overlap
    # Defaults are tuned so that, on a 10-20 session workload, the
    # expected number of pairs of sharing sessions that pick the *same*
    # shared doc is non-trivial (typically 3-5 pairs at 0.6 / 4 docs).
    overlap_fraction: float = 0.6
    num_shared_docs: int = 4
    shared_doc_min_tokens: int = 64
    shared_doc_max_tokens: int = 192
    shared_attach_position: str = "random"  # "prefix" | "mid" | "random"

    # Token universe. Tokens are ints in [0, vocab_size). These are rendered
    # to text by the backend (see `vllm_backend.token_id_to_text`).
    vocab_size: int = 4000

    seed: int = 0


@dataclass
class TurnEvent:
    """One turn to be submitted to vLLM.

    `tokens` is this turn's NEW content only (the delta). `context_tokens`
    is what actually gets sent to the engine: the whole conversation so
    far, including this turn -- which is how real multi-turn chat serving
    works and is the only reason a prefix cache has anything to reuse.

    An earlier version submitted `tokens` alone, i.e. ~32-96 tokens per
    request with no history. That made every request an independent short
    prompt, so (a) there was no growing per-session context to protect,
    and (b) the total KV footprint was far too small for
    `gpu_memory_utilization` to ever create eviction pressure. Both of
    those are premises the experiment depends on.
    """

    session_id: str
    turn_index: int
    t: float
    tokens: Tuple[int, ...]
    role: str
    is_active: bool
    # Full accumulated prompt for this turn. Defaults to `tokens` so older
    # callers still work, but the generator always populates it.
    context_tokens: Tuple[int, ...] = ()
    # True when the session's history was trimmed to fit
    # `max_context_tokens`, which resets prefix reuse for that session.
    context_truncated: bool = False

    @property
    def prompt_tokens(self) -> Tuple[int, ...]:
        """What to actually submit to the engine."""
        return self.context_tokens if self.context_tokens else self.tokens


@dataclass
class Workload:
    sessions: List[Session]
    events: List[TurnEvent]
    shared_doc_pool: List[Tuple[int, ...]] = field(default_factory=list)
    config: Optional[WorkloadConfig] = None
    sharing_sids: set = field(default_factory=set)


def _sample_shared_doc(
    rng: random.Random, cfg: WorkloadConfig
) -> Tuple[int, ...]:
    size = rng.randint(cfg.shared_doc_min_tokens, cfg.shared_doc_max_tokens)
    return tuple(rng.randrange(cfg.vocab_size) for _ in range(size))


def _maybe_attach_shared_doc(
    rng: random.Random,
    cfg: WorkloadConfig,
    turn_tokens: Tuple[int, ...],
    doc: Tuple[int, ...],
) -> Tuple[int, ...]:
    pos = cfg.shared_attach_position
    if pos == "prefix" or (pos == "random" and rng.random() < 0.5):
        return doc + turn_tokens
    if not turn_tokens:
        return doc
    offset = rng.randint(0, len(turn_tokens))
    return turn_tokens[:offset] + doc + turn_tokens[offset:]


def generate_workload(cfg: WorkloadConfig) -> Workload:
    """Generate a deterministic workload from `cfg`."""
    rng = random.Random(cfg.seed)

    shared_pool: List[Tuple[int, ...]] = [
        _sample_shared_doc(rng, cfg) for _ in range(cfg.num_shared_docs)
    ]

    sharing_sids: set = set()
    if cfg.num_sessions > 0:
        n_sharers = int(round(cfg.num_sessions * cfg.overlap_fraction))
        sharer_indices = rng.sample(range(cfg.num_sessions), k=n_sharers)
        sharing_sids = {f"s{i:03d}" for i in sharer_indices}

    start_window = max(1e-6, cfg.sim_window_s * cfg.start_window_fraction)

    sessions: List[Session] = []
    events: List[TurnEvent] = []

    for i in range(cfg.num_sessions):
        sid = f"s{i:03d}"
        start_t = rng.uniform(0.0, start_window)
        sess = Session(session_id=sid, start_t=start_t, state="IDLE")
        sessions.append(sess)

        primary_doc = shared_pool[rng.randrange(len(shared_pool))] if shared_pool else ()
        secondary_doc = (
            shared_pool[rng.randrange(len(shared_pool))]
            if shared_pool and rng.random() < 0.4
            else ()
        )

        t = start_t
        turn_idx = 0
        while t < cfg.sim_window_s:
            burst_len = sample_active_burst(rng, cfg.mean_active_burst_turns)
            sess.state = "ACTIVE"

            for _ in range(burst_len):
                if t >= cfg.sim_window_s:
                    break
                size = rng.randint(cfg.turn_min_tokens, cfg.turn_max_tokens)
                turn_toks = tuple(rng.randrange(cfg.vocab_size) for _ in range(size))

                if sid in sharing_sids and rng.random() < 0.6:
                    turn_toks = _maybe_attach_shared_doc(rng, cfg, turn_toks, primary_doc)
                if sid in sharing_sids and secondary_doc and rng.random() < 0.3:
                    turn_toks = _maybe_attach_shared_doc(rng, cfg, turn_toks, secondary_doc)

                role = "user" if (turn_idx % 2 == 0) else "assistant"
                turn = Turn(turn_index=turn_idx, t=t, tokens=turn_toks, role=role)
                sess.turns.append(turn)
                sess.last_turn_t = t

                # Build the prompt actually sent to the engine: the whole
                # conversation so far. The leading turns are byte-identical
                # to the previous request from this session, which is
                # exactly the prefix vLLM can reuse.
                if cfg.accumulate_context:
                    ctx = tuple(tok for tn in sess.turns for tok in tn.tokens)
                    truncated = False
                    if len(ctx) > cfg.max_context_tokens:
                        # Context window overflow: drop oldest turns, as a
                        # real chat app would. This shifts the prefix and
                        # so costs this session its cached blocks.
                        keep: list = []
                        total = 0
                        for tn in reversed(sess.turns):
                            if total + len(tn.tokens) > cfg.max_context_tokens:
                                break
                            keep.append(tn)
                            total += len(tn.tokens)
                        keep.reverse()
                        ctx = tuple(tok for tn in keep for tok in tn.tokens)
                        truncated = True
                else:
                    ctx = turn_toks
                    truncated = False

                events.append(
                    TurnEvent(
                        session_id=sid,
                        turn_index=turn_idx,
                        t=t,
                        tokens=turn_toks,
                        role=role,
                        is_active=True,
                        context_tokens=ctx,
                        context_truncated=truncated,
                    )
                )
                turn_idx += 1
                t += sample_active_inter_turn_dt(rng, cfg.mean_active_inter_turn_dt_s)

            sess.total_active_intervals += 1
            sess.sum_active_burst_len += burst_len
            sess.state = "IDLE"
            sess.last_active_end_t = t
            idle_gap = sample_idle_gap(rng, cfg.mean_idle_gap_s)
            sess.total_idle_intervals += 1
            sess.sum_idle_gap += idle_gap
            t += idle_gap

    events.sort(key=lambda e: (e.t, e.session_id))
    return Workload(
        sessions=sessions,
        events=events,
        shared_doc_pool=shared_pool,
        config=cfg,
        sharing_sids=sharing_sids,
    )

