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
    # Where cross-session shared content lands.
    #   "session_preamble" -- the doc opens the session, so it sits at
    #       position 0 of EVERY prompt that session issues. This is the only
    #       mode that produces cross-session reuse, because vLLM needs a
    #       contiguous match from token 0. It is also the realistic shared-
    #       document / shared-system-prompt case, and the regime Preble
    #       studied.
    #   "prefix" -- start of the individual TURN. Note this does NOT give a
    #       common prompt prefix once context accumulates: a doc opening
    #       turn 5 sits at offset len(t1..t4) in the submitted prompt.
    #       Kept for continuity; use "session_preamble" for aligned sharing.
    #   "session_mid" -- the matched CONTROL for session_preamble: same doc,
    #       attached once to the first turn but at a non-zero offset. Same
    #       token volume, same prompt lengths, zero cross-session reuse. Use
    #       this as the baseline when measuring what alignment is worth.
    #   "mid" / "random" -- re-attached inside turns throughout the session.
    #       Detectable by the overlap index, not reusable by the engine, and
    #       NOT volume-matched to session_preamble.
    shared_attach_position: str = "random"

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
    # Shared-doc placement draws from its OWN stream so that changing
    # `shared_attach_position` does not perturb the activity model. Without
    # this, the aligned and control arms get different turn counts and token
    # volumes, and the comparison stops isolating alignment.
    place_rng = random.Random((cfg.seed << 1) ^ 0x5EED)

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

                # NOTE: all shared-doc placement below uses `place_rng`,
                # never `rng`, so the activity model is identical across
                # placement modes.
                if cfg.shared_attach_position == "session_mid":
                    # Control arm for session_preamble: the SAME doc, attached
                    # exactly once to the session's first turn, but at a
                    # non-zero offset. Identical token volume and identical
                    # prompt lengths, so the only difference from
                    # session_preamble is whether the doc starts at token 0.
                    # That isolates alignment from volume -- comparing against
                    # "random" instead would confound the two, because random
                    # re-attaches the doc on ~60% of turns and inflates
                    # contexts (73% truncation vs 10%).
                    if sid in sharing_sids and turn_idx == 0:
                        off = place_rng.randint(1, max(1, len(turn_toks)))
                        turn_toks = (turn_toks[:off] + primary_doc
                                     + turn_toks[off:])
                    if sid in sharing_sids and secondary_doc and place_rng.random() < 0.3:
                        off = place_rng.randint(0, len(turn_toks))
                        turn_toks = turn_toks[:off] + secondary_doc + turn_toks[off:]
                elif cfg.shared_attach_position == "session_preamble":
                    # The doc opens the session, unconditionally, so every
                    # prompt this session issues starts with it and sessions
                    # sharing a doc have a real common prefix from token 0.
                    if sid in sharing_sids and turn_idx == 0:
                        turn_toks = primary_doc + turn_toks
                    # The secondary doc still lands mid-context, so a single
                    # run contains both the exploitable and the merely
                    # detectable kind of sharing.
                    if sid in sharing_sids and secondary_doc and place_rng.random() < 0.3:
                        off = place_rng.randint(0, len(turn_toks))
                        turn_toks = turn_toks[:off] + secondary_doc + turn_toks[off:]
                else:
                    if sid in sharing_sids and place_rng.random() < 0.6:
                        turn_toks = _maybe_attach_shared_doc(place_rng, cfg, turn_toks, primary_doc)
                    if sid in sharing_sids and secondary_doc and place_rng.random() < 0.3:
                        turn_toks = _maybe_attach_shared_doc(place_rng, cfg, turn_toks, secondary_doc)

                role = "user" if (turn_idx % 2 == 0) else "assistant"
                # The preamble-bearing turn is pinned so context-window
                # overflow cannot delete the shared prefix.
                is_pinned = (
                    cfg.shared_attach_position == "session_preamble"
                    and sid in sharing_sids
                    and turn_idx == 0
                )
                turn = Turn(turn_index=turn_idx, t=t, tokens=turn_toks,
                            role=role, pinned=is_pinned)
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
                        # Context window overflow. Keep PINNED turns at the
                        # front (system prompt / shared preamble), then fill
                        # with the most recent turns that fit, dropping from
                        # the middle -- which is what real chat apps do and
                        # what keeps the cacheable prefix stable. Dropping
                        # oldest-first instead would delete the preamble and
                        # wipe out cross-session reuse.
                        pinned = [tn for tn in sess.turns if tn.pinned]
                        total = sum(len(tn.tokens) for tn in pinned)
                        keep: list = []
                        for tn in reversed(sess.turns):
                            if tn.pinned:
                                continue
                            if total + len(tn.tokens) > cfg.max_context_tokens:
                                break
                            keep.append(tn)
                            total += len(tn.tokens)
                        keep.reverse()
                        ctx = tuple(
                            tok for tn in (pinned + keep) for tok in tn.tokens
                        )
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

