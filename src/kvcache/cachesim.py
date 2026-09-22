"""CPU model of vLLM's prefix cache, for measuring eviction headroom.

Question it answers
-------------------
Before building predictive eviction into a real engine: on this workload
and at this KV budget, how much better than LRU could ANY eviction policy
do? If an oracle that knows the future barely beats LRU, prediction has
nothing to win. If it beats LRU by a lot, that gap is the target, and we
can see how much of it a realistic (history-only) predictor captures.

Model
-----
- The prompt is split into 16-token blocks with hashes chained from token
  0 (`prefix.block_hashes`), exactly as vLLM's automatic prefix caching
  does, so a block matches only if the entire prompt before it matched.
- A request reuses the longest run of leading blocks that are all
  resident, capped so at least one prompt token is recomputed (vLLM needs
  one to produce logits). After serving, every full block of its prompt
  AND of its reply (`TurnEvent.output_tokens`, empty for synthetic
  workloads) is resident and most recently used: engines cache generated
  tokens too, and a chat's next prompt contains the reply.
- Requests are served one at a time in arrival order. Real serving
  overlaps a few requests (2 engine slots in round 4); that changes which
  blocks are pinned at any instant but not the eviction question, which
  is what this model is for.
- Capacity is a fixed number of KV blocks. Blocks of the request being
  served are never evicted to make room for itself.

Eviction policies (which resident block goes first)
--------------------------------------------------
lru             least recently used; within one request, deepest block
                first (vLLM frees a finished request's blocks tail-first).
                This is what vLLM does, and what Preble/SGLang's local
                `radix_cache.evict()` does (LRU over tree leaves).
lfu             fewest uses so far (popularity); ties by LRU.
preble-cost     fewest uses in the last 3 minutes; ties by LRU. Adapted from
                Preble's E2 routing cost (`SlidingWindowHistogram`: uses in
                a 3-minute window x prefill cost of the node). Preble uses
                it to choose a GPU, not to evict; at 16-token block
                granularity the prefill-cost factor is ~constant, so as an
                eviction rule it reduces to windowed frequency.
predictive      lowest reuse probability from a hand-written heuristic,
                `Session.return_likelihood(now)`, fed with HISTORY ONLY.
reuse           lowest predicted value from `predictor.ReusePredictor`,
                learned from earlier data: P(conversation returns within H
                | idle so far, turns so far) x P(its next prompt still
                fits). H = age of the least recently used resident block.
                History only. Requires `predictor=`.
perfect-return  the session that returns LATEST (or never) loses its
                blocks first, using the TRUE next arrival time. Upper bound
                for any predictor of *when sessions return*.
oracle          Belady: the block whose next use is furthest in the future
                goes first. Upper bound for any eviction policy (up to the
                prefix-chain subtlety that a block is only useful if its
                parent survives too).
infinite        no eviction at all: the ceiling set by compulsory misses
                (first sight of each block, context truncation).
"""

from __future__ import annotations

import bisect
import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .prefix import BLOCK_SIZE, block_hashes
from .session import LiveSessions

# The original four (synthetic study, notebooks/cpu_cache_headroom.ipynb).
POLICIES = ("lru", "predictive", "perfect-return", "oracle")
# Everything, for the WildChat study. `reuse` needs a fitted predictor.
ALL_POLICIES = ("lru", "lfu", "preble-cost", "predictive", "reuse",
                "perfect-return", "oracle")
PREBLE_WINDOW_S = 180.0   # Preble's SlidingWindowHistogram(window=3 min)

# Qwen2.5-1.5B-Instruct: 28 layers x 2 KV heads x 128 head dim x (K and V)
# x 2 bytes (fp16) = 28,672 bytes of KV per token.
KV_BYTES_PER_TOKEN = 28 * 2 * 128 * 2 * 2


def blocks_for_gib(gib: float, block_size: int = BLOCK_SIZE) -> int:
    """KV blocks that fit in `gib` GiB for Qwen2.5-1.5B at fp16."""
    return int(gib * 2**30 // (KV_BYTES_PER_TOKEN * block_size))


@dataclass
class SimResult:
    policy: str
    capacity_blocks: Optional[int]
    n_requests: int
    prompt_tokens: int
    cached_tokens: int
    evictions: int

    @property
    def cached_token_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def recomputed_tokens(self) -> int:
        """Prompt tokens that had to be prefilled because they were not cached."""
        return self.prompt_tokens - self.cached_tokens


class _Block:
    __slots__ = ("last_step", "last_t", "depth", "owners", "uses", "recent")

    def __init__(self, depth: int) -> None:
        self.last_step = 0
        self.last_t = 0.0
        self.depth = depth
        self.owners: set = set()
        self.uses = 0
        self.recent: List[float] = []   # use times, pruned to PREBLE_WINDOW_S


def simulate(events: Sequence, capacity_blocks: Optional[int], policy: str = "lru",
             block_size: int = BLOCK_SIZE, warmup_s: float = 0.0,
             predictor=None) -> SimResult:
    """Replay `events` (workload TurnEvents) through the cache model.

    `capacity_blocks=None` is the infinite cache (policy is then moot).
    Requests issued before `warmup_s` still warm the cache but are left out
    of the reported rate, matching the GPU runs' discarded first window.
    """
    if capacity_blocks is not None:
        if policy not in ALL_POLICIES:
            raise ValueError(f"unknown policy {policy!r}; expected one of {ALL_POLICIES}")
        if policy == "reuse" and predictor is None:
            raise ValueError("policy 'reuse' needs predictor=ReusePredictor(...)")
    evs = sorted(events, key=lambda e: e.t)
    # Hashes of prompt + reply; the prompt's own blocks are a prefix of these.
    hashes = [block_hashes(tuple(e.prompt_tokens) + tuple(getattr(e, "output_tokens", ())),
                           block_size) for e in evs]
    n_prompt_blocks = [len(e.prompt_tokens) // block_size for e in evs]

    # Future knowledge, used ONLY by the oracle / perfect-return policies.
    # A block counts as "used" by a later request only if it is in that
    # request's PROMPT (reply blocks are written, not read).
    uses: Dict[int, List[int]] = {}
    for i, (hs, n) in enumerate(zip(hashes, n_prompt_blocks)):
        for h in hs[:n]:
            uses.setdefault(h, []).append(i)
    arrivals: Dict[str, List[float]] = {}
    for e in evs:
        arrivals.setdefault(e.session_id, []).append(e.t)

    # History only: what a real server knows about each conversation.
    live = LiveSessions()                       # for `predictive`
    state: Dict[str, tuple] = {}                # sid -> (turns done, last t, cached length)
    cache: Dict[int, _Block] = {}
    prompt_tok = cached_tok = n_req = evictions = 0

    for i, (ev, hs) in enumerate(zip(evs, hashes)):
        now = ev.t
        step = i + 1
        n_tok = len(ev.prompt_tokens)

        # 1. Longest resident prefix of the PROMPT, leaving >= 1 token to recompute.
        matched = 0
        for h in hs[:n_prompt_blocks[i]]:
            if h not in cache:
                break
            matched += 1
        matched = min(matched, max(0, (n_tok - 1) // block_size))
        if now >= warmup_s:
            n_req += 1
            prompt_tok += n_tok
            cached_tok += matched * block_size

        # 2. Make room for this request's blocks that are not resident.
        keep = hs if capacity_blocks is None else hs[:capacity_blocks]
        mine = set(keep)
        need = sum(1 for h in keep if h not in cache)
        if capacity_blocks is not None:
            overflow = len(cache) + need - capacity_blocks
            if overflow > 0:
                key = _victim_key(policy, now, i, cache, live, state, uses,
                                  arrivals, predictor)
                victims = heapq.nsmallest(
                    overflow, (h for h in cache if h not in mine),
                    key=lambda h: key(h, cache[h]))
                for h in victims:
                    del cache[h]
                evictions += len(victims)

        # 3. Serve: all full blocks resident and most recently used.
        for depth, h in enumerate(keep):
            b = cache.get(h)
            if b is None:
                b = cache[h] = _Block(depth)
            b.last_step = step
            b.last_t = now
            b.owners.add(ev.session_id)
            b.uses += 1
            b.recent.append(now)
        live.observe(ev.session_id, ev.turn_index, ev.t, ev.tokens, ev.role)
        state[ev.session_id] = (ev.turn_index + 1, now,
                                n_tok + len(getattr(ev, "output_tokens", ())))

    return SimResult(policy if capacity_blocks is not None else "infinite",
                     capacity_blocks, n_req, prompt_tok, cached_tok, evictions)


def _victim_key(policy, now, i, cache, live, state, uses, arrivals, predictor):
    """Sort key: the smallest key is evicted first."""
    if policy == "lru":
        return lambda h, b: (b.last_step, -b.depth)

    if policy == "lfu":
        return lambda h, b: (b.uses, b.last_step, -b.depth)

    if policy == "preble-cost":
        cutoff = now - PREBLE_WINDOW_S

        def key(h, b):
            r = b.recent
            if r and r[0] < cutoff:
                del r[:bisect.bisect_left(r, cutoff)]
            return (len(r), b.last_step, -b.depth)
        return key

    if policy == "predictive":
        p_cache: Dict[str, float] = {}

        def p_return(sid: str) -> float:
            if sid not in p_cache:
                s = live.get(sid)
                p_cache[sid] = s.return_likelihood(now) if s else 0.0
            return p_cache[sid]

        def key(h, b):
            miss = 1.0
            for sid in b.owners:
                miss *= 1.0 - p_return(sid)
            return (1.0 - miss, b.last_step, -b.depth)
        return key

    if policy == "reuse":
        # H: how long the cache currently keeps an idle block under LRU.
        horizon = max(1.0, now - min(b.last_t for b in cache.values()))
        v_cache: Dict[str, float] = {}

        def value(sid: str) -> float:
            if sid not in v_cache:
                turns, last_t, length = state[sid]
                v_cache[sid] = predictor.session_value(turns, now - last_t, length, horizon)
            return v_cache[sid]

        def key(h, b):
            miss = 1.0
            for sid in b.owners:
                miss *= 1.0 - value(sid)
            return (1.0 - miss, b.last_step, -b.depth)
        return key

    if policy == "perfect-return":
        nxt: Dict[str, float] = {}

        def next_arrival(sid: str) -> float:
            if sid not in nxt:
                ts = arrivals.get(sid, [])
                j = bisect.bisect_right(ts, now)
                nxt[sid] = ts[j] if j < len(ts) else math.inf
            return nxt[sid]

        def key(h, b):
            soonest = min((next_arrival(s) for s in b.owners), default=math.inf)
            return (-soonest, b.last_step, -b.depth)
        return key

    if policy == "oracle":
        def key(h, b):
            u = uses.get(h, [])
            j = bisect.bisect_right(u, i)
            nu = u[j] if j < len(u) else math.inf
            return (-nu, -b.depth)
        return key

    raise ValueError(policy)
