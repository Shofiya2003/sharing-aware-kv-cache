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
  is resident and most recently used.
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
                This is what vLLM does.
predictive      lowest estimated reuse probability first, from HISTORY
                ONLY: each owner session's `return_likelihood(now)` from a
                `LiveSessions` table fed with served turns -- the same
                signal the GPU session-aware policy uses. A block several
                sessions hold is kept if any of them is likely to return.
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

POLICIES = ("lru", "predictive", "perfect-return", "oracle")

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


class _Block:
    __slots__ = ("last_step", "depth", "owners")

    def __init__(self, depth: int) -> None:
        self.last_step = 0
        self.depth = depth
        self.owners: set = set()


def simulate(events: Sequence, capacity_blocks: Optional[int], policy: str = "lru",
             block_size: int = BLOCK_SIZE, warmup_s: float = 0.0) -> SimResult:
    """Replay `events` (workload TurnEvents) through the cache model.

    `capacity_blocks=None` is the infinite cache (policy is then moot).
    Requests issued before `warmup_s` still warm the cache but are left out
    of the reported rate, matching the GPU runs' discarded first window.
    """
    if capacity_blocks is not None and policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    evs = sorted(events, key=lambda e: e.t)
    hashes = [block_hashes(e.prompt_tokens, block_size) for e in evs]

    # Future knowledge, used ONLY by the oracle / perfect-return policies.
    uses: Dict[int, List[int]] = {}
    for i, hs in enumerate(hashes):
        for h in hs:
            uses.setdefault(h, []).append(i)
    arrivals: Dict[str, List[float]] = {}
    for e in evs:
        arrivals.setdefault(e.session_id, []).append(e.t)

    live = LiveSessions()  # history only, for the predictive policy
    cache: Dict[int, _Block] = {}
    prompt_tok = cached_tok = n_req = evictions = 0

    for i, (ev, hs) in enumerate(zip(evs, hashes)):
        now = ev.t
        step = i + 1
        n_tok = len(ev.prompt_tokens)

        # 1. Longest resident prefix, leaving >= 1 token to recompute.
        matched = 0
        for h in hs:
            if h not in cache:
                break
            matched += 1
        matched = min(matched, max(0, (n_tok - 1) // block_size))
        if now >= warmup_s:
            n_req += 1
            prompt_tok += n_tok
            cached_tok += matched * block_size

        # 2. Make room for this prompt's blocks that are not resident.
        keep = hs if capacity_blocks is None else hs[:capacity_blocks]
        mine = set(keep)
        need = sum(1 for h in keep if h not in cache)
        if capacity_blocks is not None:
            overflow = len(cache) + need - capacity_blocks
            if overflow > 0:
                key = _victim_key(policy, now, i, cache, live, uses, arrivals)
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
            b.owners.add(ev.session_id)
        live.observe(ev.session_id, ev.turn_index, ev.t, ev.tokens, ev.role)

    return SimResult(policy if capacity_blocks is not None else "infinite",
                     capacity_blocks, n_req, prompt_tok, cached_tok, evictions)


def _victim_key(policy, now, i, cache, live, uses, arrivals):
    """Sort key: the smallest key is evicted first."""
    if policy == "lru":
        return lambda h, b: (b.last_step, -b.depth)

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
