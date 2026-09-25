"""Replay WildChat through Preble's real RadixCache: stock LRU vs our predictor.

    PYTHONPATH=src python experiments/preble_radix_eval.py --preble ~/development/preble

Preble's local eviction (`RadixCache.evict`) pops leaves from a min-heap
ordered by `last_access_time`, i.e. LRU. Here the unmodified class is loaded
from the Preble checkout and driven with the same events as `cachesim`.
The only change for the predictor arm is which leaf is popped next (the
lowest predicted value first, ties by last access time), applied by a
subclass; the tree, locking (`inc_lock_ref`), prefix matching and node
splitting are Preble's own.

Differences from the block simulator, all Preble's behaviour: capacity is in
tokens, not 16-token blocks; a whole leaf is evicted at a time (partial
eviction off, its default), so eviction can overshoot; the request being
served holds a lock on its matched prefix while room is made.
There is no GPU, scheduler or router here: requests are served in arrival
order, as in `cachesim`.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
import os
import sys
import time as _time
import types

import torch

from kvcache.cachesim import BLOCK_SIZE, simulate
from kvcache.wildchat import (build_workload, load_tokenizer, read_conversations,
                              split_by_time, system_tokens, tokenize)
from wildchat_eviction import DEFAULT_SHARD, fit_predictor, ints


class Clock:
    """Virtual clock: Preble stamps nodes with time.time()."""
    now = 0.0

    @classmethod
    def time(cls):
        return cls.now


def load_radix_cache(preble_dir):
    path = os.path.join(os.path.expanduser(preble_dir), "python", "sglang", "srt",
                        "managers", "router", "radix_cache.py")
    spec = importlib.util.spec_from_file_location("preble_radix_cache", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.time = types.SimpleNamespace(time=Clock.time)   # node timestamps follow the trace
    return mod


def make_classes(mod):
    class Tagged(mod.RadixCache):
        """Preble's RadixCache, tracking which conversations own each node."""

        def _split_node(self, key, child, split_len):
            new_node = super()._split_node(key, child, split_len)
            new_node.owners = set(getattr(child, "owners", ()))
            return new_node

        def tag(self, tokens, sid):
            node, i = self.root_node, 0
            while i < len(tokens) and tokens[i] in node.children:
                node = node.children[tokens[i]]
                if not hasattr(node, "owners"):
                    node.owners = set()
                node.owners.add(sid)
                i += len(node.key)

    class PredictorCache(Tagged):
        """Same evict() as Preble's; only the heap order differs."""
        predictor = None
        state = None            # sid -> (turns done, last time, cached length)

        def evict(self, num_tokens, evict_callback, collect_evicted_node=False):
            leaves = self._collect_leaves()
            now = Clock.now
            horizon = max(1.0, now - min(x.last_access_time for x in leaves))
            vals = {}

            def value(sid):
                if sid not in vals:
                    turns, last_t, length = self.state[sid]
                    vals[sid] = self.predictor.session_value(turns, now - last_t, length, horizon)
                return vals[sid]

            def key(x):
                miss = 1.0
                for sid in getattr(x, "owners", ()):
                    miss *= 1.0 - value(sid)
                return (1.0 - miss, x.last_access_time)

            heap = [(key(x), id(x), x) for x in leaves]
            heapq.heapify(heap)
            num_evicted = 0
            while num_evicted < num_tokens and heap:
                _, _, x = heapq.heappop(heap)
                if x == self.root_node:
                    break
                if x.lock_ref > 0:
                    continue
                n = len(x.value)
                num_evicted += evict_callback(x.value[-n:]).item()
                self._delete_leaf(x, n)
                if len(x.parent.children) == 0:
                    heapq.heappush(heap, (key(x.parent), id(x.parent), x.parent))

    return Tagged, PredictorCache


def replay(cache, events, capacity_tokens, warmup_s, sids=None):
    """Serve events in arrival order. Returns (prompt tokens, cached tokens, evicted tokens)."""
    state = getattr(cache, "state", None)
    resident = prompt_tok = cached_tok = evicted = 0
    cb = lambda v: torch.tensor(len(v))

    for ev in sorted(events, key=lambda e: e.t):
        Clock.now = ev.t
        prompt = list(ev.prompt_tokens)
        full = prompt + list(getattr(ev, "output_tokens", ()))
        value, last_node = cache.match_prefix(prompt)
        matched = min(len(value), max(0, len(prompt) - 1))
        if ev.t >= warmup_s:
            prompt_tok += len(prompt)
            cached_tok += matched

        if capacity_tokens is not None:
            cache.inc_lock_ref(last_node)           # the running request pins its prefix
            overflow = resident + len(full) - len(value) - capacity_tokens
            if overflow > 0:
                before = cache.evictable_size()
                cache.evict(overflow, cb)
                freed = before - cache.evictable_size()
                resident -= freed
                evicted += freed
            cache.dec_lock_ref(last_node)
        already = cache.insert(full, torch.arange(len(full)))
        resident += len(full) - already
        cache.tag(full, ev.session_id)
        if state is not None:
            state[ev.session_id] = (ev.turn_index + 1, ev.t, len(full))
    return prompt_tok, cached_tok, evicted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preble", default="~/development/preble")
    ap.add_argument("--shard", default=DEFAULT_SHARD)
    ap.add_argument("--rates", default="20,40")
    ap.add_argument("--capacities", default="3004", help="KV blocks; x16 = tokens")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--replay", type=int, default=2000)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--warmup-s", type=float, default=600.0)
    ap.add_argument("--out", default="results/preble/radix_eval.csv")
    a = ap.parse_args()

    mod = load_radix_cache(a.preble)
    Tagged, PredictorCache = make_classes(mod)

    convs, _ = read_conversations(a.shard)
    train, test = split_by_time(convs, 0.6)
    tok = load_tokenizer()
    predictor = fit_predictor(train, tok, a.max_context)
    system = system_tokens(tok)

    rows = []
    for seed in ints(a.seeds):
        block = test[seed * a.replay:(seed + 1) * a.replay]
        tokenize(block, tok)
        for rate in ints(a.rates):
            events = build_workload(block, system, rate, a.max_context, seed=seed)
            for cap in [None] + ints(a.capacities):
                cap_tok = None if cap is None else cap * BLOCK_SIZE
                arms = []
                t0 = _time.time()
                for name, cls in (("preble-lru", Tagged), ("preble-predictor", PredictorCache)):
                    if cap is None and name != "preble-lru":
                        continue
                    c = cls(None, None)
                    if cls is PredictorCache:
                        c.predictor, c.state = predictor, {}
                    p, ct, ev = replay(c, events, cap_tok, a.warmup_s)
                    arms.append((name if cap else "preble-infinite", p, ct, ev))
                if cap is not None:
                    for pol in ("lru", "reuse"):
                        r = simulate(events, cap, pol, warmup_s=a.warmup_s, predictor=predictor)
                        arms.append((f"sim-{pol}", r.prompt_tokens, r.cached_tokens, r.evictions))
                for name, p, ct, ev in arms:
                    rows.append(dict(rate_per_min=rate, capacity_blocks=cap or "inf", seed=seed,
                                     policy=name, cached_token_rate=round(ct / p, 6),
                                     prompt_tokens=p, evictions=ev))
                    print(f"seed {seed} rate {rate:g} cap {cap or 'inf'} {name:17s} "
                          f"cached {ct / p:.3f}", flush=True)
                print(f"  ({_time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
