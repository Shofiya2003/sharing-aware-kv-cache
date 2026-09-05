"""Phase 3 — non-prefix-aligned cross-session overlap detection.

Shared content in real multi-session LLM workloads does not start at
position 0. Two different conversations might both reference the same
shared document or tool output, but each starts with its own unique
preamble. We therefore cannot assume start-of-string alignment.

We use a token-n-gram matching scheme: any n-gram of length `n` that
appears in the token sequences of two or more live sessions is treated as
shared. When a new block is written, we look up its n-grams in a global
`OverlapIndex`; if any n-gram is already known to be referenced by
another live session, we register a cross-session reference on the
matching cached block.

This is intentionally simple (n-gram presence, not LCS) because we are
emulating a *cache-side* reuse detector, not a full deduplicating
filesystem. It is also deterministic and O(n) per write.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple


def ngrams(tokens: Tuple[int, ...], n: int) -> Iterable[Tuple[int, ...]]:
    """Yield all n-grams of `tokens`."""
    if n <= 0 or len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield tokens[i : i + n]


def ngram_id(gram: Tuple[int, ...]) -> str:
    """Stable id for a token n-gram."""
    h = hashlib.sha1()
    for tok in gram:
        h.update(int(tok).to_bytes(8, "little", signed=False))
    return h.hexdigest()[:16]


def detect_shared_ngrams(
    tokens_a: Tuple[int, ...],
    tokens_b: Tuple[int, ...],
    n: int = 8,
) -> int:
    """Return the count of distinct n-grams shared between two token sequences.

    Used in unit tests to verify the overlap detector.
    """
    set_a: Set[str] = {ngram_id(g) for g in ngrams(tokens_a, n)}
    set_b: Set[str] = {ngram_id(g) for g in ngrams(tokens_b, n)}
    return len(set_a & set_b)


class OverlapIndex:
    """Global registry of n-grams that are known to be referenced by ≥1 session.

    Layout:
      _owners: ngram_id -> set of (owner_session_id, block_id) that contain
               this n-gram and are currently cached.
      _refs:   ngram_id -> set of session_ids that have referenced this n-gram.
    """

    def __init__(self, n: int = 8) -> None:
        self.n = n
        self._owners: Dict[str, Set[Tuple[str, int]]] = defaultdict(set)
        self._refs: Dict[str, Set[str]] = defaultdict(set)

    # ------------------------------------------------------------------ writes

    def register_block(
        self,
        owner_session_id: str,
        block_id: int,
        tokens: Tuple[int, ...],
    ) -> List[Tuple[str, str]]:
        """Register a freshly cached block. Returns a list of (other_session_id, ngram_id)
        for any pre-existing cross-session matches discovered.

        Each tuple signals: `other_session_id` already references n-gram
        `ngram_id`, so the caller should register a cross-session reference
        on this block for `other_session_id`.
        """
        discoveries: List[Tuple[str, str]] = []
        for gram in ngrams(tokens, self.n):
            gid = ngram_id(gram)
            # Record this (owner, block) as containing the n-gram.
            self._owners[gid].add((owner_session_id, block_id))
            # If some other live session has referenced this n-gram before,
            # this is a sharing event.
            other_refs = self._refs.get(gid, set())
            for other_sid in other_refs:
                if other_sid != owner_session_id:
                    discoveries.append((other_sid, gid))
            # Mark the owner itself as having referenced it.
            self._refs[gid].add(owner_session_id)
        return discoveries

    # ------------------------------------------------------------------ reads

    def has_other_refs(self, owner_session_id: str, tokens: Tuple[int, ...]) -> Set[str]:
        """Return the set of other session IDs that already reference any
        n-gram in `tokens`. Used at lookup time (no new block being written)
        to register a hit-time sharing reference.
        """
        others: Set[str] = set()
        for gram in ngrams(tokens, self.n):
            gid = ngram_id(gram)
            for sid in self._refs.get(gid, ()):
                if sid != owner_session_id:
                    others.add(sid)
        return others

    def touch_session(self, owner_session_id: str, tokens: Tuple[int, ...]) -> None:
        """Mark `owner_session_id` as currently-referencing all n-grams in
        `tokens`. Useful when a session reads (but does not write) content
        that overlaps with cached content elsewhere.
        """
        for gram in ngrams(tokens, self.n):
            gid = ngram_id(gram)
            self._refs[gid].add(owner_session_id)

    def unregister_session(self, session_id: str) -> None:
        """Remove a session from all reference sets (e.g. on sim end)."""
        for gid, sids in list(self._refs.items()):
            sids.discard(session_id)
            if not sids:
                self._refs.pop(gid, None)

    # ------------------------------------------------------------------ stats

    def referenced_ngram_count(self) -> int:
        return len(self._refs)

    # ------------------------------------------------------------------ queries

    def find_other_owners(
        self,
        tokens: Tuple[int, ...],
        owner_session_id: str,
    ) -> List[Tuple[str, int]]:
        """Return all cached (owner, block_id) pairs whose blocks share an
        n-gram with `tokens` and are *not* owned by `owner_session_id`.

        Used by the benchmark driver to attribute cross-session sharing hits
        and to register references.
        """
        hits: List[Tuple[str, int]] = []
        seen: Set[Tuple[str, int]] = set()
        for gram in ngrams(tokens, self.n):
            gid = ngram_id(gram)
            for owner, bid in self._owners.get(gid, ()):
                if owner == owner_session_id:
                    continue
                key = (owner, bid)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(key)
        return hits


