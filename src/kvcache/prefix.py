"""Prefix-block index: what another session could actually let vLLM reuse.

Why this replaced the n-gram overlap detector
---------------------------------------------
The previous detector (`overlap.py`) hashed every 8-token window of every
prompt and called a request "shared" if any window had been seen in
another session, at any position. vLLM cannot reuse such content. A
token's K/V depend on its position (RoPE) and, past layer 0, on every
token before it, so identical text after a different opening produces a
different KV cache. vLLM's automatic prefix caching therefore hashes the
prompt in fixed-size blocks, each hash chained to the previous one from
token 0, and reuses a block only when every block before it matched too.

This module mirrors exactly that:

  block_hashes(tokens)  -> [h0, h1, ...]  with  h_i = hash(h_{i-1}, block_i)

so two prompts share block i if and only if their first (i+1)*block_size
tokens are identical -- which is the condition for vLLM to reuse it.

`PrefixIndex` remembers the blocks of recently served prompts and which
sessions sent them. It is a model of what the engine *might* still hold,
not a view into vLLM: when `capacity_blocks` is set it forgets the least
recently used blocks beyond that budget, roughly as vLLM would evict them.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Set, Tuple

# vLLM's default KV block size. Only FULL blocks are ever cached or reused.
BLOCK_SIZE = 16


def block_hashes(tokens: Sequence[int], block_size: int = BLOCK_SIZE) -> List[int]:
    """Chained hashes of the prompt's full blocks, from token 0.

    Deterministic across processes: Python's hash of a tuple of ints is not
    salted (only str/bytes hashing is randomized).
    """
    out: List[int] = []
    prev = 0
    for i in range(0, len(tokens) - block_size + 1, block_size):
        prev = hash((prev, tuple(tokens[i:i + block_size])))
        out.append(prev)
    return out


class PrefixIndex:
    """Block hash -> sessions whose served prompts contained that block.

    Only the chained prefix structure is stored, so a lookup can only ever
    match content that sits at the same position after an identical
    opening: the content vLLM can reuse.
    """

    def __init__(self, block_size: int = BLOCK_SIZE,
                 capacity_blocks: Optional[int] = None) -> None:
        self.block_size = block_size
        self.capacity_blocks = capacity_blocks
        # hash -> {session_id}; insertion order = recency (LRU at the front).
        self._blocks: "OrderedDict[int, Set[str]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._blocks)

    # ------------------------------------------------------------ writes

    def add(self, session_id: str, tokens: Sequence[int]) -> None:
        """Record a served prompt: all of its full blocks, as most recent."""
        for h in block_hashes(tokens, self.block_size):
            owners = self._blocks.pop(h, None) or set()
            owners.add(session_id)
            self._blocks[h] = owners
        if self.capacity_blocks is not None:
            while len(self._blocks) > self.capacity_blocks:
                self._blocks.popitem(last=False)

    # ------------------------------------------------------------- reads

    def shared_prefix(self, session_id: str,
                      tokens: Sequence[int]) -> Tuple[int, Set[str]]:
        """Leading tokens this prompt shares with OTHER sessions' prompts.

        Returns `(n_tokens, other_sessions)`: the length of the longest
        run of leading blocks that some other session also sent, and which
        sessions those were. Blocks only this session sent do not count --
        that is the session's own history, not sharing.
        """
        n = 0
        others: Set[str] = set()
        for h in block_hashes(tokens, self.block_size):
            owners = self._blocks.get(h)
            if not owners:
                break
            other = owners - {session_id}
            if not other:
                break
            others |= other
            n += self.block_size
        return n, others

    def cached_prefix(self, tokens: Sequence[int]) -> int:
        """Leading tokens present in the index from ANY session."""
        n = 0
        for h in block_hashes(tokens, self.block_size):
            if h not in self._blocks:
                break
            n += self.block_size
        return n
