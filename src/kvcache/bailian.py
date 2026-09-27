"""Loader for the Qwen-Bailian anonymized usage traces (Aliyun, ATC'25).

    https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon  (Apache-2.0)

One JSON line per request: `chat_id`, `parent_chat_id` (-1 for a session's
first request), `timestamp` (s), `input_length`, `output_length`, `turn`,
`hash_ids` (chained hashes of the prompt's 16-token blocks; the last one may
be a partial block). Each request has at most one child, so a session is a
chain of requests.

Unlike WildChat there is no text: only block hashes. That is all the cache
simulator needs (`cachesim.simulate` accepts events carrying `block_ids`).
What a request leaves in the cache is its prompt plus its reply, and the
reply is only visible through the NEXT request of the session, whose prompt
starts with it. So a request's cached blocks are its child's first
floor((input + output) / 16) block IDs; a request with no child gets unique
IDs (nothing will ever read them).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .predictor import FitModel

BLOCK = 16


@dataclass
class HashedSession:
    """Same fields the predictor's `ReturnModel.fit` reads from a WildChat conversation."""
    conv_id: str
    turn_times: List[float]
    input_lens: List[int]
    output_lens: List[int]
    type: str = "text"

    @property
    def n_turns(self) -> int:
        return len(self.turn_times)

    @property
    def start(self) -> float:
        return self.turn_times[0]


@dataclass
class HashedEvent:
    """A request for `cachesim.simulate`, described by block IDs instead of tokens."""
    session_id: str
    turn_index: int
    t: float
    block_ids: Tuple[int, ...]      # prompt + reply, full blocks (prefix of hashes)
    prompt_len: int
    output_len: int
    tokens: Tuple[int, ...] = ()
    role: str = "user"
    is_active: bool = True


def read_sessions(path: str, types: Sequence[str] = ()) -> Tuple[List[HashedSession], List[HashedEvent]]:
    """Sessions (sorted by start) and per-request events (sorted by time).

    `types` optionally keeps only sessions whose first request has that type.
    """
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[r["chat_id"]] = r
    child = {r["parent_chat_id"]: r["chat_id"] for r in rows.values() if r["parent_chat_id"] != -1}
    fresh = itertools.count(1 << 40)                 # IDs that never collide with the trace's
    sessions, events = [], []
    for root in (r for r in rows.values() if r["parent_chat_id"] == -1):
        if types and root["type"] not in types:
            continue
        chain, cur = [root], root["chat_id"]
        while cur in child:
            cur = child[cur]
            chain.append(rows[cur])
        sid = str(root["chat_id"])
        sessions.append(HashedSession(
            sid, [r["timestamp"] for r in chain], [r["input_length"] for r in chain],
            [r["output_length"] for r in chain], root["type"]))
        for k, r in enumerate(chain):
            n_full = (r["input_length"] + r["output_length"]) // BLOCK
            if k + 1 < len(chain):
                ids = tuple(chain[k + 1]["hash_ids"][:n_full])
            else:
                own = r["hash_ids"][:r["input_length"] // BLOCK]
                ids = tuple(own) + tuple(next(fresh) for _ in range(n_full - len(own)))
            events.append(HashedEvent(sid, k, r["timestamp"], ids,
                                      r["input_length"], r["output_length"]))
    sessions.sort(key=lambda s: s.start)
    events.sort(key=lambda e: e.t)
    return sessions, events


def split_sessions(sessions: Sequence[HashedSession], train_frac: float = 0.6):
    """Earliest `train_frac` of sessions (by start) -> train, rest -> test."""
    cut = int(len(sessions) * train_frac)
    return list(sessions[:cut]), list(sessions[cut:])


def fit_model_from(train: Sequence[HashedSession], max_context: int) -> FitModel:
    """Follow-up message lengths: new tokens a user adds on turn >= 2."""
    lens = sorted(s.input_lens[k] - s.input_lens[k - 1] - s.output_lens[k - 1]
                  for s in train for k in range(1, s.n_turns))
    return FitModel(followup_lengths=[x for x in lens if x > 0], max_context=max_context)
