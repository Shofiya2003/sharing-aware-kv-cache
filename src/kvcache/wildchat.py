"""WildChat -> a replayable multi-turn workload for the cache simulator.

WildChat (allenai/WildChat-1M) is ~1M real ChatGPT conversations from 2023.
What makes it useful here, and what our synthetic generator lacked:

- real conversation lengths: about half never get a second turn;
- real gaps between turns, which are heavy-tailed rather than memoryless,
  so how long a user has been idle carries information;
- real text, so prompt lengths and assistant replies are real.

Pipeline, in the order the functions are called:

    read_conversations(parquet)  text + turn times, no tokens yet
    split_by_time(convs)         earlier days -> train, later days -> test
    tokenize(convs)              tokens, only for the conversations we replay
    build_workload(convs, rate)  TurnEvents the simulator consumes

Timing. WildChat timestamps each ASSISTANT message (user messages have
none). We treat turn k's request time as its reply's timestamp: the reply is
stamped seconds after the request, and that offset is similar on every
turn, so the gaps between turns -- which is what matters -- are preserved.

Load. One shard is ~60k conversations over ~25 days, about 1.6 new
conversations per minute: too little traffic to stress one GPU's cache. So
`build_workload` keeps every conversation's internal timing exactly as
recorded but starts conversations at a chosen rate (`sessions_per_min`),
like many users arriving at one server. That rate is the load knob.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from .workload import TurnEvent

# Qwen2.5's chat markup. Every prompt starts with the same system message,
# as in a real deployment, so all sessions share their first few blocks.
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
_SYSTEM = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
_USER = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
_REPLY = "{}<|im_end|>\n"


@dataclass
class Conversation:
    conv_id: str
    start: float                 # unix seconds of the first turn
    turn_times: List[float]      # seconds since `start`; turn_times[0] == 0.0
    user_text: List[str]
    reply_text: List[str]
    # Filled by `tokenize`. Each entry already includes the chat markup, so
    # system + user[0] + reply[0] + user[1] + ... is exactly the prompt.
    user_tokens: Optional[List[Tuple[int, ...]]] = None
    reply_tokens: Optional[List[Tuple[int, ...]]] = None

    @property
    def n_turns(self) -> int:
        return len(self.turn_times)


def _unix(ts) -> float:
    if isinstance(ts, datetime):
        return ts.timestamp()
    return datetime.fromisoformat(str(ts)).timestamp()


def read_conversations(parquet_path: str, limit: Optional[int] = None):
    """Read conversations with their turn times. Returns (convs, n_skipped).

    A conversation is kept only if it strictly alternates user/assistant,
    starting with the user, and every reply has a timestamp.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["conversation_hash", "conversation"])
    rows = table.to_pylist()
    if limit:
        rows = rows[:limit]
    convs, skipped = [], 0
    for row in rows:
        msgs = row["conversation"]
        users, replies, times = [], [], []
        ok = len(msgs) >= 2 and len(msgs) % 2 == 0
        for i in range(0, len(msgs) - 1, 2) if ok else ():
            u, a = msgs[i], msgs[i + 1]
            if u["role"] != "user" or a["role"] != "assistant" or a["timestamp"] is None:
                ok = False
                break
            users.append(u["content"] or "")
            replies.append(a["content"] or "")
            times.append(_unix(a["timestamp"]))
        if not ok or any(t2 < t1 for t1, t2 in zip(times, times[1:])):
            skipped += 1
            continue
        convs.append(Conversation(
            conv_id=row["conversation_hash"], start=times[0],
            turn_times=[t - times[0] for t in times],
            user_text=users, reply_text=replies))
    convs.sort(key=lambda c: c.start)
    return convs, skipped


def split_by_time(convs: Sequence[Conversation], train_frac: float = 0.6):
    """Earliest `train_frac` of conversations (by start) -> train, rest -> test.

    Splitting by time, not at random, is what a deployed predictor faces:
    it learns from the past and is judged on the future.
    """
    ordered = sorted(convs, key=lambda c: c.start)
    cut = int(len(ordered) * train_frac)
    return list(ordered[:cut]), list(ordered[cut:])


def load_tokenizer(name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def tokenize(convs: Sequence[Conversation], tokenizer) -> None:
    """Fill `user_tokens` / `reply_tokens` in place.

    Each message is tokenized on its own, with its chat markup. Real chat
    templates are built the same way, so a turn's prompt is always the
    previous prompt + reply + new user message, token for token: the
    property prefix caching depends on.
    """
    enc = lambda texts: [tuple(x) for x in tokenizer(
        texts, add_special_tokens=False)["input_ids"]]
    for c in convs:
        c.user_tokens = enc([_USER.format(t) for t in c.user_text])
        c.reply_tokens = enc([_REPLY.format(t) for t in c.reply_text])


def system_tokens(tokenizer) -> Tuple[int, ...]:
    return tuple(tokenizer(_SYSTEM, add_special_tokens=False)["input_ids"])


def build_prompt(system: Tuple[int, ...], c: Conversation, k: int,
                 max_context_tokens: int) -> Tuple[Tuple[int, ...], bool]:
    """Prompt for turn k, and whether it had to be truncated to fit.

    Full prompt = system + (user_0 + reply_0) + ... + user_k. If that exceeds
    `max_context_tokens`, drop the OLDEST turns first (keeping the system
    prompt), as chat front-ends do. If the newest message alone still does
    not fit, keep only its end. Truncation changes the prompt's opening, so
    none of the session's cached blocks after the system prompt match.
    """
    turns = [c.user_tokens[j] + c.reply_tokens[j] for j in range(k)]
    current = c.user_tokens[k]
    budget = max_context_tokens - len(system) - len(current)
    kept, used = [], 0
    for t in reversed(turns):          # newest first
        if used + len(t) > budget:
            break
        kept.append(t)
        used += len(t)
    truncated = len(kept) < len(turns)
    if budget < 0:                     # the new message alone is too long
        current = current[-(max_context_tokens - len(system)):]
        truncated = True
    prompt = system + tuple(tok for t in reversed(kept) for tok in t) + current
    return prompt, truncated


def build_workload(convs: Sequence[Conversation], system: Tuple[int, ...],
                   sessions_per_min: float, max_context_tokens: int = 4096,
                   seed: int = 0) -> List[TurnEvent]:
    """One TurnEvent per turn, ready for `cachesim.simulate`.

    Conversations start one after another with exponential gaps at
    `sessions_per_min` (a Poisson arrival process), in their original order.
    Inside a conversation every turn keeps its recorded offset, so real
    think-time and real session endings are replayed unchanged.
    """
    rng = random.Random(seed)
    events, t0 = [], 0.0
    for c in convs:
        if c.user_tokens is None:
            raise ValueError("call tokenize() on these conversations first")
        t0 += rng.expovariate(sessions_per_min / 60.0)
        for k in range(c.n_turns):
            prompt, truncated = build_prompt(system, c, k, max_context_tokens)
            events.append(TurnEvent(
                session_id=c.conv_id, turn_index=k, t=t0 + c.turn_times[k],
                tokens=c.user_tokens[k], role="user", is_active=True,
                context_tokens=prompt, context_truncated=truncated,
                output_tokens=c.reply_tokens[k]))
    events.sort(key=lambda e: e.t)
    return events
