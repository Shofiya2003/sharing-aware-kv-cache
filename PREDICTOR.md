# Reuse predictor for KV-cache eviction

A running record of what the predictor is, why it is built this way, and
what has been implemented and measured. Companion to
[CACHE_SIMULATION.md](CACHE_SIMULATION.md), which covers the simulator it is
evaluated in.

---

## Goal

When the KV cache is full, evict the cached tokens **least likely to be
reused**, instead of the least recently used ones (LRU, what vLLM and SGLang
do). The simulator showed the prize: an oracle that knows the future beats
LRU by about 4 points of cached tokens at a T4-sized budget.

## What must be predicted, and why

Cached tokens belong to a conversation. They are reused only if **both**:

1. **the conversation returns** before the cache would have dropped them
   anyway, and
2. **its next prompt still starts with them**. If the conversation has grown
   past the context limit, the next prompt drops old turns, its opening
   changes, and none of its cached tokens match any more.

So the value of keeping a conversation's cache is

```
value = P(returns within H | idle so far, turns so far)  ×  P(next prompt still fits)
```

where **H** is how long the cache currently keeps things (see below).

Why both terms: in the synthetic study, perfect knowledge of return times
matched the oracle when nothing was truncated, but did *worse* than LRU when
37% of requests were truncated (CACHE_SIMULATION.md, finding 3). On WildChat
at a 4,096-token limit only 0.6% of requests are truncated, so the first
term should do most of the work; the second keeps the policy correct when
contexts are long.

## Term 1: will the conversation return, and soon?

Learned from the **training days** of WildChat only.

For every training conversation and every turn *k*, record what happened
next: either the gap until turn *k+1*, or **END** (no further turn). Group
by how many turns the conversation has had so far (1, 2, 3, 4–5, 6–9, 10+),
because conversations that have already continued are more likely to
continue again.

For a group, with `p_end` = the fraction that ended and `ECDF` = the
distribution of observed gaps:

```
F(x) = (1 − p_end) · ECDF(x)        probability of returning within x seconds

P(return within H | idle for a) = (F(a + H) − F(a)) / (1 − F(a))
```

The division is what makes idle time informative: once a user has been
silent for `a` seconds, only outcomes later than `a` are still possible.
With heavy-tailed gaps (WildChat's coefficient of variation is ~8), this
probability falls the longer a user stays idle. With the memoryless gaps of
the synthetic generator it would not change at all, which is why the
earlier history-based predictor learned nothing.

## Term 2: will the next prompt still fit?

The next prompt = the conversation so far (last prompt + reply, already
cached) + the user's next message. It fits if

```
current_length + next_message_length ≤ max_context
```

`next_message_length` is unknown, so use the distribution of follow-up
message lengths from the training days:

```
P(fits) = ECDF_followup_length(max_context − current_length)
```

## The horizon H

"Returns soon" only matters relative to how long the cache holds things.
H = the age of the least recently used block currently cached: the time
LRU would keep an idle conversation. It adapts automatically: under heavy
load H is short, so only conversations likely to return very soon are
worth protecting.

## How it becomes an eviction policy

Each cached block belongs to one or more conversations. Its value is
`1 − Π(1 − value_s)` over those conversations (a system prompt shared by
everyone is therefore always kept). Evict the lowest-value blocks first;
within one conversation, the deepest blocks first (they are useless
without the blocks before them); ties broken by LRU.

The predictor sees only the past: turns already served, their times and
lengths. It never sees a conversation's future.

## Baselines it is compared against

| Policy | Rule |
|---|---|
| `lru` | least recently used (vLLM; also Preble's local eviction, `radix_cache.evict()`) |
| `lfu` | least frequently used |
| `preble-cost` | adapted from Preble's routing cost model (`SlidingWindowHistogram`): uses in the last 3 minutes × recompute cost. Preble uses this to pick a GPU, not to evict; applying it to eviction is our adaptation |
| `predictive` | the earlier history-only heuristic (`Session.return_likelihood`) |
| `perfect-return` | true next-arrival times (upper bound for return-time prediction) |
| `oracle` | Belady: evicts the block needed furthest in the future (upper bound for any policy) |

Headline metric: **fraction of the LRU → oracle gap closed**, alongside
cached-token rate and recomputed tokens, over several loads and memory
budgets.

---

## Implementation log

| Date | Step | Status |
|---|---|---|
| 2026-09-21 | WildChat loader (`src/kvcache/wildchat.py`), time-based train/test split | done, commit `584a649` |
| 2026-09-21 | Predictor model (`src/kvcache/predictor.py`) and fit script (`experiments/predictor_fit.py`) | done |
| 2026-09-21 | Standalone check on the test days | done: calibrated, AUC 0.786 (results below) |
| | Eviction policies in the simulator (`reuse`, `lfu`, `preble-cost`) | |
| | Evaluation sweep over load and memory; results table | |

---

## Results

### Standalone check (2026-09-21)

`PYTHONPATH=src python experiments/predictor_fit.py` (about 25 s). Fitted on
35,914 conversations from the first 60% of days; checked on 23,943 from the
last 40%, which it never saw.

**What happens after a turn** (training days):

| Turns so far | Cases | Ended there | Median gap | p90 gap |
|---|---|---|---|---|
| 1 | 35,914 | 46.9% | 111 s | 1,474 s |
| 2 | 19,072 | 33.3% | 113 s | 1,487 s |
| 3 | 12,715 | 28.3% | 112 s | 1,423 s |
| 4–5 | 15,899 | 24.8% | 108 s | 1,358 s |
| 6–9 | 14,668 | 22.6% | 88 s | 1,103 s |
| 10+ | 7,450 | 24.8% | 63 s | 697 s |

Conversations that have already continued are less likely to end, and
longer conversations come back faster.

**P(returns within 5 minutes | idle so far)**:

| Turns so far | idle 0 s | 60 s | 120 s | 300 s | 900 s | 1,800 s | 3,600 s |
|---|---|---|---|---|---|---|---|
| 1 | 0.357 | 0.228 | 0.150 | 0.086 | 0.036 | 0.011 | 0.003 |
| 2 | 0.454 | 0.326 | 0.224 | 0.130 | 0.053 | 0.016 | 0.008 |
| 4–5 | 0.531 | 0.405 | 0.284 | 0.161 | 0.063 | 0.023 | 0.008 |
| 10+ | 0.620 | 0.426 | 0.265 | 0.127 | 0.038 | 0.013 | 0.003 |

Idle time is highly informative: a conversation silent for 5 minutes is
about 4× less likely to return in the next 5 minutes than one that just
finished a turn. (Under the synthetic generator's memoryless gaps these
rows would be flat, which is why the earlier heuristic learned nothing.)

**Held-out quality** (339,117 test situations, horizon 5 min, 27.3%
returned):

| | Value |
|---|---|
| Brier score, predictor (lower is better) | **0.160** |
| Brier score, constant average rate | 0.199 |
| AUC (0.5 = coin flip) | **0.786** |

Calibration: when it predicts 1.6% the observed rate is 1.3%; 14.8% → 14.7%;
39.0% → 41.1%; 56.1% → 59.1%. The probabilities can be used as
probabilities, not just as a ranking.

**P(next prompt still fits 4,096 tokens)**: 0.999 at 1,000 tokens, 0.993 at
3,000, 0.949 at 3,900, 0 once full. Follow-up messages are short, so this
term only matters for conversations near the limit, as expected.
