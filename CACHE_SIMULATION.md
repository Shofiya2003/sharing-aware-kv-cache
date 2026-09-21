# KV-cache eviction simulation (CPU)

A CPU-only experiment that asks: **when the GPU's KV cache is too small to
hold everything, how much better than vLLM's LRU eviction could we do, and
what would a policy need to predict to get there?**

Everything here runs without a GPU or vLLM. The full sweep takes about
3 minutes.

---

## In plain words

The GPU keeps a cache of the attention state ("KV") for every conversation
it has recently served, so a returning conversation does not have to be
recomputed from scratch. The cache is small. When it is full and a new
request needs room, something must be evicted, and the choice decides how
often the next request finds its history still cached.

We wrote a program that behaves like that cache, fed it thousands of
simulated chat requests, and counted how often the needed history was still
cached (the **hit rate**) under different rules for choosing what to evict.

## Why this experiment exists

The GPU experiments (`notebooks/kaggle_launcher.ipynb`) sit in front of an
unmodified vLLM and can only change the **order** requests are sent in. They
cannot change what vLLM evicts. The research idea under consideration
(predictive, session-aware eviction as an extension to Preble) is about
eviction, so this simulator is the only place in the project where that idea
is tested directly, and it can decide cheaply whether the idea has anything
to win before any engine work.

---

## What is simulated

`src/kvcache/cachesim.py` models vLLM's automatic prefix cache:

- **Blocks.** Prompts are split into 16-token blocks. Each block's hash chains
  from token 0 (`src/kvcache/prefix.py`), exactly as in vLLM, so a block
  matches only if the whole prompt before it matched too. Shared text in the
  middle of a prompt is never reusable: a token's KV depends on its position
  and on every token before it.
- **Reuse.** A request reuses its longest run of leading blocks that are still
  cached, leaving at least one token to recompute (vLLM needs one to produce
  output).
- **After serving,** all of the request's blocks are cached and marked most
  recently used.
- **Capacity** is a fixed number of blocks. For Qwen2.5-1.5B at fp16,
  1 GiB of KV holds 2,340 blocks.
- **Order.** Requests are served one at a time in arrival order. The real GPU
  overlaps a few requests, which shifts absolute rates slightly but not the
  comparison between eviction rules.

The workloads come from the same generator as the GPU runs
(`src/kvcache/workload.py`): multi-turn sessions whose every request carries
the whole conversation so far, capped at a context limit.

### Eviction rules compared

| Rule | Evicts first | Knows the future? | Question it answers |
|---|---|---|---|
| `lru` | least recently used block (tail first) | no | the baseline: what vLLM does |
| `predictive` | blocks of sessions least likely to return, from `return_likelihood` fed with **past turns only** | no | does a realistic return predictor help? |
| `perfect-return` | blocks of the session that returns **latest** (true next-arrival times) | when sessions return | is knowing *when users return* enough? |
| `oracle` | block whose next use is furthest away (Belady) | everything | how much could **any** rule gain? |
| `infinite` | nothing | n/a | the ceiling: misses no cache size can avoid |

How to read the gaps (all at the same budget, same seed):

- `oracle − lru` = **headroom**, the most any eviction rule could add.
- `perfect-return − lru` = the best a return-time predictor could ever add.
- `predictive − lru` = what a history-only predictor actually adds.

### Where the real T4 sits

Round 3 measured `cached_token_rate = 0.5457` on the T4 at
`gpu_memory_utilization=0.3` (`fifo_constrained`, seed 0). Simulated LRU gives
the same rate on the same workload at **3,004 blocks = 1.28 GiB of KV**, inside
the 1–1.7 GiB range the memory arithmetic predicts (15 GiB × 0.3, minus
~2.9 GiB of weights and activation overhead). So the model agrees with the
hardware, and 3,004 blocks is used as the "T4 constrained" budget below.

### Workloads

| Preset | What it is | Requests per seed | Mean prompt | Truncated |
|---|---|---|---|---|
| `arm1_round4` | GPU round 4: 8 sessions × 240 s | 174–245 | 1,147 tok | 8% |
| `arm1_round3` | GPU round 3: 12 sessions × 600 s | 778–1,140 | 2,010 tok | 37% |
| `arm1_round3_notrunc` | round 3 with no context cap (diagnostic only: prompts reach 20k tokens, past the model's 4,096 limit) | 778–1,140 | 3,393 tok | 0% |
| `arm2_preamble` | 20 sessions, a shared doc at **token 0** of every prompt | 510–631 | 1,778 tok | 21% |
| `arm2_mid` | same doc and volume, placed **mid-prompt** | 510–631 | 1,763 tok | 20% |

"Truncated" = requests whose conversation hit the context cap, so their
oldest turns were dropped.

---

## How to run

**Notebook** (charts, tables, and a written verdict):
`notebooks/cpu_cache_headroom.ipynb`, then run all cells. It works from a
local checkout or a Kaggle CPU session (the first cell clones the repo).

**Command line:**

```bash
# everything: all presets, seeds 0-4, budgets 250..8000 blocks
PYTHONPATH=src:experiments python experiments/cache_headroom.py

# one workload, a few budgets
PYTHONPATH=src:experiments python experiments/cache_headroom.py \
    --workloads arm1_round3_notrunc --capacities 3004,6000
```

Output: `results/cpu_headroom/headroom.csv` (one row per workload × seed ×
budget × rule) and `results/cpu_headroom/headroom.png`.

---

## Results (2026-09-21)

Budget 3,004 blocks (≈ T4 constrained), mean over seeds 0–4.

### Hit rate by rule

| Workload | LRU | Predictive | Perfect return | Oracle | Infinite |
|---|---|---|---|---|---|
| `arm1_round3` | 0.560 | 0.531 | 0.543 | 0.601 | 0.601 |
| `arm1_round4` | 0.801 | 0.789 | 0.795 | 0.803 | 0.803 |
| `arm2_mid` | 0.799 | 0.775 | 0.795 | 0.835 | 0.835 |
| `arm2_preamble` | 0.882 | 0.867 | 0.873 | 0.888 | 0.888 |

### Difference from LRU, paired by seed

`mean (seeds above LRU / 5)`

| Workload | Predictive | Perfect return | Oracle (headroom) |
|---|---|---|---|
| `arm1_round3` | −0.029 (0/5) | −0.016 (0/5) | **+0.041 (5/5)** |
| `arm1_round4` | −0.012 (0/5) | −0.006 (1/5) | +0.003 (3/5) |
| `arm2_mid` | −0.024 (0/5) | −0.004 (1/5) | +0.036 (5/5) |
| `arm2_preamble` | −0.015 (0/5) | −0.008 (1/5) | +0.007 (5/5) |

### Truncation on vs off (round-3 workload)

| | Predictive | Perfect return | Oracle |
|---|---|---|---|
| Truncation on (37%), 3,004 blocks | −0.029 (0/5) | −0.016 (0/5) | +0.041 (5/5) |
| **Truncation off, 3,004 blocks** | +0.001 (2/5) | **+0.016 (5/5)** | +0.016 (5/5) |
| Truncation off, 6,000 blocks | +0.000 | +0.000 | +0.000 |

### Cross-session sharing (arm 2)

| | Doc at token 0 | Doc mid-prompt | Difference |
|---|---|---|---|
| Infinite cache | 0.888 | 0.835 | +0.053 |
| T4 budget, LRU | 0.882 | 0.799 | +0.083 |

---

## Findings

1. **There is a prize.** At the T4 budget, perfect eviction beats vLLM's LRU by
   about 4 points of cached tokens on the round-3 workload (0.560 → 0.601,
   every seed). That is the most any eviction rule could add there.

2. **The round-4 GPU workload has no prize.** It fits in the T4's cache, so LRU
   is already at the ceiling (+0.003). No eviction or ordering policy can move
   its hit rate at this budget. Round 4 needs a larger workload or a smaller KV
   budget (around 1,000 blocks shows about +0.04 of headroom) to test anything.

3. **Knowing when a session returns captures the whole prize, but only if its
   cached history will still match.** With truncation on, even perfect
   knowledge of return times did *worse* than LRU. With truncation switched
   off, the same rule matched the oracle exactly (+0.016 vs +0.016, 5/5 seeds).
   The cause: when a conversation hits the context cap, its next prompt drops
   old turns, so the blocks kept for it no longer match and keeping them
   wastes space. So a useful predictor must estimate
   **P(session returns soon) × P(this block is still in its next prompt)**,
   not return time alone.

4. **The history-based predictor gains nothing, and that is the workload's
   fault.** Even with truncation off it is +0.001. The generator draws idle
   gaps from an exponential distribution, which is memoryless: how long a
   session has been idle says nothing about when it returns. Every session also
   keeps returning until the simulation ends. Real users do neither, so this
   result says nothing yet about prediction on real traffic.

5. **Cross-session sharing is real, but only at token 0.** A shared document
   at the start of every prompt is worth +0.053 (infinite cache) to +0.083 (T4
   budget) over the same document placed mid-prompt, where no session can reuse
   another's cache.

## Limitations

- Requests are served one at a time; the GPU overlaps a few.
- The oracle is Belady on blocks: a strong upper bound, not a proven optimum
  when blocks depend on their parents.
- No latency or throughput: the model has no notion of service time yet.
- Single cache only; no multi-GPU routing (Preble's E2).
- Synthetic workload with memoryless idle gaps and sessions that never end
  (finding 4).
- The truncation-off workload exceeds the model's real context limit; it is a
  diagnostic for finding 3, not a deployable setting.

## Next steps

- [ ] **Block-value predictor:** evict by P(return soon) × P(block survives into
      the next prompt). The second factor is largely computable: a session near
      the context cap will lose its middle turns next, while its pinned
      preamble stays valid. Target: close the oracle gap *with* truncation on.
- [ ] **Realistic sessions:** heavy-tailed idle gaps and sessions that end, so a
      history-based predictor has real signal. Then measure how close a
      realistic predictor gets to `perfect-return`.
- [ ] **Popularity baseline (LFU)** for comparison.
- [ ] Later: a service-time model (latency, p99, throughput), a real
      conversation trace, and multi-GPU routing.
