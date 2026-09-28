# Reuse predictor for KV-cache eviction

Replace LRU eviction with a learned estimate of *which cached conversation
is most likely to be used again*. This file is the report: what was tried, in
what order, what changes in Preble, how to reproduce every number, and the
results, including the one that did not work out well.

Related files:

- [PROBABILITY_GUIDE.md](PROBABILITY_GUIDE.md): how the probabilities are
  calculated, from scratch.
- [CACHE_SIMULATION.md](CACHE_SIMULATION.md): the simulator and the synthetic
  study that motivated this.

## Summary

- **Idea.** When the cache is full, evict the leaf with the lowest
  `value = P(conversation returns within H | idle time, turns so far) × P(its next prompt still fits)`, instead of the least recently used one.
- **Quality.** On held-out WildChat data the return probability is calibrated
  (Brier 0.160 vs 0.199 for a constant rate, AUC 0.786). On a second real
  trace (Qwen-Bailian) it is equally good (AUC 0.784), and WildChat's tables
  applied unchanged reach 0.783.
- **Eviction, WildChat.** Raises the cached-token rate over LRU by +1.1 to
  +6.8 points (about 4–11% less prefill work) in all 36 simulator seed-runs,
  and by +1.4 to +6.4 points in all 27 seed-runs inside Preble's real radix
  cache. That is 10–20% of the distance from LRU to an oracle that sees the
  future. Knowing true return times would cover 92–97% of it, so the eviction
  rule is right and prediction accuracy is the limit.
- **Eviction, Bailian.** Weak: +0.1 to +3.1 points (0.1–3.9% less prefill),
  ahead of LRU in 7 of 9 runs, and the WildChat tables do not help there.
  Causes untested.
- **Not shown.** Any effect with Preble's scheduler or router, on a GPU, or
  on latency and throughput. Requests are replayed in arrival order.

---

## Phases, in the order we did them

1. **Real serving, scheduling only.** A session-aware scheduler in front of
   an unmodified vLLM on a Kaggle T4 (README.md). It can reorder requests but
   cannot change what vLLM evicts, so it cannot test an eviction idea.
2. **CPU simulator** (`src/kvcache/cachesim.py`, CACHE_SIMULATION.md). A model
   of a prefix cache of 16-token blocks, replayed on synthetic multi-session
   traffic. An oracle that knows the future beats LRU by about 4 points of
   cached tokens at a T4-sized budget. Two lessons: a predictor must
   estimate return time *and* whether the next prompt still matches, and
   synthetic gaps are memoryless, so idle time carries no information there.
3. **Eviction rules compared in the simulator.** LRU, LFU, `preble-cost`
   (Preble's windowed-use routing cost, adapted), a hand-written history
   heuristic (`predictive`), `perfect-return` (told true next-arrival times,
   a ceiling and not deployable) and `oracle` (Belady). On synthetic
   traffic the heuristic was slightly *worse* than LRU. Reason:
   plausible-looking prediction without data hurts.
4. **Real traffic and a learned predictor** (WildChat). Return probability
   and fit probability counted from the earlier 60% of days, evaluated on the
   later 40%: first on its own (calibration, Brier, AUC), then as the
   eviction rule in the simulator.
5. **Preble's real code.** Preble's own `RadixCache` driven with the same
   events, with only the eviction order swapped.
6. **A second dataset** (Qwen-Bailian Trace A), added last as a
   generalisation check. Quality transfers; the eviction benefit does not
   (details at the end).

---

## What we change in Preble

Preble (`WukLab/preble`, commit `1a35eae`) has two separate decisions:

| Decision | Where | Touched? |
|---|---|---|
| Which GPU gets the request (global E2 router) | router | no |
| Which queued request runs next (local priority queue over the radix tree) | `python/sglang/srt/managers/router/scheduler.py` | no |
| Prefix matching, insertion, node splitting, lock pinning | `radix_cache.py` | no |
| **Which cached leaf to drop when memory is needed** | **`RadixCache.evict()`, `radix_cache.py:133`** | **yes** |

`evict()` is called from `model_rpc.py:355` when a batch needs KV memory. Stock
behaviour: collect the leaves, heapify by `last_access_time` (`TreeNode.__lt__`),
pop the oldest, skip nodes with `lock_ref > 0`, delete the leaf, and push its
parent when it becomes a leaf.

**Our change, one line of intent:** the heap is ordered by
`(1 − Π(1 − value_s), last_access_time)` instead of `last_access_time`, where
`s` ranges over the conversations that own the leaf. Everything else in
`evict()` is unchanged. `experiments/preble_radix_eval.py` does this in a
subclass (`PredictorCache.evict`); Preble's repository is never edited.

What the predictor needs that stock Preble does not have:

1. **Owners per node.** Which conversations passed through each tree node;
   copied on `_split_node`. (Harness: `Tagged.tag`.)
2. **Per-conversation state**: turns done, time of last request, cached
   length. Updated when a request is served.
3. **A session ID on each request.** In the harness the replay supplies it. In
   a live system it would have to travel with the request; that plumbing is
   not built.
4. **The fitted tables**, produced offline from a trace and loaded at startup.
5. **A clock.** Preble stamps nodes with `time.time()`; the harness replaces it
   with the trace's timestamps because a replay runs in seconds, not days.
   That is a test-harness detail, not part of the change.

Interaction with scheduling is the open question. The scheduler decides which
request pins which prefix; eviction decides what is dropped when room is
needed. We change only the second, but a scheduler that reorders requests
changes the idle gaps the predictor was fitted on. Not tested.

---

## How the predictor works

Short version; the derivation with worked numbers is in
[PROBABILITY_GUIDE.md](PROBABILITY_GUIDE.md).

```
value(conversation) = P(returns within H | idle a, turns so far)  ×  P(next prompt fits)
value(leaf)         = 1 − Π (1 − value(s))     over the conversations s owning it
```

- Return term: counted from training data. Per turn group (1, 2, 3, 4–5, 6–9,
  10+): the share of conversations that ended, and the sorted gaps to the next
  turn. `F(x) = (1 − p_end) · ECDF_gaps(x)`, and
  `P = (F(a + H) − F(a)) / (1 − F(a))`.
- Fit term: the share of past follow-up messages short enough to fit in
  `max_context − cached length`.
- `H` = age of the least recently used cached leaf: a heuristic for how long
  the cache holds anything. Not derived, and not ablated.
- Nothing in it assumes WildChat, but every number is counted from a trace, so
  the tables must be refitted on each workload (seconds).

Code: `src/kvcache/predictor.py`.

---

## Reproduce everything

Tested on macOS, Python 3.13.5, torch 2.8.0, transformers 4.57.6, numpy 2.2.6,
pandas 2.2.3, pyarrow 25.0.1. No GPU. All commands run from the repo root.

**1. Environment**

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 transformers==4.57.6 numpy pandas pyarrow
git clone https://github.com/WukLab/preble ~/development/preble
git -C ~/development/preble checkout 1a35eae     # the commit used here
export PYTHONPATH=src
```

The first run downloads the Qwen2.5-1.5B-Instruct tokenizer (tokenizer only)
from Hugging Face. (`requirements.txt` is for the GPU experiments and pins
vLLM; it is not needed here.)

**2. Data** (`data/` is gitignored)

```bash
mkdir -p data/wildchat data/bailian
curl -L -o data/wildchat/train-00000-of-00014.parquet \
  https://huggingface.co/datasets/allenai/WildChat-1M/resolve/main/data/train-00000-of-00014.parquet
curl -L -o data/bailian/qwen_traceA_blksz_16.jsonl \
  https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/main/qwen_traceA_blksz_16.jsonl
```

Use `media.githubusercontent.com` for Bailian: the `raw` URL returns a
133-byte Git LFS pointer. Checksums of the files used here (SHA-256):

```
abec2a13129db8c0e6a2d3a51ff12644873c748205a6fdf6551fbcb34430e51c  data/wildchat/train-00000-of-00014.parquet
07cedc9ed8aff301994ac68ed4aede8123b7603673575eeba9dd677de663db17  data/bailian/qwen_traceA_blksz_16.jsonl
```

**3. Run**

| Step | Command | Time | Output |
|---|---|---|---|
| Unit tests | `python -m unittest tests.test_core` | seconds | 51 tests pass |
| Predictor quality, WildChat | `python experiments/predictor_fit.py` | ~25 s | printed |
| Eviction, WildChat simulator | `python experiments/wildchat_eviction.py` | ~18 min | `results/wildchat/eviction.csv` |
| Inside Preble's radix cache | `python experiments/preble_radix_eval.py --preble ~/development/preble --rates 10,20,40 --capacities 1000,3004,6000` | ~10 min | `results/preble/radix_eval.csv` |
| Second dataset, quality + eviction | `python experiments/bailian_eval.py` | ~21 min | `results/bailian/eviction.csv` |
| Synthetic study | `python experiments/cache_headroom.py` | ~3 min | `results/cpu_headroom/headroom.csv` |

Everything is seeded (seeds 0, 1, 2), so reruns reproduce the committed CSVs.
Checked: `wildchat_eviction.py` reproduces `eviction.csv` exactly (264 rows,
maximum difference 0.0); `bailian_eval.py` reproduces `results/bailian/eviction.csv` exactly (75
rows, 0 differences); `preble_radix_eval.py` reproduces the 30 rows of its default
settings (rates 20 and 40, 3,004 blocks) exactly. `predictor_fit.py` prints
the same Brier and AUC as above.

Numbers in this file come from these CSVs and printed outputs; the summary
tables below are computed from them (for example, "gap closed" is
`(policy − LRU) / (oracle − LRU)`, paired by seed and setting).

---

## Results

**How to read the numbers.** *Cached-token rate* = share of prompt tokens
found in the cache instead of recomputed (higher is better); "points" are
percentage points of it. *Prefill work saved* = the drop in recomputed tokens
relative to LRU. *Oracle* (Belady) evicts what is needed furthest in the
future, and *perfect-return* evicts by the true next-arrival time of each
conversation; both use the future, so they are ceilings that show how much
room there is, not policies anyone can run. *Gap closed* = (policy − LRU) /
(oracle − LRU): the share of the distance from LRU to the oracle. It is the
least intuitive column, so the plain cached-token rates are given next to it.

### Predictor quality on its own (WildChat)

Fitted on 35,914 conversations (first 60% of days); checked on 23,943 from
the last 40%, 339,117 test situations, horizon 5 min, 27.3% returned.

| | Value |
|---|---|
| Brier score, predictor (lower is better) | **0.160** |
| Brier score, constant average rate | 0.199 |
| AUC (0.5 = coin flip) | **0.786** |

Calibration: predicted 1.6% → observed 1.3%; 14.8% → 14.7%; 39.0% → 41.1%;
56.1% → 59.1%.

Idle time matters: P(returns within 5 min) for a conversation with 2 turns so
far is 0.454 just after a turn, 0.326 after 60 s idle, 0.130 after 300 s,
0.016 after 1,800 s. (Full table: `predictor_fit.py` output.)

### Eviction, WildChat simulator

3 seeds (each a separate slice of 2,000 test conversations), loads of 5–40
new conversations per minute, budgets of 1,000 / 3,004 (≈ T4 at
`gpu_memory_utilization=0.3`) / 6,000 blocks, context limit 4,096, first 10
minutes excluded.

Cached-token rate at 3,004 blocks (mean of 3 seeds):

| Load (new conv/min) | LRU | Reuse predictor | Perfect return | Oracle | Infinite |
|---|---|---|---|---|---|
| 5 | 0.671 | 0.688 | 0.818 | 0.831 | 0.876 |
| 10 | 0.547 | 0.575 | 0.773 | 0.786 | 0.877 |
| 20 | 0.369 | 0.436 | 0.708 | 0.719 | 0.881 |
| 40 | 0.257 | 0.321 | 0.625 | 0.638 | 0.886 |

Share of the LRU → oracle gap closed, all 12 settings:

| Policy | Gap closed | Settings above LRU |
|---|---|---|
| `reuse` (learned predictor) | **+10% to +20%** | **12/12 (36/36 seed-runs)** |
| `perfect-return` | +92% to +97% | 12/12 |
| `preble-cost` | 0% to +8% | 7/12 |
| `predictive` (hand-written) | −170% to 0% | 0/12 |
| `lfu` | −575% to −17% | 0/12 |

The predictor saves 4–11% of prefill tokens relative to LRU (perfect return
would save 27–57%).

What this shows:

1. When a conversation returns is almost the whole story: evicting by true
   return time closes 92–97% of the gap.
2. The learned predictor beats LRU everywhere but recovers only 10–20% of
   what perfect return knowledge would; return prediction accuracy is the
   bottleneck.
3. Frequency is the wrong signal (LFU keeps long-finished conversations).
   Preble's windowed cost roughly matches LRU, which fits its purpose
   (balancing GPUs, not ranking evictions).
4. The hand-written heuristic is worse than LRU everywhere.

### Inside Preble's real radix cache (WildChat)

Method as in "What we change in Preble". Differences from the simulator are
Preble's own behaviour: capacity in tokens (budget × 16), whole leaves
evicted at a time (partial eviction off), and the running request pins its
matched prefix while room is made.

Cached-token rate, mean of 3 seeds:

| Load (new conv/min) | Blocks | Preble LRU | Preble + predictor | Gain | Simulator gain |
|---|---|---|---|---|---|
| 10 | 1,000 | 0.202 | 0.266 | +0.064 | +0.068 |
| 10 | 3,004 | 0.553 | 0.584 | +0.031 | +0.028 |
| 10 | 6,000 | 0.714 | 0.728 | +0.014 | +0.014 |
| 20 | 1,000 | 0.144 | 0.193 | +0.049 | +0.060 |
| 20 | 3,004 | 0.375 | 0.438 | +0.063 | +0.067 |
| 20 | 6,000 | 0.615 | 0.641 | +0.026 | +0.025 |
| 40 | 1,000 | 0.135 | 0.158 | +0.023 | +0.034 |
| 40 | 3,004 | 0.263 | 0.321 | +0.058 | +0.065 |
| 40 | 6,000 | 0.474 | 0.529 | +0.055 | +0.054 |

The predictor beats Preble's LRU in all 27 seed-runs (minimum gain +0.010),
its gain tracks the simulator's, and Preble's LRU matches the simulator's LRU
within 0.011, which supports the simulator as a model of the cache.

### Second dataset: Qwen-Bailian Trace A (2026-09-25)

Aliyun's anonymised production chat trace (ATC '25): 23,101 sessions, 43,058
requests, block hashes and timestamps, no text. Sessions split by start time,
first 60% train / last 40% test (`src/kvcache/bailian.py`,
`experiments/bailian_eval.py`). Trace B (single-turn API calls) has no
sessions and cannot be used.

**Predictor quality**, 97,661 held-out situations, horizon 300 s, 18.2%
returned:

| Tables | Brier | AUC |
|---|---|---|
| Constant average rate | 0.1490 | - |
| Refit on Bailian's training sessions | 0.1266 | 0.784 |
| WildChat's tables, unchanged | 0.1281 | 0.783 |

The refit is calibrated (predicted 0.37 → observed 0.39, 0.55 → 0.61). The
WildChat tables rank equally well but are overconfident in the middle
(predicted 0.08 → observed 0.04; 0.38 → 0.32). The signal, that long idle
means unlikely to return, is a property of both traces, not just WildChat.

**Eviction** (simulator; 10% of sessions per seed, 3 seeds, cache = 2% / 5% /
10% of the unique blocks; whole trace replayed, requests counted from the
start of the test period). Cached-token rate, mean of 3 seeds:

| Cache | LRU | Reuse (refit) | Reuse (WildChat tables) | Perfect return | Oracle |
|---|---|---|---|---|---|
| 2% | 0.208 | 0.239 | 0.227 | 0.487 | 0.492 |
| 5% | 0.399 | 0.399 | 0.387 | 0.648 | 0.652 |
| 10% | 0.539 | 0.544 | 0.539 | 0.701 | 0.706 |

Gap closed: refit +10.9% / +0.3% / +2.8%, ahead of LRU in 7 of 9 runs;
WildChat tables +6.6% / −4.6% / −0.4%, ahead in 4 of 9. Perfect return closes
97–99%, so as on WildChat the rule is fine and prediction is the limit, but
here the learned predictor gets far less of it.

**Not yet explained.** Candidate causes, none tested: prompts are very long
here (about 190 blocks on average, up to about 90,000 tokens), so few
requests fit per cache; many leaves may tie on value and fall back to LRU; the
32,768-token context limit for the fit term is assumed (it is not in the
trace); and the 10% session subsample changes the load. The Preble radix
harness has not been run on Bailian.

---

## Limits

- Arrival-order replay: no Preble scheduler, router, priority queue or GPU,
  and no latency or throughput measurement. The result is that the predictor
  works inside Preble's real eviction path, not that Preble with it is
  faster end to end.
- Two datasets, both chat. Preble's own benchmarks (ToolBench, LooGLE, video
  QA, APPS) are mostly independent requests over shared prefixes with
  generated arrival times, where idle time carries no signal.
- `H`, turn groups and the independence assumptions (return vs fit, owners)
  are design choices and are not ablated.
- `perfect-return` and `oracle` use the future: ceilings, not deployable.
- The predictor must be refitted per workload.
- Closest prior work: "KVCache Cache in the Wild" (ATC '25) fits a
  per-category exponential reuse-time distribution for eviction; this work
  uses non-parametric conditioning on idle time and turn count, and adds a
  fit term (as far as we know; we did not read that paper's code).
