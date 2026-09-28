# A learned reuse predictor for KV-cache eviction

When an LLM server's KV cache is full, something has to go. vLLM and
Preble/SGLang both drop the **least recently used** prefix. This project
replaces that with a **learned estimate of which cached conversation is least
likely to be reused**, counted from real chat traffic, and measures whether it
keeps more useful tokens cached.

It is evaluated in a CPU simulator and inside Preble's own, unmodified
`RadixCache`, on two independent real traces (WildChat, Qwen-Bailian).
**No GPU and no vLLM are needed to reproduce any of it.**

- **[PREDICTOR.md](PREDICTOR.md)**: the report. What was tried and in what
  order, what changes in Preble, every result, the limits, and step-by-step
  reproduction.
- **[PROBABILITY_GUIDE.md](PROBABILITY_GUIDE.md)**: the probability math
  behind the predictor, explained from scratch with hand-checkable examples.
- [CACHE_SIMULATION.md](CACHE_SIMULATION.md): the simulator and the
  synthetic study that motivated this.

## Results

**What is measured.** The *cached-token rate*: the share of each request's
prompt tokens that were still in the cache when the request arrived, so the
server did not have to recompute them. Higher is better, and every point is
prefill work not done. The baseline is **LRU** (drop the least recently used
entry), which is what vLLM and Preble do today. Requests are replayed from
real chat logs; the predictor is fitted on earlier days and tested on later
ones it has never seen.

### WildChat, in a simulator

Cache sized like a T4 (3,004 blocks of 16 tokens), at four loads. Mean of 3
seeds:

| New conversations per minute | LRU | With the predictor | Change | Prefill work saved |
|---|---|---|---|---|
| 5 | 67.1% | 68.8% | +1.7 points | 5.1% |
| 10 | 54.7% | 57.5% | +2.8 points | 6.2% |
| 20 | 36.9% | 43.6% | +6.7 points | 10.7% |
| 40 | 25.7% | 32.1% | +6.4 points | 8.8% |

Over all 12 settings (4 loads × 3 cache sizes) the predictor beats LRU every
time, by +1.1 to +6.8 points, saving about 4–11% of prefill work; it is ahead
in all 36 individual runs. The gain is largest when the cache is under
pressure (small cache or high load), because that is when eviction choices
matter.

### WildChat, inside Preble's own cache

The same replay through Preble's real, unmodified `RadixCache`, changing only
which leaf it evicts. At 3,004 blocks: 10 conversations/min 55.3% → 58.4%,
20/min 37.5% → 43.8%, 40/min 26.3% → 32.1%. Over 9 settings the gain is +1.4
to +6.4 points, ahead of Preble's LRU in all 27 runs.

### Qwen-Bailian, a second, independent trace

Predictor refitted on this trace. Cache sized as a share of the unique data
in the replay. Mean of 3 seeds:

| Cache size | LRU | With the predictor | Change | Prefill work saved |
|---|---|---|---|---|
| 2% | 20.8% | 23.9% | +3.1 points | 3.9% |
| 5% | 39.9% | 39.9% | +0.1 points | 0.1% |
| 10% | 53.9% | 54.4% | +0.5 points | 1.1% |

Ahead of LRU in 7 of 9 runs, but the gain is small, and at the two larger
cache sizes it is close to nothing. We do not yet know why.

### How good is the prediction itself?

Separately from eviction, does it tell which conversations will come back
within 5 minutes? Score: AUC, the chance that, given one conversation that
came back and one that did not, it rated the returning one higher (50% = coin
flip). **WildChat 78.6%, Bailian 78.4%**, and using WildChat's tables
unchanged on Bailian gives 78.3%. Its probabilities are also calibrated: on
WildChat, when it says 39% about 41% return.

### How much room is there?

For reference, a cache that could see the future (an impossible upper bound)
would reach 71.9% at 20 conversations/min, 3,004 blocks, against LRU's 36.9%
and the predictor's 43.6%. Eviction has a lot of headroom, and the
predictor gets part of it. Telling the cache each conversation's exact return
time would get almost all the way there (70.8%), so the remaining problem is
predicting return times better, not the eviction rule itself. The full
comparison, with the other policies tried, is in PREDICTOR.md.

## How it plugs into Preble

Only one decision is changed: which leaf `RadixCache.evict()` drops. The
router, the scheduler and prefix matching are untouched.

```
request needs KV memory
        │
        ▼
RadixCache.evict()                (radix_cache.py)
  collect leaves ── heapify ── pop the lowest key ── skip if pinned ── delete
                                     │
   stock Preble:   key = last_access_time                 oldest leaf first
   this project:   key = 1 - Π(1 - value_s)               least likely reused first
                         over the conversations s that own the leaf

   value_s = P(s returns within H | idle so far, turns so far)  ×  P(its next prompt still fits)
             └─ counted from past traces, per turn group ──────┘
```

## What is and is not shown

Shown: a predictor fitted on earlier data beats LRU on later data, replayed in
arrival order, in a simulator and inside Preble's real eviction code.

Not shown: any effect with Preble's scheduler or router, on a GPU, or on
latency or throughput. The predictor must be refitted for each workload, and
the horizon `H` and the independence assumptions are design choices that have
not been ablated. See PREDICTOR.md, "Limits".

## Quick start (CPU only, no GPU or vLLM)

Every number in [PREDICTOR.md](PREDICTOR.md) comes from the steps below.
All scripts run from the repo root and are seeded, so reruns reproduce the
committed CSVs.

```bash
# 1. environment (tested: Python 3.13, torch 2.8.0, transformers 4.57.6)
python -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 transformers==4.57.6 numpy pandas pyarrow matplotlib
export PYTHONPATH=src

# 2. Preble, at the commit used here (read, never modified)
git clone https://github.com/WukLab/preble ~/development/preble
git -C ~/development/preble checkout 1a35eae

# 3. data (gitignored): one WildChat shard and Qwen-Bailian Trace A
mkdir -p data/wildchat data/bailian
curl -L -o data/wildchat/train-00000-of-00014.parquet \
  https://huggingface.co/datasets/allenai/WildChat-1M/resolve/main/data/train-00000-of-00014.parquet
curl -L -o data/bailian/qwen_traceA_blksz_16.jsonl \
  https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/main/qwen_traceA_blksz_16.jsonl

# 4. sanity check
python -m unittest tests.test_core                     # 51 tests, ~20 s

# 5. the experiments
python experiments/predictor_fit.py                    # ~25 s   predictor quality on held-out WildChat
python experiments/wildchat_eviction.py                # ~18 min simulator sweep -> results/wildchat/eviction.csv
python experiments/preble_radix_eval.py --preble ~/development/preble \
    --rates 10,20,40 --capacities 1000,3004,6000       # ~10 min inside Preble -> results/preble/radix_eval.csv
python experiments/bailian_eval.py                     # ~21 min second dataset -> results/bailian/eviction.csv
```

Notes:

- The Bailian download must use `media.githubusercontent.com`; the `raw`
  URL returns a 133-byte Git LFS pointer, not the data.
- The first run downloads the Qwen2.5-1.5B-Instruct tokenizer (tokenizer
  only) from Hugging Face.
- `torch` is used only by the Preble harness (Preble's `RadixCache` handles
  tensors). Everything else needs numpy, pandas, pyarrow and, for WildChat,
  the tokenizer.
- File checksums, what each script prints, and the earlier synthetic study
  (`cache_headroom.py`) are in PREDICTOR.md, "Reproduce everything".

## Repo layout

```
src/kvcache/
  predictor.py        the reuse predictor (ReturnModel, FitModel, ReusePredictor)
  cachesim.py         CPU model of the prefix cache and every eviction policy
  wildchat.py         WildChat loader, time-based train/test split, workload
  bailian.py          Qwen-Bailian loader (block hashes, no text)
  prefix.py           vLLM-style chained block hashes
experiments/
  predictor_fit.py        predictor quality on held-out data (Brier, AUC, calibration)
  wildchat_eviction.py    simulator sweep on WildChat
  preble_radix_eval.py    the same replay through Preble's real RadixCache
  bailian_eval.py         quality and eviction on the second dataset
  cache_headroom.py       the earlier synthetic study
results/                  CSVs behind every table (wildchat, preble, bailian, cpu_headroom)
tests/test_core.py        51 tests
PREDICTOR.md  PROBABILITY_GUIDE.md  CACHE_SIMULATION.md
```

The rest of the repo (`policies.py`, `bench.py`, `vllm_backend.py`,
`workload.py`, `session.py`, `metrics.py`, `analysis.py`, the `phase*`
scripts, the notebooks and `requirements.txt`) belongs to the earlier work
below. That work does need a GPU and vLLM.

---

# Earlier work: session-aware request scheduling in front of vLLM (needs a GPU)

**Status: where this project started, and why it moved.** This is the first
set of experiments, run on real vLLM on a free-tier Kaggle/Colab T4. It did
not produce a conclusive result:

- The first round of numbers was discarded because of measurement defects
  (a latency-threshold stand-in for the hit rate, prompts without
  conversation history, warmup contamination; see "Two other corrections"
  below).
- The most recent matrix with the corrected measurement (the one in
  `results copy 2/`, not committed) was a single seed, and the four dispatch policies came out within 0.0014 of each other
  in cached-token rate (0.546–0.547). That is not distinguishable from
  noise, and one seed cannot say more.
- Reordering requests cannot change what vLLM evicts, so this design could
  not test an eviction idea at all.
- With only free-tier GPU sessions, more seeds, larger workloads and other
  engines were out of reach.

So the work moved to CPU-only experiments: a cache simulator, real chat
traces and Preble's real `RadixCache`, which is what the top of this README
describes. This section is kept for the record. None of the results above use GPU
output.

## What this is

A scheduling layer (in `src/kvcache/`) that sits **in front of** a real vLLM instance and decides:

- **which queued request to submit next** (FIFO / session-aware / sharing-aware / combined)
- **when to submit it** (as soon as a slot opens, vs. letting it wait)
- **with what priority** (sesssion return-likelihood + cross-session content overlap, weighted)

It does **not** touch vLLM's internal cache, block manager, or eviction. vLLM's real, built-in automatic prefix caching runs unmodified. The cache-pressure lever is `gpu_memory_utilization`: a low value forces real, observable eviction under load.

## Why this is interesting

Prior work (Preble, 2024) showed prefix-aware scheduling beats naive round-robin on workloads with long, static shared prefixes. It included a multi-step embodied-agent workload but did not evaluate a dedicated, realistic **multi-session conversational workload** where many independent sessions remain concurrently live, each accumulating context over many turns, going active/idle unpredictably, *while also sharing overlapping content with each other*. That combination — sustained irregular concurrency + cross-session sharing + real cache pressure — is the specific gap this project targets, on a single GPU rather than Preble's distributed setup.

The design tension under study: an eviction / scheduling policy based only on **session-level signals** ("keep caches for sessions likely to return soon") can starve content that's still valuable to *other* live sessions sharing it. A policy based only on **sharing signals** ("keep whatever's shared by the most sessions") can strand a legitimate, soon-returning session with a unique context. A good policy needs both.


## Quick start for the GPU scheduling experiments

### Run on a Kaggle/Colab free T4

```bash
git clone https://github.com/<you>/sharing-aware-kv-cache.git
cd sharing-aware-kv-cache
pip install -r requirements.txt

# Phase 1: confirm vLLM is up and observe pressure
python experiments/phase1_smoke.py --gpu-memory 0.3

# Phase 5: run the full 4x2 matrix (writes per-run CSVs incrementally)
python experiments/phase5_matrix.py

# Phase 6: charts + draft interpretation
python experiments/phase6_analysis.py
```

For Kaggle, see `notebooks/kaggle_launcher.ipynb` for a thin launcher.

### Run on CPU (no GPU required) for testing

The repo ships a `MockVLLMBackend` that produces qualitatively similar hit/miss patterns based on prompt length. Useful for development without a GPU.

```bash
python experiments/run_experiment.py --mock
```


## The four dispatch policies

| Policy | Submission priority | What it protects |
|---|---|---|
| `fifo` | arrival order | nothing in particular |
| `session-aware` | high session return-likelihood + large context | sessions likely to return soon |
| `sharing-aware` | longest prompt prefix (whole 16-token blocks from token 0) that other sessions recently sent | cross-session prefix blocks vLLM can actually reuse |
| `combined` | α · session_score + (1−α) · sharing_score | both signals jointly |

Default α = 0.5. Sweep α in `phase5_matrix.py --combined-alpha <v>` to find the best on your workload.

The policies operate on a `QueuedRequest` priority queue: the driver re-scores the queue before each submission. This means a request that *becomes* more valuable (because another session has just sent the same prompt opening) can be re-promoted in real time.

## Hit/miss classification — ground truth, not a proxy

We read vLLM's own per-request prefix-cache counter, `num_cached_tokens` on
`RequestOutput`: the number of prompt tokens served from an existing KV block
instead of being prefilled.

**Headline metric** (token-level, what the charts and tables report):

```
cached_token_rate = sum(num_cached_tokens) / sum(n_prompt_tokens)
```

This is the same quantity vLLM reports internally as
`gpu_prefix_cache_hit_rate`. Every run logs the engine's own value next to
ours as `engine_prefix_cache_hit_rate` in `summary_<label>.csv`; if the two
disagree materially, the run is not trustworthy and should not be reported.

**Per-request binary** (used only for the per-session fairness ECDF): a
request is a hit when `num_cached_tokens / n_prompt_tokens >=
hit_cached_fraction` (default 0.10, so trivial block-boundary reuse does not
register as a hit).

Every output row carries a `hit_basis` column: `cached_tokens` means ground
truth, `latency_proxy` means this vLLM build reported no counter and the
numbers are **not** cache hit rates. The run log prints a loud warning in
that case.

### Why not latency (and why the first round of results was wrong)

An earlier version of this harness had no ground-truth counter and
thresholded end-to-end latency instead, at 797 ms. That does not work here,
and the results it produced are not comparable to the current ones:

- With `max_new_tokens=24`, latency is dominated by decode, not prefill. The
  entire steady-state distribution was a single mode spanning ~660–1100 ms.
- A threshold placed inside that mode measures *instantaneous batch queueing*,
  not cache reuse. In one run, window 6 (p50 = 710 ms) scored a hit rate of
  1.00 while window 8 (p50 = 870 ms) scored 0.19 — a 160 ms shift in median
  latency swinging the reported "hit rate" by 0.81. No cache behaves that way.

The proxy is still computed and published as a clearly-labelled
`proxy_hit_rate` column, so the gap between it and ground truth is visible
rather than hidden.

## Two other corrections that came with this

**Requests now carry the full conversation.** `TurnEvent.context_tokens` is
the accumulated session history; earlier only the current turn's 32–96 tokens
were submitted. With no shared prefix between a session's turns there was
nothing for the prefix cache to reuse, and the total KV footprint was far too
small for `gpu_memory_utilization` to create any eviction pressure — so both
premises of the experiment were absent from the workload. Mean prompt length
goes from ~109 to ~1,980 tokens; peak live contexts now total ~1.1 GB of KV
against ~1.7 GB available at `gpu_memory_utilization=0.3`.

**Warmup windows are excluded from run summaries.** The first window of a run
carries one-time model-load and CUDA-graph-capture cost. Because
`phase5_matrix.py` iterates policy-major, FIFO ran first and absorbed it,
which is the entire reason FIFO appeared to have an 8x worse P99 (8280 ms vs
~1000 ms) in the first round. Excluding warmup, max P99 beyond window 1 was
1125 ms for FIFO against 1028 ms for sharing-aware — no tail-latency
advantage at all. Warmup windows are still written to the time series, flagged
`is_warmup=1`, so nothing is hidden.


## Key results to report

The 8-run matrix produces:

- Per-run CSV: `results/csv/time_series_<label>.csv` (per-window metrics)
- Per-run CSV: `results/csv/per_session_<label>.csv` (per-session hit rates — for fairness analysis)
- Per-run CSV: `results/csv/summary_<label>.csv` (run-level aggregates)
- Figures: `results/figures/headline_hit_rate_<cap>.png`, `p99_latency_<cap>.png`, `goodput_<cap>.png`, `fairness_<cap>.png`, `shared_hit_rate_<cap>.png`, `ablation_bars.png`
- Auto-generated interpretation starter: `results/INTERPRETATION.md`

The headline metrics:
- **Cache hit rate** (overall, and split: hit on shared content vs. session-unique)
- **P50 / P99 latency**
- **Goodput** (% requests meeting the SLA)
- **Per-session fairness** (ECDF of per-session hit rates)

## Design choices

1. **We do not modify vLLM.** The dispatch layer operates purely on submission order. This is a deliberate architectural choice — see the spec's "Architectural correction from earlier drafts" section. The contribution is *how* you use vLLM under pressure, not what vLLM does internally.
2. **Hit/miss is ground truth.** We read vLLM's per-request `num_cached_tokens` and cross-check against the engine's own `gpu_prefix_cache_hit_rate`. Every row carries a `hit_basis` column so a run that lost ground truth cannot be mistaken for one that has it. The old latency proxy is published beside it, labelled, for comparison only.
3. **Time-windowed metrics are mandatory.** The phenomenon is about behavior over sustained, irregular concurrency, not a single snapshot. Every run is bucketed at 30 sim seconds.
4. **Incremental result writing.** Free-tier GPU sessions can disconnect mid-matrix. Per-run CSVs are flushed at the end of each run, so a disconnect doesn't lose completed work.


## Limitations / honest notes

- The `MockVLLMBackend` models an LRU prefix-block pool sized off `gpu_memory_utilization`, so it exercises the ground-truth code path on CPU and responds to capacity pressure. It is still a model: its hashing is strictly prefix-aligned and it does not model batching or preemption. Real vLLM under memory pressure can differ qualitatively.
- vLLM reuses only a **contiguous prefix from token 0**, which affects the two signals very differently. *Within* a session, position is irrelevant: turn k+1's prompt literally extends turn k's, so mid-context content is reused fine (~89% of each prompt is reusable from the same session's previous turn). *Across* sessions, reuse needs a common prefix from token 0, which our default workload almost never produces (~0.15% of each prompt; 3 of 105 session pairs). So by default the sharing-aware signal has almost nothing to protect. Run the matrix at both `--shared-attach-position random` and `prefix` — that contrast is what makes the sharing half of the thesis testable, not a side ablation.
- 32% of turn transitions hit the context cap and lose their whole prefix, which is the main confound on the session-aware half. Tracked as `context_truncated_rate`; raise `--max-context-tokens` to reduce it.
- We do not sweep `num_shared_docs` or `overlap_fraction` in the main 8-run matrix. Those ablations are out of scope per the spec ("keep to 4 policies and 2 capacity levels").
- If the combined policy does NOT clearly beat both single-signal baselines, that is a *legitimate finding*. Report it and diagnose.

## Citation / use

If you use this in a paper / report, please cite the underlying Preble 2024 work and the vLLM project. See `RESEARCH_NOTE.md` for the framing.

