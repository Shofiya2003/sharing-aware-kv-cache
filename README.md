# Session- and Sharing-Aware Request Scheduling for Real LLM Serving

A request-scheduling layer built on top of [vLLM](https://github.com/vllm-project/vllm) that prioritizes and orders concurrent, multi-session, multi-turn LLM requests based on per-session activity signals and cross-session content-sharing signals. Benchmark under realistic irregular concurrent load and constrained GPU cache memory — measuring **real** cache hit rate, latency, and goodput.

> One-line: _Built a session- and sharing-aware request scheduling layer on top of vLLM, benchmarking real cache hit rate, latency, and goodput under concurrent multi-session LLM serving load with constrained GPU memory._

---

## What this is

A scheduling layer (in `src/kvcache/`) that sits **in front of** a real vLLM instance and decides:

- **which queued request to submit next** (FIFO / session-aware / sharing-aware / combined)
- **when to submit it** (as soon as a slot opens, vs. letting it wait)
- **with what priority** (sesssion return-likelihood + cross-session content overlap, weighted)

It does **not** touch vLLM's internal cache, block manager, or eviction. vLLM's real, built-in automatic prefix caching runs unmodified. The cache-pressure lever is `gpu_memory_utilization`: a low value forces real, observable eviction under load.

## Why this is interesting

Prior work (Preble, 2024) showed prefix-aware scheduling beats naive round-robin on workloads with long, static shared prefixes. It included a multi-step embodied-agent workload but did not evaluate a dedicated, realistic **multi-session conversational workload** where many independent sessions remain concurrently live, each accumulating context over many turns, going active/idle unpredictably, *while also sharing overlapping content with each other*. That combination — sustained irregular concurrency + cross-session sharing + real cache pressure — is the specific gap this project targets, on a single GPU rather than Preble's distributed setup.

The design tension under study: an eviction / scheduling policy based only on **session-level signals** ("keep caches for sessions likely to return soon") can starve content that's still valuable to *other* live sessions sharing it. A policy based only on **sharing signals** ("keep whatever's shared by the most sessions") can strand a legitimate, soon-returning session with a unique context. A good policy needs both.


## Repo layout

```
.
├── src/kvcache/                  # Core package
│   ├── session.py                # Multi-turn session lifecycle
│   ├── overlap.py                # Alignment-robust n-gram overlap detector
│   ├── workload.py               # Multi-session workload generator
│   ├── policies.py               # 4 dispatch policies (no cache mutation)
│   ├── vllm_backend.py           # Async wrapper around vLLM
│   ├── metrics.py                # Time-windowed metrics + CSV writer
│   ├── bench.py                  # Driver (asyncio) + MockVLLMBackend
│   └── analysis.py               # Headline / ablation / fairness charts
├── experiments/                  # CLI scripts (one per phase)
│   ├── run_experiment.py         # Main entry point
│   ├── phase1_smoke.py           # vLLM smoke test under pressure
│   ├── phase2_session.py         # One session, multiple real turns
│   ├── phase3_workload.py        # Workload + timeline + overlap report
│   ├── phase4_policies.py        # Policy unit checks
│   ├── phase5_matrix.py          # 4x2 matrix runner
│   └── phase6_analysis.py        # Charts from existing CSVs
├── notebooks/
│   └── kaggle_launcher.ipynb     # Thin GPU launcher
├── tests/test_core.py            # 14 unit + integration tests
├── results/                      # CSVs and figures (per-run, gitignored)
├── README.md                     # this file
├── RESEARCH_NOTE.md              # 1-2 page research note
├── pyproject.toml
└── requirements.txt
```

## Quick start

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
| `sharing-aware` | high cross-session n-gram overlap | blocks shared by ≥2 live sessions |
| `combined` | α · session_score + (1−α) · sharing_score | both signals jointly |

Default α = 0.5. Sweep α in `phase5_matrix.py --combined-alpha <v>` to find the best on your workload.

The policies operate on a `QueuedRequest` priority queue: the driver re-scores the queue before each submission. This means a request that *becomes* more valuable (because a new live session now shares its n-grams) can be re-promoted in real time.

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

## Design choices (the things you should re-read before cold-emailing a PI)

1. **We do not modify vLLM.** The dispatch layer operates purely on submission order. This is a deliberate architectural choice — see the spec's "Architectural correction from earlier drafts" section. The contribution is *how* you use vLLM under pressure, not what vLLM does internally.
2. **Hit/miss is ground truth.** We read vLLM's per-request `num_cached_tokens` and cross-check against the engine's own `gpu_prefix_cache_hit_rate`. Every row carries a `hit_basis` column so a run that lost ground truth cannot be mistaken for one that has it. The old latency proxy is published beside it, labelled, for comparison only.
3. **Time-windowed metrics are mandatory.** The phenomenon is about behavior over sustained, irregular concurrency, not a single snapshot. Every run is bucketed at 30 sim seconds.
4. **Incremental result writing.** Free-tier GPU sessions can disconnect mid-matrix. Per-run CSVs are flushed at the end of each run, so a disconnect doesn't lose completed work.


## Limitations / honest notes

- The `MockVLLMBackend` models an LRU prefix-block pool sized off `gpu_memory_utilization`, so it exercises the ground-truth code path on CPU and responds to capacity pressure. It is still a model: its hashing is strictly prefix-aligned and it does not model batching or preemption. Real vLLM under memory pressure can differ qualitatively.
- vLLM's prefix cache can only reuse a **contiguous prefix from token 0**. Our workload attaches shared documents at random positions (`shared_attach_position="random"`), so mid-context shared content is detected by our overlap index but is *by construction* invisible to vLLM's cache. Run `--shared-attach-position prefix` to get the case vLLM can actually exploit; the contrast between the two is itself a result worth reporting.
- We do not sweep `num_shared_docs` or `overlap_fraction` in the main 8-run matrix. Those ablations are out of scope per the spec ("keep to 4 policies and 2 capacity levels").
- If the combined policy does NOT clearly beat both single-signal baselines, that is a *legitimate finding*. Report it and diagnose.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

14 tests cover: session lifecycle, overlap detection (incl. mid-context), workload determinism, policy orderings, end-to-end benchmark with all 4 policies, metrics CSV writing.

## Citation / use

If you use this in a paper / report, please cite the underlying Preble 2024 work and the vLLM project. See `RESEARCH_NOTE.md` for the framing.

