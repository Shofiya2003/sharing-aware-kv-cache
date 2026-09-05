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
├── results/                      # CSVs and figures (per-run)
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

## Hit/miss classification

vLLM does not currently expose per-request cache hit/miss counters in a stable, queryable way. We use **observed latency** as a proxy:

- A request that served in `< hit_latency_threshold_ms` is treated as a cache hit (mostly prefix-cache reuse, very few tokens to prefill)
- A request above the threshold is a miss (full prefill)

Calibrate the threshold on your actual hardware with `experiments/phase1_smoke.py`, which prints recommended values.


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
2. **Latency-as-hit-proxy is honest.** vLLM doesn't expose per-request hit/miss; we use latency, calibrated on the target hardware. The threshold is reported in every run's logs.
3. **Time-windowed metrics are mandatory.** The phenomenon is about behavior over sustained, irregular concurrency, not a single snapshot. Every run is bucketed at 30 sim seconds.
4. **Incremental result writing.** Free-tier GPU sessions can disconnect mid-matrix. Per-run CSVs are flushed at the end of each run, so a disconnect doesn't lose completed work.


## Limitations / honest notes

- The `MockVLLMBackend` is for development only. It does not model prefix caching. Real vLLM behavior, especially under memory pressure, can differ qualitatively.
- Per-session hit rate from the latency proxy is approximate. For ground truth, run with vLLM's internal cache stats logging if you have the vLLM build that supports it.
- We do not sweep `num_shared_docs` or `overlap_fraction` in the main 8-run matrix. Those ablations are out of scope per the spec ("keep to 4 policies and 2 capacity levels").
- If the combined policy does NOT clearly beat both single-signal baselines, that is a *legitimate finding*. Report it and diagnose.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

14 tests cover: session lifecycle, overlap detection (incl. mid-context), workload determinism, policy orderings, end-to-end benchmark with all 4 policies, metrics CSV writing.

## Citation / use

If you use this in a paper / report, please cite the underlying Preble 2024 work and the vLLM project. See `RESEARCH_NOTE.md` for the framing.

