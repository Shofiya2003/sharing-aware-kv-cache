# Research note: scheduling sessions and shared content in front of vLLM

## 1. Motivation

Real LLM serving systems must support many independent, long-lived chat or agent sessions concurrently. Each session grows over many turns; the system has a fixed KV-cache budget; and bursty arrival patterns mean that at any moment only a subset of sessions are active while the rest are alive-but-idle. Two distinct signals can drive cache-management decisions:

1. **Per-session activity** — which sessions are likely to issue a request soon, and which have accumulated context that would be expensive to recompute if evicted.
2. **Cross-session content sharing** — which cached blocks are referenced by multiple live sessions, and which are unique to one session.

A scheduling policy that uses only one of these signals is brittle: a session-only policy will evict a heavily-shared block owned by an idle session, hurting every other live session that still references it; a sharing-only policy will protect shared content forever and starve the unique-but-soon-returning session whose context gets evicted.

## 2. Gap in Preble's evaluation

Preble (2024) showed that prefix-aware scheduling beats naive round-robin on workloads with long, static shared prefixes. It also included a multi-step embodied-agent workload where context accumulates over time. It did not, however, evaluate a dedicated realistic *multi-session conversational* workload where many independent sessions remain concurrently live, each accumulating context over many turns, going active/idle unpredictably, while *also* sharing overlapping content. The combination — sustained irregular concurrency, cross-session sharing, and real cache pressure — was untested.

## 3. Method

We build a request-dispatch layer that sits *in front of* vLLM and only controls submission order and timing. We do **not** modify vLLM's internal cache, block manager, or eviction. vLLM's real automatic prefix caching runs unmodified; the cache-pressure lever is `gpu_memory_utilization`. We work on a single vLLM instance on a free-tier T4 GPU.

The workload generator creates 10–20 sessions, each with a stochastic active/idle pattern (geometric burst lengths, exponential idle gaps, staggered start times). A configurable fraction of sessions reference shared content — a small pool of randomly-generated "documents" — with overlap appearing at random positions in the growing context (not necessarily prefix-aligned). The overlap detector is alignment-robust: it uses token-n-gram presence, so shared content that lands mid-context in one session and at the start of another is still recognized.

Four dispatch policies are evaluated:
- **FIFO** — submit in arrival order.
- **Session-aware** — prioritize sessions with high return-likelihood and large context.
- **Sharing-aware** — prioritize requests whose content is referenced by multiple live sessions.
- **Combined** — weighted sum of the two signals, with α swept.

Each request carries the session's full accumulated conversation, as a real chat API call does; this is what grows the per-session KV footprint over time and gives the prefix cache something to reuse. With 15 sessions over a 300 s window, mean prompt length is ~1,980 tokens and peak live contexts total ~1.1 GB of KV — against ~1.7 GB available at `gpu_memory_utilization=0.3`, so the capacity lever binds.

Per-request outcome is measured, not inferred: we read vLLM's own `num_cached_tokens` from each `RequestOutput` — the number of prompt tokens served from an existing KV block rather than prefilled. The headline metric is token-level, `sum(num_cached_tokens) / sum(n_prompt_tokens)`, the same quantity vLLM reports as `gpu_prefix_cache_hit_rate`; the engine's own value is logged beside ours every run as a cross-check. A per-request binary (cached fraction ≥ 0.10) is used only for the per-session fairness ECDF. Every row carries a `hit_basis` column, so a run on a vLLM build that reports no counter is marked unusable rather than silently falling back.

We log time-windowed (30 s buckets) prefix-cache hit rate, P50/P99 latency, goodput, per-session hit rates, and the share of requests whose prefix was invalidated by hitting the context window. The first window of each run is excluded from the summary as engine warmup (model load, CUDA graph capture) but still written out, flagged. Results are written incrementally per run to survive free-tier session disconnects.

### Measurement corrections from the first round

The first round of results is superseded and not comparable. Three defects, in order of severity:

1. **Latency was used as a hit proxy** (threshold 797 ms). With `max_new_tokens=24` the latency distribution is decode-dominated — a single mode spanning ~660–1100 ms — so a threshold inside it measured batch-queueing jitter, not cache reuse. A 160 ms shift in median latency moved the reported "hit rate" from 1.00 to 0.19.
2. **Requests carried only the current turn** (32–96 tokens), not the conversation. With no shared prefix across a session's turns there was nothing for the prefix cache to reuse, and the KV footprint was far too small for `gpu_memory_utilization` to create eviction pressure. Both premises of the experiment were absent from the workload.
3. **Warmup contaminated the headline P99.** `phase5_matrix.py` iterates policy-major, so FIFO ran first and absorbed one-time engine startup. Excluding warmup, FIFO's max P99 was 1125 ms against sharing-aware's 1028 ms — the apparent 8x tail-latency win (8280 ms vs 1012 ms) was entirely run order.


## 4. Ablation finding (placeholder — fill in after the GPU run)

[Replace this section with the actual numbers from `results/INTERPRETATION.md`. The headline chart is `results/figures/headline_hit_rate_constrained.png`; the ablation bars are `results/figures/ablation_bars.png`.]

Before writing anything here, clear the validity checks at the top of
`INTERPRETATION.md`: `hit_basis` must read `cached_tokens` on every row, our
`cached_token_rate` must agree with the engine's own
`engine_prefix_cache_hit_rate`, and **generous must beat constrained for the
same policy** — if it does not, the capacity axis carries no signal and no
claim about behavior "under pressure" is supported. Report effect sizes
against the seed-to-seed spread across at least three seeds, not against
zero.

**Expected pattern, based on the design:**
- At *generous* capacity, all four policies converge — there's enough KV-cache headroom that submission order barely matters.
- At *constrained* capacity, separation appears: a session-only policy starves shared content; a sharing-only policy can starvelegitimate active sessions; the combined policy balances the two.
- The per-session fairness ECDF for the combined policy should not be more skewed than either single-signal baseline. If it is, that is a starvation warning worth reporting.

## 5. Limitations

- **Single-node, single model.** The contribution is *how* you use vLLM under pressure, not what vLLM does internally. Distributed setups like Preble's are explicitly out of scope.
- **Prefix-alignment is a hard constraint, not a measurement choice.** vLLM's prefix cache reuses only a *contiguous prefix from token 0*. Our overlap detector is alignment-robust and finds shared content wherever it sits, but content the detector flags mid-context cannot be reused by the engine no matter how we schedule. Measured directly: moving shared docs to the prompt prefix raises the hit rate from ~0.58 to ~0.88. So a sharing-aware *scheduler* can only exploit sharing that happens to be prefix-aligned; exploiting mid-context sharing would require an engine-side change (block-level dedup) that is explicitly out of scope here. This is the sharpest limitation of the approach and should be stated first in any writeup.
- **Context-window truncation is a confound.** ~28% of requests hit the 3072-token cap and have their session prefix invalidated. That is realistic (real chat apps do this) but it is a miss cause unrelated to the policy, so it is tracked and reported as `context_truncated_rate`.
- **Modest session count.** Free-tier GPU quota caps us at 10–20 concurrent sessions. Production serving systems see 10²–10³ concurrent sessions; the qualitative behavior should hold, but the absolute numbers will differ.
- **Single model.** We use Qwen2.5-1.5B-Instruct. Behavior with much larger models (where KV-cache memory is even more constrained) is an open question.

## 6. Future work

- Sweep `overlap_fraction`, `num_shared_docs`, and `num_sessions` as second-order ablations.
- Sweep the `combined` policy's α in finer increments to find the optimal mix on a given workload regime.
- Add per-block hit counters if/when vLLM exposes them stably; replace the latency proxy.
- Extend to multi-vLLM-instance setups, where the dispatch layer becomes a router and the policy must additionally decide *which instance* to submit to.
- Add a learning-augmented policy (e.g. learn return-likelihood from observed idle-gap distribution) and compare against the hand-coded session_score used here.

## 7. Resume framing (draft bullets, fill in real numbers)

- Built a request-scheduling layer on top of vLLM that prioritizes concurrent LLM session dispatch by session activity and cross-session content-sharing signals, measured against a FIFO baseline on real hardware using vLLM's own per-request prefix-cache counters.
- Designed and ran a benchmark simulating 15–20 concurrent, irregularly active multi-turn chat sessions against a real vLLM instance under constrained GPU KV-cache memory, measuring prefix-cache hit rate, latency, and goodput across [N] seeds; [result, including a negative one if that is what the data says].
- Implemented alignment-robust cross-session content overlap detection, and quantified the gap between detectable sharing and *exploitable* sharing given that vLLM reuses only contiguous prompt prefixes (hit rate ~0.58 random placement vs ~0.88 prefix-aligned).
- Found and corrected three measurement defects in an earlier round of the same experiment — a latency-threshold hit proxy standing in for cache instrumentation, prompts that omitted conversation history, and engine warmup contaminating tail-latency numbers — each of which had produced a result that did not survive re-measurement.

