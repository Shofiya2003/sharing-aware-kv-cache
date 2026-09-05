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

Per-request outcome is classified as a hit or miss using observed latency, calibrated on the target hardware. We log time-windowed (30 s buckets) cache hit rate, P50/P99 latency, goodput, and per-session hit rates (for fairness analysis). Results are written incrementally per run to survive free-tier session disconnects.


## 4. Ablation finding (placeholder — fill in after the GPU run)

[Replace this section with the actual numbers from `results/INTERPRETATION.md`. The headline chart is `results/figures/headline_hit_rate_constrained.png`; the ablation bars are `results/figures/ablation_bars.png`.]

**Expected pattern, based on the design:**
- At *generous* capacity, all four policies converge — there's enough KV-cache headroom that submission order barely matters.
- At *constrained* capacity, separation appears: a session-only policy starves shared content; a sharing-only policy can starvelegitimate active sessions; the combined policy balances the two.
- The per-session fairness ECDF for the combined policy should not be more skewed than either single-signal baseline. If it is, that is a starvation warning worth reporting.

## 5. Limitations

- **Single-node, single model.** The contribution is *how* you use vLLM under pressure, not what vLLM does internally. Distributed setups like Preble's are explicitly out of scope.
- **Latency-as-hit-proxy.** vLLM does not currently expose per-request cache hit/miss. We use a calibrated latency threshold. Real per-block hit counters would be strictly better; that is a vLLM-engine-side change we are not making.
- **Modest session count.** Free-tier GPU quota caps us at 10–20 concurrent sessions. Production serving systems see 10²–10³ concurrent sessions; the qualitative behavior should hold, but the absolute numbers will differ.
- **Single model.** We use Qwen2.5-1.5B-Instruct. Behavior with much larger models (where KV-cache memory is even more constrained) is an open question.

## 6. Future work

- Sweep `overlap_fraction`, `num_shared_docs`, and `num_sessions` as second-order ablations.
- Sweep the `combined` policy's α in finer increments to find the optimal mix on a given workload regime.
- Add per-block hit counters if/when vLLM exposes them stably; replace the latency proxy.
- Extend to multi-vLLM-instance setups, where the dispatch layer becomes a router and the policy must additionally decide *which instance* to submit to.
- Add a learning-augmented policy (e.g. learn return-likelihood from observed idle-gap distribution) and compare against the hand-coded session_score used here.

## 7. Resume framing (draft bullets, fill in real numbers)

- Built a request-scheduling layer on top of vLLM that prioritizes concurrent LLM session dispatch by session activity and cross-session content-sharing signals, measured against FIFO baseline on real hardware.
- Designed and ran a benchmark simulating 15–20 concurrent, irregularly active chat/agent sessions against a real vLLM serving instance under constrained GPU cache memory, measuring real cache hit rate, latency, and goodput; combined scheduling signals improved [metric] by [X]% over single-signal baselines at [condition].
- Implemented alignment-robust cross-session content overlap detection to identify shareable KV-cache content across dynamically growing multi-turn contexts.

