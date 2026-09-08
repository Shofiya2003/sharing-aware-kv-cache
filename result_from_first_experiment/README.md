# Results — first GPU experiment (frozen archive, do not modify)

8-run matrix (4 policies × generous/constrained) on a Kaggle T4, run with
the code as of the Phase-1 warmup/threshold fixes (pre matrix-run warmup:
each run booted a fresh engine, so window-0 P99 includes cold-start
compilation — see caveat below).

Run config (all 8 runs):
- model Qwen/Qwen2.5-1.5B-Instruct, 15 sessions, 300 sim-seconds, seed 0
- generous gpu_mem 0.7 / constrained gpu_mem 0.3, max_num_seqs 4,
  max_new_tokens 24, speed x10, max_model_len 4096
- sla_latency_ms 2000, hit_latency_threshold_ms 797 (from Phase 1:
  baseline 168ms, burst median 4403ms)

Headline (constrained hit rate): sharing-aware 0.687 > combined 0.625 >
session-aware 0.605 > FIFO 0.555. Combined did NOT beat both baselines;
see INTERPRETATION.md and the analysis in the chat history. Goodput 0.99
everywhere (SLA ceiling — verdict rests on hit rate + P99).

Contents: csv/ (24 per-run CSVs), figures/ (11 charts), INTERPRETATION.md.
Live results/ continues in the repo root for follow-up runs (reseed,
alpha sweep), which use --label-suffix so they never overwrite these.
