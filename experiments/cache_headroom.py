"""Eviction headroom on CPU: LRU vs predictive vs oracles, across KV budgets.

Replays each workload through `kvcache.cachesim` (a block-level model of
vLLM's prefix cache) for every eviction policy, seed and capacity, and
writes one CSV row per run. No GPU, no vLLM.

    python experiments/cache_headroom.py                 # all presets
    python experiments/cache_headroom.py --workloads arm1_round3 --seeds 0,1,2

See notebooks/cpu_cache_headroom.ipynb for the charts and how to read them.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List

from kvcache.cachesim import POLICIES, simulate
from kvcache.workload import WorkloadConfig, generate_workload

# Workloads as the GPU launcher runs them.
_ARM1 = dict(max_context_tokens=3072, turn_min_tokens=16, turn_max_tokens=48)
_ARM2 = dict(shared_doc_min_tokens=800, shared_doc_max_tokens=1200,
             num_shared_docs=2, overlap_fraction=0.8,
             turn_min_tokens=16, turn_max_tokens=48,
             mean_active_burst_turns=3.0, mean_idle_gap_s=35.0,
             max_context_tokens=3900)
PRESETS: Dict[str, dict] = {
    # Round 4 (cell 3b): 8 sessions x 240 s.
    "arm1_round4": dict(num_sessions=8, sim_window_s=240, **_ARM1),
    # Round 3: 12 sessions x 600 s. The T4 measured cached_token_rate
    # 0.546 on seed 0 (fifo_constrained), which anchors the capacity axis.
    "arm1_round3": dict(num_sessions=12, sim_window_s=600, **_ARM1),
    # Arm 2 (cell 3e): shared doc at token 0 of every prompt vs mid-prompt.
    "arm2_preamble": dict(num_sessions=20, sim_window_s=300,
                          shared_attach_position="session_preamble", **_ARM2),
    "arm2_mid": dict(num_sessions=20, sim_window_s=300,
                     shared_attach_position="session_mid", **_ARM2),
}
DEFAULT_CAPACITIES = [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
FIELDS = ["workload", "seed", "capacity_blocks", "policy", "cached_token_rate",
          "n_requests", "prompt_tokens", "cached_tokens", "evictions"]


def run(workloads: List[str], seeds: List[int], capacities: List[int],
        warmup_s: float = 30.0, out_csv: str = "", verbose: bool = True) -> List[dict]:
    rows: List[dict] = []
    for name in workloads:
        for seed in seeds:
            t0 = time.time()
            events = generate_workload(WorkloadConfig(seed=seed, **PRESETS[name])).events
            runs = [(None, "infinite")] + [(c, p) for c in capacities for p in POLICIES]
            for cap, pol in runs:
                r = simulate(events, cap, pol, warmup_s=warmup_s)
                rows.append(dict(
                    workload=name, seed=seed,
                    capacity_blocks=("inf" if cap is None else cap),
                    policy=r.policy, cached_token_rate=round(r.cached_token_rate, 6),
                    n_requests=r.n_requests, prompt_tokens=r.prompt_tokens,
                    cached_tokens=r.cached_tokens, evictions=r.evictions))
            if verbose:
                print(f"[headroom] {name} seed {seed}: {len(events)} requests, "
                      f"{len(runs)} runs in {time.time() - t0:.1f}s", flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        if verbose:
            print(f"[headroom] wrote {len(rows)} rows to {out_csv}")
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workloads", default=",".join(PRESETS),
                   help=f"comma-separated, from: {', '.join(PRESETS)}")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--capacities", default=",".join(map(str, DEFAULT_CAPACITIES)),
                   help="KV budgets in 16-token blocks (Qwen2.5-1.5B: 2340 per GiB)")
    p.add_argument("--warmup-s", type=float, default=30.0,
                   help="Leave requests before this sim time out of the rate "
                        "(the GPU runs discard their first 30 s window).")
    p.add_argument("--out", default="results/cpu_headroom/headroom.csv")
    a = p.parse_args()
    names = [w for w in a.workloads.split(",") if w]
    bad = [w for w in names if w not in PRESETS]
    if bad:
        print(f"unknown workload(s): {bad}")
        return 2
    run(names, [int(s) for s in a.seeds.split(",") if s],
        [int(c) for c in a.capacities.split(",") if c], a.warmup_s, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
