"""Phase 3 — Workload generator + timeline + overlap report.

Usage:
    python experiments/phase3_workload.py
    python experiments/phase3_workload.py --num-sessions 20 --sim-window 600

Generates a workload, plots the sessions' active/idle timeline, and
reports the measured cross-session overlap (per the spec's
Checkpoint 3a / 3b). This is a CPU-only step.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kvcache.workload import WorkloadConfig, generate_workload
from kvcache.prefix import PrefixIndex


def plot_timeline(workload, out_path: str) -> None:
    """One row per session; horizontal bars for ACTIVE bursts, gaps are IDLE."""
    fig, ax = plt.subplots(figsize=(12, max(4, len(workload.sessions) * 0.25)))
    for i, sess in enumerate(workload.sessions):
        # Walk the session's turns to reconstruct ACTIVE bursts
        bursts = []
        cur_burst = None
        cur_t = sess.start_t
        for ev in workload.events:
            if ev.session_id != sess.session_id:
                continue
            if cur_burst is None:
                cur_burst = [ev.t, ev.t]
            else:
                cur_burst[1] = ev.t
            # Decide burst end heuristically: inter-turn gap > 5s ends burst
            next_ev_t = None
            for e2 in workload.events:
                if e2.session_id == sess.session_id and e2.t > ev.t:
                    next_ev_t = e2.t
                    break
            if next_ev_t is None or (next_ev_t - ev.t) > 5.0:
                bursts.append(tuple(cur_burst))
                cur_burst = None
        if cur_burst is not None:
            bursts.append(tuple(cur_burst))
        for (s, e) in bursts:
            ax.barh(i, e - s, left=s, height=0.6, color="steelblue", edgecolor="none")
    ax.set_yticks(range(len(workload.sessions)))
    ax.set_yticklabels([s.session_id for s in workload.sessions], fontsize=6)
    ax.set_xlabel("Simulated time (s)")
    ax.set_title("Session ACTIVE bursts (idle gaps are blank)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def report_overlap(workload) -> None:
    """Report cross-session sharing the engine could actually reuse.

    Replays the prompts in arrival order through a PrefixIndex and, for
    each request, counts the leading tokens it shares with prompts OTHER
    sessions sent earlier (whole 16-token blocks from token 0, as vLLM
    matches them). Shared text anywhere else in a prompt counts for
    nothing, because vLLM cannot reuse it.
    """
    idx = PrefixIndex()
    n_shared = 0
    shared_tok = 0
    prompt_tok = 0
    for ev in sorted(workload.events, key=lambda e: e.t):
        n, _others = idx.shared_prefix(ev.session_id, ev.prompt_tokens)
        n_shared += n > 0
        shared_tok += n
        prompt_tok += len(ev.prompt_tokens)
        idx.add(ev.session_id, ev.prompt_tokens)
    n_ev = max(1, len(workload.events))
    print(f"[phase3] requests opening with another session's blocks: "
          f"{n_shared}/{len(workload.events)} ({n_shared / n_ev:.1%})")
    print(f"[phase3] cross-session reusable prefix: "
          f"{shared_tok / max(1, prompt_tok):.2%} of all prompt tokens")
    print(
        f"[phase3] sharing sessions in workload: {len(workload.sharing_sids)} "
        f"/ {len(workload.sessions)} total"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--num-sessions", type=int, default=15)
    p.add_argument("--sim-window", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/figures/workload_timeline.png")
    args = p.parse_args()

    cfg = WorkloadConfig(
        num_sessions=args.num_sessions,
        sim_window_s=args.sim_window,
        seed=args.seed,
    )
    w = generate_workload(cfg)
    print(
        f"[phase3] generated workload: {len(w.sessions)} sessions, "
        f"{len(w.events)} events, {len(w.shared_doc_pool)} shared docs"
    )
    plot_timeline(w, args.out)
    print(f"[phase3] wrote timeline -> {args.out}")
    report_overlap(w)
    return 0


if __name__ == "__main__":
    sys.exit(main())

