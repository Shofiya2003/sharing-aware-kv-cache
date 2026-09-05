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
from kvcache.overlap import ngrams, ngram_id


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
    """Report the measured cross-session overlap.

    For each point in time, count the fraction of currently-live sessions
    whose most recent n-grams are referenced by at least one other live
    session. Then average across the simulation.
    """
    # Build per-session n-gram sets over time.
    n = 8
    per_sess_grams = defaultdict(set)
    samples = []
    sim_window = workload.config.sim_window_s
    sample_dt = 5.0
    t = 0.0
    event_idx = 0
    sorted_events = sorted(workload.events, key=lambda e: e.t)

    while t <= sim_window:
        while event_idx < len(sorted_events) and sorted_events[event_idx].t <= t:
            ev = sorted_events[event_idx]
            for gram in ngrams(ev.tokens, n):
                per_sess_grams[ev.session_id].add(ngram_id(gram))
            event_idx += 1
        live_sids = [s.session_id for s in workload.sessions if s.start_t <= t]
        if live_sids:
            sharing = 0
            for sid in live_sids:
                others = set()
                for gid in per_sess_grams[sid]:
                    # Is this gid in any other session's set?
                    for other in live_sids:
                        if other != sid and gid in per_sess_grams[other]:
                            others.add(other)
                            break
                if others:
                    sharing += 1
            frac = sharing / len(live_sids)
        else:
            frac = 0.0
        samples.append(frac)
        t += sample_dt

    avg = sum(samples) / max(1, len(samples))
    print(f"[phase3] average fraction of live sessions with cross-session overlap: {avg:.3f}")
    print(f"[phase3] samples: {len(samples)} across sim window {sim_window}s")
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

