"""Phase 4 — Verify the 4 dispatch policies produce different orderings.

This is a CPU-only test. It does NOT need a GPU.

Usage:
    python experiments/phase4_policies.py
"""

from __future__ import annotations

import argparse
import sys

from kvcache.overlap import OverlapIndex
from kvcache.policies import (
    CombinedPolicy,
    FIFOPolicy,
    QueuedRequest,
    SessionAwarePolicy,
    SharingAwarePolicy,
)
from kvcache.session import Session
from kvcache.workload import WorkloadConfig, generate_workload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--num-sessions", type=int, default=5)
    p.add_argument("--sim-window", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    w = generate_workload(
        WorkloadConfig(
            num_sessions=args.num_sessions,
            sim_window_s=args.sim_window,
            seed=args.seed,
            overlap_fraction=0.6,
        )
    )
    events = w.events[:8]
    queue = [
        QueuedRequest(event=e, arrival_t=e.t, enqueue_seq=i) for i, e in enumerate(events)
    ]
    sessions = {s.session_id: s for s in w.sessions}
    oi = OverlapIndex(n=8)
    # Pretend earlier content is "in the index" so sharing signals are visible
    for e in events:
        if e.session_id in w.sharing_sids:
            oi.touch_session(e.session_id, e.tokens)

    print(f"[phase4] testing {len(queue)} queued requests with {len(w.sharing_sids)} sharers")
    orderings = {}
    for name, policy in [
        ("fifo", FIFOPolicy()),
        ("session-aware", SessionAwarePolicy()),
        ("sharing-aware", SharingAwarePolicy()),
        ("combined", CombinedPolicy(alpha=0.5)),
    ]:
        d = policy.score_queue(queue, sessions, oi, now=0.0)
        orderings[name] = [q.event.session_id + "/" + str(q.event.turn_index) for q in d.ordered]
        print(f"  {name:14s} -> {orderings[name]}")

    # Sanity: at least session-aware and sharing-aware should differ on this
    # constructed example, otherwise the policies are not doing their job.
    s = orderings["session-aware"]
    sh = orderings["sharing-aware"]
    differs = s != sh
    print()
    print(f"[phase4] session-aware vs sharing-aware differ? {differs}")
    if not differs:
        print(
            "[phase4] WARNING: on this constructed example session-aware and "
            "sharing-aware produced the same ordering. Try a different seed "
            "or workload config to construct a more discriminating example."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

