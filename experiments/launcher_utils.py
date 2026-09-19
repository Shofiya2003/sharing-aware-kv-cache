"""Helpers shared by the Kaggle launcher's matrix cells (3b, 3d, 3e).

Kept in the repo rather than pasted into each cell so the three cells
agree on what "done", "usable" and "retry" mean.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

# run_experiment.py exit codes.
RC_OVERLOADED = 2      # offered load exceeds capacity: config error, stop
RC_ENGINE_FAILED = 3   # engine died mid-run: usually transient, retry

CALIBRATION = "results/load_calibration.json"


def arrival_rate() -> float:
    """The calibrated offered load (req/s) from cell 3a2.

    Refuses to guess. Round 2's guessed load was ~2.8x capacity, and a
    speed factor alone does not transfer between workloads anyway.
    """
    if not os.path.exists(CALIBRATION):
        raise SystemExit(f"[launcher] no {CALIBRATION}: run cell 3a2 first.")
    cal = json.load(open(CALIBRATION))
    if "arrival_rate_req_s" not in cal:
        raise SystemExit(f"[launcher] {CALIBRATION} predates arrival-rate "
                         f"calibration: re-run cell 3a2.")
    rate = float(cal["arrival_rate_req_s"])
    print(f"[launcher] offered load {rate:.3f} req/s "
          f"({cal.get('target_utilization', '?')} of measured "
          f"{cal.get('service_rate_req_s', 0):.3f} req/s)")
    return rate


def read_summary(path: str) -> Dict[str, str]:
    try:
        with open(path, newline="") as fh:
            return next(iter(csv.DictReader(fh)), {}) or {}
    except OSError:
        return {}


def why_unusable(row: Dict[str, str]) -> List[str]:
    """Reasons a summary is not a data point; empty when it is usable."""
    why = []
    if row.get("hit_basis") != "cached_tokens":
        why.append(f"basis={row.get('hit_basis')}")
    try:
        cov = float(row.get("cache_ground_truth_coverage") or 0)
    except ValueError:
        cov = 0.0
    if cov < 0.99:
        why.append(f"coverage={cov:.2f}")
    try:
        err = float(row.get("error_rate") or 0)
    except ValueError:
        err = 0.0
    if err > 0.01:
        why.append(f"engine failed {err:.0%} of requests")
    if str(row.get("saturated")) == "1":
        why.append("saturated (backlog grew all run)")
    if "usable" not in row:
        why.append("pre-round-3 CSV (no `usable` column)")
    return why


def done_labels(csv_dir: str, set_aside_dir: str) -> set:
    """Labels with a USABLE summary. Unusable ones are moved aside.

    Anything left in `csv_dir` is treated as finished and never re-run, so
    an unusable summary there would permanently hold its slot in the matrix
    (that is how a run like round 2's `fifo_generous_s2` would survive).
    """
    done = set()
    moved = []
    for f in sorted(glob.glob(os.path.join(csv_dir, "summary_*.csv"))):
        label = os.path.basename(f)[len("summary_"):-len(".csv")]
        why = why_unusable(read_summary(f))
        if not why:
            done.add(label)
            continue
        os.makedirs(set_aside_dir, exist_ok=True)
        for kind in ("summary", "time_series", "per_session"):
            p = os.path.join(csv_dir, f"{kind}_{label}.csv")
            if os.path.exists(p):
                shutil.move(p, os.path.join(set_aside_dir, os.path.basename(p)))
        moved.append((label, "; ".join(why)))
    for label, why in moved:
        print(f"[launcher] {label}: not usable ({why}) -> moved to "
              f"{set_aside_dir}/, will re-run")
    return done


def gpu_used_mib() -> Optional[int]:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return max(int(x) for x in r.stdout.split())


def wait_for_gpu_free(limit_mib: int = 1024, timeout_s: float = 180.0) -> None:
    """Block until the previous run's engine has released the GPU.

    vLLM's engine core is a separate process; if it is still exiting when
    the next run starts, the new engine sizes its KV cache against a GPU
    that is partly taken, or dies mid-run.
    """
    t0 = time.monotonic()
    used = gpu_used_mib()
    while used is not None and used > limit_mib:
        if time.monotonic() - t0 > timeout_s:
            print(f"[launcher] WARNING: GPU still has {used} MiB in use after "
                  f"{timeout_s:.0f}s; starting anyway")
            return
        time.sleep(5)
        used = gpu_used_mib()
    if time.monotonic() - t0 > 1:
        print(f"[launcher] GPU free after {time.monotonic() - t0:.0f}s")


def run(cmd: List[str], env: Dict[str, str], label: str) -> int:
    """Run one experiment; retry once if the engine died. Returns the rc."""
    for attempt in (1, 2):
        wait_for_gpu_free()
        r = subprocess.run([sys.executable] + cmd, env=env, text=True)
        if r.returncode != RC_ENGINE_FAILED or attempt == 2:
            return r.returncode
        print(f"[launcher] {label}: engine failed mid-run; retrying once "
              f"on a fresh engine")
    return r.returncode
