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
RC_TIMEOUT = 4         # killed by the launcher to stay inside the session

CALIBRATION = "results/load_calibration.json"

# Kaggle kills a GPU session at 9 h wall and, in batch mode, then saves NO
# output. Round 3 needed 56 h for 24 runs and was cut off at run 5, so every
# matrix cell now budgets against this clock instead of trusting an estimate.
SESSION_START_FILE = "/kaggle/working/.kvcache_session_start"
SESSION_LIMIT_S = 9 * 3600
# Left for analysis (3c), zip + push (4) and slack in the per-run estimate.
SESSION_RESERVE_S = 30 * 60
# Engine start + warmup + drain on top of events / arrival rate.
RUN_OVERHEAD_S = 150


def mark_session_start() -> float:
    """Record when this Kaggle session began (cell 1). Idempotent.

    A marker older than the session limit belongs to a previous session
    (interactive sessions can keep /kaggle/working), so it is replaced.
    """
    now = time.time()
    try:
        t0 = float(open(SESSION_START_FILE).read().strip())
        if 0 <= now - t0 < SESSION_LIMIT_S:
            return t0
    except (OSError, ValueError):
        pass
    try:
        os.makedirs(os.path.dirname(SESSION_START_FILE), exist_ok=True)
        with open(SESSION_START_FILE, "w") as fh:
            fh.write(str(now))
    except OSError:
        pass  # not on Kaggle; seconds_left() falls back to "now"
    return now


def seconds_left() -> float:
    """Wall seconds a run may still use before the session reserve."""
    t0 = mark_session_start()
    return t0 + SESSION_LIMIT_S - SESSION_RESERVE_S - time.time()


def calibration() -> dict:
    if not os.path.exists(CALIBRATION):
        raise SystemExit(f"[launcher] no {CALIBRATION}: run cell 3a2 first.")
    return json.load(open(CALIBRATION))


def peak_rate() -> float:
    """The calibrated offered load from cell 3a2, in prompt tokens/s in the
    workload's busiest 30 s window (run_experiment --target-peak-prompt-tok-s).

    Refuses to guess. Round 2's guessed load was ~2.8x capacity, and a
    mean-based rate overloads the end of every run (contexts grow).
    """
    cal = calibration()
    if "peak_prompt_tok_s" not in cal:
        raise SystemExit(f"[launcher] {CALIBRATION} predates peak-based "
                         f"calibration: re-run cell 3a2.")
    rate = float(cal["peak_prompt_tok_s"])
    print(f"[launcher] offered load: busiest window at {rate:.0f} prompt tok/s "
          f"({cal.get('target_peak_utilization', '?')} of measured "
          f"{cal.get('capacity_prompt_tok_s', 0):.0f} tok/s), "
          f"max_num_seqs={cal.get('max_num_seqs')}, workload "
          f"{cal.get('num_sessions')} sessions x {cal.get('sim_window_s')} s")
    return rate


def estimate_run_s(seed: int) -> float:
    """Wall seconds one run of `seed` should take, from the calibration."""
    per = calibration().get("per_seed", {})
    if str(seed) in per:
        return float(per[str(seed)]["est_run_s"])
    return max((float(v["est_run_s"]) for v in per.values()), default=3600.0)


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
    # Only FIFO's saturation says the load is wrong; see metrics.py
    # saturation_blocks_usable. Older CSVs lack the column: treat as blocking.
    if (str(row.get("saturated")) == "1"
            and str(row.get("saturation_blocks_usable", "1")) != "0"):
        why.append("saturated (backlog grew all run)")
    if "usable" not in row:
        why.append("pre-round-3 CSV (no `usable` column)")
    return why


def _move_run(csv_dir: str, label: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    for kind in ("summary", "time_series", "per_session"):
        p = os.path.join(csv_dir, f"{kind}_{label}.csv")
        if os.path.exists(p):
            shutil.move(p, os.path.join(dest, os.path.basename(p)))


def done_labels(csv_dir: str, set_aside_dir: str, run_tag: str = "",
                superseded_dir: str = "") -> set:
    """Labels with a USABLE summary. Unusable ones are moved aside.

    Anything left in `csv_dir` is treated as finished and never re-run, so
    an unusable summary there would permanently hold its slot in the matrix
    (that is how a run like round 2's `fifo_generous_s2` would survive).

    With `run_tag`, a summary from another round (different run_tag) is
    moved to `superseded_dir` too: round 3's fifo_constrained ran a 12-
    session workload at max_num_seqs 8 and would otherwise count as a
    finished round-4 run.
    """
    done = set()
    moved = []
    for f in sorted(glob.glob(os.path.join(csv_dir, "summary_*.csv"))):
        label = os.path.basename(f)[len("summary_"):-len(".csv")]
        row = read_summary(f)
        if run_tag and row.get("run_tag", "") != run_tag:
            dest = superseded_dir or set_aside_dir
            _move_run(csv_dir, label, dest)
            print(f"[launcher] {label}: from run_tag "
                  f"{row.get('run_tag') or '(none)'!r}, not {run_tag!r} -> "
                  f"moved to {dest}/")
            continue
        why = why_unusable(row)
        if not why:
            done.add(label)
            continue
        _move_run(csv_dir, label, set_aside_dir)
        moved.append((label, "; ".join(why)))
    for label, why in moved:
        print(f"[launcher] {label}: not usable ({why}) -> moved to "
              f"{set_aside_dir}/, will re-run")
    return done


def gpu_used_mib() -> Optional[int]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return max(int(x) for x in r.stdout.split())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


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


def run_logged(cmd: List[str], env: Dict[str, str], log_path: str) -> int:
    """Run `cmd`, streaming its output live AND into `log_path`.

    The old capture_output + print-the-tail pattern hid the one line that
    mattered when the calibration died: the notebook showed only
    "calibration failed".
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"\n===== {time.strftime('%H:%M:%S')} {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, env={**env, "PYTHONUNBUFFERED": "1"},
                                text=True, bufsize=1,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def tail(path: str, n: int = 60) -> str:
    try:
        return "".join(open(path).readlines()[-n:])
    except OSError:
        return ""


def _run_with_deadline(cmd: List[str], env: Dict[str, str],
                       timeout_s: Optional[float]) -> int:
    """subprocess.run, but a timeout kills the whole process group.

    The vLLM engine core is a grandchild; killing only the direct child
    would leave it holding the GPU for the next run.
    """
    proc = subprocess.Popen(cmd, env=env, text=True, start_new_session=True)
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        import signal
        for sig, grace in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                proc.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        return RC_TIMEOUT


def run(cmd: List[str], env: Dict[str, str], label: str,
        deadline_s: Optional[float] = None) -> int:
    """Run one experiment; retry once if the engine died. Returns the rc.

    `deadline_s` is a time.monotonic() value the run must not outlive; it
    is killed (RC_TIMEOUT) rather than letting Kaggle kill the notebook.
    """
    for attempt in (1, 2):
        wait_for_gpu_free()
        timeout = None if deadline_s is None else deadline_s - time.monotonic()
        if timeout is not None and timeout <= 0:
            return RC_TIMEOUT
        rc = _run_with_deadline([sys.executable] + cmd, env, timeout)
        if rc == RC_TIMEOUT:
            print(f"[launcher] {label}: killed at the session deadline")
            return rc
        if rc != RC_ENGINE_FAILED or attempt == 2:
            return rc
        print(f"[launcher] {label}: engine failed mid-run; retrying once "
              f"on a fresh engine")
    return rc
