"""Phase 6 — analysis + charts.

Loads the per-run time-series CSVs produced by the benchmark harness and
emits the headline + ablation + fairness plots called for in the build
spec, plus a short interpretation note.

Inputs (one CSV per (policy, capacity) run):
  results/csv/time_series_<label>.csv
  results/csv/per_session_<label>.csv
  results/csv/summary_<label>.csv

Outputs:
  results/figures/headline_hit_rate.png
  results/figures/ablation_p99_latency.png
  results/figures/goodput_over_time.png
  results/figures/fairness_per_session_hit_rate.png
  results/figures/shared_at_eviction.png    (proxy: shared-content hit rate)
  results/figures/alpha_sweep.png           (only if alpha-sweep runs are present)
  results/INTERPRETATION.md
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

POLICY_ORDER = ["fifo", "session-aware", "sharing-aware", "combined"]
POLICY_COLORS = {
    "fifo": "#888888",
    "session-aware": "#1f77b4",
    "sharing-aware": "#2ca02c",
    "combined": "#d62728",
}
POLICY_LABELS = {
    "fifo": "FIFO (naive)",
    "session-aware": "Session-aware",
    "sharing-aware": "Sharing-aware",
    "combined": "Combined",
}


@dataclass
class RunSet:
    """A collection of completed runs, one per (policy, capacity)."""

    csv_dir: str
    runs: Dict[str, pd.DataFrame]  # label -> time_series
    summaries: Dict[str, pd.DataFrame]  # label -> summary
    per_session: Dict[str, pd.DataFrame]  # label -> per-session

    @classmethod
    def load(cls, csv_dir: str, label_pattern: str = "") -> "RunSet":
        runs = {}
        sums = {}
        ps = {}
        for fn in sorted(os.listdir(csv_dir)):
            if not fn.endswith(".csv"):
                continue
            full = os.path.join(csv_dir, fn)
            if fn.startswith("time_series_"):
                label = fn[len("time_series_"):-len(".csv")]
                runs[label] = pd.read_csv(full)
            elif fn.startswith("summary_"):
                label = fn[len("summary_"):-len(".csv")]
                sums[label] = pd.read_csv(full)
            elif fn.startswith("per_session_"):
                label = fn[len("per_session_"):-len(".csv")]
                ps[label] = pd.read_csv(full)
        if label_pattern:
            runs = {k: v for k, v in runs.items() if label_pattern in k}
            sums = {k: v for k, v in sums.items() if label_pattern in k}
            ps = {k: v for k, v in ps.items() if label_pattern in k}
        return cls(csv_dir=csv_dir, runs=runs, summaries=sums, per_session=ps)


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


import re

# Run labels look like "combined_constrained" or, for follow-up runs,
# "combined_constrained_a025" / "fifo_generous_s1" (--label-suffix).
_LABEL_RE = re.compile(
    r"^(fifo|session-aware|sharing-aware|combined)_(generous|constrained)(_.*)?$"
)


def _split_policy_and_capacity(label: str):
    """Parse a run label into (policy, capacity, variant).

    Variant is "" for base-matrix runs, else the --label-suffix
    (e.g. "_a025", "_s1"). Unknown labels fall back to (label,
    "constrained", "") so old files never crash the analysis.
    """
    m = _LABEL_RE.match(label)
    if m:
        return m.group(1), m.group(2), (m.group(3) or "")
    for cap in ("constrained", "generous"):
        if label.endswith("_" + cap):
            return label[: -len(cap) - 1], cap, ""
    return label, "constrained", ""


def _matching(rs: RunSet, policy: str, capacity: str):
    """All (variant, label, df) for a policy x capacity, base ("") first."""
    out = []
    for label, df in rs.runs.items():
        p, c, v = _split_policy_and_capacity(label)
        if p == policy and c == capacity:
            out.append((v, label, df))
    out.sort(key=lambda t: (t[0] != "", t[0]))
    return out


def _legend_name(policy: str, variant: str) -> str:
    base = POLICY_LABELS.get(policy, policy)
    if variant:
        return f"{base} [{variant.lstrip('_')}]"
    return base


def _filter(rs: RunSet, policy: str, capacity: str) -> Optional[pd.DataFrame]:
    """Base ("") run for a policy x capacity, or None. Variants are handled
    by _matching; this keeps single-run call sites deterministic."""
    for variant, _label, df in _matching(rs, policy, capacity):
        if variant == "":
            return df
    return None


def plot_headline_hit_rate(rs: RunSet, out_path: str, capacity: str = "constrained") -> None:
    """Real cache hit rate over time, all 4 policies, on one chart.

    This is the headline chart for the project. Under constrained capacity,
    we expect the policies to separate.
    """
    plt.figure(figsize=(10, 5))
    plotted = 0
    for policy in POLICY_ORDER:
        for variant, _label, df in _matching(rs, policy, capacity):
            # Use a rolling mean to smooth out single-window noise
            smoothed = df["hit_rate"].rolling(window=2, min_periods=1).mean()
            plt.plot(
                df["t_start"],
                smoothed,
                label=_legend_name(policy, variant),
                color=POLICY_COLORS.get(policy, None),
                linewidth=2.0,
                linestyle="--" if variant else "-",
            )
            plotted += 1
    if plotted == 0:
        plt.close()
        return
    plt.xlabel("Simulated time (s)")
    plt.ylabel("Cache hit rate (rolling)")
    plt.title(f"Real cache hit rate over time — {capacity} capacity")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_p99_latency(rs: RunSet, out_path: str, capacity: str = "constrained") -> None:
    plt.figure(figsize=(10, 5))
    for policy in POLICY_ORDER:
        for variant, _label, df in _matching(rs, policy, capacity):
            plt.plot(
                df["t_start"],
                df["p99_latency_ms"],
                label=_legend_name(policy, variant),
                color=POLICY_COLORS.get(policy, None),
                linewidth=2.0,
                linestyle="--" if variant else "-",
            )
    plt.xlabel("Simulated time (s)")
    plt.ylabel("P99 latency (ms)")
    plt.title(f"P99 latency over time — {capacity} capacity")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_goodput(rs: RunSet, out_path: str, capacity: str = "constrained") -> None:
    plt.figure(figsize=(10, 5))
    for policy in POLICY_ORDER:
        for variant, _label, df in _matching(rs, policy, capacity):
            smoothed = df["goodput"].rolling(window=2, min_periods=1).mean()
            plt.plot(
                df["t_start"],
                smoothed,
                label=_legend_name(policy, variant),
                color=POLICY_COLORS.get(policy, None),
                linewidth=2.0,
                linestyle="--" if variant else "-",
            )
    plt.xlabel("Simulated time (s)")
    plt.ylabel("Goodput (SLA-meeting request fraction)")
    plt.title(f"Goodput over time — {capacity} capacity")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_fairness(rs: RunSet, out_path: str, capacity: str = "constrained") -> None:
    """Distribution of per-session hit rates per policy.

    If a policy is starving some sessions (e.g. session-aware ignoring
    sharing signals and starving popular shared content), the distribution
    will be more skewed. We use ECDFs because the per-session N is small.

    Base-matrix runs only; follow-up variants are compared via the
    time-series charts and the interpretation table.
    """
    plt.figure(figsize=(10, 5))
    for policy in POLICY_ORDER:
        label = f"{policy}_{capacity}"
        if label not in rs.per_session:
            continue
        df = rs.per_session[label]
        if "hit_rate" not in df.columns:
            continue
        vals = df["hit_rate"].dropna().values
        if len(vals) == 0:
            continue
        xs = sorted(vals)
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        plt.plot(
            xs, ys,
            label=POLICY_LABELS.get(policy, policy),
            color=POLICY_COLORS.get(policy, None),
            linewidth=2.0,
        )
    plt.xlabel("Per-session cache hit rate")
    plt.ylabel("ECDF (fraction of sessions)")
    plt.title(f"Per-session fairness (ECDF) — {capacity} capacity")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_shared_hit_rate(rs: RunSet, out_path: str, capacity: str = "constrained") -> None:
    """Diagnostic: hit rate on detected-shared content per policy.

    This is the closest proxy we have to "evictions that destroyed shared
    value" — if a policy evicts a shared block, the next read of that
    content is a miss, so shared_hit_rate drops.
    """
    plt.figure(figsize=(10, 5))
    for policy in POLICY_ORDER:
        for variant, _label, df in _matching(rs, policy, capacity):
            if "shared_hit_rate" not in df.columns:
                continue
            smoothed = df["shared_hit_rate"].rolling(window=2, min_periods=1).mean()
            plt.plot(
                df["t_start"],
                smoothed,
                label=_legend_name(policy, variant),
                color=POLICY_COLORS.get(policy, None),
                linewidth=2.0,
                linestyle="--" if variant else "-",
            )
    plt.xlabel("Simulated time (s)")
    plt.ylabel("Shared-content hit rate (rolling)")
    plt.title(f"Hit rate on shared content over time — {capacity} capacity")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_ablation_bar(rs: RunSet, out_path: str) -> None:
    """Single bar chart: overall hit rate per policy at constrained capacity.

    The simplest "ablation" view: which policy wins?
    Base-matrix runs only; follow-up variants live in the interpretation
    table and the dashed time-series lines.
    """
    rows = []
    for policy in POLICY_ORDER:
        for cap in ("constrained", "generous"):
            label = f"{policy}_{cap}"
            if label in rs.summaries:
                s = rs.summaries[label].iloc[0].to_dict()
                # NOTE: s contains "policy": <run_label> (e.g.
                # "fifo_constrained"), which must NOT overwrite the short
                # policy name — a previous version spread **s last, the
                # reindex below matched nothing, and the chart rendered
                # empty axes.
                rows.append({**s, "policy": policy, "capacity": cap})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cap in zip(axes, ("constrained", "generous")):
        sub = df[df["capacity"] == cap]
        if sub.empty:
            ax.set_title(f"{cap} (no data)")
            continue
        sub = sub.set_index("policy").reindex(POLICY_ORDER).reset_index()
        colors = [POLICY_COLORS.get(p, "#444") for p in sub["policy"]]
        ax.bar(sub["policy"], sub["hit_rate"], color=colors)
        ax.set_title(f"Overall hit rate — {cap}")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Hit rate")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def write_interpretation(rs: RunSet, out_path: str) -> None:
    """Auto-generate a short interpretation note from the run summaries.

    The note is a starting point for the project report. It surfaces the
    numbers; the user is expected to write the surrounding prose.
    """
    lines: List[str] = []
    lines.append("# Headline numbers — auto-generated\n")
    lines.append(
        "This is a draft of the headline result table. Fill in narrative "
        "around it.\n"
    )
    lines.append("| Policy | Capacity | Lookups | Hit rate | Shared hit rate | P50 (ms) | P99 (ms) | Goodput |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    def row(policy: str, cap: str, s: dict) -> str:
        return (
            f"| {_legend_name(policy, s.pop('_variant', ''))} | {cap} | "
            f"{int(s.get('lookups', 0))} | "
            f"{s.get('hit_rate', 0):.3f} | "
            f"{s.get('shared_hit_rate', 0):.3f} | "
            f"{s.get('p50_latency_ms', 0):.0f} | "
            f"{s.get('p99_latency_ms', 0):.0f} | "
            f"{s.get('goodput', 0):.2f} |"
        )

    for policy in POLICY_ORDER:
        for cap in ("constrained", "generous"):
            label = f"{policy}_{cap}"
            if label not in rs.summaries:
                continue
            s = rs.summaries[label].iloc[0].to_dict()
            lines.append(row(policy, cap, s))
    # Follow-up runs (--label-suffix), if any: same columns, variant tagged.
    variants = []
    for label in sorted(rs.summaries):
        p, c, v = _split_policy_and_capacity(label)
        if v and p in POLICY_ORDER and c in ("constrained", "generous"):
            variants.append((p, c, v, label))
    if variants:
        lines.append("")
        lines.append("### Follow-up runs (--label-suffix)")
        lines.append("")
        for p, c, v, label in variants:
            s = rs.summaries[label].iloc[0].to_dict()
            s["_variant"] = v
            lines.append(row(p, c, s))
    lines.append("")
    lines.append("## Diagnostic questions to address in the writeup")
    lines.append("")
    lines.append(
        "- Does the **Combined** policy beat both single-signal baselines on "
        "the constrained-capacity hit rate? By how much?"
    )
    lines.append(
        "- Does the advantage shrink or disappear at the **generous** capacity? "
        "(expected: yes — at low pressure, ordering doesn't matter much.)"
    )
    lines.append(
        "- Is the per-session fairness ECDF for Combined reasonable, or is it "
        "skewed toward protecting some sessions at the cost of others?"
    )
    lines.append(
        "- Does the **Sharing-aware** policy's hit rate on shared content stay "
        "higher than **Session-aware**'s? (sanity check on the policy's design.)"
    )
    lines.append(
        "- If Combined does NOT clearly beat both baselines, what does the "
        "diagnosis say? (e.g. 'combining helps only at high overlap_fraction; "
        "at low overlap, session-awareness alone is sufficient'.)"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def ensure_base_results(csv_dir: str) -> int:
    """Copy frozen experiment-1 CSVs into `csv_dir` when it has none.

    Fresh Kaggle/Colab sessions start with an empty working dir, while the
    base matrix is archived on GitHub under
    `result_from_first_experiment/csv/`. Without this, cell 3b would redo
    the hour-long base matrix and analysis would find nothing. Fires only
    when `csv_dir` contains no `summary_*.csv` yet, so partial local runs
    are never touched. Set `KVCACHE_NO_RESTORE=1` to disable (clean redo).
    Returns the number of files restored.
    """
    import glob as _glob
    import shutil as _shutil

    if os.environ.get("KVCACHE_NO_RESTORE"):
        return 0
    if _glob.glob(os.path.join(csv_dir, "summary_*.csv")):
        return 0
    arch = os.path.normpath(os.path.join(
        os.path.abspath(csv_dir), "..", "..",
        "result_from_first_experiment", "csv"))
    srcs = sorted(_glob.glob(os.path.join(arch, "*.csv")))
    if not srcs:
        return 0
    os.makedirs(csv_dir, exist_ok=True)
    for f in srcs:
        _shutil.copy(f, os.path.join(csv_dir, os.path.basename(f)))
    print(f"[results] restored {len(srcs)} experiment-1 CSVs from {arch} "
          f"(base runs show as done; KVCACHE_NO_RESTORE=1 disables this)")
    return len(srcs)


def run_analysis(csv_dir: str, fig_dir: str) -> None:
    """Top-level entry point. Loads CSVs, emits all charts + interpretation."""
    rs = RunSet.load(csv_dir)
    if not rs.runs:
        print(f"[analysis] no runs found in {csv_dir}")
        return
    print(
        f"[analysis] loaded {len(rs.runs)} runs: {sorted(rs.runs.keys())}"
    )
    for cap in ("constrained", "generous"):
        plot_headline_hit_rate(rs, os.path.join(fig_dir, f"headline_hit_rate_{cap}.png"), capacity=cap)
        plot_p99_latency(rs, os.path.join(fig_dir, f"p99_latency_{cap}.png"), capacity=cap)
        plot_goodput(rs, os.path.join(fig_dir, f"goodput_{cap}.png"), capacity=cap)
        plot_fairness(rs, os.path.join(fig_dir, f"fairness_{cap}.png"), capacity=cap)
        plot_shared_hit_rate(rs, os.path.join(fig_dir, f"shared_hit_rate_{cap}.png"), capacity=cap)
    plot_ablation_bar(rs, os.path.join(fig_dir, "ablation_bars.png"))
    write_interpretation(rs, os.path.join(fig_dir, "..", "INTERPRETATION.md"))
    print(f"[analysis] wrote charts to {fig_dir}")

