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
import statistics
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
    """Token-level prefix-cache hit rate over time, all 4 policies.

    This is the headline chart. The plotted quantity is
    `cached_token_rate` = cached_tokens / prompt_tokens, read straight from
    vLLM's per-request `num_cached_tokens`. Warmup windows are shaded out:
    they carry one-time engine startup cost, not policy behavior.

    Falls back to the legacy `hit_rate` column only for older CSVs that
    predate ground-truth measurement, and says so in the axis label.
    """
    plt.figure(figsize=(10, 5))
    plotted = 0
    metric = None
    for policy in POLICY_ORDER:
        for variant, _label, df in _matching(rs, policy, capacity):
            col = "cached_token_rate" if "cached_token_rate" in df.columns else "hit_rate"
            metric = metric or col
            # Use a rolling mean to smooth out single-window noise
            smoothed = df[col].rolling(window=2, min_periods=1).mean()
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
    # Shade the warmup region so it is never read as a policy effect.
    for _v, _l, df in _matching(rs, POLICY_ORDER[0], capacity):
        if "is_warmup" in df.columns and df["is_warmup"].any():
            warm = df[df["is_warmup"] == 1]
            plt.axvspan(float(warm["t_start"].min()),
                        float(warm["t_end"].max()),
                        color="0.85", alpha=0.6, zorder=0)
            plt.text(float(warm["t_start"].min()), 0.02, " warmup (excluded)",
                     fontsize=8, color="0.35", va="bottom")
        break
    plt.xlabel("Simulated time (s)")
    if metric == "cached_token_rate":
        plt.ylabel("Prefix-cache hit rate (cached tokens / prompt tokens)")
        plt.title(f"Prefix-cache hit rate over time — {capacity} capacity\n"
                  f"ground truth: vLLM num_cached_tokens")
    else:
        plt.ylabel("Hit rate (LEGACY latency proxy — not ground truth)")
        plt.title(f"Hit rate over time — {capacity} capacity "
                  f"(legacy latency proxy)")
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
            # End-to-end, not engine latency: engine latency hides the time
            # a reordering policy leaves requests waiting in the dispatcher.
            col = ("p99_e2e_latency_ms" if "p99_e2e_latency_ms" in df.columns
                   else "p99_latency_ms")
            plt.plot(
                df["t_start"],
                df[col],
                label=_legend_name(policy, variant),
                color=POLICY_COLORS.get(policy, None),
                linewidth=2.0,
                linestyle="--" if variant else "-",
            )
    plt.xlabel("Simulated time (s)")
    plt.ylabel("P99 end-to-end latency (ms)")
    plt.title(f"P99 end-to-end latency over time — {capacity} capacity")
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


def _seed_of(label: str) -> Optional[int]:
    """Seed index for a base/`_s<N>` label; None for other variants.

    The matrix writes seed 0 with no suffix and seeds 1..N as `_s<N>`.
    Alpha sweeps and alignment arms carry other suffixes and are not
    seeds, so they must not be pooled as replicates.
    """
    p, c, v = _split_policy_and_capacity(label)
    if p not in POLICY_ORDER or c not in ("constrained", "generous"):
        return None
    if not v:
        return 0
    tail = v.lstrip("_")
    if tail.startswith("s") and tail[1:].isdigit():
        return int(tail[1:])
    return None


def _usable(s: dict) -> bool:
    """Is this run admissible as a data point?

    Written defensively: `usable` is a newer column, so fall back to the
    underlying conditions when reading CSVs from an older run.
    """
    if "usable" in s and str(s.get("usable")) not in ("", "nan"):
        try:
            return int(float(s["usable"])) == 1
        except (TypeError, ValueError):
            pass
    basis_ok = str(s.get("hit_basis", "")) == "cached_tokens"
    try:
        cov_ok = float(s.get("cache_ground_truth_coverage", 0.0)) >= 0.99
    except (TypeError, ValueError):
        cov_ok = False
    try:
        err_ok = float(s.get("error_rate", 0.0) or 0.0) <= 0.01
    except (TypeError, ValueError):
        err_ok = True
    return basis_ok and cov_ok and err_ok and not _legacy_saturated(s)


# Pre-`saturated`-column CSVs (round 2) are judged on median dispatch wait
# instead. 10 s is 4x the 2.5 s SLA, the same bar metrics.py applies.
_LEGACY_SATURATED_QUEUE_WAIT_MS = 10_000.0


def _legacy_saturated(s: dict) -> bool:
    if str(s.get("saturated", "")) not in ("", "nan"):
        return False  # new CSV: `usable` already accounts for it
    try:
        return float(s.get("p50_queue_wait_ms", 0.0)) > _LEGACY_SATURATED_QUEUE_WAIT_MS
    except (TypeError, ValueError):
        return False


def _quarantine_section(rs: RunSet) -> List[str]:
    """List runs excluded from the pooled statistics, and why.

    Round 2 shipped `fifo_generous_s2` in the headline table with 18%
    ground-truth coverage and a p50 end-to-end latency of 1.5 ms against
    60-360 s everywhere else. It did not serve the workload, and because
    it was pooled in it made FIFO look like the best policy at generous
    capacity. Excluded runs are reported here rather than dropped
    silently.
    """
    bad = []
    for label in sorted(rs.summaries):
        if _seed_of(label) is None:
            continue
        s = rs.summaries[label].iloc[0].to_dict()
        if _usable(s):
            continue
        reasons = []
        if str(s.get("hit_basis", "")) != "cached_tokens":
            reasons.append(f"basis={s.get('hit_basis')}")
        try:
            cov = float(s.get("cache_ground_truth_coverage", 0.0))
            if cov < 0.99:
                reasons.append(f"coverage={cov:.2f}")
        except (TypeError, ValueError):
            pass
        try:
            if float(s.get("error_rate", 0.0) or 0.0) > 0.01:
                reasons.append(
                    f"engine failed {int(float(s.get('n_failed', 0)))} requests "
                    f"({float(s['error_rate']):.0%})"
                )
        except (TypeError, ValueError):
            pass
        if _legacy_saturated(s):
            reasons.append(
                f"saturated (p50 dispatch wait "
                f"{float(s['p50_queue_wait_ms'])/1000:.0f}s; pre-watchdog run)"
            )
        try:
            if int(float(s.get("saturated", 0))) == 1:
                reasons.append(
                    f"saturated (queue wait "
                    f"{float(s.get('queue_wait_first_window_ms', 0))/1000:.0f}s"
                    f"->{float(s.get('queue_wait_last_window_ms', 0))/1000:.0f}s)"
                )
        except (TypeError, ValueError):
            pass
        bad.append((label, ", ".join(reasons) or "unknown"))
    if not bad:
        return []
    out = ["", "## Quarantined runs — EXCLUDED from the pooled numbers below", ""]
    out.append(
        "These runs did not measure what the matrix intends to measure. "
        "They are listed so the exclusion is visible, not to be reported "
        "as results."
    )
    out.append("")
    out.append("| Run | Why excluded |")
    out.append("|---|---|")
    for label, why in bad:
        out.append(f"| `{label}` | {why} |")
    return out


def _paired_section(rs: RunSet) -> List[str]:
    """Per-seed paired deltas against FIFO — the comparison that resolves.

    Pooling each cell as mean +/- sd across seeds cannot separate the
    policy effect from seed noise here: the seed-to-seed spread of
    `cached_token_rate` is ~0.066 while the policy effect is ~0.02.
    But seed variation is a property of the WORKLOAD, and every policy
    runs the same workload for a given seed, so it cancels when each
    policy is differenced against FIFO within a seed. The paired spread
    is roughly 8x smaller, which is what makes the effect readable.
    """
    out: List[str] = ["", "## Paired comparison vs FIFO (same seed)", ""]
    out.append(
        "Each cell is `cached_token_rate(policy, seed) - "
        "cached_token_rate(fifo, seed)`. Seed noise is shared by both "
        "terms and cancels; the unpaired per-seed spread (~0.066) is an "
        "artifact of differing workloads, not of the policies. A mean "
        "several times its own sd, with a consistent sign across seeds, "
        "is a real effect. Quarantined runs and seeds missing a FIFO "
        "partner are skipped."
    )
    out.append("")
    for cap in ("constrained", "generous"):
        rows = []
        for policy in POLICY_ORDER:
            if policy == "fifo":
                continue
            deltas = []
            for label in sorted(rs.summaries):
                seed = _seed_of(label)
                p, c, _v = _split_policy_and_capacity(label)
                if seed is None or p != policy or c != cap:
                    continue
                base_label = f"fifo_{cap}" + ("" if seed == 0 else f"_s{seed}")
                if base_label not in rs.summaries:
                    continue
                a = rs.summaries[label].iloc[0].to_dict()
                b = rs.summaries[base_label].iloc[0].to_dict()
                if not (_usable(a) and _usable(b)):
                    continue
                deltas.append((seed, float(a["cached_token_rate"])
                               - float(b["cached_token_rate"])))
            if deltas:
                rows.append((policy, sorted(deltas)))
        if not rows:
            continue
        out.append(f"### {cap}")
        out.append("")
        out.append("| Policy | per-seed Δ | n | mean Δ | sd | same sign? |")
        out.append("|---|---|---:|---:|---:|---|")
        for policy, deltas in rows:
            vals = [d for _s, d in deltas]
            mean = sum(vals) / len(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            same = "yes" if all(v > 0 for v in vals) or all(v < 0 for v in vals) else "NO"
            per = " ".join(f"{v:+.4f}" for v in vals)
            sd_s = "—" if len(vals) < 2 else f"{sd:.4f}"
            out.append(
                f"| {_legend_name(policy, '')} | `{per}` | {len(vals)} | "
                f"{mean:+.4f} | {sd_s} | {same} |"
            )
        out.append("")
    return out


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
    lines.append(
        "**Primary metric is `cached_tok`** — vLLM's own per-request "
        "`num_cached_tokens` summed over prompt tokens "
        "(`cached_tokens / prompt_tokens`). `req_hit` is the per-request "
        "binary used for the fairness ECDF. `proxy` is what the old, "
        "discredited latency threshold would have reported on the same "
        "requests; it is shown only so the gap is visible. `trunc` is the "
        "share of requests whose prefix was invalidated by hitting the "
        "context window (a miss cause unrelated to the policy). Latencies "
        "are END-TO-END (from the user's turn, including dispatch-queue "
        "wait); engine-only latency flatters policies that reorder, so it "
        "is not shown. `goodput` is the share of requests within the SLA. "
        "Rows marked ✗ are not usable (see Quarantined runs). Summaries "
        "exclude warmup windows.\n"
    )
    lines.append(
        "| Policy | Capacity | Reqs | cached_tok | shared_cached_tok | "
        "req_hit | P50 e2e (ms) | P99 e2e (ms) | goodput | trunc | proxy | "
        "basis | usable |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")

    def row(policy: str, cap: str, s: dict) -> str:
        basis = str(s.get("hit_basis", "?"))
        flag = "" if basis == "cached_tokens" else " **(NOT ground truth)**"
        return (
            f"| {_legend_name(policy, s.pop('_variant', ''))} | {cap} | "
            f"{int(s.get('lookups', 0))} | "
            f"{s.get('cached_token_rate', float('nan')):.3f} | "
            f"{s.get('shared_cached_token_rate', float('nan')):.3f} | "
            f"{s.get('hit_rate', 0):.3f} | "
            f"{s.get('p50_e2e_latency_ms', s.get('p50_latency_ms', 0)):.0f} | "
            f"{s.get('p99_e2e_latency_ms', s.get('p99_latency_ms', 0)):.0f} | "
            f"{s.get('goodput', float('nan')):.2f} | "
            f"{s.get('context_truncated_rate', float('nan')):.2f} | "
            f"{s.get('proxy_hit_rate', float('nan')):.3f} | "
            f"{basis}{flag} | {'✓' if _usable(s) else '✗'} |"
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
    lines.extend(_quarantine_section(rs))
    lines.extend(_paired_section(rs))
    lines.append("")
    lines.append("## Validity checks — do these FIRST")
    lines.append("")
    lines.append(
        "- Is `basis` = `cached_tokens` for every row? If any row says "
        "`latency_proxy`, that run has no usable hit rate and must not be "
        "reported as one."
    )
    lines.append(
        "- Does `cached_tok` agree with `engine_prefix_cache_hit_rate` in the "
        "summary CSV? They measure the same thing two different ways; a large "
        "gap means the per-request counters are being read wrong."
    )
    lines.append(
        "- How far apart are `cached_tok` and `proxy`? A large gap is the "
        "evidence that the old latency-threshold results were measuring "
        "queueing jitter rather than cache reuse."
    )
    lines.append(
        "- Does **generous** beat **constrained** for the same policy? If not, "
        "the capacity lever is not binding and the constrained/generous axis "
        "carries no signal."
    )
    lines.append(
        "- Is `trunc` similar across policies? If one policy truncates far "
        "more, part of its miss rate is the context window, not the policy."
    )
    lines.append("")
    lines.append("## Diagnostic questions to address in the writeup")
    lines.append("")
    lines.append(
        "- Does the **Combined** policy beat both single-signal baselines on "
        "the constrained-capacity `cached_tok`? By how much, and is the gap "
        "larger than the seed-to-seed spread?"
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


def summary_has_ground_truth(path: str) -> bool:
    """True when a summary CSV was produced with ground-truth hit measurement.

    Pre-ground-truth summaries have no `hit_basis` column at all (their hit
    rate came from a latency threshold). Those must never be mixed with
    current runs or restored as if they were done.
    """
    import csv as _csv

    try:
        with open(path, newline="") as fh:
            row = next(iter(_csv.DictReader(fh)), None)
    except OSError:
        return False
    if row is None:
        return False
    return row.get("hit_basis") == "cached_tokens"


def ensure_base_results(csv_dir: str) -> int:
    """Copy frozen archived CSVs into `csv_dir` when it has none.

    Fresh Kaggle/Colab sessions start with an empty working dir, while an
    earlier matrix may be archived under `result_from_first_experiment/csv/`.
    Restoring it lets the runner skip work it has already done.

    IMPORTANT: we only restore archives that carry ground-truth hit
    measurement (`hit_basis=cached_tokens`). The experiment-1 archive does
    not: its hit rates came from a 797 ms latency threshold. Restoring it
    would be actively harmful, because the runner's "skip what's already
    done" check keys on `summary_<label>.csv` existing -- so stale,
    invalid runs would cause the new matrix to be skipped entirely and the
    analysis would silently report the old numbers.

    Fires only when `csv_dir` contains no `summary_*.csv` yet, so partial
    local runs are never touched. Set `KVCACHE_NO_RESTORE=1` to disable.
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

    summaries = [f for f in srcs if os.path.basename(f).startswith("summary_")]
    legacy = [f for f in summaries if not summary_has_ground_truth(f)]
    if legacy:
        print(f"[results] NOT restoring {len(srcs)} CSVs from {arch}: "
              f"{len(legacy)}/{len(summaries)} summaries predate ground-truth "
              f"hit measurement (no hit_basis=cached_tokens).")
        print("[results] Those runs used a latency-threshold hit proxy and are "
              "superseded. Restoring them would make the runner skip the new "
              "matrix and the analysis report the old numbers. Re-run the "
              "matrix instead.")
        return 0

    os.makedirs(csv_dir, exist_ok=True)
    for f in srcs:
        _shutil.copy(f, os.path.join(csv_dir, os.path.basename(f)))
    print(f"[results] restored {len(srcs)} CSVs from {arch} "
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

    # Refuse to silently mix measurement regimes. A legacy run's "hit rate"
    # is a latency threshold; a current run's is vLLM's cached-token
    # counter. Plotting them on one axis would be meaningless.
    legacy, current = [], []
    for label, df in rs.summaries.items():
        row = df.iloc[0].to_dict() if len(df) else {}
        (current if row.get("hit_basis") == "cached_tokens" else legacy).append(label)
    if legacy and current:
        print("\n[analysis] *** MIXED MEASUREMENT REGIMES IN ONE DIRECTORY ***")
        print(f"[analysis] ground truth ({len(current)}): {sorted(current)}")
        print(f"[analysis] LEGACY latency-proxy ({len(legacy)}): {sorted(legacy)}")
        print("[analysis] These are not comparable. Move the legacy CSVs out of")
        print("[analysis] this directory before reporting anything from it.")
    elif legacy and not current:
        print("\n[analysis] WARNING: every run here predates ground-truth hit")
        print("[analysis] measurement. The 'hit rate' in these charts is a")
        print("[analysis] latency threshold, NOT a cache hit rate.")

    for cap in ("constrained", "generous"):
        plot_headline_hit_rate(rs, os.path.join(fig_dir, f"headline_hit_rate_{cap}.png"), capacity=cap)
        plot_p99_latency(rs, os.path.join(fig_dir, f"p99_latency_{cap}.png"), capacity=cap)
        plot_goodput(rs, os.path.join(fig_dir, f"goodput_{cap}.png"), capacity=cap)
        plot_fairness(rs, os.path.join(fig_dir, f"fairness_{cap}.png"), capacity=cap)
        plot_shared_hit_rate(rs, os.path.join(fig_dir, f"shared_hit_rate_{cap}.png"), capacity=cap)
    plot_ablation_bar(rs, os.path.join(fig_dir, "ablation_bars.png"))
    write_interpretation(rs, os.path.join(fig_dir, "..", "INTERPRETATION.md"))
    print(f"[analysis] wrote charts to {fig_dir}")

