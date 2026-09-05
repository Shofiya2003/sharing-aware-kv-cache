"""Phase 6 — Produce the headline + ablation + fairness charts.

Reads the CSVs written by phase5_matrix (or by run_experiment.py) and
emits figures + a draft interpretation note.

Usage:
    python experiments/phase6_analysis.py
    python experiments/phase6_analysis.py --csv-dir results/csv --fig-dir results/figures
"""

from __future__ import annotations

import argparse
import sys

from kvcache.analysis import run_analysis


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv-dir", default="results/csv")
    p.add_argument("--fig-dir", default="results/figures")
    args = p.parse_args()
    run_analysis(args.csv_dir, args.fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

