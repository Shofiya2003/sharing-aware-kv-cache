"""Zip matrix results into one archive for one-click download.

On Kaggle/Colab the results live on the ephemeral container disk. This
script bundles results/csv/, results/figures/ and results/INTERPRETATION.md
(whatever exists) into a single zip you download from the file browser.

Usage (from the repo root, on the GPU host):
    python experiments/zip_results.py
    python experiments/zip_results.py --out /kaggle/working/kvcache_results.zip
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=".",
                   help="Repo root (contains results/). Defaults to cwd.")
    p.add_argument("--out", default=None,
                   help="Output zip path. Default: <repo>/kvcache_results.zip "
                        "(on Kaggle you may prefer /kaggle/working/kvcache_results.zip).")
    args = p.parse_args()

    repo = os.path.abspath(args.repo)
    targets = ["results/csv", "results/figures", "results/INTERPRETATION.md"]
    existing = [t for t in targets if os.path.exists(os.path.join(repo, t))]
    if not existing:
        print(f"[zip] nothing to zip: no results/ found under {repo}")
        return 1

    out = args.out or os.path.join(repo, "kvcache_results.zip")
    n_files = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for t in existing:
            full = os.path.join(repo, t)
            if os.path.isdir(full):
                for root, _, files in os.walk(full):
                    for f in sorted(files):
                        fp = os.path.join(root, f)
                        z.write(fp, os.path.relpath(fp, repo))
                        n_files += 1
            else:
                z.write(full, t)
                n_files += 1
    size_kb = os.path.getsize(out) / 1024.0
    print(f"[zip] wrote {out} ({n_files} files, {size_kb:.0f} KB)")
    print(f"[zip] included: {', '.join(existing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
