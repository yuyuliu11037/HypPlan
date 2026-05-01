"""Bar plot of ProofWriter accuracy stratified by question depth (QDep 0-3).

Source: logs/proofwriter_multihop_summary.txt (Qwen-14B-Instruct, 200 test records).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SYSTEMS = ["Few Shot", "Self-Consistency", "Planning-Token", "HypPlan"]
QDEPS = ["QDep=0", "QDep=1", "QDep=2", "QDep=3"]
ACC = np.array([
    [79.3, 68.5, 62.2, 59.3],
    [82.9, 75.9, 59.5, 63.0],
    [35.4, 38.9, 43.2, 48.1],
    [11.0, 48.1, 75.7, 96.3],
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figs/proofwriter_qdep.png")
    args = ap.parse_args()

    n_sys, n_dep = ACC.shape
    x = np.arange(n_dep)
    width = 0.8 / n_sys
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (sys_name, row) in enumerate(zip(SYSTEMS, ACC)):
        offset = (i - (n_sys - 1) / 2) * width
        bars = ax.bar(x + offset, row, width, label=sys_name, color=colors[i])
        for b, v in zip(bars, row):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(QDEPS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("ProofWriter accuracy by question depth (Qwen-14B, 200 test)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
