"""Method diagram: the extraction-under-quantization protocol.

Redrawn as a clean left-to-right pipeline with numbered stages, bold headers,
and the plum/crimson theme of the paper.

Usage:  python -m quantmem.method_figure --out figures/method.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

INK = "#241b2f"
MUTED = "#6f6480"
SURFACE = "#faf8fc"
PLUM = "#8338a0"
DEEP = "#45115a"
LIGHT = "#b57edc"
CRIMSON = "#d1495b"


def stage(ax, x, y, w, h, num, title, lines, accent, fill="#f3eef8"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fill, ec=accent, lw=2.2, zorder=2))
    # numbered chip
    ax.add_patch(Circle((x + 0.24, y + h - 0.24), 0.16, fc=accent, ec="none",
                        zorder=4))
    ax.text(x + 0.24, y + h - 0.24, str(num), ha="center", va="center",
            color="white", fontsize=10, fontweight="bold", zorder=5)
    # left-aligned bold title
    ax.text(x + 0.46, y + h - 0.20, title, ha="left", va="center",
            fontsize=11, color=INK, fontweight="bold")
    # body lines, left aligned
    for i, ln in enumerate(lines):
        ax.text(x + 0.22, y + h - 0.62 - i * 0.30, ln, ha="left", va="center",
                fontsize=8.6, color=MUTED, fontfamily="monospace")


def arrow(ax, x0, y0, x1, y1, label="", color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=16, color=color, lw=1.8,
                                 zorder=1, shrinkA=2, shrinkB=2))
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.12, label, ha="center", fontsize=8,
                color=color, fontstyle="italic")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures/method.png"))
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(13, 4.0), facecolor=SURFACE)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    stage(ax, 0.15, 1.35, 2.35, 1.7, 1, "Ground truth",
          ["64-token sequences", "the model is known", "to have memorized",
           "(public Pythia set)"], PLUM)
    stage(ax, 3.0, 1.35, 2.05, 1.7, 2, "Split",
          ["prompt = tok 1-32", "target = tok 33-64"], LIGHT)
    stage(ax, 5.55, 0.9, 2.15, 2.6, 3, "Quantize",
          ["Pythia 160M/410M/1B", "", "FP32 FP16 INT8", "NF4  FP4   (bnb)",
           "RTN8 RTN4  (ours)"], DEEP, fill="#efe6f5")
    stage(ax, 8.2, 2.15, 2.55, 1.35, 4, "Greedy decode",
          ["32 tokens, no", "sampling; compare", "to target"], LIGHT)
    stage(ax, 8.2, 0.35, 2.55, 1.35, 5, "Perplexity",
          ["WikiText-2 + Pile", "= capability", "control"], PLUM)
    stage(ax, 11.15, 0.9, 1.95, 2.6, 6, "Metrics",
          ["extraction", "rate", "", "vs. perplexity", "= selectivity"],
          CRIMSON, fill="#f9e9ec")

    arrow(ax, 2.5, 2.2, 3.0, 2.2)
    arrow(ax, 5.05, 2.2, 5.55, 2.2)
    arrow(ax, 7.7, 2.5, 8.2, 2.75)
    arrow(ax, 7.7, 1.7, 8.2, 1.0)
    arrow(ax, 10.75, 2.75, 11.15, 2.35)
    arrow(ax, 10.75, 1.0, 11.15, 1.75)

    ax.text(0.15, 3.72, "How we measure memorization and capability under quantization",
            fontsize=12.5, color=INK, fontweight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE, bbox_inches="tight",
                pad_inches=0.15)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
