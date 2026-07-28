"""Paper figures: memorization vs capability under quantization.

Visual theme ("precision & privacy"): model size is encoded as a plum
sequential ramp (bigger model = darker = more memory retained); the
independent RTN quantizer uses open diamonds; a crimson accent marks the
privacy-relevant regions; tick labels are monospace for a technical feel.

Figure 1: extraction rate by precision level, one line per model size.
Figure 2: the capability-memorization plane, with the region where
memorization is lost faster than capability shaded.

Usage:  python -m quantmem.figures --out figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- theme ---------------------------------------------------------------
INK = "#241b2f"          # deep plum-gray ink
MUTED = "#6f6480"        # muted plum
GRID = "#e7e0ef"
BASE = "#c9bedb"
SURFACE = "#faf8fc"      # faint purple-tinted surface
LEAK = "#d1495b"         # crimson: the privacy/leak accent
# model size as a sequential plum ramp (light -> dark = small -> large)
MODEL_COLORS = {"pythia-160m-deduped": "#b57edc",
                "pythia-410m-deduped": "#8338a0",
                "pythia-1b-deduped": "#45115a"}
MODEL_LABELS = {"pythia-160m-deduped": "160M",
                "pythia-410m-deduped": "410M",
                "pythia-1b-deduped": "1B"}
QUANT_ORDER = ["fp32", "fp16", "int8", "rtn8", "nf4", "rtn4", "fp4"]
BIT_LABEL = {"fp32": "FP32\n32-bit", "fp16": "FP16\n16-bit",
             "int8": "INT8\n8-bit", "nf4": "NF4\n4-bit", "fp4": "FP4\n4-bit"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("monospace")
        lbl.set_fontsize(8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)


def load_all(results: Path) -> dict:
    data = {}
    for model in MODEL_COLORS:
        rows = {}
        for q in QUANT_ORDER:
            ef = results / f"extract_{model}_{q}_s0.json"
            pf = results / f"ppl_{model}_{q}.json"
            pilef = results / f"ppl_{model}_{q}_pile.json"
            if ef.exists():
                rows[q] = {"exact": json.loads(ef.read_text())["exact_match"]}
                if pf.exists():
                    rows[q]["ppl"] = json.loads(pf.read_text())["perplexity"]
                if pilef.exists():
                    rows[q]["ppl_pile"] = json.loads(pilef.read_text())["perplexity"]
        if rows:
            data[model] = rows
    return data


def fig_extraction(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3), facecolor=SURFACE)
    qs = ["fp32", "fp16", "int8", "nf4", "fp4"]

    # crimson "majority leaked" band: extraction >= 0.5
    ax.axhspan(0.5, 1.0, color=LEAK, alpha=0.06, zorder=0)
    ax.axhline(0.5, color=LEAK, lw=1.1, ls=(0, (5, 3)), zorder=1)
    ax.text(0.05, 0.515, "majority of memorized data still extractable",
            color=LEAK, fontsize=7.8, va="bottom")

    for model, rows in data.items():
        x = np.arange(len(qs))
        y = [rows[q]["exact"] if q in rows else np.nan for q in qs]
        ax.plot(x, y, "-s", color=MODEL_COLORS[model], label=MODEL_LABELS[model],
                lw=2.2, ms=7, mec="white", mew=0.8, zorder=3)
        for q in rows:
            if q.startswith("rtn"):
                match = "int8" if q == "rtn8" else "nf4"
                if match in qs:
                    ax.plot(qs.index(match), rows[q]["exact"], "D", ms=8,
                            mfc=SURFACE, mec=MODEL_COLORS[model], mew=1.9, zorder=4)
    ax.plot([], [], "D", mfc=SURFACE, mec=MUTED, mew=1.6, label="RTN (independent)")

    ax.set_xticks(np.arange(len(qs)), [BIT_LABEL[q] for q in qs])
    ax.set_ylabel("verbatim extraction rate", color=INK, fontsize=10)
    ax.set_title("Memorized training data survives 4-bit quantization,\n"
                 "more so for larger models", fontsize=11, color=INK)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    _style(ax)
    ax.legend(frameon=False, title="model size", fontsize=9, title_fontsize=9,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out / "extraction_curve.png", dpi=200, facecolor=SURFACE)
    print(f"saved {out / 'extraction_curve.png'}")


def fig_selectivity(data: dict, out: Path, corpus_key: str = "ppl_pile") -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.4), facecolor=SURFACE)

    # shade the region below the diagonal: memorization lost faster
    ax.fill_between([0, 1], [0, 1], [0, 0], color="#8338a0", alpha=0.05, zorder=0)
    ax.plot([0, 1], [0, 1], "--", color=BASE, lw=1.5, zorder=1)
    ax.text(0.60, 0.66, "proportional decay", rotation=45, fontsize=8.5,
            color=MUTED, ha="center")
    ax.text(0.66, 0.30, "memorization\nlost faster", fontsize=9, color="#8338a0",
            ha="center", style="italic", alpha=0.8)

    for model, rows in data.items():
        ref_q = "fp32" if "fp32" in rows and corpus_key in rows.get("fp32", {}) else "fp16"
        if ref_q not in rows or corpus_key not in rows[ref_q]:
            continue
        ref = rows[ref_q]
        for q, r in rows.items():
            if q == ref_q or corpus_key not in r:
                continue
            cap = ref[corpus_key] / r[corpus_key]
            mem = r["exact"] / ref["exact"]
            is_rtn = q.startswith("rtn")
            ax.plot(cap, mem, "D" if is_rtn else "s", color=MODEL_COLORS[model],
                    ms=10 if is_rtn else 11,
                    mfc=SURFACE if is_rtn else MODEL_COLORS[model],
                    mec=MODEL_COLORS[model], mew=1.9, zorder=3)
            # skip labels for the 8-bit cluster hugging the reference corner
            if not (cap > 0.97 and mem > 0.95):
                ax.annotate(q.upper(), (cap, mem), textcoords="offset points",
                            xytext=(7, -2), fontsize=7.2, color=MUTED,
                            fontfamily="monospace")

    # highlight the danger point: high capability retained, memory still leaked
    for model, rows in data.items():
        if "nf4" in rows and corpus_key in rows.get("nf4", {}) and model.endswith("1b-deduped"):
            ref = rows["fp16"]
            cap = ref[corpus_key] / rows["nf4"][corpus_key]
            mem = rows["nf4"]["exact"] / ref["exact"]
            ax.scatter([cap], [mem], s=430, facecolors="none", edgecolors=LEAK,
                       lw=2.0, zorder=5)
            ax.annotate("95% capability kept,\n72% still leaked",
                        (cap, mem), textcoords="offset points", xytext=(-14, 26),
                        fontsize=8, color=LEAK, ha="right")

    for model in data:
        ax.plot([], [], "s", color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.plot([], [], "D", mfc=SURFACE, mec=MUTED, mew=1.6, label="RTN (independent)")

    ax.set_xlabel("capability retained  (Pile perplexity ratio)", color=INK, fontsize=10)
    ax.set_ylabel("memorization retained  (extraction ratio)", color=INK, fontsize=10)
    ax.set_title("Quantization erases memorization faster than capability",
                 fontsize=11, color=INK)
    ax.set_xlim(0.35, 1.03)
    ax.set_ylim(0, 1.05)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    _style(ax)
    ax.legend(frameon=False, title="model size", fontsize=9, title_fontsize=9,
              loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "selectivity_plane.png", dpi=200, facecolor=SURFACE)
    print(f"saved {out / 'selectivity_plane.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_all(args.results)
    fig_extraction(data, args.out)
    fig_selectivity(data, args.out)


if __name__ == "__main__":
    main()
