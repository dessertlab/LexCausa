import argparse
import csv
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt

DIMS = [
    ("Cogency", "A_contra_cogency_avg", "B_contra_cogency_avg"),
    ("Semantics", "A_contra_semantics_avg", "B_contra_semantics_avg"),
    ("NormSupport", "A_contra_norm_support_avg", "B_contra_norm_support_avg"),
]
DEFAULT_CSV = Path("experiments/causal_taxonomy_ablation/runs/run_summary.csv")
DEFAULT_OUT = Path("experiments/causal_taxonomy_ablation/analysis/figures/ablation_dims.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()

    rows = [
        r
        for r in csv.DictReader(a.csv.open(encoding="utf-8"))
        if float(r["A_aqa_net_contra"]) and float(r["B_aqa_net_contra"])
    ]
    labels = [d[0] for d in DIMS]
    av = [fmean(float(r[d[1]]) for r in rows) for d in DIMS]
    bv = [fmean(float(r[d[2]]) for r in rows) for d in DIMS]

    x = range(len(labels))
    w = 0.38
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ba = ax.bar([i - w / 2 for i in x], av, w, label="Unguided",
                color="#BFBFBF", edgecolor="#8A8A8A", linewidth=0.6)
    bb = ax.bar([i + w / 2 for i in x], bv, w, label="Taxonomy-guided",
                color="#6C6CF0", edgecolor="#3F3FC0", linewidth=0.6)
    ax.set_ylim(0.5, 0.98)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Contra-side score")
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bars in (ba, bb):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.3f}", (r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5,
              handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.0, dpi=300)


if __name__ == "__main__":
    main()
