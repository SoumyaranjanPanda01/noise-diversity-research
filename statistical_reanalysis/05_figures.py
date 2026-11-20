"""Step 5: the three figures used in the paper, generated only from the run records and the outputs of steps 03-04.

Writes vector PDFs (TrueType fonts embedded, no Type 3) to revised_manuscript/figures/.
IEEE sizing: single column 3.5 in, double column 7.16 in. No plotted value is typed by hand.

    python statistical_reanalysis/05_figures.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import ALL_FAMILIES, OUT, ROOT  # noqa: E402

FIG = ROOT / "revised_manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "grid.color": "#e4e3df", "grid.linewidth": 0.5, "lines.linewidth": 1.5, "savefig.dpi": 300,
})
SINGLE, DOUBLE = 3.5, 7.16


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def fig_primary(r3):
    """Primary effect per fold and pooled, with item-only, passage-disjoint, and seed+item intervals."""
    S1 = r3["settings"]["S1 cluster bootstrap"]["primary"]
    S3 = r3["settings"]["S3 cluster, exclude verbatim+near-duplicate"]["primary"]
    rows = [("Fold A", "fold_a"), ("Fold B", "fold_b"), ("Pooled", "pooled")]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.05))
    for i, (lab, k) in enumerate(rows):
        y = len(rows) - 1 - i
        a, b = S1[k], S3[k]
        ax.plot(a["combined_ci95_pp"], [y + 0.18] * 2, color=GRAY, lw=4, solid_capstyle="butt", alpha=0.55)
        ax.plot(a["ci95_pp"], [y + 0.18] * 2, color=BLUE, lw=1.6)
        ax.plot([a["effect_pp"]], [y + 0.18], "o", color=BLUE, ms=4.5, mec="white", mew=0.6)
        ax.plot(b["ci95_pp"], [y - 0.14] * 2, color=ORANGE, lw=1.6)
        ax.plot([b["effect_pp"]], [y - 0.14], "s", color=ORANGE, ms=4.2, mec="white", mew=0.6)
        for v in a["per_seed_pp"]:
            ax.plot([v], [y + 0.40], "|", color=INK, ms=5, mew=0.9)
    ax.axvline(0, color=MUTED, lw=0.7, ls=(0, (2, 2)))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlabel("Mixed-3 minus mean single-family accuracy on unseen families (pp)")
    ax.set_ylim(-0.55, len(rows) - 0.35)
    ax.grid(axis="x")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=BLUE, marker="o", ms=4, lw=1.6, label="all 1,000 items (cluster CI)"),
         Line2D([], [], color=ORANGE, marker="s", ms=4, lw=1.6, label="passage-disjoint, 827 items"),
         Line2D([], [], color=GRAY, lw=4, alpha=0.55, label="+ seed variance (approx.)"),
         Line2D([], [], color=INK, marker="|", ls="", ms=5, label="individual seeds")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False,
              handlelength=1.6, columnspacing=1.0)
    save(fig, "fig_primary_effect")


def fig_transfer(r3):
    tm = r3["transfer_matrix_S1"]
    fams = ALL_FAMILIES
    M = np.array([[tm[r][c]["effect_pp"] for c in fams] for r in fams])
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#c0392b", "#f1f0ec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    im = ax.imshow(np.clip(M, -3, 3), cmap=cmap, norm=TwoSlopeNorm(0, -3, 3))
    for i in range(6):
        for j in range(6):
            lo, hi = tm[fams[i]][fams[j]]["ci95_pp"]
            excl = lo > 0 or hi < 0
            ax.text(j, i, f"{M[i, j]:+.1f}" + ("*" if excl else ""), ha="center", va="center", fontsize=6.3,
                    color="white" if abs(M[i, j]) > 2.2 else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(6))
    ax.set_xticklabels(fams, rotation=35, ha="right")
    ax.set_yticks(range(6))
    ax.set_yticklabels(fams)
    ax.set_xlabel("Test family")
    ax.set_ylabel("Training family (single-family model)")
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("pp vs clean baseline (clipped at ±3)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    save(fig, "fig_transfer_matrix")


def fig_distractor(r4):
    c = r4["conditions"]
    order = [("untuned", "untuned"), ("shared/clean", "clean"), ("shared/neftune", "NEFTune"),
             ("fold_a/mixed3_f0.50", "A: Mixed-3"), ("fold_b/single_shuffle", "B: single shuffle"),
             ("fold_b/mixed3_f0.10", "B: Mixed-3 0.10"), ("fold_b/mixed3_f0.50", "B: Mixed-3 0.50"),
             ("fold_b/single_distractor", "B: single distractor")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.15), gridspec_kw={"width_ratios": [1.25, 1]})
    for i, (k, lab) in enumerate(order):
        y = len(order) - 1 - i
        yc, yd = 100 * c[k]["yes_rate_clean"][0], 100 * c[k]["yes_rate_distractor"][0]
        a1.plot([yc, yd], [y, y], color="#cfcdc6", lw=2.2, zorder=1)
        a1.plot(yc, y, "o", color=INK, ms=4.5, mec="white", mew=0.5, zorder=2)
        a1.plot(yd, y, "s", color=ORANGE, ms=4.5, mec="white", mew=0.5, zorder=2)
        for rr in r4["per_run"]:
            if rr["condition"] == k and k != "untuned":
                a1.plot(100 * rr["yes_rate_distractor"], y + 0.22, "|", color=ORANGE, ms=4, mew=0.8)
    a1.axvline(100 * r4["gold_yes_rate"], color=MUTED, lw=0.7, ls=(0, (2, 2)))
    a1.text(100 * r4["gold_yes_rate"] + 1, len(order) - 0.45, "gold Yes rate", color=MUTED, fontsize=6.3)
    a1.set_yticks(range(len(order)))
    a1.set_yticklabels([o[1] for o in order][::-1])
    a1.set_xlabel("Predicted \"Yes\" (%)")
    a1.set_xlim(-2, 75)
    a1.set_ylim(-0.6, len(order) - 0.2)
    a1.grid(axis="x")
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([], [], marker="o", color=INK, ls="", ms=4, label="clean questions"),
                       Line2D([], [], marker="s", color=ORANGE, ls="", ms=4, label="with distractor (| = seeds)")],
              loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False, fontsize=6.5)
    a1.set_title("(a) Answer preference", loc="left")

    dr = r4["dose_response"]
    pts = [(v["n_distractor_train"], 100 * v["acc_distractor"][0], 100 * v["acc_distractor"][1], k) for k, v in dr.items()]
    zero = [p for p in pts if p[0] == 0]
    pos = sorted([p for p in pts if p[0] > 0])
    xs0 = 60
    for j, p in enumerate(zero):
        a2.errorbar(xs0 * (0.85 + 0.1 * j), p[1], yerr=p[2], fmt="o", color=GRAY, ms=3.5, capsize=1.5, lw=0.8)
    a2.errorbar([p[0] for p in pos], [p[1] for p in pos], yerr=[p[2] for p in pos], fmt="-o", color=BLUE, ms=3.5,
                capsize=1.5, lw=1.3)
    a2.axhline(100 * c["shared/clean"]["acc_clean"][0], color=MUTED, lw=0.7, ls=(0, (2, 2)))
    a2.text(2400, 100 * c["shared/clean"]["acc_clean"][0] + 0.8, "clean-input accuracy\n(clean model)", color=MUTED,
            fontsize=6.0, ha="right", va="bottom")
    a2.set_xscale("log")
    a2.set_xticks([xs0, 133, 333, 667, 1333, 2000])
    a2.set_xticklabels(["0", "133", "333", "667", "1333", "2000"], rotation=35)
    a2.minorticks_off()
    a2.set_xlabel("Distractor examples in training (of 4,000)")
    a2.set_ylabel("Distractor-test accuracy (%)")
    a2.set_ylim(54, 84)
    a2.grid(axis="y")
    a2.set_title("(b) In-family exposure", loc="left")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig_distractor_bias")


def main():
    r3 = json.loads((OUT / "03_contrasts_reanalysis.json").read_text(encoding="utf-8"))
    r4 = json.loads((OUT / "04_distractor_bias.json").read_text(encoding="utf-8"))
    fig_primary(r3)
    fig_transfer(r3)
    fig_distractor(r4)


if __name__ == "__main__":
    main()
