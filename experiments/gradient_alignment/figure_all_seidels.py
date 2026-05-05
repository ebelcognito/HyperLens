"""Two-panel figure comparing all five Seidel coefficients across topologies.

Panel A: triplet (19-D)  — distribution of |cos(−∇L, ∇M_k)| for k=S1..S5.
Panel B: sixlet (37-D)   — same.

Each panel: violin/box plot per Seidel term plus the analytic random null.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEIDEL_LABELS = ["$S_1$\nspherical", "$S_2$\ncoma", "$S_3$\nastig.",
                 "$S_4$\nPetzval", "$S_5$\ndistortion"]
COLORS = ["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def panel(ax, npz_path: Path, title: str):
    d = np.load(npz_path, allow_pickle=True)
    cos = d["cos"]                # (N, 5)
    theta_dim = int(d["theta_dim"])
    null = float(np.sqrt(2.0 / (np.pi * theta_dim)))
    abs_cos = np.abs(cos)

    parts = ax.violinplot(
        [abs_cos[:, k] for k in range(5)],
        positions=range(5),
        showmeans=False, showmedians=False, showextrema=False,
        widths=0.85,
    )
    for k, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[k])
        body.set_alpha(0.45)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)

    # Means (solid) and medians (dotted) on top
    means = abs_cos.mean(axis=0)
    medians = np.median(abs_cos, axis=0)
    for k in range(5):
        ax.plot(
            [k - 0.32, k + 0.32], [means[k], means[k]],
            color="black", linewidth=2.0,
        )
        ax.plot(
            [k - 0.22, k + 0.22], [medians[k], medians[k]],
            color="black", linewidth=1.0, linestyle=":",
        )
        ax.text(
            k, means[k] + 0.04, f"{means[k]:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.axhline(
        null, color="grey", linestyle="--", linewidth=1.0,
        label=f"random null ({theta_dim}-D) = {null:.3f}",
    )
    ax.set_xticks(range(5))
    ax.set_xticklabels(SEIDEL_LABELS, fontsize=9)
    ax.set_ylabel(r"$|\cos(-\nabla L,\ \nabla M_k)|$")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--triplet",
        default="experiments/gradient_alignment/all_seidels_triplet.npz",
    )
    ap.add_argument(
        "--sixlet",
        default="experiments/gradient_alignment/all_seidels_sixlet.npz",
    )
    ap.add_argument(
        "--septlet",
        default="experiments/gradient_alignment/all_seidels_septlet.npz",
    )
    ap.add_argument(
        "--septlet-nofnum",
        default="experiments/gradient_alignment/all_seidels_septlet_nofnum.npz",
    )
    ap.add_argument(
        "--out",
        default="experiments/gradient_alignment/figure_all_seidels.pdf",
    )
    args = ap.parse_args()

    have_septlet = args.septlet and Path(args.septlet).exists()
    have_sept_nf = args.septlet_nofnum and Path(args.septlet_nofnum).exists()
    panels = [(Path(args.triplet), "(a)  Cooke triplet (19-D, S2/S3 used as BCs)")]
    panels.append((Path(args.sixlet), "(b)  Sixlet (37-D, S2/S3 used as BCs)"))
    if have_septlet:
        panels.append((
            Path(args.septlet),
            "(c)  Septlet APS-C (41-D, S4/S2 BCs, F# penalty ON)",
        ))
    if have_sept_nf:
        panels.append((
            Path(args.septlet_nofnum),
            "(d)  Septlet APS-C (41-D, S4/S2 BCs, F# penalty OFF — control)",
        ))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.6), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (path, title) in zip(axes, panels):
        panel(ax, path, title)
    fig.suptitle(
        "Gradient alignment of all five Seidel coefficients with the loss\n"
        r"Each violin: distribution of $|\cos(-\nabla L,\ \nabla M_k)|$ over"
        " the search trajectory. Solid bar = mean, dotted = median.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
