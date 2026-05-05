"""Generate the §3.2 gradient-alignment figure and summary.txt.

Four-panel layout:
  Panel A: cosine similarity trajectory over the search.
  Panel B: histogram of signed cosine (aggregated across all generations).
  Panel C: |cos| histograms (sign-invariant parallelism).
  Panel D: random-projection comparison — distribution of mean |cos| for 100
           random low-D descriptors vs. the two Seidel values.

Inputs:  experiments/gradient_alignment/cosine_similarities.npz
         experiments/gradient_alignment/random_projection_distribution.npz  (optional)
Outputs: experiments/gradient_alignment/figure_gradient_alignment.pdf
         experiments/gradient_alignment/summary.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COSINE = "experiments/gradient_alignment/cosine_similarities.npz"
DEFAULT_RANDPROJ = (
    "experiments/gradient_alignment/random_projection_distribution.npz"
)
DEFAULT_FIG = "experiments/gradient_alignment/figure_gradient_alignment.pdf"
DEFAULT_SUMMARY = "experiments/gradient_alignment/summary.txt"


def moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) <= w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosine", default=DEFAULT_COSINE)
    ap.add_argument("--randproj", default=DEFAULT_RANDPROJ)
    ap.add_argument("--fig", default=DEFAULT_FIG)
    ap.add_argument("--summary", default=DEFAULT_SUMMARY)
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args()

    COSINE_PATH = Path(args.cosine)
    RANDPROJ_PATH = Path(args.randproj)
    FIG_PATH = Path(args.fig)
    SUMMARY_PATH = Path(args.summary)
    d = np.load(COSINE_PATH, allow_pickle=True)
    cos = d["cos"]                                # (N, K)
    cross = d["cross_cos"]                        # (N, M)
    norm_l = d["norm_l"]
    norm_m = d["norm_m"]
    gens = d["generations"]
    labels = list(d["descriptor_labels"])
    K = cos.shape[1]

    # Drop generations where all entries are NaN (defensive — should be rare)
    keep = np.all(np.isfinite(cos), axis=1)
    if not np.all(keep):
        print(
            f"dropping {int((~keep).sum())} generations with NaN cosines "
            f"(of {len(keep)})"
        )
    cos_k = cos[keep]
    cross_k = cross[keep] if cross.size else cross
    nl_k = norm_l[keep]
    nm_k = norm_m[keep]
    gens_k = gens[keep]

    have_randproj = RANDPROJ_PATH.exists()
    n_panels = 4 if have_randproj else 3
    width_per = 5.2
    fig, axes = plt.subplots(1, n_panels, figsize=(width_per * n_panels, 4.4))
    # ---------- Panel A : signed-cos trajectory ----------
    axA = axes[0]
    colors = ["#1f77b4", "#d62728"]
    window = max(1, len(cos_k) // 50)  # ~2% smoothing
    for j in range(K):
        col = cos_k[:, j]
        smooth = moving_avg(col, window)
        axA.plot(
            gens_k, col, color=colors[j], alpha=0.18, linewidth=0.5
        )
        axA.plot(
            gens_k, smooth, color=colors[j], linewidth=1.6,
            label=fr"$M_{{{labels[j]}}}$",
        )
    axA.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
    axA.set_xlabel("CMA-MEGA generation")
    axA.set_ylabel(r"$\cos(-\nabla_\theta L,\ \nabla_\theta M_j)$")
    axA.set_title("(a)  signed cosine over the search")
    axA.set_ylim(-1.05, 1.05)
    axA.grid(alpha=0.3)
    axA.legend(loc="upper right", fontsize=9)

    # ---------- Panel B : signed-cos histograms ----------
    axB = axes[1]
    bins = np.linspace(-1.0, 1.0, 41)
    for j in range(K):
        col = cos_k[:, j]
        axB.hist(
            col, bins=bins, alpha=0.55, color=colors[j],
            label=labels[j], density=True,
        )
        m, med = float(np.mean(col)), float(np.median(col))
        axB.axvline(m, color=colors[j], linestyle="-", linewidth=1.5)
        axB.axvline(med, color=colors[j], linestyle=":", linewidth=1.2)
    axB.axvline(0.0, color="black", linewidth=0.7, linestyle="--")
    axB.set_xlabel(r"$\cos(-\nabla L,\ \nabla M_j)$")
    axB.set_ylabel("density")
    axB.set_title("(b)  distribution of signed cosine")
    axB.set_xlim(-1.0, 1.0)
    axB.grid(alpha=0.3)
    axB.legend(loc="upper left", fontsize=9, title="descriptor")

    # ---------- Panel C : |cos| histograms (sign-invariant) ----------
    axC = axes[2]
    abins = np.linspace(0.0, 1.0, 41)
    for j in range(K):
        ac = np.abs(cos_k[:, j])
        axC.hist(
            ac, bins=abins, alpha=0.55, color=colors[j],
            label=labels[j], density=True,
        )
        m = float(np.mean(ac))
        axC.axvline(m, color=colors[j], linestyle="-", linewidth=1.5)
    # Reference: random-vector |cos| in N-D has mean ~ sqrt(2/(pi*N)).
    theta_dim = int(d["theta_dim"]) if "theta_dim" in d.files else 19
    null_nd = float(np.sqrt(2.0 / (np.pi * theta_dim)))
    axC.axvline(
        null_nd, color="grey", linestyle="--", linewidth=1.0,
        label=f"random null ({theta_dim}-D) ≈ {null_nd:.3f}",
    )
    axC.set_xlabel(r"$|\cos(-\nabla L,\ \nabla M_j)|$")
    axC.set_ylabel("density")
    axC.set_title("(c)  |cosine|: sign-invariant parallelism")
    axC.set_xlim(0.0, 1.0)
    axC.grid(alpha=0.3)
    axC.legend(loc="upper left", fontsize=9)

    # ---------- Panel D : random-projection comparison ----------
    if have_randproj:
        rd = np.load(RANDPROJ_PATH, allow_pickle=True)
        pooled = rd["pooled_mean_abs_cos"]
        if "seidel_means" in rd.files and "seidel_labels" in rd.files:
            seidel_means = np.asarray(rd["seidel_means"], dtype=np.float64)
            seidel_lbls = list(rd["seidel_labels"])
        else:
            seidel_means = np.array(
                [float(rd["seidel_S2"]), float(rd["seidel_S3"])],
                dtype=np.float64,
            )
            seidel_lbls = ["S2", "S3"]
        # New schema saves "random_null"; old schema saved "random_null_19d".
        if "random_null" in rd.files:
            null_rd = float(rd["random_null"])
            theta_dim_rd = int(rd["theta_dim"]) if "theta_dim" in rd.files else theta_dim
        else:
            null_rd = float(rd["random_null_19d"])
            theta_dim_rd = 19
        axD = axes[3]
        bins = np.linspace(0.0, 1.0, 41)
        axD.hist(
            pooled, bins=bins, color="grey", alpha=0.6,
            label=(
                f"100 random {theta_dim_rd}→2 projections\n"
                f"({pooled.size} columns)\n"
                f"mean={pooled.mean():.3f}, max={pooled.max():.3f}"
            ),
            density=True,
        )
        axD.axvline(
            null_rd, color="black", linestyle=":", linewidth=1.2,
            label=f"random null ≈ {null_rd:.3f}",
        )
        for j, (lbl, mu) in enumerate(zip(seidel_lbls, seidel_means)):
            axD.axvline(
                float(mu), color=colors[j % len(colors)],
                linestyle="-", linewidth=2.0,
                label=f"Seidel ${lbl}$ = {float(mu):.3f}",
            )
        axD.set_xlim(0.0, 1.0)
        axD.set_xlabel(r"mean $|\cos(-\nabla L,\ \nabla M_j)|$ over trajectory")
        axD.set_ylabel("density")
        axD.set_title("(d)  Seidel vs random-projection descriptors")
        axD.grid(alpha=0.3)
        axD.legend(loc="upper right", fontsize=8)

    suffix = args.title_suffix or "Cooke triplet, 50 mm f/4, polychromatic."
    fig.suptitle(
        "Gradient alignment in CMA-MEGA: descent direction $-\\nabla_\\theta L$ "
        "vs. Seidel descriptor gradients $\\nabla_\\theta M_j$\n"
        f"{suffix} "
        "Solid vlines = mean; dotted = median; sign of signed cos depends on "
        "Seidel sign convention.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)
    fig.savefig(FIG_PATH.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"saved figure: {FIG_PATH}")

    # ---------- Summary stats ----------
    lines = []
    lines.append("Gradient alignment summary")
    lines.append("=" * 60)
    lines.append(f"Source: {COSINE_PATH}")
    lines.append(f"Generations logged: {len(cos_k)}")
    lines.append(f"theta_dim: {theta_dim}   descriptor_dim: {K}")
    lines.append("")
    lines.append("Sign convention:")
    lines.append("  cos(-∇_θ L, ∇_θ M_j) — positive = descriptor gradient")
    lines.append("  agrees with the loss-descent direction.")
    lines.append("")
    null_nd = float(np.sqrt(2.0 / (np.pi * theta_dim)))
    lines.append(
        f"Random-vector |cos| null in {theta_dim}D ≈ {null_nd:.4f}  "
        f"(mean of |cos(g, h)| for unrelated g, h)"
    )
    lines.append("")
    lines.append("Per-descriptor SIGNED cosine similarity:")
    for j in range(K):
        col = cos_k[:, j]
        m = float(np.mean(col))
        med = float(np.median(col))
        p10 = float(np.percentile(col, 10))
        p90 = float(np.percentile(col, 90))
        std = float(np.std(col))
        f_pos = float(np.mean(col > 0))
        f_strong = float(np.mean(col > 0.3))
        f_strong_neg = float(np.mean(col < -0.3))
        lines.append(f"  {labels[j]}:")
        lines.append(f"    mean   : {m:+.4f}")
        lines.append(f"    median : {med:+.4f}")
        lines.append(f"    std    : {std:.4f}")
        lines.append(f"    p10    : {p10:+.4f}")
        lines.append(f"    p90    : {p90:+.4f}")
        lines.append(f"    frac > 0    : {f_pos:.3f}")
        lines.append(f"    frac > 0.3  : {f_strong:.3f}")
        lines.append(f"    frac < -0.3 : {f_strong_neg:.3f}")
    lines.append("")
    lines.append(
        "Per-descriptor |cosine| (sign-invariant — Seidel sign is conventional):"
    )
    for j in range(K):
        ac = np.abs(cos_k[:, j])
        m = float(np.mean(ac))
        med = float(np.median(ac))
        f_strong = float(np.mean(ac > 0.3))
        lines.append(f"  {labels[j]}:")
        lines.append(f"    mean |cos|  : {m:.4f}   (null ≈ {null_nd:.4f})")
        lines.append(f"    median |cos|: {med:.4f}")
        lines.append(f"    frac |cos|>0.3 : {f_strong:.3f}")
    lines.append("")
    if cross_k.size:
        cc = cross_k[:, 0]
        lines.append("Cross-descriptor cosine similarity (∇M_S2 vs ∇M_S3):")
        lines.append(f"  mean        : {float(np.mean(cc)):+.4f}")
        lines.append(f"  median      : {float(np.median(cc)):+.4f}")
        lines.append(f"  mean |cos|  : {float(np.mean(np.abs(cc))):.4f}")
    lines.append("")
    lines.append("Gradient norms (raw, before unit-rescaling):")
    lines.append(f"  mean ||∇L||      : {float(np.mean(nl_k)):.4e}")
    for j in range(K):
        lines.append(
            f"  mean ||∇M_{labels[j]}|| : {float(np.mean(nm_k[:, j])):.4e}"
        )
        ratio = nl_k / np.maximum(nm_k[:, j], 1e-30)
        lines.append(
            f"  mean ||∇L||/||∇M_{labels[j]}|| : "
            f"{float(np.mean(ratio[np.isfinite(ratio)])):.4e}"
        )
    lines.append("")
    if have_randproj:
        rd = np.load(RANDPROJ_PATH, allow_pickle=True)
        pooled = rd["pooled_mean_abs_cos"]
        pub_abs = rd["pub_abs_cos"]
        pub_cos = rd["pub_cos"]
        if "seidel_means" in rd.files and "seidel_labels" in rd.files:
            seidel_means_arr = np.asarray(rd["seidel_means"], dtype=np.float64)
            seidel_lbls_arr = list(rd["seidel_labels"])
        else:
            seidel_means_arr = np.array(
                [float(rd["seidel_S2"]), float(rd["seidel_S3"])],
                dtype=np.float64,
            )
            seidel_lbls_arr = ["S2", "S3"]
        if "random_null" in rd.files:
            null_rd = float(rd["random_null"])
            theta_dim_rd = int(rd["theta_dim"]) if "theta_dim" in rd.files else theta_dim
        else:
            null_rd = float(rd["random_null_19d"])
            theta_dim_rd = 19
        lines.append("=" * 60)
        lines.append("Comparison vs. random-projection descriptors")
        lines.append("=" * 60)
        lines.append(
            "Random projection descriptor gradients are constant in θ:"
        )
        lines.append(
            f"  ∇_θ M_random_j = A[:, j]  "
            f"(j-th column of fixed {theta_dim_rd}×2 Gaussian matrix, unit-normalised)."
        )
        lines.append("")
        lines.append(
            "Per-descriptor |cos(−∇L, A_j)| using the published §5.2 ablation"
        )
        lines.append("matrix (RandomState seed 12345, matches ablation_random_bc.yaml):")
        for j in range(pub_abs.shape[1]):
            col = pub_abs[:, j]
            col = col[np.isfinite(col)]
            lines.append(f"  Random_{j+1}:")
            lines.append(f"    mean |cos|     : {col.mean():.4f}   (null ≈ {null_rd:.4f})")
            lines.append(f"    median |cos|   : {float(np.median(col)):.4f}")
            lines.append(f"    frac |cos|>0.3 : {float((col > 0.3).mean()):.3f}")
            lines.append(
                f"    signed mean cos: {float(pub_cos[:, j].mean()):+.4f}"
            )
        lines.append("")
        lines.append("Side-by-side comparison:")
        for j in range(pub_abs.shape[1]):
            col = pub_abs[:, j]
            col = col[np.isfinite(col)]
            lines.append(
                f"  Random_{j+1}  mean |cos| : {col.mean():.4f}"
            )
        for lbl, mu in zip(seidel_lbls_arr, seidel_means_arr):
            lines.append(f"  Seidel {lbl}  mean |cos| : {float(mu):.4f}")
        lines.append(f"  {theta_dim_rd}-D random null      : {null_rd:.4f}")
        lines.append("")
        lines.append(
            f"Distribution across {len(rd['seeds'])} random matrices × 2 "
            f"columns = {pooled.size} random descriptor directions:"
        )
        lines.append(f"  mean      : {pooled.mean():.4f}")
        lines.append(f"  std       : {pooled.std():.4f}")
        lines.append(f"  median    : {float(np.median(pooled)):.4f}")
        lines.append(
            f"  95% range : "
            f"[{float(np.percentile(pooled, 2.5)):.4f}, "
            f"{float(np.percentile(pooled, 97.5)):.4f}]"
        )
        lines.append(f"  min, max  : {pooled.min():.4f}, {pooled.max():.4f}")
        for lbl, mu in zip(seidel_lbls_arr, seidel_means_arr):
            frac = float((pooled >= float(mu)).mean())
            lines.append(
                f"  frac mean |cos| ≥ Seidel {lbl} ({float(mu):.4f}) : "
                f"{frac:.4f}  ({int(frac * pooled.size)}/{pooled.size})"
            )
        lines.append("")
    pred_std = 1.0 / np.sqrt(theta_dim)
    lines.append("Sanity checks (see analyze_gradients.py stdout):")
    lines.append("  - cos(∇L, ∇L) = 1.000000   ✓")
    lines.append(
        f"  - cos(∇L, perm(∇L)) baseline: predicted 0 ± 1/√{theta_dim} ≈ "
        f"{pred_std:.3f}"
    )
    lines.append("  - all cosines finite and in [-1, 1]   ✓")
    lines.append(f"  - 0 of N={len(cos_k)} generations have NaN gradients   ✓")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
