"""
3-D version of the CMA-MEGA sampling demo.

Uses top-3 PCA on curvature dims (same setup as v2), and shows:
  - a 3-D scatter plot with samples and basin stars
  - three orthographic 2-D projections (PC1-PC2, PC1-PC3, PC2-PC3)

Renders a single 4-panel figure: 3-D + three 2-D projections.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

KNOWN_JSON = "/home/hisham26/Desktop/RL/paper_figures/sample_efficiency_subregion/known_minima.json"
LINEAGE = (
    "/home/hisham26/Desktop/RL/results/lensmega/"
    "cooke_triplet_seidel3d_s2s3s4_20260423_031324/lineage.jsonl"
)
OUT_DIR = "/home/hisham26/Desktop/RL/paper_figures/sample_efficiency_subregion"

CURV_IDX = list(range(6))
MIN_MEMBERS = 2
MAX_RMS_UM = 20.0
TOP_FITNESS_FRAC = 0.05


def load_reliable_minima():
    with open(KNOWN_JSON) as f:
        d = json.load(f)
    basins = [b for b in d["basins"]
              if b["n_members"] >= MIN_MEMBERS
              and b["best_rms_um"] <= MAX_RMS_UM]
    centroids = np.array([b["centroid_theta"] for b in basins], dtype=float)
    labels = [f"b{b['basin_id']} {b['best_rms_um']:.1f}μm"
              for b in basins]
    return centroids, labels


def load_lineage(path):
    seeds, offs, seed_fits, off_fits = [], [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            th = r.get("theta")
            if th is None:
                continue
            if r["source"] == "seed":
                seeds.append(th); seed_fits.append(float(r.get("fitness", np.nan)))
            else:
                offs.append(th); off_fits.append(float(r.get("fitness", np.nan)))
    return (np.array(seeds), np.array(offs),
            np.array(seed_fits), np.array(off_fits))


def main():
    cents_full, labels = load_reliable_minima()
    seeds, offs, seed_fits, off_fits = load_lineage(LINEAGE)
    print(f"minima: {len(cents_full)}  seeds: {len(seeds)}  offspring: {len(offs)}")

    # Curvature subset
    cents = cents_full[:, CURV_IDX]
    seeds_c = seeds[:, CURV_IDX]
    offs_c = offs[:, CURV_IDX]

    # Standardise by std of everything combined
    all_c = np.vstack([cents, seeds_c, offs_c])
    mu = all_c.mean(axis=0); sd = all_c.std(axis=0) + 1e-9
    cents_s = (cents - mu) / sd
    seeds_s = (seeds_c - mu) / sd
    offs_s = (offs_c - mu) / sd

    # PCA (3 components)
    c_mean = cents_s.mean(axis=0)
    X = cents_s - c_mean
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    basis = Vt[:3]
    explained = (S ** 2) / (S ** 2).sum()
    print(f"PCA top-3 explains {explained[:3].sum() * 100:.2f}% of centroid variance")

    def proj(P):
        return (P - c_mean) @ basis.T
    p_cents = proj(cents_s)
    p_seeds = proj(seeds_s)
    p_offs = proj(offs_s)

    # Top-fitness offspring
    valid = np.isfinite(off_fits)
    q = np.quantile(off_fits[valid], 1.0 - TOP_FITNESS_FRAC)
    top_mask = valid & (off_fits >= q)
    p_offs_top = p_offs[top_mask]
    off_fits_top = off_fits[top_mask]

    # ---------- FIGURE ----------
    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.2, 1.0],
        hspace=0.28, wspace=0.28,
        left=0.04, right=0.98, top=0.94, bottom=0.06,
    )

    # Main: 3-D scatter (spans left 2/3 of top row)
    ax3d = fig.add_subplot(gs[0, :2], projection="3d")

    # Extent for axes
    all_p = np.vstack([p_cents, p_seeds, p_offs])
    lim = np.percentile(np.abs(all_p), 98) * 1.1

    # Random seeds (very sparse background)
    sub_seed = np.random.default_rng(0).choice(
        len(p_seeds), min(600, len(p_seeds)), replace=False,
    )
    ax3d.scatter(
        p_seeds[sub_seed, 0], p_seeds[sub_seed, 1], p_seeds[sub_seed, 2],
        s=3, alpha=0.12, c="C1",
        label=f"random seeds (sub {len(sub_seed):,})",
        depthshade=False,
    )
    # Top-5% offspring — smaller/denser dots
    ax3d.scatter(
        p_offs_top[:, 0], p_offs_top[:, 1], p_offs_top[:, 2],
        s=4, alpha=0.5, c="C2",
        label=f"top-{int(TOP_FITNESS_FRAC*100)}% priors (N={top_mask.sum():,})",
        depthshade=False,
    )

    # Tight attraction-region wireframes around each minimum (r=0.5 std unit)
    # so they don't overlap into a single blob
    SPHERE_R = 0.5
    u = np.linspace(0, 2 * np.pi, 22)
    v = np.linspace(0, np.pi, 14)
    xs = SPHERE_R * np.outer(np.cos(u), np.sin(v))
    ys = SPHERE_R * np.outer(np.sin(u), np.sin(v))
    zs = SPHERE_R * np.outer(np.ones_like(u), np.cos(v))
    for pc in p_cents:
        ax3d.plot_wireframe(
            pc[0] + xs, pc[1] + ys, pc[2] + zs,
            color="red", alpha=0.25, linewidth=0.6, rcount=7, ccount=11,
        )

    # Known minima as big red stars — plotted last so they render on top
    ax3d.scatter(
        p_cents[:, 0], p_cents[:, 1], p_cents[:, 2],
        s=600, c="red", marker="*", edgecolor="black", linewidths=2.0,
        zorder=100,
        label=f"known minima ({len(p_cents)})",
    )
    # Minima labels (offset up-and-away)
    for lab, pc in zip(labels, p_cents):
        ax3d.text(pc[0], pc[1], pc[2] + 0.6, lab,
                  fontsize=10, color="darkred",
                  ha="center", va="bottom", fontweight="bold", zorder=101)

    ax3d.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)", fontsize=10)
    ax3d.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)", fontsize=10)
    ax3d.set_zlabel(f"PC3 ({explained[2]*100:.1f}%)", fontsize=10)
    ax3d.set_xlim(-lim, lim); ax3d.set_ylim(-lim, lim); ax3d.set_zlim(-lim, lim)
    ax3d.set_title(
        "Curvature-PCA 3-D view — red stars = known minima, "
        "wireframes = r=0.5 attraction shell\n"
        "green = top-5% CMA-MEGA priors, orange = random seeds",
        fontsize=10, pad=15,
    )
    ax3d.legend(loc="upper right", fontsize=9)
    ax3d.view_init(elev=22, azim=35)

    # Headline text panel (top-right). Distances in full 6-D curvature space
    # (not the 3-D PCA projection) — consistent with v2's metric.
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    def _nearest_full(pts):
        N = len(pts); out = np.empty(N)
        chunk = 2000
        for i in range(0, N, chunk):
            sub = pts[i:i + chunk]
            diff = sub[np.newaxis, :, :] - cents_s[:, np.newaxis, :]
            out[i:i + chunk] = np.linalg.norm(diff, axis=2).min(axis=0)
        return out
    d_seeds = _nearest_full(seeds_s)
    d_offs = _nearest_full(offs_s)
    d_offs_top = d_offs[top_mask]

    lines = [
        "SETUP",
        f"  {len(p_cents)} known minima (≥{MIN_MEMBERS} members, RMS ≤ {MAX_RMS_UM} μm)",
        f"  PCA on 6-D curvatures → top-3 PCs",
        f"  explained: {explained[0]*100:.1f} + {explained[1]*100:.1f} + "
        f"{explained[2]*100:.1f}% = {explained[:3].sum()*100:.2f}%",
        "",
        "NEAREST-MINIMUM DISTANCE (std curvature units)",
        f"  random seed       mean {d_seeds.mean():.2f}",
        f"  CMA-MEGA all      mean {d_offs.mean():.2f}  (×{d_seeds.mean()/d_offs.mean():.2f})",
        f"  CMA-MEGA top-{int(TOP_FITNESS_FRAC*100)}%   mean {d_offs_top.mean():.2f}  "
        f"(×{d_seeds.mean()/d_offs_top.mean():.2f})",
        "",
        "FRACTION WITHIN RADIUS r",
        f"  {'r':>5}  {'seed':>8}  {'off(all)':>9}  {'top-5%':>8}",
    ]
    for r in [0.5, 1.0, 1.5, 2.0, 3.0]:
        fs = (d_seeds <= r).mean() * 100
        fo = (d_offs <= r).mean() * 100
        ft = (d_offs_top <= r).mean() * 100
        lines.append(f"  {r:>5.1f}  {fs:>6.2f}%  {fo:>7.2f}%  {ft:>7.2f}%")
    ax.text(0.0, 0.98, "\n".join(lines), family="monospace", fontsize=9,
            va="top", transform=ax.transAxes)

    # Three orthographic projections along the bottom
    proj_pairs = [(0, 1, "PC1", "PC2"),
                  (0, 2, "PC1", "PC3"),
                  (1, 2, "PC2", "PC3")]
    for i, (a, b, la, lb) in enumerate(proj_pairs):
        ax = fig.add_subplot(gs[1, i])
        # Seed as orange background
        ax.scatter(p_seeds[sub_seed, a], p_seeds[sub_seed, b],
                   s=5, alpha=0.15, c="C1", label="random seed")
        # Top-5% as green
        ax.scatter(p_offs_top[:, a], p_offs_top[:, b],
                   s=8, alpha=0.35, c="C2",
                   label=f"top-{int(TOP_FITNESS_FRAC*100)}% priors")
        # Minima as red stars
        ax.scatter(p_cents[:, a], p_cents[:, b],
                   s=320, c="red", marker="*",
                   edgecolor="black", linewidths=1.2, zorder=5)
        ax.set_xlabel(la); ax.set_ylabel(lb)
        ax.set_title(f"({la}, {lb}) projection", fontsize=11)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.axhline(0, color="k", lw=0.3, alpha=0.3)
        ax.axvline(0, color="k", lw=0.3, alpha=0.3)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Where do CMA-MEGA+Seidel priors land? 3-D curvature-PCA view\n"
        "top-5% fitness emissions (green) vs known minima (red stars) — "
        "raw θ, no refinement",
        fontsize=13, fontweight="bold",
    )
    out_png = os.path.join(OUT_DIR, "cma_mega_samples_3d.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
