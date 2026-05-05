"""
"CMA-MEGA samples land near known minima" — the mechanism demonstration.

Uses:
  - 26 basin centroids from the clean 2-D Cooke archive (known_minima.json)
  - 50k raw emissions + 3k random seeds from the 3-D S2S3S4 Cooke run
    (lineage.jsonl)

Outputs a multi-panel figure:
  A. Histogram of "distance to nearest known minimum" — seed vs offspring.
  B. Survival curve: fraction of samples within radius r of nearest minimum.
  C. PCA scatter of basin centroids + density heatmap of offspring samples.
  D. PCA scatter of centroids + density heatmap of seed (random) samples.
  E. Headline numbers.

Distances in STANDARDISED theta-space (each dim divided by its std across the
basin-centroid set) so each dim contributes comparably.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KNOWN_JSON = "/home/hisham26/Desktop/RL/paper_figures/sample_efficiency_subregion/known_minima.json"
LINEAGE = (
    "/home/hisham26/Desktop/RL/results/lensmega/"
    "cooke_triplet_seidel3d_s2s3s4_20260423_031324/lineage.jsonl"
)
OUT_DIR = "/home/hisham26/Desktop/RL/paper_figures/sample_efficiency_subregion"
os.makedirs(OUT_DIR, exist_ok=True)


def load_known_minima():
    with open(KNOWN_JSON) as f:
        d = json.load(f)
    centroids = np.array([b["centroid_theta"] for b in d["basins"]], dtype=float)
    names = [f"basin_{b['basin_id']}_rms{b['best_rms_um']:.1f}" for b in d["basins"]]
    rms = np.array([b["best_rms_um"] for b in d["basins"]])
    return centroids, names, rms


def load_lineage_thetas(path):
    seeds, offs = [], []
    seed_fits, off_fits = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            th = r.get("theta")
            if th is None:
                continue
            if r["source"] == "seed":
                seeds.append(th)
                seed_fits.append(float(r.get("fitness", np.nan)))
            else:
                offs.append(th)
                off_fits.append(float(r.get("fitness", np.nan)))
    return (np.array(seeds), np.array(offs),
            np.array(seed_fits), np.array(off_fits))


def pca_2d(centroids_std):
    """PCA on standardised centroid set; return 2-PC basis."""
    mean = centroids_std.mean(axis=0)
    X = centroids_std - mean
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    basis = Vt[:2]  # shape (2, D)
    explained = (S ** 2) / (S ** 2).sum()
    return mean, basis, explained


def project(points_std, mean, basis):
    return (points_std - mean) @ basis.T


def nearest_dist(points_std, centroids_std):
    # For memory, compute nearest in chunks
    N = len(points_std)
    out = np.empty(N)
    chunk = 2000
    for i in range(0, N, chunk):
        sub = points_std[i:i + chunk]
        # (chunk, D) - (K, 1, D) -> (K, chunk) after norm
        diff = sub[np.newaxis, :, :] - centroids_std[:, np.newaxis, :]
        d = np.linalg.norm(diff, axis=2)
        out[i:i + chunk] = d.min(axis=0)
    return out


def main():
    centroids, basin_names, basin_rms = load_known_minima()
    print(f"loaded {len(centroids)} basin centroids (19-D theta)")
    seeds, offs, seed_fits, off_fits = load_lineage_thetas(LINEAGE)
    print(f"loaded {len(seeds):,} seed samples, {len(offs):,} offspring samples")

    # Standardise everything using std of the BASIN set
    mu = centroids.mean(axis=0)
    sd = centroids.std(axis=0) + 1e-9
    centroids_std = (centroids - mu) / sd
    seeds_std = (seeds - mu) / sd
    offs_std = (offs - mu) / sd

    # Nearest-basin distances
    print("computing nearest-basin distances ...")
    d_seed = nearest_dist(seeds_std, centroids_std)
    d_off = nearest_dist(offs_std, centroids_std)
    print(f"  seed   : mean={d_seed.mean():.3f}  median={np.median(d_seed):.3f}")
    print(f"  offspr : mean={d_off.mean():.3f}  median={np.median(d_off):.3f}")

    # Also compute nearest-basin distance for the BEST 10% of offspring
    # (by fitness) — the top samples should be closest
    top_mask = off_fits > np.quantile(off_fits[np.isfinite(off_fits)], 0.9)
    d_off_top = d_off[top_mask]
    print(f"  offspr top-10%: mean={d_off_top.mean():.3f}  "
          f"median={np.median(d_off_top):.3f}")

    # PCA basis
    pmean, pbasis, explained = pca_2d(centroids_std)
    print(f"PCA of centroids: top-2 explains {explained[:2].sum() * 100:.1f}% of variance")

    p_cents = project(centroids_std, pmean, pbasis)
    p_seeds = project(seeds_std, pmean, pbasis)
    p_offs = project(offs_std, pmean, pbasis)

    # Figure
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.30,
                          left=0.06, right=0.98, top=0.93, bottom=0.06)

    # ---- A. Histogram of nearest-basin distance ----
    ax = fig.add_subplot(gs[0, 0])
    lo = 0; hi = max(d_seed.max(), d_off.max())
    bins = np.linspace(lo, hi, 80)
    ax.hist(d_seed, bins=bins, alpha=0.55, color="C1",
            label=f"seed (random) — mean {d_seed.mean():.2f}",
            density=True)
    ax.hist(d_off, bins=bins, alpha=0.55, color="C0",
            label=f"CMA-MEGA offspring — mean {d_off.mean():.2f}",
            density=True)
    ax.axvline(d_seed.mean(), color="C1", ls="--", alpha=0.7)
    ax.axvline(d_off.mean(), color="C0", ls="--", alpha=0.7)
    ax.set_xlabel("distance to nearest known minimum (std θ units)")
    ax.set_ylabel("density")
    ax.set_title("A. Distance to nearest known minimum")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- B. Survival: fraction within radius r ----
    ax = fig.add_subplot(gs[0, 1])
    rr = np.linspace(0, hi, 200)
    cum_s = np.array([(d_seed <= r).mean() for r in rr]) * 100
    cum_o = np.array([(d_off <= r).mean() for r in rr]) * 100
    ax.plot(rr, cum_s, color="C1", lw=2, label="seed")
    ax.plot(rr, cum_o, color="C0", lw=2, label="CMA-MEGA offspring")
    ax.set_xlabel("radius r (std θ units)")
    ax.set_ylabel("% of samples within r of nearest minimum")
    ax.set_title("B. Cumulative — how close do samples get?")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # ---- C. PCA density (CMA-MEGA offspring) ----
    ax = fig.add_subplot(gs[0, 2])
    lim = max(abs(p_cents).max(), abs(p_offs).max()) * 1.05
    # Hexbin density for offspring
    hb = ax.hexbin(p_offs[:, 0], p_offs[:, 1], gridsize=60,
                   cmap="Blues", mincnt=1, extent=(-lim, lim, -lim, lim))
    ax.scatter(p_cents[:, 0], p_cents[:, 1], s=200, c="red", marker="*",
               edgecolor="black", linewidths=0.8, zorder=5,
               label=f"known minima (n={len(p_cents)})")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var of centroids)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var of centroids)")
    ax.set_title("C. CMA-MEGA offspring density vs known minima")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    plt.colorbar(hb, ax=ax, label="# offspring samples")

    # ---- D. PCA density (random seeds) — same axes ----
    ax = fig.add_subplot(gs[1, 0])
    hb = ax.hexbin(p_seeds[:, 0], p_seeds[:, 1], gridsize=60,
                   cmap="Oranges", mincnt=1, extent=(-lim, lim, -lim, lim))
    ax.scatter(p_cents[:, 0], p_cents[:, 1], s=200, c="red", marker="*",
               edgecolor="black", linewidths=0.8, zorder=5,
               label=f"known minima (n={len(p_cents)})")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var of centroids)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var of centroids)")
    ax.set_title("D. Random-seed density vs known minima\n"
                 "(same axes — uniform-θ baseline)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    plt.colorbar(hb, ax=ax, label="# seed samples")

    # ---- E. Top-10%-fitness CMA-MEGA samples alone ----
    ax = fig.add_subplot(gs[1, 1])
    p_top = p_offs[top_mask]
    ax.scatter(p_offs[:, 0], p_offs[:, 1], s=4, alpha=0.15,
               color="lightgray", label="all offspring")
    ax.scatter(p_top[:, 0], p_top[:, 1], s=6, alpha=0.5,
               color="C0", label=f"top-10% fitness (n={len(p_top):,})")
    ax.scatter(p_cents[:, 0], p_cents[:, 1], s=220, c="red", marker="*",
               edgecolor="black", linewidths=0.8, zorder=5,
               label="known minima")
    ax.set_xlabel(f"PC1"); ax.set_ylabel("PC2")
    ax.set_title("E. Top-10% CMA-MEGA samples\n→ should sit on the minima")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.3)

    # ---- F. Distance histogram for top 10% ----
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(d_seed, bins=bins, alpha=0.45, color="C1",
            label=f"seed (all)  mean {d_seed.mean():.2f}", density=True)
    ax.hist(d_off, bins=bins, alpha=0.45, color="C0",
            label=f"offspring (all)  mean {d_off.mean():.2f}", density=True)
    ax.hist(d_off_top, bins=bins, alpha=0.7, color="C2",
            label=f"offspring top 10%  mean {d_off_top.mean():.2f}",
            density=True)
    ax.axvline(d_off_top.mean(), color="C2", ls="--", alpha=0.7)
    ax.set_xlabel("distance to nearest known minimum")
    ax.set_ylabel("density")
    ax.set_title("F. Top-10% offspring are closest to minima")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- G. Headline metrics ----
    ax = fig.add_subplot(gs[2, :]); ax.axis("off")
    lines = [
        "HEADLINE",
        f"  Basin centroids (known minima, 19-D θ): n = {len(centroids)}",
        f"  Seed (random-θ) samples:       n = {len(seeds):,}  "
        f"mean nearest-basin dist = {d_seed.mean():.3f}  median = {np.median(d_seed):.3f}",
        f"  CMA-MEGA offspring samples:    n = {len(offs):,}  "
        f"mean nearest-basin dist = {d_off.mean():.3f}  median = {np.median(d_off):.3f}",
        f"  ↳ ratio mean(seed)/mean(offspring) = "
        f"{d_seed.mean() / max(d_off.mean(), 1e-9):.3f}×",
        "",
        f"  CMA-MEGA offspring, top 10% by fitness:  "
        f"mean nearest-basin dist = {d_off_top.mean():.3f}  median = {np.median(d_off_top):.3f}",
        f"  ↳ ratio mean(seed)/mean(top-10% offspring) = "
        f"{d_seed.mean() / max(d_off_top.mean(), 1e-9):.3f}×",
        "",
        "Radius-shell containment — fraction within distance r:",
        f"  {'r':>6}  {'seed %':>8}  {'offspring %':>12}  {'ratio':>8}",
    ]
    for r in [0.5, 1.0, 1.5, 2.0, 3.0]:
        fs = (d_seed <= r).mean() * 100
        fo = (d_off <= r).mean() * 100
        ratio = fo / max(fs, 1e-9)
        lines.append(f"  {r:>6.2f}  {fs:>7.3f}%  {fo:>11.3f}%  {ratio:>7.1f}×")
    ax.text(0.0, 0.95, "\n".join(lines), family="monospace", fontsize=10,
            va="top", transform=ax.transAxes)

    fig.suptitle(
        "CMA-MEGA+Seidel sampling vs 26 known minima (Cooke, 19-D θ)\n"
        "Raw emissions — no refinement — concentration toward known minima",
        fontsize=13, fontweight="bold",
    )
    out_png = os.path.join(OUT_DIR, "cma_mega_samples_vs_known_minima.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
