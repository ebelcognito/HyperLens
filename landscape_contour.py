"""
Proper fitness-landscape figure:
  * 2-D plane in 19-D θ-space spanned by the top-2 PCA axes of the 5
    known minima's curvature centroids.
  * Actual fitness function evaluated on a grid over that plane.
  * Contour plot + known-minima stars + CMA-MEGA offspring projected onto
    the plane (top-5% by fitness).

This replaces the earlier sample-density landscape which was a proxy.
"""
import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/hisham26/Desktop/RL")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.run import build_all


CFG_PATH = (
    "/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
    "config/examples/cooke_triplet_seidel3d_s2s3s4.yaml"
)
KNOWN_JSON = (
    "/home/hisham26/Desktop/RL/paper_figures/"
    "sample_efficiency_subregion/known_minima.json"
)
LINEAGE = (
    "/home/hisham26/Desktop/RL/results/lensmega/"
    "cooke_triplet_seidel3d_s2s3s4_20260423_031324/lineage.jsonl"
)
OUT = "/home/hisham26/Desktop/RL/paper_figures/landscape_contour"
os.makedirs(OUT, exist_ok=True)

CURV_IDX = list(range(6))
MIN_MEMBERS = 2
MAX_RMS_UM = 20.0
GRID = 40              # grid resolution — 40x40 = 1600 fitness evals
TOP_FITNESS_FRAC = 0.05


def load_minima():
    with open(KNOWN_JSON) as f:
        d = json.load(f)
    basins = [b for b in d["basins"]
              if b["n_members"] >= MIN_MEMBERS
              and b["best_rms_um"] <= MAX_RMS_UM]
    cents = np.array([b["centroid_theta"] for b in basins], dtype=float)
    labels = [f"b{b['basin_id']}·{b['best_rms_um']:.1f}μm" for b in basins]
    return cents, labels


def load_lineage():
    seeds, offs, off_fits = [], [], []
    with open(LINEAGE) as f:
        for line in f:
            r = json.loads(line)
            th = r.get("theta")
            if th is None:
                continue
            if r["source"] == "seed":
                seeds.append(th)
            else:
                offs.append(th); off_fits.append(float(r.get("fitness", np.nan)))
    return np.array(seeds), np.array(offs), np.array(off_fits)


def main():
    cfg = load_config(CFG_PATH)
    problem, descriptor, archive, emitter, seeder = build_all(cfg)
    problem._ensure_backend()
    print(f"theta_dim = {problem.theta_dim}")

    cents, labels = load_minima()
    seeds, offs, off_fits = load_lineage()
    print(f"minima: {len(cents)}  seeds: {len(seeds):,}  offspring: {len(offs):,}")

    # Mean θ over the minima — this is the "centre" of the plane.
    theta_mean = cents.mean(axis=0)

    # Standardise the CURVATURE subspace only (6 dims) for PCA.
    curv = cents[:, CURV_IDX]
    mu = curv.mean(axis=0); sd = curv.std(axis=0) + 1e-9

    # PCA of centroids in standardised curvature space.
    X = (curv - mu) / sd
    _, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    pca_basis = Vt[:2]   # (2, 6) — in standardised-curvature space
    print(f"PCA top-2 explains {explained[:2].sum()*100:.1f}% of centroid curvature variance")

    def proj(curv_arr):
        return ((curv_arr - mu) / sd - X.mean(axis=0)) @ pca_basis.T

    p_cents = proj(curv)

    # Grid extent
    lim = max(abs(p_cents).max() * 1.8, 2.0)
    grid_ax = np.linspace(-lim, lim, GRID)
    XX, YY = np.meshgrid(grid_ax, grid_ax, indexing="ij")

    # For each grid point: lift back to 19-D θ, evaluate fitness.
    # Non-curvature dims stay at theta_mean.
    print(f"evaluating fitness on {GRID}x{GRID}={GRID*GRID} grid…")
    fit_grid = np.full((GRID, GRID), np.nan, dtype=float)
    rms_grid = np.full((GRID, GRID), np.nan, dtype=float)

    done = 0
    for i in range(GRID):
        for j in range(GRID):
            # Inverse PCA: PC1, PC2 → standardised 6-D curvatures → raw curvatures
            pc = np.array([XX[i, j], YY[i, j]])
            # point = mean_std + pc @ basis  (in standardised space)
            curv_std = X.mean(axis=0) + pc @ pca_basis
            curv_raw = curv_std * sd + mu

            # Build full 19-D θ (keep non-curvatures at minima-mean)
            theta = theta_mean.copy()
            theta[CURV_IDX] = curv_raw

            try:
                res = problem.evaluate(theta, with_grads=False)
                fit_grid[i, j] = float(res.fitness)
                rms_grid[i, j] = float(res.metrics.get("rms_spot", np.nan))
            except Exception as e:
                pass
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{GRID*GRID}  latest fit={fit_grid[i,j]:.4f}")

    print(f"fitness range on grid: {np.nanmin(fit_grid):.3f} to {np.nanmax(fit_grid):.3f}")
    print(f"RMS range on grid:     {np.nanmin(rms_grid):.4f} to {np.nanmax(rms_grid):.4f}")

    # Project CMA-MEGA offspring (top 5% by fitness)
    valid = np.isfinite(off_fits)
    q = np.quantile(off_fits[valid], 1.0 - TOP_FITNESS_FRAC)
    top_mask = valid & (off_fits >= q)
    p_offs_top = proj(offs[top_mask, :6])
    print(f"projected {top_mask.sum():,} top-5% offspring onto the plane")

    # Also project a sample of random seeds
    sub_seed_idx = np.random.default_rng(0).choice(
        len(seeds), min(800, len(seeds)), replace=False
    )
    p_seeds = proj(seeds[sub_seed_idx, :6])

    # --- plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5),
                             gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.25))

    # Clip fitness for visualisation (lots of outliers at big negative)
    fit_vis = np.clip(fit_grid,
                      np.nanpercentile(fit_grid, 2),
                      np.nanpercentile(fit_grid, 95))

    # Panel 1: fitness landscape + samples overlay
    ax = axes[0]
    cf = ax.contourf(XX, YY, fit_vis, levels=30, cmap="viridis")
    ax.contour(XX, YY, fit_vis, levels=12, colors="white",
               linewidths=0.5, alpha=0.35)
    # Seeds
    ax.scatter(p_seeds[:, 0], p_seeds[:, 1], s=3, c="orange", alpha=0.35,
               label=f"random seeds (sub {len(p_seeds)})", zorder=3)
    # Top-5% offspring
    ax.scatter(p_offs_top[:, 0], p_offs_top[:, 1], s=6, c="#8fff8f", alpha=0.45,
               label=f"CMA-MEGA top-5% (N={top_mask.sum():,})", zorder=4)
    # Known minima
    ax.scatter(p_cents[:, 0], p_cents[:, 1], s=420, c="red", marker="*",
               edgecolor="black", linewidths=1.5, zorder=10,
               label=f"known minima ({len(p_cents)})")
    for lab, pc in zip(labels, p_cents):
        ax.annotate(lab, pc, fontsize=9, color="white",
                    xytext=(0, 11), textcoords="offset points",
                    ha="center", fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.7, pad=1.5,
                              boxstyle="round"))
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax.set_title("Fitness landscape (actual evaluation on the plane)\n"
                 "with CMA-MEGA top-5% priors and random seeds overlaid",
                 fontsize=12)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    cb = plt.colorbar(cf, ax=ax); cb.set_label("fitness (higher = better)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Panel 2: RMS spot landscape (log scale)
    ax = axes[1]
    rms_vis = np.log10(np.clip(rms_grid, 1e-3, 1e3))
    cf = ax.contourf(XX, YY, rms_vis, levels=30, cmap="magma_r")
    ax.contour(XX, YY, rms_vis, levels=10, colors="white",
               linewidths=0.5, alpha=0.4)
    ax.scatter(p_offs_top[:, 0], p_offs_top[:, 1], s=6, c="#8fff8f", alpha=0.5,
               label="top-5% priors", zorder=4)
    ax.scatter(p_cents[:, 0], p_cents[:, 1], s=420, c="red", marker="*",
               edgecolor="black", linewidths=1.5, zorder=10,
               label="known minima")
    ax.set_xlabel(f"PC1"); ax.set_ylabel("PC2")
    ax.set_title("Spot-RMS landscape (log₁₀, mm)\n"
                 "darker = smaller spot = better lens",
                 fontsize=12)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    cb = plt.colorbar(cf, ax=ax); cb.set_label("log₁₀(RMS / mm)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Cooke-triplet fitness landscape on the curvature-PCA plane "
        f"({GRID}×{GRID} grid, real ray-trace evaluations)\n"
        "— red stars are known minima, green dots are top-5% CMA-MEGA raw emissions",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(OUT, "landscape_contour.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_png}")

    # Persist grids for downstream analysis
    np.savez(os.path.join(OUT, "landscape_grids.npz"),
             XX=XX, YY=YY, fit_grid=fit_grid, rms_grid=rms_grid,
             p_cents=p_cents, p_offs_top=p_offs_top, p_seeds=p_seeds,
             explained=explained)
    print(f"saved {os.path.join(OUT, 'landscape_grids.npz')}")


if __name__ == "__main__":
    main()
