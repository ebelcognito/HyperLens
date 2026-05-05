"""
3-D fitness-landscape view of the 2-D curvature-PCA plane.

Ground plane: PC1, PC2 (curvatures only, computed on the 5 reliable minima).
Z-axis: fitness achieved by CMA-MEGA at each (PC1, PC2) location.
   valleys = regions where CMA-MEGA found good priors
   flat plateau = empty / sub-optimal regions

Known minima from the prior 2-D Cooke archive are shown as red stars
anchored on the landscape — they should sit at the bottom of visible valleys.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D  # noqa

KNOWN_JSON = "/home/hisham26/Desktop/RL/paper_figures/sample_efficiency_subregion/known_minima.json"
LINEAGE = (
    "/home/hisham26/Desktop/RL/results/lensmega/"
    "cooke_triplet_seidel3d_s2s3s4_20260423_031324/lineage.jsonl"
)
OUT_DIR = "/home/hisham26/Desktop/RL/paper_figures/sample_efficiency_subregion"

CURV_IDX = list(range(6))
MIN_MEMBERS = 2
MAX_RMS_UM = 20.0
GRID = 60  # 2-D grid resolution on the PC1-PC2 plane


def load_reliable_minima():
    with open(KNOWN_JSON) as f:
        d = json.load(f)
    basins = [b for b in d["basins"]
              if b["n_members"] >= MIN_MEMBERS
              and b["best_rms_um"] <= MAX_RMS_UM]
    centroids = np.array([b["centroid_theta"] for b in basins], dtype=float)
    labels = [f"b{b['basin_id']}·{b['best_rms_um']:.1f}μm"
              for b in basins]
    return centroids, labels


def load_lineage(path):
    offs, off_fits = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["source"] == "seed":
                continue
            th = r.get("theta")
            if th is None:
                continue
            offs.append(th); off_fits.append(float(r.get("fitness", np.nan)))
    return np.array(offs), np.array(off_fits)


def main():
    cents_full, labels = load_reliable_minima()
    offs, off_fits = load_lineage(LINEAGE)
    print(f"minima: {len(cents_full)}  offspring: {len(offs):,}")

    cents = cents_full[:, CURV_IDX]
    offs_c = offs[:, CURV_IDX]

    # Standardise by std of everything combined
    all_c = np.vstack([cents, offs_c])
    mu = all_c.mean(axis=0); sd = all_c.std(axis=0) + 1e-9
    cents_s = (cents - mu) / sd
    offs_s = (offs_c - mu) / sd

    # PCA on centroids
    c_mean = cents_s.mean(axis=0)
    X = cents_s - c_mean
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    basis = Vt[:2]
    explained = (S ** 2) / (S ** 2).sum()
    print(f"PCA top-2 explains {explained[:2].sum() * 100:.1f}% of variance")

    def proj(P):
        return (P - c_mean) @ basis.T
    p_cents = proj(cents_s)
    p_offs = proj(offs_s)

    # Choose landscape extent
    all_p = np.vstack([p_cents, p_offs])
    lim = np.percentile(np.abs(all_p), 98) * 1.05
    xs = np.linspace(-lim, lim, GRID)
    ys = np.linspace(-lim, lim, GRID)
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]

    # For each 2-D grid cell, compute the best fitness of offspring landing in it.
    # Fitness is negative; "best" = max (closest to 0). Use -log(-fitness) so
    # extreme-bad samples get compressed, not swamp the landscape.
    fit_grid = np.full((GRID, GRID), np.nan, dtype=float)
    valid = np.isfinite(off_fits)
    off_fits_pos = -off_fits[valid]       # > 0; smaller = better
    # clip and log: use log1p to make variance visible at small-fitness scale
    off_fits_log = np.log1p(np.clip(off_fits_pos, 0, 1e6))

    p_valid = p_offs[valid]
    xi = ((p_valid[:, 0] + lim) / (2 * lim) * (GRID - 1)).astype(int)
    yi = ((p_valid[:, 1] + lim) / (2 * lim) * (GRID - 1)).astype(int)
    xi = np.clip(xi, 0, GRID - 1); yi = np.clip(yi, 0, GRID - 1)

    # For each cell keep the MIN log-pos-fitness (= best fitness)
    for i in range(len(p_valid)):
        v = off_fits_log[i]
        if np.isnan(fit_grid[xi[i], yi[i]]) or v < fit_grid[xi[i], yi[i]]:
            fit_grid[xi[i], yi[i]] = v

    # Fill empty cells with max value (plateau). Smooth slightly.
    cap = np.nanpercentile(fit_grid, 95)
    fit_grid_filled = np.where(np.isnan(fit_grid), cap, fit_grid)
    fit_grid_smooth = gaussian_filter(fit_grid_filled, sigma=1.0)

    # For Z-axis we want valleys DOWN, so Z = fit_grid (already "higher = worse")
    # Actually let's invert: Z = cap - fit_grid, so valleys become DEEP.
    Z = cap - fit_grid_smooth   # 0 at plateau; deeper (more negative? no, higher) at valleys
    # Ugh — Z=cap−smooth: small smooth → Z big (peak); large smooth → Z≈0 (plateau).
    # That makes valleys appear as PEAKS, wrong direction.
    # Flip: Z = -(cap - smooth) = smooth - cap.  Now peaks become valleys.
    Z = fit_grid_smooth - cap   # <= 0; more negative = better, flatter = plateau

    # Mesh for surface
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    # --- plot ---
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.12,
                          left=0.03, right=0.98, top=0.93, bottom=0.07)

    ax3d = fig.add_subplot(gs[0], projection="3d")
    # Surface coloured by fitness (deeper = better)
    surf = ax3d.plot_surface(
        XX, YY, Z,
        cmap="viridis_r",     # dark = good (valleys), light = plateau
        linewidth=0, antialiased=True, alpha=0.95,
        vmin=Z.min(), vmax=0,
    )

    # Anchor each minimum to the surface height at that (PC1, PC2)
    def _height_at(px, py):
        i = int(np.clip((px + lim) / (2 * lim) * (GRID - 1), 0, GRID - 1))
        j = int(np.clip((py + lim) / (2 * lim) * (GRID - 1), 0, GRID - 1))
        return Z[i, j]

    z_stars = np.array([_height_at(p[0], p[1]) for p in p_cents])
    ax3d.scatter(
        p_cents[:, 0], p_cents[:, 1], z_stars - 0.2,
        s=600, c="red", marker="*", edgecolor="black", linewidths=1.5,
        zorder=100, label="known minima",
    )
    # Stems dropping to surface so stars look anchored
    for pc, zs in zip(p_cents, z_stars):
        ax3d.plot([pc[0], pc[0]], [pc[1], pc[1]], [zs + 0.05, zs - 0.2],
                  color="red", lw=1.2)
    for lab, pc, zs in zip(labels, p_cents, z_stars):
        ax3d.text(pc[0], pc[1], zs - 0.6, lab, fontsize=9, color="darkred",
                  ha="center", va="top", fontweight="bold")

    ax3d.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax3d.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax3d.set_zlabel("best fitness here\n(lower = CMA-MEGA found good design)")
    ax3d.set_title(
        "Fitness landscape carved by CMA-MEGA+Seidel\n"
        "valleys = regions with good priors · flat = unexplored / bad",
        fontsize=11,
    )
    ax3d.view_init(elev=35, azim=-55)
    ax3d.legend(loc="upper right", fontsize=9)
    fig.colorbar(surf, ax=ax3d, shrink=0.55, label="fitness achieved (deeper = better)")

    # 2-D top-down view (contour + known-minima stars)
    ax2d = fig.add_subplot(gs[1])
    cm_img = ax2d.contourf(XX, YY, Z, levels=20, cmap="viridis_r")
    ax2d.contour(XX, YY, Z, levels=10, colors="white", linewidths=0.4, alpha=0.35)
    ax2d.scatter(p_cents[:, 0], p_cents[:, 1], s=420, c="red", marker="*",
                 edgecolor="black", linewidths=1.2, zorder=5)
    for lab, pc in zip(labels, p_cents):
        ax2d.annotate(lab, pc, fontsize=9, color="white",
                      ha="center", va="bottom",
                      xytext=(0, 10), textcoords="offset points",
                      fontweight="bold",
                      bbox=dict(facecolor="black", alpha=0.6, pad=1.5, boxstyle="round"))
    ax2d.set_xlabel(f"PC1"); ax2d.set_ylabel("PC2")
    ax2d.set_title("Top-down: landscape contour with known minima", fontsize=11)
    ax2d.set_xlim(-lim, lim); ax2d.set_ylim(-lim, lim)
    ax2d.set_aspect("equal")
    fig.colorbar(cm_img, ax=ax2d, label="fitness achieved")

    fig.suptitle(
        "CMA-MEGA+Seidel fitness landscape on the curvature-PCA plane\n"
        "Valleys carved by the emitter — red stars are the 5 known minima "
        "from the prior Cooke run",
        fontsize=13, fontweight="bold",
    )
    out_png = os.path.join(OUT_DIR, "cma_mega_landscape_3d.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
