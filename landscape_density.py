"""
Landscape with density heatmaps of CMA-MEGA offspring samples (no fitness cherry-pick).

Reuses the plane-through-3-minima grid computed by landscape_plane_through_minima.py,
then overlays three sample-density heatmaps on the RMS landscape:

  1. ALL offspring samples — unbiased full picture.
  2. Offspring that were INSERTED into the archive (new cell OR replaced elite).
  3. Offspring that REPLACED an existing elite (strict fitness improvement).

None of these use fitness-based filtering; (2) and (3) use the runner's own
archive-insertion decisions.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KNOWN_JSON = (
    "/home/hisham26/Desktop/RL/paper_figures/"
    "sample_efficiency_subregion/known_minima.json"
)
LINEAGE = (
    "/home/hisham26/Desktop/RL/results/lensmega/"
    "cooke_triplet_seidel3d_s2s3s4_20260423_031324/lineage.jsonl"
)
GRIDS_NPZ = (
    "/home/hisham26/Desktop/RL/paper_figures/landscape_plane3/"
    "landscape_plane3_grids.npz"
)
OUT = "/home/hisham26/Desktop/RL/paper_figures/landscape_plane3"
os.makedirs(OUT, exist_ok=True)

MIN_MEMBERS = 2
MAX_RMS_UM = 20.0
ANCHOR_BASIN_IDS = [7, 10, 9]


def load_minima():
    with open(KNOWN_JSON) as f:
        d = json.load(f)
    all_b = [b for b in d["basins"]
             if b["n_members"] >= MIN_MEMBERS
             and b["best_rms_um"] <= MAX_RMS_UM]
    anchors = [b for b in all_b if b["basin_id"] in ANCHOR_BASIN_IDS]
    return anchors


def load_lineage_tagged():
    """Return four arrays: all_theta, all_fitness, all_insert, all_source (seed/offspring)."""
    rows = []
    with open(LINEAGE) as f:
        for line in f:
            r = json.loads(line)
            th = r.get("theta")
            if th is None:
                continue
            rows.append({
                "theta": th,
                "fitness": float(r.get("fitness", np.nan)),
                "insert": int(r.get("insert_status", 0)),
                "source": r["source"],
            })
    return rows


def project_arr(theta_arr, origin, e1, e2):
    d = theta_arr - origin
    a = d @ e1
    b = d @ e2
    resid = np.linalg.norm(d - a[:, None] * e1 - b[:, None] * e2, axis=1)
    return a, b, resid


def density_hexbin(ax, x, y, extent, cmap, label):
    hb = ax.hexbin(x, y, gridsize=40, cmap=cmap, mincnt=1,
                   extent=extent)
    cb = plt.colorbar(hb, ax=ax, shrink=0.9)
    cb.set_label(f"# samples ({label})")


def main():
    anchors = load_minima()
    print(f"anchor basins: {[b['basin_id'] for b in anchors]}")

    grids = np.load(GRIDS_NPZ)
    AA = grids["AA"]; BB = grids["BB"]
    rms_grid = grids["rms_grid"]
    origin = grids["origin"]
    e1 = grids["e1"]; e2 = grids["e2"]
    anchor_xy = grids["anchor_coords"]

    rows = load_lineage_tagged()
    theta = np.array([r["theta"] for r in rows])
    insert = np.array([r["insert"] for r in rows])
    source = np.array([r["source"] for r in rows])
    print(f"total lineage rows: {len(rows):,}")

    mask_seed = source == "seed"
    mask_off = source == "offspring"
    mask_inserted = mask_off & (insert > 0)
    mask_replaced = mask_off & (insert == 2)
    print(f"  seed: {mask_seed.sum():,}")
    print(f"  offspring (all): {mask_off.sum():,}")
    print(f"  offspring inserted into archive: {mask_inserted.sum():,}  "
          f"({100*mask_inserted.sum()/max(mask_off.sum(),1):.1f}%)")
    print(f"  offspring that replaced an elite: {mask_replaced.sum():,}  "
          f"({100*mask_replaced.sum()/max(mask_off.sum(),1):.1f}%)")

    # Project all theta onto plane
    a_all, b_all, r_all = project_arr(theta, origin, e1, e2)

    # in-plane filter: keep bottom 50% residual
    close_thresh = np.percentile(r_all, 50)
    in_plane = r_all < close_thresh

    # Grid extent for the overlay
    a_lo = AA.min(); a_hi = AA.max()
    b_lo = BB.min(); b_hi = BB.max()
    extent = (a_lo, a_hi, b_lo, b_hi)

    # Bin densities on a shared grid (no in-plane filter now — we want the
    # full projection for both seeds and offspring so the projection-geometry
    # bias cancels in the RATIO map).
    nbin = 50
    a_edges = np.linspace(a_lo, a_hi, nbin + 1)
    b_edges = np.linspace(b_lo, b_hi, nbin + 1)

    def density(mask):
        if mask.sum() == 0:
            return np.zeros((nbin, nbin))
        H, _, _ = np.histogram2d(a_all[mask], b_all[mask],
                                 bins=[a_edges, b_edges])
        return H

    dens_seed = density(mask_seed)
    dens_off = density(mask_off)
    dens_ins = density(mask_inserted)
    dens_rep = density(mask_replaced)

    # Normalise to relative density (each sums to 1) so ratio cancels count.
    def norm(H):
        s = H.sum()
        return H / s if s > 0 else H
    n_seed = norm(dens_seed)
    n_off = norm(dens_off)
    n_ins = norm(dens_ins)
    n_rep = norm(dens_rep)

    # Ratio with small additive prior so zero-seed bins don't blow up.
    prior = (n_seed[n_seed > 0].min() * 0.5) if (n_seed > 0).any() else 1e-6
    ratio_off = (n_off + prior) / (n_seed + prior)
    ratio_ins = (n_ins + prior) / (n_seed + prior)
    ratio_rep = (n_rep + prior) / (n_seed + prior)

    # For plotting: use log-ratio centred at 0
    log_ratio_off = np.log10(np.clip(ratio_off, 1e-2, 1e2))
    log_ratio_ins = np.log10(np.clip(ratio_ins, 1e-2, 1e2))
    log_ratio_rep = np.log10(np.clip(ratio_rep, 1e-2, 1e2))

    a_mid = 0.5 * (a_edges[:-1] + a_edges[1:])
    b_mid = 0.5 * (b_edges[:-1] + b_edges[1:])
    AA_bin, BB_bin = np.meshgrid(a_mid, b_mid, indexing="ij")

    # ---- figure: 4 panels (A: raw seed & offspring density side by side,
    # C/D: ratio maps for ALL offspring and inserted/replaced)
    fig, axes = plt.subplots(2, 2, figsize=(17, 14),
                             gridspec_kw=dict(hspace=0.32, wspace=0.25))

    rms_vis = np.log10(np.clip(rms_grid, 1e-4, 1e3))
    vmin = np.nanpercentile(rms_vis, 2); vmax = np.nanpercentile(rms_vis, 98)

    # Panel A: raw random-seed projected density (shows projection bias)
    ax = axes[0, 0]
    ax.contourf(AA, BB, rms_vis, levels=30, cmap="magma_r",
                vmin=vmin, vmax=vmax, alpha=0.55)
    ax.pcolormesh(AA_bin, BB_bin, dens_seed, cmap="Oranges",
                  shading="nearest", alpha=0.75)
    for (a, b), _ in zip(anchor_xy, anchors):
        ax.scatter(a, b, s=420, c="red", marker="*",
                   edgecolor="black", linewidths=1.6, zorder=20)
    ax.set_xlabel("plane coord a"); ax.set_ylabel("plane coord b")
    ax.set_title(f"A. Random-seed density projected onto plane\n"
                 f"(N={mask_seed.sum():,}, concentration is projection-geometry bias,\n"
                 f"NOT a method signal — this is the null)",
                 fontsize=11)
    ax.set_xlim(a_lo, a_hi); ax.set_ylim(b_lo, b_hi); ax.set_aspect("equal")

    # Panel B: raw CMA-MEGA offspring density
    ax = axes[0, 1]
    ax.contourf(AA, BB, rms_vis, levels=30, cmap="magma_r",
                vmin=vmin, vmax=vmax, alpha=0.55)
    ax.pcolormesh(AA_bin, BB_bin, dens_off, cmap="Greens",
                  shading="nearest", alpha=0.75)
    for (a, b), _ in zip(anchor_xy, anchors):
        ax.scatter(a, b, s=420, c="red", marker="*",
                   edgecolor="black", linewidths=1.6, zorder=20)
    ax.set_xlabel("plane coord a"); ax.set_ylabel("plane coord b")
    ax.set_title(f"B. CMA-MEGA offspring density (raw)\n"
                 f"(N={mask_off.sum():,}, contains projection bias + method signal)",
                 fontsize=11)
    ax.set_xlim(a_lo, a_hi); ax.set_ylim(b_lo, b_hi); ax.set_aspect("equal")

    # Panel C: ratio — CMA-MEGA offspring / random seed (log scale)
    ax = axes[1, 0]
    ax.contourf(AA, BB, rms_vis, levels=30, cmap="magma_r",
                vmin=vmin, vmax=vmax, alpha=0.3)
    pc = ax.pcolormesh(AA_bin, BB_bin, log_ratio_off,
                       cmap="PuOr_r", shading="nearest",
                       vmin=-1.0, vmax=1.0, alpha=0.85)
    for (a, b), _ in zip(anchor_xy, anchors):
        ax.scatter(a, b, s=420, c="red", marker="*",
                   edgecolor="black", linewidths=1.6, zorder=20)
    cb = plt.colorbar(pc, ax=ax, shrink=0.85)
    cb.set_label("log₁₀(CMA-MEGA density / seed density)")
    ax.set_xlabel("plane coord a"); ax.set_ylabel("plane coord b")
    ax.set_title("C. Density RATIO: all offspring / seed\n"
                 "orange = over-sampled, purple = under-sampled\n"
                 "projection bias cancels in this ratio",
                 fontsize=11)
    ax.set_xlim(a_lo, a_hi); ax.set_ylim(b_lo, b_hi); ax.set_aspect("equal")

    # Panel D: ratio — replaced-elite offspring / seed
    ax = axes[1, 1]
    ax.contourf(AA, BB, rms_vis, levels=30, cmap="magma_r",
                vmin=vmin, vmax=vmax, alpha=0.3)
    pc = ax.pcolormesh(AA_bin, BB_bin, log_ratio_rep,
                       cmap="PuOr_r", shading="nearest",
                       vmin=-1.0, vmax=1.0, alpha=0.85)
    for (a, b), _ in zip(anchor_xy, anchors):
        ax.scatter(a, b, s=420, c="red", marker="*",
                   edgecolor="black", linewidths=1.6, zorder=20)
    cb = plt.colorbar(pc, ax=ax, shrink=0.85)
    cb.set_label("log₁₀(replaced-elite density / seed density)")
    ax.set_xlabel("plane coord a"); ax.set_ylabel("plane coord b")
    ax.set_title(f"D. Density RATIO: elite-replacing offspring / seed\n"
                 f"(N={mask_replaced.sum():,}, strict fitness improvements)\n"
                 "orange = CMA-MEGA concentrated here",
                 fontsize=11)
    ax.set_xlim(a_lo, a_hi); ax.set_ylim(b_lo, b_hi); ax.set_aspect("equal")

    fig.suptitle(
        "CMA-MEGA sampling vs random-seed null — density ratio on the plane through 3 minima\n"
        "Background = log₁₀(RMS/mm) · ratio maps remove projection-geometry bias",
        fontsize=13, fontweight="bold",
    )
    out_png = os.path.join(OUT, "landscape_density.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
