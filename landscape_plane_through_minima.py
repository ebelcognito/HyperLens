"""
Plane-through-three-minima fitness landscape, LDG-EA Fig 1 style.

We choose 3 specific known minima (the best 3 by refined RMS), compute the
affine 2-D plane that passes through all three, and evaluate the actual
fitness function on a grid over that plane.  The 3 chosen minima
therefore sit at exact, known coordinates on the landscape by construction.
Other known minima are also projected onto the plane (with residual so we
can see how far off-plane they live).

Top-5% CMA-MEGA offspring are projected onto the same plane and shown as
dots. Random seeds shown for reference.
"""
import json
import os
import sys
import numpy as np
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
OUT = "/home/hisham26/Desktop/RL/paper_figures/landscape_plane3"
os.makedirs(OUT, exist_ok=True)

MIN_MEMBERS = 2
MAX_RMS_UM = 20.0
GRID = 60               # 60x60 = 3,600 fitness evals
TOP_FITNESS_FRAC = 0.05

# IDs of the 3 minima to define the plane. Picked: best-RMS basin 7, mid
# basin 10, and mid-high basin 9. All multi-member.
ANCHOR_BASIN_IDS = [7, 10, 9]


def load_all_minima():
    with open(KNOWN_JSON) as f:
        d = json.load(f)
    all_b = [b for b in d["basins"]
             if b["n_members"] >= MIN_MEMBERS
             and b["best_rms_um"] <= MAX_RMS_UM]
    return all_b


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

    all_basins = load_all_minima()
    anchors = [b for b in all_basins if b["basin_id"] in ANCHOR_BASIN_IDS]
    assert len(anchors) == 3, f"expected 3 anchors, got {len(anchors)}: {[b['basin_id'] for b in anchors]}"
    p0 = np.array(anchors[0]["centroid_theta"], dtype=float)
    p1 = np.array(anchors[1]["centroid_theta"], dtype=float)
    p2 = np.array(anchors[2]["centroid_theta"], dtype=float)

    # Define an orthonormal basis for the plane through p0, p1, p2.
    #  origin = p0
    #  e1     = (p1 - p0) / ||p1 - p0||
    #  e2     = orthogonal component of (p2 - p0), normalised
    origin = p0
    v1 = p1 - p0
    e1 = v1 / np.linalg.norm(v1)
    v2 = p2 - p0
    v2_perp = v2 - (v2 @ e1) * e1
    e2 = v2_perp / np.linalg.norm(v2_perp)
    print(f"anchor basins: {[b['basin_id'] for b in anchors]}")
    print(f"anchor RMS (μm): {[b['best_rms_um'] for b in anchors]}")
    print(f"||p1-p0|| = {np.linalg.norm(v1):.3f}, ||p2-p0|| = {np.linalg.norm(v2):.3f}")
    print(f"angle(p1-p0, p2-p0) = {np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1))):.1f}°")

    # Coordinates of each anchor on this plane (a = e1-projection, b = e2-projection)
    def project_to_plane(theta):
        d = theta - origin
        a = d @ e1
        b = d @ e2
        resid = d - a * e1 - b * e2
        return a, b, np.linalg.norm(resid)

    ancor_coords = [project_to_plane(np.array(b["centroid_theta"])) for b in anchors]
    print(f"anchor plane-coords: {[(f'{a:.3f}', f'{bv:.3f}', f'resid={r:.2e}') for a, bv, r in ancor_coords]}")

    # Plane extents: give ~50% margin around anchor positions
    a_vals = [c[0] for c in ancor_coords]
    b_vals = [c[1] for c in ancor_coords]
    a_lo = min(a_vals) - 0.5 * (max(a_vals) - min(a_vals) + 1e-6)
    a_hi = max(a_vals) + 0.5 * (max(a_vals) - min(a_vals) + 1e-6)
    b_lo = min(b_vals) - 0.5 * (max(b_vals) - min(b_vals) + 1e-6)
    b_hi = max(b_vals) + 0.5 * (max(b_vals) - min(b_vals) + 1e-6)
    # Make symmetric enough
    pad = max(abs(a_hi - a_lo), abs(b_hi - b_lo)) * 0.2
    a_lo -= pad; a_hi += pad; b_lo -= pad; b_hi += pad

    grid_a = np.linspace(a_lo, a_hi, GRID)
    grid_b = np.linspace(b_lo, b_hi, GRID)
    AA, BB = np.meshgrid(grid_a, grid_b, indexing="ij")

    print(f"evaluating fitness on {GRID}×{GRID}={GRID*GRID} grid...")
    fit_grid = np.full((GRID, GRID), np.nan, dtype=float)
    rms_grid = np.full((GRID, GRID), np.nan, dtype=float)

    done = 0
    for i in range(GRID):
        for j in range(GRID):
            theta = origin + AA[i, j] * e1 + BB[i, j] * e2
            try:
                res = problem.evaluate(theta, with_grads=False)
                fit_grid[i, j] = float(res.fitness)
                rms_grid[i, j] = float(res.metrics.get("rms_spot", np.nan))
            except Exception:
                pass
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{GRID*GRID}  latest fit={fit_grid[i,j]:.3f}, rms={rms_grid[i,j]:.4f}")

    print(f"fit range: [{np.nanmin(fit_grid):.3f}, {np.nanmax(fit_grid):.3f}]")
    print(f"rms range: [{np.nanmin(rms_grid):.4f}, {np.nanmax(rms_grid):.4f}]")

    # Project ALL known minima onto the plane (anchors sit at residual=0, others have residuals)
    other_basins = [b for b in all_basins if b["basin_id"] not in ANCHOR_BASIN_IDS]
    other_coords = [(project_to_plane(np.array(b["centroid_theta"])), b) for b in other_basins]

    # Project CMA-MEGA offspring (top 5%)
    seeds, offs, off_fits = load_lineage()
    valid = np.isfinite(off_fits)
    q = np.quantile(off_fits[valid], 1.0 - TOP_FITNESS_FRAC)
    top_mask = valid & (off_fits >= q)
    def proj_arr(theta_arr):
        diffs = theta_arr - origin
        a = diffs @ e1
        b = diffs @ e2
        resid = np.linalg.norm(diffs - a[:, None] * e1 - b[:, None] * e2, axis=1)
        return a, b, resid
    off_a, off_b, off_r = proj_arr(offs[top_mask])
    seed_sub_idx = np.random.default_rng(0).choice(
        len(seeds), min(800, len(seeds)), replace=False
    )
    seed_a, seed_b, seed_r = proj_arr(seeds[seed_sub_idx])

    # Compute max residual for "in-plane" filtering
    in_plane_threshold = 3.0    # std units - heuristic
    # Actually use relative threshold: median off-plane distance of the anchor basins
    median_off_r = np.median(off_r) if len(off_r) else 0.0
    # Show only samples close to the plane (within some factor)
    close_mask_off = off_r < np.percentile(off_r, 70)
    close_mask_seed = seed_r < np.percentile(seed_r, 70)

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5),
                             gridspec_kw=dict(width_ratios=[1, 1], wspace=0.22))

    # ----- left: fitness landscape (clipped to readable band) -----
    ax = axes[0]
    # Clip fitness tightly: huge negatives come from invalid-design penalties
    # and wash out every useful contrast. Anchor minima have fitness ~-0.0 to -0.02;
    # the "good" neighbourhood is bounded in [-5, 0]. Outside that window, the
    # design is essentially broken and we render it as "off-landscape".
    fit_clipped = np.clip(fit_grid, -5, 0)
    cf = ax.contourf(AA, BB, fit_clipped, levels=40, cmap="viridis",
                     vmin=-5, vmax=0)
    ax.contour(AA, BB, fit_clipped, levels=12, colors="white",
               linewidths=0.4, alpha=0.35)

    # Samples — plot CMA-MEGA top-5% only, with tighter in-plane filter
    close_off = off_r < np.percentile(off_r, 50)
    ax.scatter(off_a[close_off], off_b[close_off],
               s=16, c="#79ff79", alpha=0.75, zorder=3, edgecolor="darkgreen",
               linewidths=0.3,
               label=f"CMA-MEGA top-5% (closest to plane, n={close_off.sum()})")

    for (a, b, r), ab in zip(ancor_coords, anchors):
        ax.scatter(a, b, s=620, c="red", marker="*",
                   edgecolor="black", linewidths=2.0, zorder=10)
        ax.annotate(f"basin {ab['basin_id']}\n{ab['best_rms_um']:.1f} μm", (a, b),
                    fontsize=11, color="white", ha="center",
                    xytext=(0, 18), textcoords="offset points",
                    fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.85, pad=3, boxstyle="round"))

    ax.set_xlabel("plane coordinate a (basin 7 → basin 10 axis)", fontsize=11)
    ax.set_ylabel("plane coordinate b (⊥ toward basin 9)", fontsize=11)
    ax.set_title("Fitness landscape on the plane through 3 minima\n"
                 "(clipped to [-5, 0] — purple = broken / invalid design)",
                 fontsize=12)
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, label="fitness (higher = better, clipped)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    # ----- right: RMS landscape (log) -----
    ax = axes[1]
    rms_vis = np.log10(np.clip(rms_grid, 1e-4, 1e3))
    cf = ax.contourf(AA, BB, rms_vis, levels=40, cmap="magma_r",
                     vmin=np.nanpercentile(rms_vis, 2),
                     vmax=np.nanpercentile(rms_vis, 98))
    ax.contour(AA, BB, rms_vis, levels=10, colors="white",
               linewidths=0.4, alpha=0.35)

    ax.scatter(off_a[close_off], off_b[close_off],
               s=16, c="#79ff79", alpha=0.75, zorder=3, edgecolor="darkgreen",
               linewidths=0.3,
               label="CMA-MEGA top-5% priors")

    for (a, b, r), ab in zip(ancor_coords, anchors):
        ax.scatter(a, b, s=620, c="red", marker="*",
                   edgecolor="black", linewidths=2.0, zorder=10)
        ax.annotate(f"basin {ab['basin_id']}\n{ab['best_rms_um']:.1f} μm", (a, b),
                    fontsize=11, color="white", ha="center",
                    xytext=(0, 18), textcoords="offset points",
                    fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.85, pad=3, boxstyle="round"))

    ax.set_xlabel("plane coordinate a", fontsize=11)
    ax.set_ylabel("plane coordinate b", fontsize=11)
    ax.set_title("Spot-RMS landscape (log₁₀, mm)\n"
                 "dark = small spot = good lens · valleys connect the minima",
                 fontsize=12)
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, label="log₁₀(RMS / mm)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    fig.suptitle(
        "Cooke-triplet design landscape on the plane through 3 known minima\n"
        "— green dots are raw CMA-MEGA priors (no refinement) — they pool into the valleys",
        fontsize=13, fontweight="bold",
    )
    out_png = os.path.join(OUT, "landscape_plane3.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_png}")

    np.savez(os.path.join(OUT, "landscape_plane3_grids.npz"),
             AA=AA, BB=BB, fit_grid=fit_grid, rms_grid=rms_grid,
             origin=origin, e1=e1, e2=e2,
             anchor_coords=np.array([[a, b] for a, b, _ in ancor_coords]))
    print(f"saved grids")


if __name__ == "__main__":
    main()
