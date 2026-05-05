"""
Clean hero figure:

Plane defined by 3 best-RMS refined elites (all at residual = 0 by construction).
RMS landscape evaluated on a grid over the plane.
Random seeds and CMA-MEGA offspring plotted as scatter points at equal N = 3,000.

No ratio panel (noisy at sparse bin counts). Two clean panels only.
"""
import glob
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


CFG = ("/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
       "config/examples/cooke_triplet_seidel3d_s2s3s4.yaml")
RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "cooke_triplet_seidel3d_s2s3s4_20260423_031324")
OUT = "/home/hisham26/Desktop/RL/paper_figures/landscape_plane_elites"
os.makedirs(OUT, exist_ok=True)

GRID = 60
TARGET_N = 3000
RESID_KEEP_FRAC = 0.50       # show only the half of samples closest to the plane


def load_refined_elites():
    paths = sorted(glob.glob(os.path.join(RUN_DIR, "elite_designs", "*.json")))
    out = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        ref = d.get("refined") or {}
        if "theta" not in ref or "rms_mm" not in ref:
            continue
        out.append({
            "name": os.path.basename(p)[:-5],
            "theta": np.array(ref["theta"], dtype=float),
            "rms_um": float(ref["rms_mm"]) * 1000.0,
        })
    return out


def load_lineage():
    seeds, offs = [], []
    with open(os.path.join(RUN_DIR, "lineage.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            th = r.get("theta")
            if th is None:
                continue
            (seeds if r["source"] == "seed" else offs).append(th)
    return np.array(seeds), np.array(offs)


def main():
    elites = load_refined_elites()
    print(f"refined elites: {len(elites)}")
    for e in elites:
        print(f"  {e['name']:<28} RMS={e['rms_um']:.2f} μm")

    # Use the three BEST-RMS elites as plane anchors.
    elites_sorted = sorted(elites, key=lambda e: e["rms_um"])
    anchors = elites_sorted[:3]
    print("\nplane anchors (3 best-RMS refined elites):")
    for i, a in enumerate(anchors):
        print(f"  anchor {i}: {a['name']} · RMS = {a['rms_um']:.2f} μm")

    p0, p1, p2 = anchors[0]["theta"], anchors[1]["theta"], anchors[2]["theta"]
    origin = p0
    v1 = p1 - p0
    e1 = v1 / np.linalg.norm(v1)
    v2 = p2 - p0
    v2_perp = v2 - (v2 @ e1) * e1
    e2 = v2_perp / np.linalg.norm(v2_perp)

    def proj_1(theta):
        d = theta - origin
        a = float(d @ e1); b = float(d @ e2)
        resid = float(np.linalg.norm(d - a * e1 - b * e2))
        return a, b, resid

    def proj_many(theta_arr):
        d = theta_arr - origin
        a = d @ e1; b = d @ e2
        resid = np.linalg.norm(d - a[:, None] * e1 - b[:, None] * e2, axis=1)
        return a, b, resid

    anchor_xy = np.array([proj_1(a["theta"])[:2] for a in anchors])
    print(f"anchor plane coords: {anchor_xy.tolist()}")

    # Plane extent — tight around anchors with modest margin
    a_min, a_max = anchor_xy[:, 0].min(), anchor_xy[:, 0].max()
    b_min, b_max = anchor_xy[:, 1].min(), anchor_xy[:, 1].max()
    span_a = max(a_max - a_min, 1.0)
    span_b = max(b_max - b_min, 1.0)
    pad = 0.8 * max(span_a, span_b)
    a_lo, a_hi = a_min - pad, a_max + pad
    b_lo, b_hi = b_min - pad, b_max + pad
    print(f"plane extent: a ∈ [{a_lo:.2f}, {a_hi:.2f}]  b ∈ [{b_lo:.2f}, {b_hi:.2f}]")

    # Evaluate fitness on grid
    cfg = load_config(CFG)
    problem, _, _, _, _ = build_all(cfg)
    problem._ensure_backend()

    grid_a = np.linspace(a_lo, a_hi, GRID)
    grid_b = np.linspace(b_lo, b_hi, GRID)
    AA, BB = np.meshgrid(grid_a, grid_b, indexing="ij")

    print(f"evaluating on {GRID}×{GRID} grid...")
    rms_grid = np.full((GRID, GRID), np.nan, dtype=float)
    done = 0
    for i in range(GRID):
        for j in range(GRID):
            theta = origin + AA[i, j] * e1 + BB[i, j] * e2
            try:
                res = problem.evaluate(theta, with_grads=False)
                rms_grid[i, j] = float(res.metrics.get("rms_spot", np.nan))
            except Exception:
                pass
            done += 1
            if done % 600 == 0:
                print(f"  {done}/{GRID*GRID}")
    print(f"rms range [{np.nanmin(rms_grid):.4f}, {np.nanmax(rms_grid):.4f}]")

    # Project samples, keep only the ones closest to the plane
    seeds, offs = load_lineage()
    rng = np.random.default_rng(0)
    seeds_N = seeds[rng.choice(len(seeds), TARGET_N, replace=False)]
    offs_N = offs[rng.choice(len(offs), TARGET_N, replace=False)]
    sa, sb, sr = proj_many(seeds_N)
    oa, ob, or_ = proj_many(offs_N)

    # Only show samples close to the plane
    s_close = sr < np.percentile(sr, RESID_KEEP_FRAC * 100)
    o_close = or_ < np.percentile(or_, RESID_KEEP_FRAC * 100)
    # Also restrict to samples inside the displayed extent
    def in_extent(a, b):
        return (a >= a_lo) & (a <= a_hi) & (b >= b_lo) & (b <= b_hi)
    s_mask = s_close & in_extent(sa, sb)
    o_mask = o_close & in_extent(oa, ob)
    print(f"displayed seeds: {s_mask.sum():,} / {TARGET_N}")
    print(f"displayed offspring: {o_mask.sum():,} / {TARGET_N}")

    # Project ALL refined elites to show where the others live
    elite_thetas = np.array([e["theta"] for e in elites])
    ea, eb, er = proj_many(elite_thetas)

    # --- figure: two panels, seeds left, CMA-MEGA right ---
    fig, axes = plt.subplots(1, 2, figsize=(17, 8),
                             gridspec_kw=dict(wspace=0.18))

    rms_vis = np.log10(np.clip(rms_grid, 1e-4, 1e3))
    vmin_r = np.nanpercentile(rms_vis, 2)
    vmax_r = np.nanpercentile(rms_vis, 98)

    def draw_landscape(ax):
        cf = ax.contourf(AA, BB, rms_vis, levels=30, cmap="magma_r",
                         vmin=vmin_r, vmax=vmax_r)
        ax.contour(AA, BB, rms_vis, levels=10, colors="white",
                   linewidths=0.4, alpha=0.35)
        return cf

    # panel A: random seeds
    ax = axes[0]
    cf = draw_landscape(ax)
    ax.scatter(sa[s_mask], sb[s_mask], s=10, c="orange", alpha=0.55,
               edgecolor="darkorange", linewidths=0.2, zorder=3,
               label=f"random seeds ({s_mask.sum():,})")
    # anchors
    for (a, b), anc in zip(anchor_xy, anchors):
        ax.scatter(a, b, s=560, c="red", marker="*",
                   edgecolor="black", linewidths=1.8, zorder=20)
    ax.set_xlabel("plane coord a", fontsize=11)
    ax.set_ylabel("plane coord b", fontsize=11)
    ax.set_title(f"A. Random seeds (N = {TARGET_N:,})\n"
                 f"closest {int(RESID_KEEP_FRAC * 100)}% to the plane shown",
                 fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(a_lo, a_hi); ax.set_ylim(b_lo, b_hi)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)

    # panel B: CMA-MEGA offspring
    ax = axes[1]
    cf = draw_landscape(ax)
    ax.scatter(oa[o_mask], ob[o_mask], s=10, c="#33dd33", alpha=0.6,
               edgecolor="darkgreen", linewidths=0.2, zorder=3,
               label=f"CMA-MEGA priors ({o_mask.sum():,})")
    for (a, b), anc in zip(anchor_xy, anchors):
        ax.scatter(a, b, s=560, c="red", marker="*",
                   edgecolor="black", linewidths=1.8, zorder=20)
    ax.set_xlabel("plane coord a", fontsize=11)
    ax.set_ylabel("plane coord b", fontsize=11)
    ax.set_title(f"B. CMA-MEGA offspring (N = {TARGET_N:,} subsample)\n"
                 f"closest {int(RESID_KEEP_FRAC * 100)}% to the plane shown",
                 fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(a_lo, a_hi); ax.set_ylim(b_lo, b_hi)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)

    # Shared colorbar
    cb = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.80, pad=0.015,
                      label="log₁₀(spot RMS / mm) · dark = better lens")

    fig.suptitle(
        f"Cooke design landscape on the plane through the 3 best-RMS refined elites "
        f"({anchors[0]['rms_um']:.2f} / {anchors[1]['rms_um']:.2f} / "
        f"{anchors[2]['rms_um']:.2f} μm) — matched N = {TARGET_N:,}",
        fontsize=13, fontweight="bold",
    )
    out_png = os.path.join(OUT, "landscape_plane_elites.png")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
