"""
Hero composite v2: plane through the 3 STRUCTURALLY DISTINCT family anchors
(not farthest-point-by-θ). All 3 family stars sit at residual = 0 by
construction — landscape positions match physical reality.
"""
import glob
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import ConnectionPatch

sys.path.insert(0, "/home/hisham26/Desktop/RL")
from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.run import build_all


CFG = ("/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
       "config/examples/sixlet_discovery.yaml")
GALLERY_JSON = ("/home/hisham26/Desktop/RL/paper_figures/sixlet_design_gallery/"
                "design_family_summary.json")
RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "sixlet_discovery_20260423_132001")
RENDERS_DIR = "/home/hisham26/Desktop/RL/paper_figures/sixlet_top3_imaging"
OUT = "/home/hisham26/Desktop/RL/paper_figures/sixlet_hero_composite"
os.makedirs(OUT, exist_ok=True)

GRID = 60


def load_basins():
    with open(GALLERY_JSON) as f:
        d = json.load(f)
    return d["basins"]


def pick_top3_distinct(basins):
    by_pattern = {}
    for b in basins:
        sp = b["sign_pattern"]
        if sp not in by_pattern or b["rms_um_refined"] < by_pattern[sp]["rms_um_refined"]:
            by_pattern[sp] = b
    return sorted(by_pattern.values(), key=lambda b: b["rms_um_refined"])[:3]


def load_full_theta(rep_name):
    path = os.path.join(RUN_DIR, "elite_designs", f"{rep_name}.json")
    with open(path) as f:
        d = json.load(f)
    return np.array(d["refined"]["theta"], dtype=float)


def main():
    all_basins = load_basins()
    top3 = pick_top3_distinct(all_basins)
    family_colors = ["#E45756", "#4C78A8", "#54A24B"]
    print("Family anchors (= plane anchors):")
    for i, b in enumerate(top3):
        print(f"  Family {i+1}  basin {b['id']}  {b['rep_name']}  "
              f"RMS {b['rms_um_refined']:.2f}μm")

    # Load full thetas for the 3 anchors and define plane through them
    p0 = load_full_theta(top3[0]["rep_name"])
    p1 = load_full_theta(top3[1]["rep_name"])
    p2 = load_full_theta(top3[2]["rep_name"])

    origin = p0
    v1 = p1 - p0
    e1 = v1 / np.linalg.norm(v1)
    v2 = p2 - p0
    v2_perp = v2 - (v2 @ e1) * e1
    e2 = v2_perp / np.linalg.norm(v2_perp)
    print(f"\n||p1-p0||={np.linalg.norm(v1):.2f}, ||p2-p0||={np.linalg.norm(v2):.2f}")

    def proj_one(theta):
        d_ = theta - origin
        a = float(d_ @ e1); b = float(d_ @ e2)
        resid = float(np.linalg.norm(d_ - a * e1 - b * e2))
        return a, b, resid

    anchor_xy = [proj_one(p)[:2] for p in (p0, p1, p2)]
    for (a, b), r in zip(anchor_xy, top3):
        _, _, residual = proj_one(load_full_theta(r["rep_name"]))
        print(f"  anchor for Family {top3.index(r)+1}: (a,b)=({a:.2f},{b:.2f}) resid={residual:.2e}")

    # Project ALL basins for star placement on the landscape
    other_basins = []
    for b in all_basins:
        if b["id"] in {x["id"] for x in top3}:
            continue
        th = load_full_theta(b["rep_name"])
        a, bb, resid = proj_one(th)
        other_basins.append({**b, "plane_a": a, "plane_b": bb, "plane_resid": resid})

    # Plane extent from anchors with margin
    anchor_a = np.array([c[0] for c in anchor_xy])
    anchor_b = np.array([c[1] for c in anchor_xy])
    span_a = max(anchor_a.max() - anchor_a.min(), 1.0)
    span_b = max(anchor_b.max() - anchor_b.min(), 1.0)
    pad = 0.6 * max(span_a, span_b)
    a_lo = anchor_a.min() - pad; a_hi = anchor_a.max() + pad
    b_lo = anchor_b.min() - pad; b_hi = anchor_b.max() + pad
    print(f"\nplane extent: a ∈ [{a_lo:.1f}, {a_hi:.1f}]  b ∈ [{b_lo:.1f}, {b_hi:.1f}]")

    # Evaluate landscape on grid
    cfg = load_config(CFG)
    problem, _, _, _, _ = build_all(cfg)
    problem._ensure_backend()
    grid_a = np.linspace(a_lo, a_hi, GRID)
    grid_b = np.linspace(b_lo, b_hi, GRID)
    AA, BB = np.meshgrid(grid_a, grid_b, indexing="ij")
    rms_grid = np.full((GRID, GRID), np.nan, dtype=float)
    print(f"evaluating landscape on {GRID}x{GRID} grid...")
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
    print(f"rms range: [{np.nanmin(rms_grid):.4f}, {np.nanmax(rms_grid):.4f}]")

    # ========== figure ==========
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[2.4, 1.0, 1.0],
        height_ratios=[1, 1, 1],
        wspace=0.10, hspace=0.20,
        left=0.04, right=0.99, top=0.94, bottom=0.04,
    )
    ax_landscape = fig.add_subplot(gs[:, 0])
    side_axes = [
        (fig.add_subplot(gs[i, 1]), fig.add_subplot(gs[i, 2])) for i in range(3)
    ]

    rms_vis = np.log10(np.clip(rms_grid, 1e-4, 1e3))
    cf = ax_landscape.contourf(AA, BB, rms_vis, levels=30, cmap="magma_r",
                               vmin=np.nanpercentile(rms_vis, 2),
                               vmax=np.nanpercentile(rms_vis, 98))
    ax_landscape.contour(AA, BB, rms_vis, levels=10, colors="white",
                         linewidths=0.4, alpha=0.35)

    # Other basins as small grey stars (with residual displayed via alpha)
    for b in other_basins:
        if not (a_lo <= b["plane_a"] <= a_hi and b_lo <= b["plane_b"] <= b_hi):
            continue
        # Higher residual → more transparent
        alpha = max(0.25, min(0.85, 1.0 - b["plane_resid"] / 30.0))
        ax_landscape.scatter(b["plane_a"], b["plane_b"], s=120, c="#cccccc",
                             marker="*", edgecolor="black", linewidths=0.8,
                             zorder=15, alpha=alpha)

    star_positions = []
    for (a, b), color in zip(anchor_xy, family_colors):
        ax_landscape.scatter(a, b, s=620, c=color, marker="*",
                             edgecolor="black", linewidths=2.0, zorder=20)
        star_positions.append((a, b))

    ax_landscape.set_xlabel("plane coord a", fontsize=11)
    ax_landscape.set_ylabel("plane coord b", fontsize=11)
    ax_landscape.set_xlim(a_lo, a_hi); ax_landscape.set_ylim(b_lo, b_hi)
    ax_landscape.set_aspect("equal")
    ax_landscape.set_title(
        "Sixlet design landscape on the plane through 3 family anchors\n"
        "(coloured stars = the 3 families; grey stars = other basins, dimmer = farther off-plane)",
        fontsize=12, fontweight="bold",
    )
    plt.colorbar(cf, ax=ax_landscape, shrink=0.85,
                 label="log₁₀(RMS / mm) · yellow = better")

    # Side rows
    for i, (b, color) in enumerate(zip(top3, family_colors)):
        ax_lens, ax_render = side_axes[i]
        lens_png = os.path.join(RUN_DIR, "elite_designs",
                                f"{b['rep_name']}_refined.png")
        if os.path.exists(lens_png):
            ax_lens.imshow(imread(lens_png))
        ax_lens.set_xticks([]); ax_lens.set_yticks([])
        for sp in ax_lens.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3)
        ax_lens.set_title(
            f"Family {i+1}  ·  basin {b['id']}\n"
            f"sign {b['sign_pattern']}  ·  RMS {b['rms_um_refined']:.2f} μm",
            fontsize=11, color=color, fontweight="bold",
        )
        render_png = os.path.join(RENDERS_DIR, f"basin_{b['id']}_USAF_render.png")
        if os.path.exists(render_png):
            ax_render.imshow(imread(render_png))
        ax_render.set_xticks([]); ax_render.set_yticks([])
        for sp in ax_render.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3)

    # Arrows
    for (sx, sy), (ax_lens, _), color in zip(star_positions, side_axes, family_colors):
        con = ConnectionPatch(
            xyA=(sx, sy), coordsA=ax_landscape.transData,
            xyB=(0.0, 0.5), coordsB=ax_lens.transAxes,
            arrowstyle="-|>", mutation_scale=20,
            linewidth=2.4, color=color,
            zorder=30, alpha=0.9,
            shrinkA=10, shrinkB=4,
        )
        fig.add_artist(con)

    fig.suptitle(
        "Sixlet hero composite — plane through 3 distinct family anchors "
        "(all anchors at residual = 0 by construction)",
        fontsize=14, fontweight="bold", y=0.99,
    )
    out_png = os.path.join(OUT, "sixlet_hero_composite_v2.png")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
