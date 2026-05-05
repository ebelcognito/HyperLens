"""
Composite hero figure: landscape with stars at every basin, with three of
them connected by arrows to side rows showing the corresponding lens layout
and through-the-lens USAF chart rendering.

Layout:
   |                                    |  lens diagram | rendered image  |   row 1 (Family 1)
   |                                    |  lens diagram | rendered image  |   row 2 (Family 2)
   |   LANDSCAPE  (stars for all 11)    |  lens diagram | rendered image  |   row 3 (Family 3)
   |        ↑ arrows from 3 stars       |                                  |
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

GALLERY_JSON = ("/home/hisham26/Desktop/RL/paper_figures/sixlet_design_gallery/"
                "design_family_summary.json")
RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "sixlet_discovery_20260423_132001")
GRIDS_NPZ = ("/home/hisham26/Desktop/RL/paper_figures/landscape_sixlet/"
             "grids.npz")
RENDERS_DIR = ("/home/hisham26/Desktop/RL/paper_figures/sixlet_top3_imaging")
OUT = "/home/hisham26/Desktop/RL/paper_figures/sixlet_hero_composite"
os.makedirs(OUT, exist_ok=True)


def load_basins():
    with open(GALLERY_JSON) as f:
        d = json.load(f)
    return d["basins"]


def pick_top3_distinct(basins):
    """One per sign pattern, best-RMS, top-3 by RMS."""
    by_pattern = {}
    for b in basins:
        sp = b["sign_pattern"]
        if sp not in by_pattern or b["rms_um_refined"] < by_pattern[sp]["rms_um_refined"]:
            by_pattern[sp] = b
    return sorted(by_pattern.values(), key=lambda b: b["rms_um_refined"])[:3]


def main():
    all_basins = load_basins()
    top3 = pick_top3_distinct(all_basins)
    top3_ids = {b["id"] for b in top3}
    family_colors = ["#E45756", "#4C78A8", "#54A24B"]

    grids = np.load(GRIDS_NPZ)
    AA = grids["AA"]; BB = grids["BB"]; rms_grid = grids["rms_grid"]
    a_lo = AA.min(); a_hi = AA.max()
    b_lo = BB.min(); b_hi = BB.max()

    # ---------- figure layout ----------
    fig = plt.figure(figsize=(20, 11))
    # 3 rows × 3 cols: landscape spans rows 0:3, col 0; per-row lens & render in cols 1, 2
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[2.4, 1.0, 1.0],
        height_ratios=[1, 1, 1],
        wspace=0.10, hspace=0.20,
        left=0.04, right=0.99, top=0.94, bottom=0.04,
    )

    ax_landscape = fig.add_subplot(gs[:, 0])
    side_axes = []   # list of (ax_lens, ax_render) per row
    for i in range(3):
        ax_lens = fig.add_subplot(gs[i, 1])
        ax_render = fig.add_subplot(gs[i, 2])
        side_axes.append((ax_lens, ax_render))

    # ---------- landscape ----------
    rms_vis = np.log10(np.clip(rms_grid, 1e-4, 1e3))
    cf = ax_landscape.contourf(AA, BB, rms_vis, levels=30, cmap="magma_r",
                               vmin=np.nanpercentile(rms_vis, 2),
                               vmax=np.nanpercentile(rms_vis, 98))
    ax_landscape.contour(AA, BB, rms_vis, levels=10, colors="white",
                         linewidths=0.4, alpha=0.35)

    # Plot stars for ALL basins, smaller and grey for non-top-3
    for b in all_basins:
        if b["id"] in top3_ids:
            continue
        ax_landscape.scatter(b["plane_a"], b["plane_b"], s=140,
                             c="#cccccc", marker="*",
                             edgecolor="black", linewidths=1.0,
                             zorder=15, alpha=0.85)

    # Top-3 stars: bigger, coloured
    star_positions = []
    for i, (b, color) in enumerate(zip(top3, family_colors)):
        s = ax_landscape.scatter(b["plane_a"], b["plane_b"],
                                  s=540, c=color, marker="*",
                                  edgecolor="black", linewidths=2.0,
                                  zorder=20)
        star_positions.append((b["plane_a"], b["plane_b"]))

    ax_landscape.set_xlabel("plane coord a", fontsize=11)
    ax_landscape.set_ylabel("plane coord b", fontsize=11)
    ax_landscape.set_xlim(a_lo, a_hi); ax_landscape.set_ylim(b_lo, b_hi)
    ax_landscape.set_aspect("equal")
    ax_landscape.set_title(
        "Sixlet design landscape — stars mark every distinct refined basin\n"
        "(coloured stars are the three families shown on the right)",
        fontsize=12, fontweight="bold",
    )
    plt.colorbar(cf, ax=ax_landscape, shrink=0.85,
                 label="log₁₀(RMS / mm) · yellow = better")

    # ---------- side rows ----------
    for i, (b, color) in enumerate(zip(top3, family_colors)):
        ax_lens, ax_render = side_axes[i]

        # lens diagram
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

        # render
        render_png = os.path.join(RENDERS_DIR, f"basin_{b['id']}_USAF_render.png")
        if os.path.exists(render_png):
            ax_render.imshow(imread(render_png))
        ax_render.set_xticks([]); ax_render.set_yticks([])
        for sp in ax_render.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3)
        ax_render.set_title(f"USAF chart through Family {i+1}",
                            fontsize=10, color=color)

    # ---------- arrows from star → start of each row ----------
    for (sx, sy), (ax_lens, _), color in zip(star_positions, side_axes, family_colors):
        # arrow target: left middle of ax_lens, in axes coordinates of ax_lens
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
        "Sixlet design landscape — every basin discovered (stars) and three "
        "structurally distinct families (lens layout + USAF chart rendering)",
        fontsize=14, fontweight="bold", y=0.99,
    )
    out_png = os.path.join(OUT, "sixlet_hero_composite.png")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
