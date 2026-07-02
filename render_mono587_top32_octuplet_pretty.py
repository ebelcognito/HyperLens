"""Re-render the top 32 octuplet designs using the SAME pipeline that produced
figures/octuplet_top17_eps030_lenses/* (render_one from
render_spotlight_sixlet_designs.py + make_geolens from fig5_extract).

Why: the default draw_layout(style="filled") path produces broken/skinny
glass polygons for some cemented octuplet elements. The render_one path uses
lens_fill_polygons + draw_lens_clean and fills cleanly.

Outputs to mono587_top32_octuplet_renders/:
  - rank_{NN}_mono_*um_filled.png   (1700x850 each)
  - grid_8x4.png
"""
import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = "/nas/longleaf/home/hisham26/Desktop/hyperlens"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "DeepLens"))

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)
from scripts_throwaway.render_spotlight_sixlet_designs import (
    render_one, BLUE_FILL,
)

SRC_JSON = os.path.join(
    REPO, "scripts_throwaway/mono587_top32_octuplet_eps030_sub20.json"
)
OUT_DIR = os.path.join(REPO, "mono587_top32_octuplet_renders")
os.makedirs(OUT_DIR, exist_ok=True)

# Target a 2:1 aspect, ~1700x850 px (matches the sixlet crop window).
FIG_W_IN = 8.5
FIG_H_IN = 4.25
DPI = 200


def render_design(theta, out_path, prob_cfg, p_problem, lens_path,
                  template, solver):
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")
    ax.set_facecolor("white")
    ax.set_position([0.0, 0.0, 1.0, 1.0])  # axes fills the figure exactly
    lens = make_geolens(prob_cfg, p_problem, lens_path,
                        np.asarray(theta, dtype=np.float64),
                        template, solver)
    render_one(ax, lens, fill_color=BLUE_FILL)
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)


def main():
    bundle = json.load(open(SRC_JSON))
    designs = bundle["designs"][:32]
    assert len(designs) == 32

    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["octuplet"]["cfg"]
    )

    renders = []
    for d in designs:
        rank = int(d["rank"])
        mono_rms = float(d["mono_rms_um"])
        stem = f"rank_{rank:02d}_mono_{mono_rms:06.2f}um"
        out_path = os.path.join(OUT_DIR, f"{stem}_filled.png")
        print(f"  {stem}", flush=True)
        try:
            render_design(d["theta"], out_path,
                          prob_cfg, p_problem, lens_path, template, solver)
        except Exception as e:
            print(f"    FAIL: {e}")
            continue
        renders.append({"png": out_path, "rank": rank, "mono_rms_um": mono_rms})

    nrows, ncols = 8, 4
    tile_w = 3.6
    tile_h = tile_w * FIG_H_IN / FIG_W_IN  # match aspect
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * tile_w, nrows * tile_h),
        gridspec_kw=dict(hspace=0.05, wspace=0.04),
    )
    for ax, it in zip(axes.flat, renders):
        ax.imshow(Image.open(it["png"]))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    # blank any unused axes (shouldn't happen at 32 = 8*4)
    for ax in axes.flat[len(renders):]:
        ax.set_axis_off()
    grid_path = os.path.join(OUT_DIR, "grid_8x4.png")
    fig.savefig(grid_path, dpi=200, bbox_inches="tight",
                pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"\ngrid -> {grid_path}")


if __name__ == "__main__":
    main()
