"""Render the top 24 designs from scripts_throwaway/mono587_top30_octuplet_eps030_sub20.json
using the filled ("pretty") pipeline, individually + 6x4 grid, and crop
each PNG to the central 1700x850 window.

Outputs to mono587_top24_octuplet_renders/:
  - rank_{NN}_mono_*um_filled.png   (24 individual filled-style, cropped)
  - grid_6x4.png                     (composite, 6 rows x 4 cols, no labels)
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

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.full_dof import FullDOFParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from deeplens.optics import GeoLens

# Matches the config used to build the source JSON
# (scripts_throwaway/count_mono587_diverse.py SYSTEMS["octuplet"])
CFG_PATH = os.path.join(
    REPO,
    "experiments/gradient_alignment/octuplet_apsc_champion_config.yaml",
)
SRC_JSON = os.path.join(
    REPO, "scripts_throwaway/mono587_top30_octuplet_eps030_sub20.json"
)
OUT_DIR = os.path.join(REPO, "mono587_top24_octuplet_renders")
os.makedirs(OUT_DIR, exist_ok=True)

GLASS_FACECOLOR = "#BFE3F2"
GLASS_EDGECOLOR = "black"
GLASS_LW = 1.2
BUNDLE_ALPHA = 0.22
APERTURE_SHRINK = 0.92

W_FULL, H_FULL = 1920, 1440
W_CROP, H_CROP = 1700, 850
CROP_BOX = ((W_FULL - W_CROP)//2, (H_FULL - H_CROP)//2,
            (W_FULL - W_CROP)//2 + W_CROP,
            (H_FULL - H_CROP)//2 + H_CROP)


def build_lens(theta, cfg, aperture_shrink=APERTURE_SHRINK, device="cpu"):
    p = cfg.problem
    template, _ = load_lens_template(
        p.lens_path, wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index, device=device,
        cemented_interfaces=tuple(p.cemented_interfaces),
    )
    solver = FullDOFParaxialSolver(template)
    lens = GeoLens(filename=p.lens_path, device=device)
    _apply_theta_generic(
        lens, np.asarray(theta, dtype=float),
        template=template, solver=solver,
        focal_length_target_mm=p.focal_length_target_mm, device=device,
        parameterization=p.parameterization,
        free_curvature_indices=p.free_curvature_indices,
        free_gap_indices=p.free_gap_indices,
        variable_flange=p.variable_flange,
        variable_materials=p.variable_materials,
        free_material_indices=p.free_material_indices,
    )
    if aperture_shrink and aperture_shrink != 1.0 and lens.aper_idx is not None:
        r_old = float(lens.surfaces[lens.aper_idx].r)
        lens.surfaces[lens.aper_idx].r = r_old * float(aperture_shrink)
        try:
            lens.post_computation()
        except Exception:
            pass
    return lens


def main():
    cfg = load_config(CFG_PATH)
    with open(SRC_JSON) as f:
        bundle = json.load(f)
    designs = bundle["designs"][:24]
    assert len(designs) == 24, f"expected 24, got {len(designs)}"

    renders = []
    for d in designs:
        rank = int(d["rank"])
        mono_rms = float(d["mono_rms_um"])
        theta = d["theta"]
        stem = f"rank_{rank:02d}_mono_{mono_rms:06.2f}um"
        out_path = os.path.join(OUT_DIR, f"{stem}_filled.png")
        print(f"rendering {stem}")
        lens = build_lens(theta, cfg)
        lens.draw_layout(
            filename=out_path,
            depth=float("inf"),
            style="filled",
            lens_title="",
            glass_facecolor=GLASS_FACECOLOR,
            glass_edgecolor=GLASS_EDGECOLOR,
            glass_lw=GLASS_LW,
            bundle_alpha=BUNDLE_ALPHA,
        )
        # crop in place to central 1700x850
        im = Image.open(out_path)
        if im.size == (W_FULL, H_FULL):
            im.crop(CROP_BOX).save(out_path)
        renders.append({"png": out_path, "rank": rank, "mono_rms_um": mono_rms})

    # 6 rows x 4 cols grid, minimal text — no labels
    nrows, ncols = 6, 4
    tile_w = 3.6
    tile_h = tile_w * H_CROP / W_CROP
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
    grid_path = os.path.join(OUT_DIR, "grid_6x4.png")
    fig.savefig(grid_path, dpi=200, bbox_inches="tight",
                pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"\ngrid -> {grid_path}")


if __name__ == "__main__":
    main()
