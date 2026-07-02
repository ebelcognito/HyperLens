"""Render the 24 design_jsons/ sixlet APS-C v2 designs using the filled
("pretty") pipeline, individually and as a 6x4 grid. Minimal text.

Outputs to design_jsons_renders/:
  - rank_{NN}_rms_*um_filled.png   (24 individual filled-style diagrams)
  - grid_6x4.png                    (composite, 6 rows x 4 cols, no labels)
"""
import json
import os
import sys
import glob
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
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from deeplens.optics import GeoLens

CFG_PATH = os.path.join(
    REPO,
    "optical_rl/qd/lensmega/config/examples/sixlet_apsc_discovery_v2.yaml",
)
JSON_DIR = os.path.join(REPO, "design_jsons")
OUT_DIR = os.path.join(REPO, "design_jsons_renders")
os.makedirs(OUT_DIR, exist_ok=True)

GLASS_FACECOLOR = "#BFE3F2"
GLASS_EDGECOLOR = "black"
GLASS_LW = 1.2
BUNDLE_ALPHA = 0.22
APERTURE_SHRINK = 0.92  # sixlet recipe from render_filled_collections.py


def build_lens(theta, cfg, aperture_shrink=APERTURE_SHRINK, device="cpu"):
    p = cfg.problem
    template, _ = load_lens_template(
        p.lens_path, wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index, device=device,
    )
    solver = GeneralizedParaxialSolver(template)
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
    paths = sorted(glob.glob(os.path.join(JSON_DIR, "rank_*.json")))
    assert len(paths) == 24, f"expected 24 designs, found {len(paths)}"

    renders = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        with open(p) as f:
            d = json.load(f)
        ref = d.get("refined") or d
        theta = ref["theta"]
        rms_um = float(ref["rms_um"])
        out_path = os.path.join(OUT_DIR, f"{stem}_filled.png")
        print(f"rendering {stem}  rms={rms_um:.2f} um")
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
        renders.append({"png": out_path, "rms_um": rms_um, "stem": stem})

    # 6 rows x 4 cols grid, minimal text — no labels at all
    nrows, ncols = 6, 4
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.6, nrows * 1.7),
        gridspec_kw=dict(hspace=0.05, wspace=0.04),
    )
    for ax, it in zip(axes.flat, renders):
        ax.imshow(Image.open(it["png"]))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    grid_path = os.path.join(OUT_DIR, "grid_6x4.png")
    fig.savefig(grid_path, dpi=200, bbox_inches="tight",
                pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"\ngrid -> {grid_path}")


if __name__ == "__main__":
    main()
