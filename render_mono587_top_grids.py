"""Render filled-style ("pretty") diagrams for the eps-net top picks:
  sixlet:   top 40 from mono587_top40_sixlet_eps030_sub10.json  -> 10x4 grid
  octuplet: top 32 from mono587_top32_octuplet_eps030_sub20.json ->  8x4 grid

Each individual PNG is cropped in place to the central 1700x850 window.
Outputs:
  mono587_top40_sixlet_renders/   (40 pngs + grid_10x4.png)
  mono587_top32_octuplet_renders/ (32 pngs + grid_8x4.png)
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
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.lensmega.problem.full_dof import FullDOFParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from deeplens.optics import GeoLens

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

SIXLET = dict(
    src=os.path.join(REPO, "scripts_throwaway/mono587_top40_sixlet_eps030_sub10.json"),
    cfg=os.path.join(REPO,
        "optical_rl/qd/lensmega/config/examples/sixlet_apsc_discovery_v2.yaml"),
    out_dir=os.path.join(REPO, "mono587_top40_sixlet_renders"),
    nrows=10, ncols=4, n=40,
    has_cemented=False,
)

OCTUPLET = dict(
    src=os.path.join(REPO, "scripts_throwaway/mono587_top32_octuplet_eps030_sub20.json"),
    cfg=os.path.join(REPO,
        "experiments/gradient_alignment/octuplet_apsc_champion_config.yaml"),
    out_dir=os.path.join(REPO, "mono587_top32_octuplet_renders"),
    nrows=8, ncols=4, n=32,
    has_cemented=True,
)


def build_lens(theta, cfg, has_cemented, device="cpu"):
    p = cfg.problem
    if has_cemented:
        template, _ = load_lens_template(
            p.lens_path, wavelength_um=p.wavelength_um,
            stop_surface_index=p.stop_surface_index,
            free_curvature_indices=tuple(p.free_curvature_indices),
            derived_curvature_index=p.derived_curvature_index, device=device,
            cemented_interfaces=tuple(p.cemented_interfaces),
        )
        solver = FullDOFParaxialSolver(template)
    else:
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
    if lens.aper_idx is not None:
        r_old = float(lens.surfaces[lens.aper_idx].r)
        lens.surfaces[lens.aper_idx].r = r_old * APERTURE_SHRINK
        try:
            lens.post_computation()
        except Exception:
            pass
    return lens


def render_system(spec):
    print(f"\n========== {os.path.basename(spec['out_dir'])} ==========", flush=True)
    os.makedirs(spec["out_dir"], exist_ok=True)
    cfg = load_config(spec["cfg"])
    bundle = json.load(open(spec["src"]))
    designs = bundle["designs"][: spec["n"]]
    assert len(designs) == spec["n"], f"want {spec['n']} got {len(designs)}"

    renders = []
    for d in designs:
        rank = int(d["rank"])
        mono_rms = float(d["mono_rms_um"])
        theta = d["theta"]
        stem = f"rank_{rank:02d}_mono_{mono_rms:06.2f}um"
        out_path = os.path.join(spec["out_dir"], f"{stem}_filled.png")
        print(f"  {stem}", flush=True)
        lens = build_lens(theta, cfg, spec["has_cemented"])
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
        im = Image.open(out_path)
        if im.size == (W_FULL, H_FULL):
            im.crop(CROP_BOX).save(out_path)
        renders.append({"png": out_path, "rank": rank, "mono_rms_um": mono_rms})

    nrows, ncols = spec["nrows"], spec["ncols"]
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
    grid_path = os.path.join(spec["out_dir"], f"grid_{nrows}x{ncols}.png")
    fig.savefig(grid_path, dpi=200, bbox_inches="tight",
                pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"grid -> {grid_path}", flush=True)


def main():
    render_system(SIXLET)
    render_system(OCTUPLET)


if __name__ == "__main__":
    main()
