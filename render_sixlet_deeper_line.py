"""Re-render sixlet deeper-basin lens diagrams in regular DeepLens line style
with 12 rays per FOV (3 fields, R/G/B per FOV).

Overwrites:
  result_figs_all/sixlet_deeper_lens_1_rms_12.7um.png
  result_figs_all/sixlet_deeper_lens_2_rms_59.3um.png
  result_figs_all/sixlet_deeper_lens_3_rms_179.6um.png

Uses GeoLens.draw_lens_2d + draw_ray_2d directly so we control num_rays.
"""
import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/hisham26/Desktop/RL")
sys.path.insert(0, "/home/hisham26/Desktop/RL/DeepLens")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from deeplens.optics import GeoLens

CFG = ("/home/hisham26/Desktop/RL/results/lensmega/"
       "sixlet_discovery_materials_tightBC_20260426_051752/config.yaml")
ELITE_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
             "sixlet_discovery_materials_tightBC_20260426_051752/elite_designs")
OUT = "/home/hisham26/Desktop/RL/result_figs_all"

NUM_RAYS_PER_FIELD = 5
NUM_FIELDS = 3
WAVE_RGB = [0.486, 0.587, 0.656]
COLORS_LIST = ["#CC0000", "#006600", "#0066CC"]

PICKS = [
    "elite_0033_97248k",   # 12.7 µm
    "elite_0026_83520k",   # 59.3 µm
    "elite_0003_32384k",   # 179.6 µm
]


def build_lens(theta, cfg, device="cpu"):
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
    return lens


def render_line(lens, out_path):
    """Standard line-style render: draw_lens_2d + 12 rays/field × 3 fields RGB."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax, fig = lens.draw_lens_2d(ax=ax, fig=fig, zmx_format=True)

    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    fovs = np.linspace(0.0, rfov_deg * 0.99, NUM_FIELDS)

    for fov, color, wvln in zip(fovs, COLORS_LIST, WAVE_RGB):
        ray = lens.sample_parallel_2D(
            fov=fov, num_rays=NUM_RAYS_PER_FIELD, wvln=wvln,
            depth=-1.0, plane="sagittal",
        )
        _, ray_o_record = lens.trace2sensor(ray=ray, record=True)
        ax, fig = lens.draw_ray_2d(ray_o_record, ax=ax, fig=fig, color=color)

    ax.axis("off")
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)


def main():
    cfg = load_config(CFG)
    items = []
    for stem in PICKS:
        with open(os.path.join(ELITE_DIR, f"{stem}.json")) as f:
            d = json.load(f)
        ref = d["refined"]
        rms_um = float(ref["rms_mm"]) * 1000.0
        items.append((stem, rms_um, ref["theta"]))
    items.sort(key=lambda t: t[1])

    for rank, (stem, rms_um, theta) in enumerate(items, start=1):
        out_path = os.path.join(
            OUT, f"sixlet_deeper_lens_{rank}_rms_{rms_um:.1f}um.png"
        )
        print(f"  rank {rank}: {stem}  rms={rms_um:.2f} µm")
        lens = build_lens(theta, cfg)
        render_line(lens, out_path)
        print(f"    -> {os.path.basename(out_path)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
