"""Render two specific sixlet elites (lens diagram in filled style + USAF chart).

Saves to result_figs_all/:
  extra_sixlet_A_<rms>um_lens_filled.png
  extra_sixlet_A_<rms>um_render.png
  extra_sixlet_B_<rms>um_lens_filled.png
  extra_sixlet_B_<rms>um_render.png

Uses the patched DeepLens draw_layout(style='filled') with bundle filtering
(survivor-based extreme picks) so off-axis fields don't drop out.
"""
import json
import os
import sys
import tempfile
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, "/home/hisham26/Desktop/RL")
sys.path.insert(0, "/home/hisham26/Desktop/RL/DeepLens")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from deeplens.optics import GeoLens

OUT = "/home/hisham26/Desktop/RL/result_figs_all"
CHART_PATH = "/home/hisham26/Desktop/RL/DeepLens/datasets/charts/USAF_1951_1k.png"

PICKS = [
    dict(
        label="A",
        elite=("/home/hisham26/Desktop/RL/results/lensmega/"
               "sixlet_discovery_20260413_155547/elite_designs/elite_0006_26586k.json"),
        cfg=("/home/hisham26/Desktop/RL/results/lensmega/"
             "sixlet_discovery_20260413_155547/config.yaml"),
    ),
    dict(
        label="B",
        elite=("/home/hisham26/Desktop/RL/results/lensmega/"
               "sixlet_discovery_20260421_154452/elite_designs/BEST_0001_1998k.json"),
        cfg=("/home/hisham26/Desktop/RL/results/lensmega/"
             "sixlet_discovery_20260421_154452/config.yaml"),
    ),
]

# Filled style aesthetics (matching the rest of result_figs_all)
GLASS_FACECOLOR = "#BFE3F2"
GLASS_EDGECOLOR = "black"
GLASS_LW = 1.2
BUNDLE_ALPHA = 0.22

# Image render config
SENSOR_RES = (1024, 1024)
DEPTH = -20000.0
PSF_GRID = (12, 12)
PSF_KS = 51


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


def load_chart(target_size):
    img = Image.open(CHART_PATH).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def render_image_through_lens(lens, chart, out_path):
    lens.sensor_res = SENSOR_RES
    lens.set_sensor(sensor_res=SENSOR_RES, sensor_size=lens.sensor_size)
    with torch.no_grad():
        rendered = lens.render(
            chart, depth=DEPTH, method="psf_map",
            psf_grid=PSF_GRID, psf_ks=PSF_KS,
        )
    arr = rendered.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr, 0, 1)
    Image.fromarray((arr * 255).astype(np.uint8)).save(out_path)


def load_cfg_compat(cfg_path):
    """Load YAML and migrate older run configs to the current pydantic schema.

    Migrations:
      - `variable_d_sensor: true` → `variable_flange: true`  (renamed)
      - drop other deprecated keys
    """
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    if "problem" in raw and isinstance(raw["problem"], dict):
        prob = raw["problem"]
        if "variable_d_sensor" in prob and "variable_flange" not in prob:
            prob["variable_flange"] = bool(prob["variable_d_sensor"])
        for stale in ("variable_d_sensor", "d_sensor_init", "d_sensor_bounds"):
            prob.pop(stale, None)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        yaml.safe_dump(raw, tf)
        tmp = tf.name
    try:
        return load_config(tmp)
    finally:
        os.unlink(tmp)


def main():
    chart = load_chart(SENSOR_RES)
    print(f"USAF chart loaded at {SENSOR_RES}\n")

    for pick in PICKS:
        with open(pick["elite"]) as f:
            d = json.load(f)
        ref = d["refined"]
        rms_um = float(ref["rms_mm"]) * 1000.0
        theta = ref["theta"]

        cfg = load_cfg_compat(pick["cfg"])
        print(f"Pick {pick['label']}: {os.path.basename(pick['elite'])}  "
              f"refined RMS = {rms_um:.2f} µm  sign={d.get('sign_pattern')}")

        # 1) filled lens diagram
        lens = build_lens(theta, cfg)
        lens_path = os.path.join(
            OUT, f"extra_sixlet_{pick['label']}_{rms_um:.1f}um_lens_filled.png"
        )
        lens.draw_layout(
            filename=lens_path,
            depth=float("inf"),
            style="filled",
            lens_title="",
            glass_facecolor=GLASS_FACECOLOR,
            glass_edgecolor=GLASS_EDGECOLOR,
            glass_lw=GLASS_LW,
            bundle_alpha=BUNDLE_ALPHA,
        )
        print(f"  -> {os.path.basename(lens_path)}")

        # 2) USAF chart through lens
        lens_img = build_lens(theta, cfg)
        img_path = os.path.join(
            OUT, f"extra_sixlet_{pick['label']}_{rms_um:.1f}um_render.png"
        )
        render_image_through_lens(lens_img, chart, img_path)
        print(f"  -> {os.path.basename(img_path)}\n")

    print("Done.")


if __name__ == "__main__":
    main()
