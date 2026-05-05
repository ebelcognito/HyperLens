"""Render lens diagrams + USAF chart images for the paper Results section.

Three sets of designs are rendered and saved to result_figs_all/:

  Set 1 — sixlet scaling (3 picks from sixlet_discovery_materials_tightBC):
    posthoc_0089_r133_c156, elite_0033_97248k, posthoc_0005_r162_c132

  Set 2 — sixlet deeper basins (3 picks from same run):
    elite_0003_32384k, elite_0026_83520k, elite_0033_97248k

  Set 3 — seventuplet scaling (3 picks from septlet_F3_20260425_224405):
    BEST_0001_4944k, BEST_0002_8304k, elite_0020_34872k

Same visual style as render_method_overview.py + render_sixlet_discovery.py
(no titles, no axes, white bg, three field bundles colored by FOV).

Selection of which set to render is via --set argument (or --all).
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
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

# ============================================================================
SETS = {
    "sixlet_scaling": dict(
        cfg=("/home/hisham26/Desktop/RL/results/lensmega/"
             "sixlet_discovery_materials_tightBC_20260426_051752/config.yaml"),
        run_dir=("/home/hisham26/Desktop/RL/results/lensmega/"
                 "sixlet_discovery_materials_tightBC_20260426_051752"),
        prefix="sixlet_scaling",
        picks=[
            "posthoc_0089_r133_c156",
            "elite_0033_97248k",
            "posthoc_0005_r162_c132",
        ],
        title="Sixlet scaling (start of scaling section)",
    ),
    "sixlet_deeper": dict(
        cfg=("/home/hisham26/Desktop/RL/results/lensmega/"
             "sixlet_discovery_materials_tightBC_20260426_051752/config.yaml"),
        run_dir=("/home/hisham26/Desktop/RL/results/lensmega/"
                 "sixlet_discovery_materials_tightBC_20260426_051752"),
        prefix="sixlet_deeper",
        picks=[
            "elite_0003_32384k",
            "elite_0026_83520k",
            "elite_0033_97248k",
        ],
        title="Sixlet deeper basins (gradual refinement section)",
    ),
    "septlet_scaling": dict(
        cfg=("/home/hisham26/Desktop/RL/results/lensmega/"
             "septlet_F3_20260425_224405/config.yaml"),
        run_dir=("/home/hisham26/Desktop/RL/results/lensmega/"
                 "septlet_F3_20260425_224405"),
        prefix="septlet",
        picks=[
            "BEST_0001_4944k",
            "BEST_0002_8304k",
            "elite_0020_34872k",
        ],
        title="Seventuplet scaling section",
    ),
}

CHART_PATH = "/home/hisham26/Desktop/RL/DeepLens/datasets/charts/USAF_1951_1k.png"
OUT = "/home/hisham26/Desktop/RL/result_figs_all"
os.makedirs(OUT, exist_ok=True)

WAVELENGTH_UM = 0.587
FIELD_COLORS = ["#0072B2", "#D55E00", "#009E73"]
RAY_ALPHA = 0.85
SENSOR_RES = (1024, 1024)
DEPTH = -20000.0
PSF_GRID = (12, 12)
PSF_KS = 51
LINEWIDTH_LENS = 1.9
LINEWIDTH_RAY = 1.4
NUM_RAYS_PER_FIELD = 3
NUM_FIELDS = 3


def build_lens_from_theta(theta, cfg, device="cpu"):
    p = cfg.problem
    template, _ = load_lens_template(
        p.lens_path,
        wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        device=device,
    )
    solver = GeneralizedParaxialSolver(template)
    lens = GeoLens(filename=p.lens_path, device=device)
    _apply_theta_generic(
        lens, np.asarray(theta, dtype=float),
        template=template, solver=solver,
        focal_length_target_mm=p.focal_length_target_mm,
        device=device,
        parameterization=p.parameterization,
        free_curvature_indices=p.free_curvature_indices,
        free_gap_indices=p.free_gap_indices,
        variable_flange=p.variable_flange,
        variable_materials=p.variable_materials,
        free_material_indices=p.free_material_indices,
    )
    return lens


def draw_lens_clean(lens, ax, color="black", lw_lens=LINEWIDTH_LENS):
    for s in lens.surfaces:
        cls = s.__class__.__name__
        if cls == "Aperture":
            d = float(s.d.item()); r = float(s.r)
            wedge_l = 0.05 * r; wedge_h = 0.18 * r
            for sign in (-1, +1):
                ax.plot([d - wedge_l, d + wedge_l], [sign * r, sign * r],
                        color="#666", lw=lw_lens)
                ax.plot([d, d], [sign * r, sign * (r + wedge_h)],
                        color="#666", lw=lw_lens)
        else:
            r_eff = float(s.draw_r())
            r = torch.linspace(-r_eff, r_eff, 256, device=s.device)
            z = s.surface_with_offset(
                r, torch.zeros(len(r), device=s.device), valid_check=False
            ).cpu().detach().numpy()
            r_np = r.cpu().detach().numpy()
            ax.plot(z, r_np, color=color, lw=lw_lens)
    surfs = lens.surfaces
    for i in range(len(surfs)):
        if hasattr(surfs[i], "mat2") and surfs[i].mat2.n > 1.1:
            s_prev = surfs[i]; s_next = surfs[i + 1]
            r_prev = float(s_prev.draw_r()); r_next = float(s_next.draw_r())
            sag_prev = s_prev.surface_with_offset(r_prev, 0.0, valid_check=False).item()
            sag_next = s_next.surface_with_offset(r_next, 0.0, valid_check=False).item()
            for sign in (-1, +1):
                ax.plot([sag_prev, sag_next], [sign * r_prev, sign * r_next],
                        color=color, lw=lw_lens)
    d_sensor = float(lens.d_sensor.item())
    r_sensor = float(lens.r_sensor)
    ax.plot([d_sensor, d_sensor], [-r_sensor, r_sensor], color=color, lw=lw_lens)


def trace_3rays_field(lens, fov_deg, wvln_um):
    rays = lens.sample_parallel_2D(
        fov=fov_deg, num_rays=NUM_RAYS_PER_FIELD,
        wvln=wvln_um, depth=-1.0, plane="sagittal",
    )
    _, ray_o_record = lens.trace2sensor(ray=rays, record=True)
    pts = torch.stack(ray_o_record, dim=-2).cpu().numpy()
    if pts.ndim == 3: pts = pts[None, ...]
    return pts


def render_lens_diagram(lens, out_path):
    fig = plt.figure(figsize=(6.0, 3.0), dpi=300, facecolor="white")
    ax = fig.add_subplot(111); ax.set_facecolor("white")
    draw_lens_clean(lens, ax)
    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    fovs = np.linspace(0.0, rfov_deg * 0.999, NUM_FIELDS)
    for fov, color in zip(fovs, FIELD_COLORS):
        pts = trace_3rays_field(lens, fov, WAVELENGTH_UM)
        for ray_idx in range(pts.shape[1]):
            ax.plot(pts[0, ray_idx, :, 2], pts[0, ray_idx, :, 0],
                    color=color, lw=LINEWIDTH_RAY, alpha=RAY_ALPHA)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("")
    for sp in ax.spines.values(): sp.set_visible(False)
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                pad_inches=0.02, facecolor="white", transparent=False)
    plt.close(fig)


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


def render_set(set_name):
    spec = SETS[set_name]
    print(f"\n========== {spec['title']} ==========")
    cfg = load_config(spec["cfg"])
    elite_dir = os.path.join(spec["run_dir"], "elite_designs")

    items = []
    for stem in spec["picks"]:
        path = os.path.join(elite_dir, f"{stem}.json")
        with open(path) as f:
            d = json.load(f)
        r = d.get("refined") or d.get("refined_F8") or {}
        rms_um = float(r["rms_mm"]) * 1000.0
        items.append((stem, rms_um, r["theta"]))
    items.sort(key=lambda t: t[1])

    print(f"Picks for {set_name}:")
    for i, (stem, rms, _) in enumerate(items, start=1):
        print(f"  rank {i}: {stem}  rms={rms:.2f} um")

    chart = load_chart(SENSOR_RES)
    for rank, (stem, rms_um, theta) in enumerate(items, start=1):
        rms_tag = f"{rms_um:.1f}"
        prefix = spec["prefix"]
        lens_path = os.path.join(OUT, f"{prefix}_lens_{rank}_rms_{rms_tag}um.png")
        img_path  = os.path.join(OUT, f"{prefix}_render_{rank}_rms_{rms_tag}um.png")

        print(f"\n[{set_name} rank {rank}] {stem}  rms={rms_um:.2f} um")
        lens = build_lens_from_theta(theta, cfg, device="cpu")
        render_lens_diagram(lens, lens_path)
        print(f"  -> {os.path.basename(lens_path)}")

        lens_img = build_lens_from_theta(theta, cfg, device="cpu")
        render_image_through_lens(lens_img, chart, img_path)
        print(f"  -> {os.path.basename(img_path)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS.keys()) + ["all"], default="all")
    args = ap.parse_args()
    if args.set == "all":
        for s in SETS:
            render_set(s)
    else:
        render_set(args.set)


if __name__ == "__main__":
    main()
