"""Render lens diagrams + USAF chart images for the sixlet discovery run.

Produces 8 standalone PNGs (4 elites x {lens diagram, rendered image}) for the
top-4 geometrically distinct refined elites of the sixlet_discovery run, picked
by exhaustive choose-4 over the top-15-by-RMS pool to maximize the minimum
pairwise unit-theta L2 distance.

Style matches paper_figures/method_overview_renders/ (no titles, no axes,
white background, three field bundles colored by FOV).
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

sys.path.insert(0, "/home/hisham26/Desktop/RL")
sys.path.insert(0, "/home/hisham26/Desktop/RL/DeepLens")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from deeplens.optics import GeoLens


CFG = ("/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
       "config/examples/sixlet_discovery.yaml")
RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "sixlet_discovery_20260423_132001")
ELITE_DIR = os.path.join(RUN_DIR, "elite_designs")
CHART_PATH = "/home/hisham26/Desktop/RL/DeepLens/datasets/charts/USAF_1951_1k.png"
OUT = "/home/hisham26/Desktop/RL/paper_figures/sixlet_discovery_renders"
os.makedirs(OUT, exist_ok=True)

# Top-4 geometrically distinct refined elites for the sixlet_discovery run,
# selected via exhaustive choose-4 over the top-15-by-refined-RMS pool, max-
# imizing min pairwise unit-theta L2 distance. min-dist = 2.35.
PICKS = [
    "elite_0011_5904k",     # rms 4.7  μm — full ~110 mm length, balanced doublets
    "elite_0023_9792k",     # rms 26.1 μm — strong all-positive front, all-negative middle
    "elite_0038_15786k",    # rms 38.8 μm — negative front meniscus, compact ~95 mm
    "elite_0034_14706k",    # rms 39.4 μm — shortest envelope ~75 mm, dense packing
]

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
            d = float(s.d.item())
            r = float(s.r)
            wedge_l = 0.05 * r
            wedge_h = 0.18 * r
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
            s_prev = surfs[i]
            s_next = surfs[i + 1]
            r_prev = float(s_prev.draw_r())
            r_next = float(s_next.draw_r())
            sag_prev = s_prev.surface_with_offset(
                r_prev, 0.0, valid_check=False
            ).item()
            sag_next = s_next.surface_with_offset(
                r_next, 0.0, valid_check=False
            ).item()
            for sign in (-1, +1):
                ax.plot([sag_prev, sag_next],
                        [sign * r_prev, sign * r_next],
                        color=color, lw=lw_lens)

    d_sensor = float(lens.d_sensor.item())
    r_sensor = float(lens.r_sensor)
    ax.plot([d_sensor, d_sensor], [-r_sensor, r_sensor], color=color, lw=lw_lens)


def trace_3rays_field(lens, fov_deg, wvln_um):
    rays = lens.sample_parallel_2D(
        fov=fov_deg,
        num_rays=NUM_RAYS_PER_FIELD,
        wvln=wvln_um,
        depth=-1.0,
        plane="sagittal",
    )
    _, ray_o_record = lens.trace2sensor(ray=rays, record=True)
    pts = torch.stack(ray_o_record, dim=-2).cpu().numpy()
    if pts.ndim == 3:
        pts = pts[None, ...]
    return pts


def render_lens_diagram(lens, out_path):
    fig = plt.figure(figsize=(6.0, 3.0), dpi=300, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    draw_lens_clean(lens, ax)

    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    fovs = np.linspace(0.0, rfov_deg * 0.999, NUM_FIELDS)

    for fov, color in zip(fovs, FIELD_COLORS):
        pts = trace_3rays_field(lens, fov, WAVELENGTH_UM)
        for ray_idx in range(pts.shape[1]):
            ax.plot(
                pts[0, ray_idx, :, 2],
                pts[0, ray_idx, :, 0],
                color=color, lw=LINEWIDTH_RAY, alpha=RAY_ALPHA,
            )

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("")
    for sp in ax.spines.values():
        sp.set_visible(False)

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


def lens_specs(lens):
    foclen = float(lens.foclen)
    _, pupilr = lens.calc_entrance_pupil()
    pupilr = float(pupilr)
    fnum = foclen / (2.0 * pupilr) if pupilr > 0 else float("nan")
    dfov_deg = float(np.rad2deg(float(lens.dfov)))
    sensor_diag = 2.0 * float(lens.r_sensor)
    return foclen, fnum, dfov_deg, sensor_diag


def main():
    cfg = load_config(CFG)

    # Load + sort picks by refined RMS
    items = []
    for stem in PICKS:
        path = os.path.join(ELITE_DIR, f"{stem}.json")
        with open(path) as f:
            d = json.load(f)
        r = d["refined"]
        rms_um = float(r["rms_mm"]) * 1000.0
        items.append((stem, rms_um, r["theta"]))
    items.sort(key=lambda t: t[1])

    print("Rendering top-4 distinct elites:")
    for i, (stem, rms, _) in enumerate(items, start=1):
        print(f"  rank {i}: {stem}  rms={rms:.2f} um")

    print("\nLens specs (computed from built GeoLens at d-line):")
    print(f"  {'rank':>4} {'source':22s} {'EFL_mm':>8} {'F#':>6} {'dFOV deg':>8} {'sensor_diag':>11}")
    spec_cache = []
    for i, (stem, _, theta) in enumerate(items, start=1):
        L = build_lens_from_theta(theta, cfg, device="cpu")
        efl, fnum, dfov, diag = lens_specs(L)
        spec_cache.append((efl, fnum, dfov, diag))
        print(f"  {i:>4d} {stem:22s} {efl:8.3f} {fnum:6.2f} {dfov:8.2f} {diag:11.2f}")

    chart = load_chart(SENSOR_RES)
    print(f"\nUSAF chart loaded at {SENSOR_RES}")

    summary = []
    for rank, (stem, rms_um, theta) in enumerate(items, start=1):
        rms_tag = f"{rms_um:.1f}"
        lens_path = os.path.join(OUT, f"lens_design_{rank}_rms_{rms_tag}um.png")
        img_path  = os.path.join(OUT, f"render_{rank}_rms_{rms_tag}um.png")

        print(f"\n[rank {rank}] {stem}  rms={rms_um:.2f} um")
        lens = build_lens_from_theta(theta, cfg, device="cpu")

        print(f"  drawing lens diagram -> {os.path.basename(lens_path)}")
        render_lens_diagram(lens, lens_path)

        lens_img = build_lens_from_theta(theta, cfg, device="cpu")
        print(f"  rendering USAF chart -> {os.path.basename(img_path)}")
        render_image_through_lens(lens_img, chart, img_path)

        summary.append((rank, stem, rms_um, lens_path, img_path))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for (rank, stem, rms, lp, ip), (efl, fnum, dfov, _) in zip(summary, spec_cache):
        print(f"\nrank {rank}  source={stem}  refined_rms={rms:.2f} um  "
              f"EFL={efl:.2f}mm  F/{fnum:.2f}  dFOV={dfov:.2f} deg")
        print(f"  {os.path.basename(lp)}  - lens cross-section, 9 rays @ d-line, "
              f"colored by field (0H/0.5H/1.0H)")
        print(f"  {os.path.basename(ip)}  - USAF 1951 imaged through this lens")


if __name__ == "__main__":
    main()
