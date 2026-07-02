#!/usr/bin/env python
"""Render the multi-panel per-spotlight figures requested in HANDOFF.md §4.

Layout (per spotlight design):
  +-----------------------------+----------------------+
  |                             |  Seidel bar chart    |
  |   Filled-bundle lens render |  (S1..S5)            |
  |   (RECIPE §6 style C)       +----------------------+
  |                             |                      |
  |                             |  Spot diagram        |
  |                             |  (R/G/B overlaid)    |
  +                             +----------------------+
  |                             |                      |
  |                             |  Color test image    |
  |                             |  (chromatic blur)    |
  +-----------------------------+----------------------+

Inputs (per topology):
  results/figure_archive_demo[_octuplet]/spotlights.json
  results/figure_archive_demo/test_images/<image>.png       (shared across topologies)

Outputs:
  results/figure_archive_demo/spotlight_grids/sixlet_<label>_grid.png
  results/figure_archive_demo_octuplet/spotlight_grids/octuplet_<label>_grid.png

The per-wavelength PSF used for image blurring is derived from the actual spot
diagram at the on-axis field (field 0). This is a coarse approximation — it
ignores field dependence — but for a "this is what you'd see through this
lens" visual it's faithful enough that chromatic differences are visible.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
from scipy.ndimage import convolve as nd_convolve
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)

# ---- Recipe constants (verbatim from RENDERING_RECIPE.md) ----
BLUE_FILL = "#7BAFD4"
RED_FILL  = "#E07B7B"
OUTLINE   = "#1F1F1F"
SENSOR    = "#000000"
LINEWIDTH_LENS   = 1.2
LINEWIDTH_SENSOR = 1.6
FILL_NUM_SAMPLES = 192
FIELD_COLORS = ["#1565C0", "#2E7D32", "#C62828"]
FILLED_BUNDLE_ALPHA = 0.32
FILLED_BUNDLE_NUM_RAYS = 9
RAY_WVLN_UM = 0.587
RAY_DEPTH_MM = -20000.0

WVLN_RGB = {0.486: "blue", 0.587: "green", 0.656: "red"}
WVLN_HEX = {0.486: "#1565C0", 0.587: "#2E7D32", 0.656: "#C62828"}


# ---- Recipe primitives (verbatim from render_spotlight_*_designs.py) ----
def lens_fill_polygons(lens, num_samples=FILL_NUM_SAMPLES):
    polys = []
    for i in range(len(lens.surfaces) - 1):
        s0 = lens.surfaces[i]; s1 = lens.surfaces[i + 1]
        if not (hasattr(s0, "mat2") and float(s0.mat2.n) > 1.1):
            continue
        r0 = float(s0.draw_r()) * 0.999
        r1 = float(s1.draw_r()) * 0.999
        if r0 <= 1e-8 or r1 <= 1e-8:
            continue
        rs0 = np.linspace(-r0, r0, num_samples, dtype=np.float64)
        rs1 = np.linspace(-r1, r1, num_samples, dtype=np.float64)
        rt0 = torch.tensor(rs0, dtype=torch.float32, device=lens.device)
        rt1 = torch.tensor(rs1, dtype=torch.float32, device=lens.device)
        z0 = s0.surface_with_offset(rt0, torch.zeros_like(rt0), valid_check=False).detach().cpu().numpy()
        z1 = s1.surface_with_offset(rt1, torch.zeros_like(rt1), valid_check=False).detach().cpu().numpy()
        polys.append((np.concatenate([z0, z1[::-1]]), np.concatenate([rs0, rs1[::-1]])))
    return polys


def draw_lens_clean(lens, ax, color=OUTLINE, lw=LINEWIDTH_LENS):
    for s in lens.surfaces:
        if s.__class__.__name__ == "Aperture":
            d = float(s.d.item()) if torch.is_tensor(s.d) else float(s.d)
            r = float(s.r)
            wedge_l = 0.05 * r; wedge_h = 0.18 * r
            for sign in (-1, +1):
                ax.plot([d - wedge_l, d + wedge_l], [sign * r, sign * r],
                        color="#666", lw=lw, solid_capstyle="round")
                ax.plot([d, d], [sign * r, sign * (r + wedge_h)],
                        color="#666", lw=lw, solid_capstyle="round")
        else:
            r_eff = float(s.draw_r())
            r = torch.linspace(-r_eff, r_eff, 256, device=s.device)
            z = s.surface_with_offset(r, torch.zeros(len(r), device=s.device),
                                      valid_check=False).cpu().detach().numpy()
            ax.plot(z, r.cpu().detach().numpy(), color=color, lw=lw, solid_capstyle="round")
    surfs = lens.surfaces
    for i in range(len(surfs) - 1):
        s_prev = surfs[i]
        if not (hasattr(s_prev, "mat2") and float(s_prev.mat2.n) > 1.1):
            continue
        s_next = surfs[i + 1]
        r_prev = float(s_prev.draw_r()); r_next = float(s_next.draw_r())
        sag_prev = s_prev.surface_with_offset(
            torch.tensor(r_prev, device=s_prev.device),
            torch.tensor(0.0, device=s_prev.device), valid_check=False).item()
        sag_next = s_next.surface_with_offset(
            torch.tensor(r_next, device=s_next.device),
            torch.tensor(0.0, device=s_next.device), valid_check=False).item()
        for sign in (-1, +1):
            ax.plot([sag_prev, sag_next], [sign * r_prev, sign * r_next],
                    color=color, lw=lw, solid_capstyle="round")
    d_sensor = float(lens.d_sensor.item()) if torch.is_tensor(lens.d_sensor) else float(lens.d_sensor)
    r_sensor = float(lens.r_sensor)
    ax.plot([d_sensor, d_sensor], [-r_sensor, r_sensor],
            color=SENSOR, lw=LINEWIDTH_SENSOR, solid_capstyle="round")


def hfov_deg_for(lens):
    if hasattr(lens, "rfov") and lens.rfov is not None:
        try: return float(np.rad2deg(float(lens.rfov)))
        except Exception: pass
    half_sensor = (float(lens.sensor_size[0]) / 2.0
                   if hasattr(lens, "sensor_size") else float(lens.r_sensor))
    efl = float(getattr(lens, "foclen", 50.0))
    return math.degrees(math.atan(half_sensor / abs(efl)))


def draw_filled_bundles(lens, ax):
    hfov = hfov_deg_for(lens)
    for fov_frac, color in zip([0.0, 0.5, 1.0], FIELD_COLORS):
        try:
            ray = lens.sample_point_source_2D(
                fov=float(fov_frac * hfov), depth=RAY_DEPTH_MM,
                num_rays=FILLED_BUNDLE_NUM_RAYS, wvln=RAY_WVLN_UM)
            ray.prop_to(-1.0)
            _, rec = lens.trace2sensor(ray=ray, record=True)
            lens.draw_ray_2d_bundle(rec, ax=ax, fig=ax.figure,
                                    color=color, alpha=FILLED_BUNDLE_ALPHA)
        except Exception as e:
            print(f"    filled-bundle draw failed for fov={fov_frac}: {e}")


def render_lens_panel(ax, lens, fill_color):
    draw_filled_bundles(lens, ax)
    for z_poly, x_poly in lens_fill_polygons(lens):
        ax.fill(z_poly, x_poly, color=fill_color, alpha=0.78,
                linewidth=0.0, zorder=3)
    draw_lens_clean(lens, ax)
    ax.set_aspect("equal"); ax.set_axis_off()


# ---- Spot diagram + per-wavelength PSF ----
def compute_spot_endpoints_3wvln(lens, p_problem, num_rays=512):
    """Returns dict {wvln: (F=3, num_rays, 2)} sensor-plane (x, y) in mm.
    Invalid rays are NaN."""
    from optical_rl.qd.lensmega.problem.tracing import (
        disk_samples, trace_fields_batch,
    )
    device = "cpu"
    trace_dtype = lens.surfaces[
        p_problem.free_curvature_indices[0]
    ].c.dtype
    aperture = torch.from_numpy(disk_samples(num_rays)).to(
        dtype=trace_dtype, device=device,
    )
    with torch.no_grad():
        pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
    aperture_xy = aperture * float(pupilr)
    pupil_z = float(pupilz)

    template_sensor_half = (lens.sensor_size[0] / 2.0
                            if hasattr(lens, "sensor_size")
                            else float(lens.r_sensor))
    efl = float(getattr(lens, "foclen", float(p_problem.focal_length_target_mm)))
    hfov_deg = math.degrees(math.atan(template_sensor_half / abs(efl)))
    fields = ((0.0, 0.0), (0.0, 0.5 * hfov_deg), (0.0, hfov_deg))

    out = {}
    for w in p_problem.fitness_wavelengths_um:
        ray = trace_fields_batch(
            lens, fields, aperture_xy=aperture_xy, pupil_z=pupil_z,
            object_depth_mm=p_problem.object_depth_mm,
            wavelength_um=float(w), device=device, dtype=trace_dtype,
        )
        valid = ray.is_valid.detach().cpu().numpy()
        xy = ray.o[..., :2].detach().cpu().numpy()
        out[float(w)] = np.where(valid[..., None], xy, np.nan)
    return out


def render_spot_panel(ax, spot_xy_per_wvln, *, label_um=True):
    """Spot diagram, on-axis field, R/G/B overlaid, centroid-aligned."""
    field_idx = 0
    max_extent = 0.0
    pts_per_wvln = {}
    for w, xy_F in spot_xy_per_wvln.items():
        pts = xy_F[field_idx]
        valid = ~np.isnan(pts).any(axis=-1)
        pts = pts[valid]
        if len(pts) == 0:
            continue
        c = pts.mean(axis=0)
        pts = pts - c  # centroid-align
        pts_per_wvln[w] = pts
        max_extent = max(max_extent, float(np.abs(pts).max()))
    # Plot from blue→green→red so red is on top
    for w in sorted(pts_per_wvln.keys()):
        pts = pts_per_wvln[w]
        ax.scatter(pts[:, 0] * 1000, pts[:, 1] * 1000,  # mm -> um
                   s=3, alpha=0.55, color=WVLN_HEX[w],
                   label=f"{int(round(w*1000))} nm",
                   linewidths=0.0)
    if max_extent > 0:
        lim = max_extent * 1000 * 1.15
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [μm]", fontsize=8); ax.set_ylabel("y [μm]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.85,
              borderpad=0.3, handletextpad=0.3, markerscale=2)
    ax.set_title("On-axis spot, R/G/B", fontsize=9, pad=2)
    ax.grid(True, lw=0.3, alpha=0.4)


# ---- Seidel bar chart ----
def render_seidel_panel(ax, design_metrics, fill_color):
    """5 bars: |S1|..|S5|. design_metrics = dict from refined json or
    spotlights.json containing seidel_S1..S5."""
    keys = ["seidel_S1", "seidel_S2", "seidel_S3", "seidel_S4", "seidel_S5"]
    labels = ["S₁\n(spherical)", "S₂\n(coma)", "S₃\n(astig)",
              "S₄\n(field curv)", "S₅\n(distortion)"]
    vals = [abs(float(design_metrics[k])) for k in keys]
    xs = np.arange(5)
    bars = ax.bar(xs, vals, color=fill_color, edgecolor="#333", linewidth=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_yscale("log")
    ax.grid(True, axis="y", lw=0.3, alpha=0.4, which="both")
    ax.set_title("|Seidel coefficients|", fontsize=9, pad=2)
    # Annotate each bar with value
    for x, v in zip(xs, vals):
        ax.text(x, v, f"{v:.1e}", ha="center", va="bottom",
                fontsize=6, rotation=0)


# ---- Image render (chromatic blur via per-wvln PSF kernels) ----
def make_psf_kernel_from_spot(pts_xy_mm, kernel_pix=51, mm_per_pix=0.005):
    """pts_xy_mm: (N, 2) sensor-plane spot points (mm), centroid-aligned.
    Returns a normalised 2D kernel of shape (kernel_pix, kernel_pix)."""
    valid = ~np.isnan(pts_xy_mm).any(axis=-1)
    pts = pts_xy_mm[valid]
    if len(pts) == 0:
        k = np.zeros((kernel_pix, kernel_pix), dtype=np.float64)
        k[kernel_pix//2, kernel_pix//2] = 1.0
        return k
    pts = pts - pts.mean(axis=0)
    # Discretise into a histogram on a kernel grid
    half = kernel_pix // 2
    extent = half * mm_per_pix
    hist, _, _ = np.histogram2d(
        pts[:, 1], pts[:, 0],          # y -> rows, x -> cols
        bins=[kernel_pix, kernel_pix],
        range=[[-extent, extent], [-extent, extent]],
    )
    s = hist.sum()
    if s <= 0:
        k = np.zeros_like(hist)
        k[kernel_pix//2, kernel_pix//2] = 1.0
        return k
    return hist / s


def render_image_panel(ax, spot_xy_per_wvln, test_image_rgb_uint8):
    """Convolve each RGB channel of the test image with a kernel built from
    the on-axis spot at the matching wavelength, then display the result."""
    # Build per-wvln kernel from on-axis spot
    field_idx = 0
    # Choose sensible kernel scale: ~5x the largest spot radius across wvlns
    all_pts = []
    for w, xy_F in spot_xy_per_wvln.items():
        pts = xy_F[field_idx]
        valid = ~np.isnan(pts).any(axis=-1)
        pts = pts[valid]
        if len(pts):
            pts = pts - pts.mean(axis=0)
            all_pts.append(pts)
    if not all_pts:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no valid spots", ha="center", va="center",
                transform=ax.transAxes)
        return
    catted = np.concatenate(all_pts, axis=0)
    spot_extent = float(np.percentile(np.linalg.norm(catted, axis=-1), 99))

    # Want kernel covering ~3x spot extent. Pick mm_per_pix so kernel_pix ≈ 51.
    kernel_pix = 51
    mm_per_pix = max(spot_extent * 3.0 / (kernel_pix // 2), 1e-4)

    kernels_rgb = {}  # in order R, G, B
    chan_to_wvln = {"R": 0.656, "G": 0.587, "B": 0.486}
    for chan, w in chan_to_wvln.items():
        if w not in spot_xy_per_wvln:
            kernels_rgb[chan] = None
            continue
        pts = spot_xy_per_wvln[w][field_idx]
        kernels_rgb[chan] = make_psf_kernel_from_spot(
            pts, kernel_pix=kernel_pix, mm_per_pix=mm_per_pix,
        )

    # Convolve each channel and recombine
    img = np.asarray(test_image_rgb_uint8).astype(np.float64) / 255.0
    out = np.empty_like(img)
    for ci, chan in enumerate(("R", "G", "B")):
        k = kernels_rgb[chan]
        if k is None:
            out[..., ci] = img[..., ci]
            continue
        out[..., ci] = nd_convolve(img[..., ci], k, mode="reflect")
    out = np.clip(out, 0.0, 1.0)
    ax.imshow(out, interpolation="nearest")
    ax.set_axis_off()
    # Annotate the underlying spot extent so the reader knows the scale
    ax.set_title(f"Render through this lens (PSF-convolved)\n"
                 f"on-axis spot extent ≈ {spot_extent*1000:.0f} μm "
                 f"(p99 radius)", fontsize=8, pad=4)


# ---- Octuplet theta lookup ----
def _octuplet_label_to_theta():
    """Build a {label: theta} map for octuplet by joining spotlights.json
    against the HL refined-design JSON via raw_fitness."""
    OUT = ROOT / "results" / "figure_archive_demo_octuplet"
    HL = ROOT / "results" / "lensmega" / "octuplet_apsc_2cement_v2_seed42_20260506_040025"
    spots = json.loads((OUT / "spotlights.json").read_text())
    hl_ref = json.loads((HL / "refined_top500_seed42_hl.json").read_text())["designs"]
    cells = json.loads((HL / "archive_cells.json").read_text())["cells"]
    fit_to_measures = {round(c["fitness"], 8): c["measures"] for c in cells}
    out = {}
    for sp in spots:
        best = None
        for d in hl_ref:
            f = round(d["raw_fitness"], 8)
            if f not in fit_to_measures: continue
            m = fit_to_measures[f]
            if (abs(m[0]-sp["S2"]) < 1e-7 and abs(m[1]-sp["S3"]) < 1e-7
                    and abs(d["rms_um"]-sp["rms_um"]) < 0.05):
                best = d; break
        if best is None:
            cands = []
            for d in hl_ref:
                f = round(d["raw_fitness"], 8)
                if f not in fit_to_measures: continue
                m = fit_to_measures[f]
                if abs(m[0]-sp["S2"]) < 1e-7 and abs(m[1]-sp["S3"]) < 1e-7:
                    cands.append((abs(d["rms_um"]-sp["rms_um"]), d))
            if cands:
                cands.sort(key=lambda t: t[0])
                best = cands[0][1]
        if best is not None:
            out[sp["label"]] = (np.asarray(best["theta"], dtype=np.float64),
                                best["metrics"])
    return out


def _sixlet_label_to_theta_and_metrics():
    """Sixlet spotlights.json has theta. We also need Seidel metrics — those
    are in the HL refined json keyed by raw_fitness."""
    OUT = ROOT / "results" / "figure_archive_demo"
    HL = ROOT / "results" / "lensmega" / "sixlet_apsc_discovery_v2_seed42_20260506_034601"
    spots = json.loads((OUT / "spotlights.json").read_text())
    hl_ref = json.loads((HL / "refined_top500_seed42_hl.json").read_text())["designs"]
    out = {}
    for sp in spots:
        theta = np.asarray(sp["refined_theta"], dtype=np.float64)
        # Find matching design by theta L2-distance (cheap, exact match expected)
        best, best_d = None, float("inf")
        for d in hl_ref:
            t = np.asarray(d["theta"], dtype=np.float64)
            if t.shape != theta.shape: continue
            dd = float(np.linalg.norm(t - theta))
            if dd < best_d:
                best_d, best = dd, d
        if best is None or best_d > 1e-3:
            print(f"  ! sixlet '{sp['label']}': no theta match (best d={best_d:.4e}); "
                  "Seidel metrics will be missing")
            metrics = None
        else:
            metrics = best["metrics"]
        out[sp["label"]] = (theta, metrics)
    return out


def render_one_grid(*, sp, theta, metrics, lens, p_problem, test_image,
                    out_path, fill_color, title_color, topology):
    """Compose the 4-panel figure for a single spotlight."""
    spot_xy = compute_spot_endpoints_3wvln(lens, p_problem, num_rays=512)

    # Layout: left big lens (col 0, rows 0-1), right column = Seidel/spot/img
    fig = plt.figure(figsize=(11.5, 6.4), facecolor="white")
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        width_ratios=[1.6, 1.0],
        height_ratios=[1.0, 1.0, 1.4],
        hspace=0.35, wspace=0.18,
    )
    ax_lens = fig.add_subplot(gs[:, 0])
    ax_seidel = fig.add_subplot(gs[0, 1])
    ax_spot   = fig.add_subplot(gs[1, 1])
    ax_img    = fig.add_subplot(gs[2, 1])

    render_lens_panel(ax_lens, lens, fill_color=fill_color)
    if metrics is not None:
        render_seidel_panel(ax_seidel, metrics, fill_color=fill_color)
    else:
        ax_seidel.set_axis_off()
        ax_seidel.text(0.5, 0.5, "Seidel metrics N/A",
                       ha="center", va="center", transform=ax_seidel.transAxes)
    render_spot_panel(ax_spot, spot_xy)
    render_image_panel(ax_img, spot_xy, test_image)

    # Per-wavelength RMS strip in the suptitle
    rms_per_wvln = {}
    for w, xy_F in spot_xy.items():
        # Field-averaged RMS in um (matches the verifier)
        per_field_rms_um = []
        for f_i in range(xy_F.shape[0]):
            pts = xy_F[f_i]
            valid = ~np.isnan(pts).any(axis=-1)
            pp = pts[valid]
            if len(pp) > 0:
                c = pp.mean(axis=0)
                per_field_rms_um.append(
                    float(np.sqrt(((pp - c) ** 2).sum(axis=-1).mean())) * 1000.0
                )
        rms_per_wvln[w] = (np.mean(per_field_rms_um) if per_field_rms_um
                           else float("nan"))
    rms_strip = "    ".join(
        f"{int(round(w*1000))} nm = {rms_per_wvln[w]:5.1f} μm"
        for w in sorted(rms_per_wvln)
    )
    fig.suptitle(
        f"{topology.upper()} — {sp['label'].upper()}    "
        f"reported RMS = {sp['rms_um']:.1f} μm    [{rms_strip}]    "
        f"{'in SA+GA hull' if sp.get('in_saga_hull') else 'HL-only'}",
        fontsize=11, color=title_color, fontweight="bold", y=0.995,
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_topology(topology, test_image):
    print(f"\n=== {topology.upper()} ===")
    if topology == "sixlet":
        out_dir = ROOT / "results" / "figure_archive_demo"
        label_to_data = _sixlet_label_to_theta_and_metrics()
    elif topology == "octuplet":
        out_dir = ROOT / "results" / "figure_archive_demo_octuplet"
        label_to_data = _octuplet_label_to_theta()
    else:
        raise ValueError(topology)

    spots = json.loads((out_dir / "spotlights.json").read_text())
    grids_dir = out_dir / "spotlight_grids"
    grids_dir.mkdir(exist_ok=True)

    template, solver, prob_cfg, lens_path, p_problem = (
        build_lens_template_and_solver(SYSTEMS[topology]["cfg"])
    )

    for sp in spots:
        label = sp["label"]
        if label not in label_to_data:
            print(f"  [{label}] no theta available; skipping")
            continue
        theta, metrics = label_to_data[label]
        is_shared = bool(sp.get("in_saga_hull"))
        fill = RED_FILL if is_shared else BLUE_FILL
        title_color = "#9A2A2A" if is_shared else "#1A4F8B"
        print(f"  [{label}] rendering grid (RMS={sp['rms_um']:.1f}μm, "
              f"{'shared' if is_shared else 'HL-only'})")
        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path,
                                np.asarray(theta, dtype=np.float64),
                                template, solver)
        except Exception as e:
            print(f"    FAILED to build lens: {e}")
            continue
        out_path = grids_dir / f"{topology}_{label}_grid.png"
        render_one_grid(
            sp=sp, theta=theta, metrics=metrics, lens=lens,
            p_problem=p_problem, test_image=test_image,
            out_path=out_path, fill_color=fill,
            title_color=title_color, topology=topology,
        )
        print(f"    -> {out_path}")


def main():
    test_image_path = (
        ROOT / "results" / "figure_archive_demo" / "test_images" / "kodim23.png"
    )
    test_image = np.asarray(Image.open(test_image_path).convert("RGB"),
                            dtype=np.uint8)
    print(f"Loaded test image: {test_image_path} shape={test_image.shape}")

    render_topology("sixlet", test_image)
    render_topology("octuplet", test_image)


if __name__ == "__main__":
    main()
