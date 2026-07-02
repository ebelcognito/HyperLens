#!/usr/bin/env python
"""Render 3 separate panels per spotlight (replaces the cluttered 4-panel grid):

  1. <topology>_<label>_lens.png      Lens cross-section (Style C bundles)
  2. <topology>_<label>_seidel.png    Two-bar |S2|/|S3| chart, minimal axes
  3. <topology>_<label>_robust.png    Stability ribbon (single joint metric):
                                       min/max-across-wvln RMS vs field, with
                                       the per-field median as a centre line.

The ribbon's vertical position shows spot performance (lower=sharper), and its
thickness + slope show stability (thin+flat = robust across wavelengths and
across the field).

Glass colour convention: shared (in_saga_hull=True) → red; HL-only → blue.
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
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)

# Recipe constants
BLUE_FILL = "#7BAFD4"
RED_FILL  = "#E07B7B"
OUTLINE   = "#1F1F1F"
SENSOR    = "#000000"
LINEWIDTH_LENS   = 1.2
LINEWIDTH_SENSOR = 1.6
FILL_NUM_SAMPLES = 192

FIELD_COLORS = ["#1565C0", "#2E7D32", "#C62828"]
RAY_WVLN_UM = 0.587
RAY_DEPTH_MM = -20000.0
FILLED_BUNDLE_ALPHA = 0.32
FILLED_BUNDLE_NUM_RAYS = 9


# ---- Recipe primitives (verbatim) ----
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


# ---- Per-field per-wvln RMS (for the stability ribbon) ----
def per_field_per_wvln_rms_um(lens, p_problem, n_fields=7, num_rays=256):
    """Returns (field_fracs, wvlns, rms_um) where rms_um[f, w] is the field-f
    RMS at wvln-w in micrometres. Field 0 = on-axis, 1 = full HFOV."""
    from optical_rl.qd.lensmega.problem.tracing import (
        disk_samples, trace_fields_batch, rms_spot_batch,
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
    field_fracs = np.linspace(0.0, 1.0, n_fields)
    fields = [(0.0, float(f * hfov_deg)) for f in field_fracs]

    wvlns = list(p_problem.fitness_wavelengths_um)
    rms_arr = np.zeros((n_fields, len(wvlns)), dtype=np.float64)
    for wi, w in enumerate(wvlns):
        ray = trace_fields_batch(
            lens, fields, aperture_xy=aperture_xy, pupil_z=pupil_z,
            object_depth_mm=p_problem.object_depth_mm,
            wavelength_um=float(w), device=device, dtype=trace_dtype,
        )
        rms, _ = rms_spot_batch(ray)
        rms_arr[:, wi] = rms.detach().cpu().numpy() * 1000.0  # mm -> um
    return field_fracs, wvlns, rms_arr


# ---- The three render functions ----
def render_lens(lens, fill_color, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.4), facecolor="white")
    ax.set_facecolor("white")
    draw_filled_bundles(lens, ax)
    for z_poly, x_poly in lens_fill_polygons(lens):
        ax.fill(z_poly, x_poly, color=fill_color, alpha=0.78,
                linewidth=0.0, zorder=3)
    draw_lens_clean(lens, ax)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_seidel_s2_s3(s2_val, s3_val, fill_color, out_path):
    """Two bars: |S2| and |S3|, archive-cell measures. Compact, bold axes."""
    vals = [abs(float(s2_val)), abs(float(s3_val))]
    labels = ["S₂", "S₃"]
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.0), facecolor="white")
    ax.bar([0, 1], vals, color=fill_color, edgecolor="#333",
           linewidth=0.8, width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=20, fontweight="bold")
    vmax = max(vals) * 1.15
    ax.set_ylim(0, vmax)
    ax.set_yticks([0, vmax / 2.0, vmax])
    if vmax >= 1e-3:
        scale, exp_lbl = 1e3, "×10⁻³"
    elif vmax >= 1e-6:
        scale, exp_lbl = 1e6, "×10⁻⁶"
    else:
        scale, exp_lbl = 1e9, "×10⁻⁹"
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v*scale:.3g}")
    )
    ax.set_ylabel(f"|S|  ({exp_lbl})", fontsize=15, fontweight="bold")
    ax.tick_params(axis="y", labelsize=14)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#444"); ax.spines["bottom"].set_linewidth(1.3)
    ax.spines["left"].set_color("#444"); ax.spines["left"].set_linewidth(1.3)
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_robust(field_fracs, wvlns, rms_arr, fill_color, out_path):
    """Single joint metric: ribbon = min/max-across-wavelengths RMS vs field,
    with the per-field median as a centre line. Compact, bold axes."""
    rms_arr = np.asarray(rms_arr)
    field_fracs = np.asarray(field_fracs)
    rms_min = rms_arr.min(axis=1)
    rms_max = rms_arr.max(axis=1)
    rms_mid = np.median(rms_arr, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.0), facecolor="white")
    ax.fill_between(field_fracs, rms_min, rms_max,
                    color=fill_color, alpha=0.45, linewidth=0.0)
    ax.plot(field_fracs, rms_mid, color=fill_color, lw=2.4,
            solid_capstyle="round")
    ax.plot(field_fracs, rms_min, color=fill_color, lw=0.8, alpha=0.7)
    ax.plot(field_fracs, rms_max, color=fill_color, lw=0.8, alpha=0.7)

    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["axis", "½", "edge"], fontsize=15, fontweight="bold")
    ax.set_xlabel("field", fontsize=15, fontweight="bold")
    ymax = float(rms_max.max()) * 1.15
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, ymax / 2.0, ymax])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_ylabel("RMS  [µm]", fontsize=15, fontweight="bold")
    ax.tick_params(axis="y", labelsize=14)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#444"); ax.spines["bottom"].set_linewidth(1.3)
    ax.spines["left"].set_color("#444"); ax.spines["left"].set_linewidth(1.3)
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---- Topology runners ----
def _octuplet_label_to_theta():
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
            out[sp["label"]] = np.asarray(best["theta"], dtype=np.float64)
    return out


def render_topology(topology):
    print(f"\n=== {topology.upper()} ===")
    if topology == "sixlet":
        out_dir = ROOT / "results" / "figure_archive_demo"
        spots = json.loads((out_dir / "spotlights.json").read_text())
        # sixlet stores theta directly
        label_to_theta = {
            sp["label"]: np.asarray(sp["refined_theta"], dtype=np.float64)
            for sp in spots
        }
    elif topology == "octuplet":
        out_dir = ROOT / "results" / "figure_archive_demo_octuplet"
        spots = json.loads((out_dir / "spotlights.json").read_text())
        label_to_theta = _octuplet_label_to_theta()
    else:
        raise ValueError(topology)

    panels_dir = out_dir / "spotlight_panels"
    panels_dir.mkdir(exist_ok=True)

    template, solver, prob_cfg, lens_path, p_problem = (
        build_lens_template_and_solver(SYSTEMS[topology]["cfg"])
    )

    for sp in spots:
        label = sp["label"]
        if label not in label_to_theta:
            print(f"  [{label}] no theta available; skipping")
            continue
        is_shared = bool(sp.get("in_saga_hull"))
        fill = RED_FILL if is_shared else BLUE_FILL

        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path,
                                label_to_theta[label],
                                template, solver)
        except Exception as e:
            print(f"  [{label}] FAILED to build lens: {e}")
            continue

        print(f"  [{label}] {'shared' if is_shared else 'HL-only'}  "
              f"S2={sp['S2']:+.5f} S3={sp['S3']:+.5f}")
        prefix = panels_dir / f"{topology}_{label}"
        render_lens(lens, fill, str(prefix) + "_lens.png")
        render_seidel_s2_s3(sp["S2"], sp["S3"], fill, str(prefix) + "_seidel.png")

        field_fracs, wvlns, rms_arr = per_field_per_wvln_rms_um(
            lens, p_problem, n_fields=7, num_rays=256,
        )
        render_robust(field_fracs, wvlns, rms_arr, fill,
                      str(prefix) + "_robust.png")
        print(f"    → {prefix}_{{lens,seidel,robust}}.png")


def main():
    render_topology("sixlet")
    render_topology("octuplet")


if __name__ == "__main__":
    main()
