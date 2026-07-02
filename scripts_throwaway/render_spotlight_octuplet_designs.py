#!/usr/bin/env python
"""Render the 4 spotlight octuplet designs from the v16oct main panel as
separate beautiful lens cross-sections, following the v16 rendering recipe at
figures/method_library_data/RENDERING_RECIPE.md.

Inputs:
  results/figure_archive_demo_octuplet/spotlights.json  (4 spotlights)
  HL refined designs JSON                               (full theta vectors)
  HL archive_cells.json                                 (S2/S3 lookup)

Outputs:
  results/figure_archive_demo_octuplet/spot_<label>_<ray_style>.png
"""
from __future__ import annotations
import json, math, os, sys
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

# ---------- Recipe constants (verbatim from RENDERING_RECIPE.md §5/§6) ----------
BLUE_FILL = "#7BAFD4"
RED_FILL  = "#E07B7B"
OUTLINE   = "#1F1F1F"
SENSOR    = "#000000"
LINEWIDTH_LENS   = 1.2
LINEWIDTH_SENSOR = 1.6
FILL_NUM_SAMPLES = 192

FIELD_COLORS = ["#1565C0", "#2E7D32", "#C62828"]
RAY_LW = 0.55
RAY_ALPHA = 0.85
RAY_NUM_PER_FIELD = 11
RAY_WVLN_UM = 0.587
RAY_DEPTH_MM = -20000.0

FILLED_BUNDLE_ALPHA = 0.32
FILLED_BUNDLE_NUM_RAYS = 9

# Glass-fill convention: HL-only discoveries → blue; shared/global basin → red,
# matching the "anchor (blue) vs incumbent (red)" convention from the recipe.
SPOTLIGHT_FILL = {
    "out1":   "#7BAFD4",   # blue — HL-only discovery
    "out2":   "#7BAFD4",   # blue — HL-only discovery
    "out3":   "#7BAFD4",   # blue — HL-only discovery
    "shared": "#E07B7B",   # red — common to HL and SA+GA
}


# ---------- Recipe rendering primitives (verbatim) ----------
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


def draw_line_rays(lens, ax):
    hfov = hfov_deg_for(lens)
    for fov_frac, color in zip([0.0, 0.5, 1.0], FIELD_COLORS):
        try:
            ray = lens.sample_point_source_2D(
                fov=float(fov_frac * hfov), depth=RAY_DEPTH_MM,
                num_rays=RAY_NUM_PER_FIELD, wvln=RAY_WVLN_UM)
            ray.prop_to(-1.0)
            _, rec = lens.trace2sensor(ray=ray, record=True)
            n_before = len(ax.lines)
            lens.draw_ray_2d(rec, ax=ax, fig=ax.figure, color=color)
            for line in ax.lines[n_before:]:
                line.set_linewidth(RAY_LW); line.set_alpha(RAY_ALPHA)
        except Exception as e:
            print(f"    line-ray draw failed for fov={fov_frac}: {e}")


def draw_filled_bundles(lens, ax):
    """Recipe Style C — filled envelope between extreme marginals per field."""
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


def render_one(ax, lens, fill_color, *, ray_style="filled"):
    if ray_style == "lines":
        draw_line_rays(lens, ax)
    elif ray_style == "filled":
        draw_filled_bundles(lens, ax)
    fill_alpha = (1.0 if ray_style == "none"
                  else 0.85 if ray_style == "lines"
                  else 0.78)
    for z_poly, x_poly in lens_fill_polygons(lens):
        ax.fill(z_poly, x_poly, color=fill_color, alpha=fill_alpha,
                linewidth=0.0, zorder=3)
    draw_lens_clean(lens, ax)
    ax.set_aspect("equal"); ax.set_axis_off()


# ---------- Match each spotlight to its full-theta HL refined design ----------
def match_spotlights_to_theta(spotlights, hl_ref, fit_to_measures, tol_S=1e-7, tol_rms=0.05):
    """For each spotlight (with S2/S3/rms_um), find the HL refined design with
    matching archive-cell measures and rms. Returns list of (spotlight, design)."""
    matched = []
    for sp in spotlights:
        best = None
        for d in hl_ref:
            f = round(d["raw_fitness"], 8)
            if f not in fit_to_measures:
                continue
            m = fit_to_measures[f]
            ds2 = abs(m[0] - sp["S2"])
            ds3 = abs(m[1] - sp["S3"])
            drms = abs(d["rms_um"] - sp["rms_um"])
            if ds2 < tol_S and ds3 < tol_S and drms < tol_rms:
                best = d; break
        if best is None:
            print(f"  ! No exact match for spotlight {sp['label']} "
                  f"(S2={sp['S2']:+.5f} S3={sp['S3']:+.5f} rms={sp['rms_um']:.2f}μm) — "
                  f"falling back to closest by rms")
            # Fallback: closest by rms among designs with same archive measures
            cands = []
            for d in hl_ref:
                f = round(d["raw_fitness"], 8)
                if f not in fit_to_measures: continue
                m = fit_to_measures[f]
                if abs(m[0]-sp["S2"]) < tol_S and abs(m[1]-sp["S3"]) < tol_S:
                    cands.append((abs(d["rms_um"]-sp["rms_um"]), d))
            if cands:
                cands.sort(key=lambda t: t[0])
                best = cands[0][1]
        if best is None:
            print(f"  ! STILL no match for {sp['label']}, skipping")
            continue
        matched.append((sp, best))
    return matched


# ---------- Main ----------
def main():
    OUT = ROOT / "results" / "figure_archive_demo_octuplet"
    HL = ROOT / "results" / "lensmega" / "octuplet_apsc_2cement_v2_seed42_20260506_040025"

    spotlights = json.loads((OUT / "spotlights.json").read_text())
    hl_ref     = json.loads((HL / "refined_top500_seed42_hl.json").read_text())["designs"]
    cells      = json.loads((HL / "archive_cells.json").read_text())["cells"]
    fit_to_measures = {round(c["fitness"], 8): c["measures"] for c in cells}

    matched = match_spotlights_to_theta(spotlights, hl_ref, fit_to_measures)
    print(f"Matched {len(matched)}/{len(spotlights)} spotlights to theta vectors")

    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["octuplet"]["cfg"]
    )

    for sp, d in matched:
        label = sp["label"]
        fill = SPOTLIGHT_FILL.get(label, BLUE_FILL)
        rms = float(d["rms_um"])
        print(f"\nRendering spotlight '{label}': RMS={rms:.2f} μm  fill={fill}")

        lens = make_geolens(prob_cfg, p_problem, lens_path,
                            np.asarray(d["theta"], dtype=np.float64),
                            template, solver)

        # Larger landscape figure to show the optical stack clearly.
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.6), facecolor="white")
        ax.set_facecolor("white")
        render_one(ax, lens, fill_color=fill, ray_style="filled")

        # Title with label + RMS. Title color follows the glass-fill convention:
        # blue for HL-only discoveries, deep red for the shared basin.
        title_color = "#9A2A2A" if label == "shared" else "#1A4F8B"
        ax.set_title(
            f"{label.upper()}    RMS = {rms:.1f} μm",
            fontsize=12, color=title_color, fontweight="bold", pad=4,
        )

        out_png = OUT / f"spot_{label}_bundles.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  → {out_png}")


if __name__ == "__main__":
    main()
