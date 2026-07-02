#!/usr/bin/env python
"""Classical transverse ray-fan plots for the 3+3 diverse picks.

For each design, plot Δy_image vs normalized pupil-y (tangential / Y-fan)
and Δx_image vs normalized pupil-x (sagittal / X-fan), at three field
angles (on-axis, mid-field, full-field), with three wavelengths overlaid.

Δ is computed against the chief ray (pupil-center) at the same wavelength
and field, so the reference is achromatic per (field, wavelength) and the
plots show standard "transverse ray aberration" curves.

Layout: 6-row x 6-col grid of small panels.
  rows  = 6 designs (sixlet #1, #2, #3, octuplet #1, #2, #3)
  cols  = 3 fields x 2 directions (F0-tan, F0-sag, F1-tan, F1-sag, F2-tan, F2-sag)

Output: figures/fig5_three_diverse_rayfans.png
        figures/fig5_data/three_diverse_rayfan_data.npz  (raw curves)
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
DATA = ROOT / "figures" / "fig5_data"
OUT_PNG = ROOT / "figures" / "fig5_three_diverse_rayfans.png"
OUT_NPZ = DATA / "three_diverse_rayfan_data.npz"

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)
from deeplens.optics.light.ray import Ray

WAVELENGTHS_UM = (0.486, 0.587, 0.656)
WAVE_COLORS = {0.486: "#1f77b4", 0.587: "#2ca02c", 0.656: "#d62728"}
WAVE_LABELS = {0.486: "486 nm", 0.587: "587 nm", 0.656: "656 nm"}
N_PUPIL = 41  # samples along each fan
FIELD_FRACS = (0.0, 0.5, 1.0)  # on-axis, mid, full


def trace_pupil_sweep(lens, field_y_deg, pupil_z, pupil_r, sweep_axis,
                      object_depth_mm, wavelength_um, n=N_PUPIL):
    """Trace a fan of rays for one (field, wavelength, fan-direction).

    sweep_axis = 'y' for tangential (meridional), 'x' for sagittal.
    Returns (pupil_norm, dx, dy, valid_mask, x_image_chief, y_image_chief)
    where dx/dy are transverse aberrations in mm relative to the chief
    ray (pupil center) at the same field and wavelength.
    """
    device = "cpu"
    dtype = torch.float64

    pupil_norm = np.linspace(-1.0, 1.0, n)
    aperture_xy = np.zeros((n, 2), dtype=np.float64)
    if sweep_axis == "y":
        aperture_xy[:, 1] = pupil_norm * pupil_r
    elif sweep_axis == "x":
        aperture_xy[:, 0] = pupil_norm * pupil_r
    else:
        raise ValueError(sweep_axis)

    # Object point at (0, depth*tan(field_y), depth)
    z_obj = float(object_depth_mm)
    y_obj = z_obj * math.tan(math.radians(float(field_y_deg)))
    x_obj = 0.0

    origins = torch.zeros((n, 3), dtype=dtype, device=device)
    origins[:, 0] = x_obj
    origins[:, 1] = y_obj
    origins[:, 2] = z_obj
    targets = torch.zeros((n, 3), dtype=dtype, device=device)
    targets[:, 0] = torch.from_numpy(aperture_xy[:, 0]).to(dtype=dtype, device=device)
    targets[:, 1] = torch.from_numpy(aperture_xy[:, 1]).to(dtype=dtype, device=device)
    targets[:, 2] = pupil_z
    ray = Ray(origins, targets - origins, wvln=wavelength_um, device=device)
    out = lens.trace2sensor(ray)
    valid = out.is_valid.detach().cpu().numpy()
    xy = out.o[..., :2].detach().cpu().numpy()  # (n, 2)
    # Chief ray = ray with pupil_norm == 0 (the central index)
    chief_idx = int(np.argmin(np.abs(pupil_norm)))
    if not valid[chief_idx]:
        # Fallback: use the centroid of valid rays as reference
        if valid.any():
            ref = xy[valid].mean(axis=0)
        else:
            ref = np.array([np.nan, np.nan])
    else:
        ref = xy[chief_idx]
    dx = xy[:, 0] - ref[0]
    dy = xy[:, 1] - ref[1]
    dx = np.where(valid, dx, np.nan)
    dy = np.where(valid, dy, np.nan)
    return pupil_norm, dx, dy, valid, ref[0], ref[1]


def build_lens_for_design(sys_name, theta):
    info = SYSTEMS[sys_name]
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(info["cfg"])
    lens = make_geolens(prob_cfg, p_problem, lens_path, np.asarray(theta, dtype=np.float64),
                        template, solver)
    return lens, p_problem


def hfov_deg(lens):
    half_sensor = float(lens.sensor_size[0]) / 2.0 if hasattr(lens, "sensor_size") else float(lens.r_sensor)
    efl = float(getattr(lens, "foclen", 50.0))
    return math.degrees(math.atan(half_sensor / abs(efl)))


def main():
    picks = json.loads((DATA / "three_diverse_below_thr.json").read_text())

    designs_in_order = []
    for sys_name in ("sixlet", "octuplet"):
        for d in picks[sys_name]["designs"]:
            designs_in_order.append((sys_name, d))

    n_designs = len(designs_in_order)  # 6
    fig, axes = plt.subplots(
        n_designs, 6, figsize=(13, 1.7 * n_designs),
        sharex=True,
    )
    if n_designs == 1:
        axes = axes[None, :]

    # NPZ collector
    npz_data = {}

    # Determine a per-design symmetric y-limit by data, but use a shared per-design scale
    # so the three field columns are visually comparable.
    for row, (sys_name, design) in enumerate(designs_in_order):
        lens, p_problem = build_lens_for_design(sys_name, design["theta"])
        with torch.no_grad():
            pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
        pupil_z = float(pupilz)
        pupil_r = float(pupilr)
        H = hfov_deg(lens)

        # Per-design max |aberration| across all curves (for ylim)
        all_abs = []
        per_field_curves = {}  # field_idx -> list of (axis, wvln, dx, dy)
        for fi, frac in enumerate(FIELD_FRACS):
            field_y = frac * H
            curves = {"tan": {}, "sag": {}}
            for w in WAVELENGTHS_UM:
                p_y, dx_y, dy_y, _, _, _ = trace_pupil_sweep(
                    lens, field_y, pupil_z, pupil_r, "y",
                    p_problem.object_depth_mm, w,
                )
                curves["tan"][w] = (p_y, dy_y)
                p_x, dx_x, dy_x, _, _, _ = trace_pupil_sweep(
                    lens, field_y, pupil_z, pupil_r, "x",
                    p_problem.object_depth_mm, w,
                )
                curves["sag"][w] = (p_x, dx_x)
                # Collect for ylim (in mm, but plot in um => x1000)
                for arr in (dy_y, dx_x):
                    finite = arr[np.isfinite(arr)]
                    if finite.size:
                        all_abs.append(np.abs(finite).max())
            per_field_curves[fi] = curves
            npz_data[f"{sys_name}_{design['rank']}_F{fi}_tan_pupilnorm"] = curves["tan"][0.587][0]
            for w in WAVELENGTHS_UM:
                npz_data[f"{sys_name}_{design['rank']}_F{fi}_tan_dy_um_w{int(w*1000)}"] = curves["tan"][w][1] * 1000.0
                npz_data[f"{sys_name}_{design['rank']}_F{fi}_sag_dx_um_w{int(w*1000)}"] = curves["sag"][w][1] * 1000.0

        ylim_um = max(all_abs) * 1000.0 * 1.1 if all_abs else 50.0
        ylim_um = max(20.0, min(ylim_um, 500.0))  # clamp to reasonable range

        # Plot per (field, direction)
        for fi, frac in enumerate(FIELD_FRACS):
            field_y = frac * H
            for di, (direction, ax_offset) in enumerate([("tan", 0), ("sag", 1)]):
                col = fi * 2 + di
                ax = axes[row, col]
                for w in WAVELENGTHS_UM:
                    p, dval = per_field_curves[fi][direction][w]
                    ax.plot(p, dval * 1000.0, color=WAVE_COLORS[w], lw=1.0,
                            label=WAVE_LABELS[w] if (row == 0 and col == 0) else None)
                ax.axhline(0, color="grey", lw=0.4, alpha=0.6)
                ax.axvline(0, color="grey", lw=0.4, alpha=0.6)
                ax.set_ylim(-ylim_um, ylim_um)
                ax.set_xlim(-1.05, 1.05)
                ax.grid(alpha=0.3, lw=0.4)

                if row == 0:
                    ax.set_title(
                        f"F{fi} ({frac:.1f} HFOV)\n{'tangential (Δy)' if direction=='tan' else 'sagittal (Δx)'}",
                        fontsize=8,
                    )
                if col == 0:
                    ax.set_ylabel(
                        f"{sys_name} #{design['rank']+1}\nid={design['design_id']}\n"
                        f"RMS={design['refined_RMS_um']:.1f} um\n"
                        f"Δ_image (um)",
                        fontsize=7,
                    )
                if row == n_designs - 1:
                    ax.set_xlabel("normalized pupil", fontsize=7)
                ax.tick_params(axis="both", labelsize=6)

    # Single global legend (top of figure)
    handles = [plt.Line2D([0], [0], color=WAVE_COLORS[w], lw=2, label=WAVE_LABELS[w])
               for w in WAVELENGTHS_UM]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=9,
               bbox_to_anchor=(0.99, 0.998))
    fig.suptitle(
        "Transverse ray-aberration plots ('ray fans') for 3 diverse refined designs per system\n"
        "rows = designs;  cols = 3 fields x {tangential, sagittal};  colors = 3 wavelengths",
        fontsize=11, y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    np.savez(OUT_NPZ, **npz_data)
    print(f"wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
