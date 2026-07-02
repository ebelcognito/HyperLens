#!/usr/bin/env python
"""Polished render of the 7 sixlet method-library designs WITHOUT pruning.

Same workflow as method_library_render_unpruned_octuplet.py, but pulled
from sixlet_parents_and_incumbent.json and built with the sixlet config.

Outputs:
  figures/method_library_sixlet_unpruned.png                    no rays
  figures/method_library_sixlet_unpruned_with_rays.png          line rays
  figures/method_library_sixlet_unpruned_with_filled_rays.png   filled bundles
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
DATA_IN = ROOT / "figures" / "fig5_data" / "sixlet_parents_and_incumbent.json"
OUT_NO_RAYS     = ROOT / "figures" / "method_library_sixlet_unpruned.png"
OUT_LINE_RAYS   = ROOT / "figures" / "method_library_sixlet_unpruned_with_rays.png"
OUT_FILLED_RAYS = ROOT / "figures" / "method_library_sixlet_unpruned_with_filled_rays.png"

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)

# ---- styling (same as octuplet for consistency across figures) ----
BLUE_FILL = "#7BAFD4"
RED_FILL  = "#E07B7B"
OUTLINE   = "#1F1F1F"
SENSOR    = "#000000"
LINEWIDTH_LENS = 1.2
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


# --------------------------------------------------------------------
# Helpers (identical to octuplet version)
# --------------------------------------------------------------------

def lens_fill_polygons(lens, num_samples=FILL_NUM_SAMPLES):
    polys = []
    for i in range(len(lens.surfaces) - 1):
        s0 = lens.surfaces[i]
        s1 = lens.surfaces[i + 1]
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
        z_poly = np.concatenate([z0, z1[::-1]], axis=0)
        x_poly = np.concatenate([rs0, rs1[::-1]], axis=0)
        polys.append((z_poly, x_poly))
    return polys


def draw_lens_clean(lens, ax, color=OUTLINE, lw=LINEWIDTH_LENS):
    for s in lens.surfaces:
        cls = s.__class__.__name__
        if cls == "Aperture":
            d = float(s.d.item()) if torch.is_tensor(s.d) else float(s.d)
            r = float(s.r)
            wedge_l = 0.05 * r
            wedge_h = 0.18 * r
            for sign in (-1, +1):
                ax.plot([d - wedge_l, d + wedge_l], [sign * r, sign * r],
                        color="#666", lw=lw, solid_capstyle="round")
                ax.plot([d, d], [sign * r, sign * (r + wedge_h)],
                        color="#666", lw=lw, solid_capstyle="round")
        else:
            r_eff = float(s.draw_r())
            r = torch.linspace(-r_eff, r_eff, 256, device=s.device)
            z = s.surface_with_offset(
                r, torch.zeros(len(r), device=s.device), valid_check=False
            ).cpu().detach().numpy()
            r_np = r.cpu().detach().numpy()
            ax.plot(z, r_np, color=color, lw=lw, solid_capstyle="round")

    surfs = lens.surfaces
    for i in range(len(surfs) - 1):
        s_prev = surfs[i]
        if not (hasattr(s_prev, "mat2") and float(s_prev.mat2.n) > 1.1):
            continue
        s_next = surfs[i + 1]
        r_prev = float(s_prev.draw_r())
        r_next = float(s_next.draw_r())
        sag_prev = s_prev.surface_with_offset(
            torch.tensor(r_prev, device=s_prev.device),
            torch.tensor(0.0, device=s_prev.device),
            valid_check=False,
        ).item()
        sag_next = s_next.surface_with_offset(
            torch.tensor(r_next, device=s_next.device),
            torch.tensor(0.0, device=s_next.device),
            valid_check=False,
        ).item()
        for sign in (-1, +1):
            ax.plot([sag_prev, sag_next],
                    [sign * r_prev, sign * r_next],
                    color=color, lw=lw, solid_capstyle="round")

    d_sensor = float(lens.d_sensor.item()) if torch.is_tensor(lens.d_sensor) else float(lens.d_sensor)
    r_sensor = float(lens.r_sensor)
    ax.plot([d_sensor, d_sensor], [-r_sensor, r_sensor],
            color=SENSOR, lw=LINEWIDTH_SENSOR, solid_capstyle="round")


def hfov_deg_for(lens):
    if hasattr(lens, "rfov") and lens.rfov is not None:
        try:
            return float(np.rad2deg(float(lens.rfov)))
        except Exception:
            pass
    half_sensor = float(lens.sensor_size[0]) / 2.0 if hasattr(lens, "sensor_size") else float(lens.r_sensor)
    efl = float(getattr(lens, "foclen", 50.0))
    return math.degrees(math.atan(half_sensor / abs(efl)))


def draw_line_rays(lens, ax):
    hfov = hfov_deg_for(lens)
    for fov_frac, color in zip([0.0, 0.5, 1.0], FIELD_COLORS):
        try:
            ray = lens.sample_point_source_2D(
                fov=float(fov_frac * hfov), depth=RAY_DEPTH_MM,
                num_rays=RAY_NUM_PER_FIELD, wvln=RAY_WVLN_UM,
            )
            ray.prop_to(-1.0)
            _, rec = lens.trace2sensor(ray=ray, record=True)
            n_before = len(ax.lines)
            lens.draw_ray_2d(rec, ax=ax, fig=ax.figure, color=color)
            for line in ax.lines[n_before:]:
                line.set_linewidth(RAY_LW)
                line.set_alpha(RAY_ALPHA)
        except Exception as e:
            print(f"    line-ray draw failed for fov={fov_frac}: {e}")


def draw_filled_bundles(lens, ax):
    hfov = hfov_deg_for(lens)
    for fov_frac, color in zip([0.0, 0.5, 1.0], FIELD_COLORS):
        try:
            ray = lens.sample_point_source_2D(
                fov=float(fov_frac * hfov), depth=RAY_DEPTH_MM,
                num_rays=FILLED_BUNDLE_NUM_RAYS, wvln=RAY_WVLN_UM,
            )
            ray.prop_to(-1.0)
            _, rec = lens.trace2sensor(ray=ray, record=True)
            lens.draw_ray_2d_bundle(
                rec, ax=ax, fig=ax.figure,
                color=color, alpha=FILLED_BUNDLE_ALPHA,
            )
        except Exception as e:
            print(f"    filled-bundle draw failed for fov={fov_frac}: {e}")


def render_one(ax, lens, fill_color, *, ray_style="none"):
    if ray_style == "lines":
        draw_line_rays(lens, ax)
    elif ray_style == "filled":
        draw_filled_bundles(lens, ax)
    fill_alpha = 1.0 if ray_style == "none" else (0.85 if ray_style == "lines" else 0.78)
    for z_poly, x_poly in lens_fill_polygons(lens):
        ax.fill(z_poly, x_poly, color=fill_color, alpha=fill_alpha,
                linewidth=0.0, zorder=3)
    draw_lens_clean(lens, ax)
    ax.set_aspect("equal")
    ax.set_axis_off()


def collect_designs():
    payload = json.loads(DATA_IN.read_text())
    by_key = {}
    for k, a in enumerate(payload["anchors_used"]):
        by_key[("anchor", k)] = {
            "theta": a["theta"], "refined_RMS_um": a["refined_RMS_um"],
            "source_run": a["source_run"], "design_id": a["design_id"],
        }
    for d in payload["designs"]:
        if d["role"] == "parent":
            by_key[("parent", int(d["rank"]))] = {
                "theta": d["theta"], "refined_RMS_um": d["refined_RMS_um"],
                "source_run": d["source_run"], "design_id": d["design_id"],
            }
        elif d["role"] == "incumbent":
            by_key[("incumbent", int(d.get("rank", 0)))] = {
                "theta": d["theta"], "refined_RMS_um": d["refined_RMS_um"],
                "source_run": d["source_run"], "design_id": d["design_id"],
            }
    return by_key


def render_grid(out_png, ray_style):
    by_key = collect_designs()
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["sixlet"]["cfg"]
    )
    layout = [
        ("anchor",     0, 0, 0, "anchor #1"),
        ("anchor",     1, 0, 1, "anchor #2"),
        ("anchor",     2, 0, 2, "anchor #3"),
        ("incumbent",  0, 0, 3, "incumbent"),
        ("parent",     0, 1, 0, "near-parent #1"),
        ("parent",     1, 1, 1, "near-parent #2"),
        ("parent",     2, 1, 2, "near-parent #3"),
        (None,         0, 1, 3, ""),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6), facecolor="white")
    for role, rank, row, col, label in layout:
        ax = axes[row, col]
        ax.set_facecolor("white")
        if role is None:
            ax.set_axis_off()
            continue
        des = by_key.get((role, rank))
        if des is None:
            ax.text(0.5, 0.5, f"{label}\nMISSING",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        fill_color = RED_FILL if role == "incumbent" else BLUE_FILL
        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path,
                                np.asarray(des["theta"]), template, solver)
            render_one(ax, lens, fill_color=fill_color, ray_style=ray_style)
        except Exception as e:
            ax.text(0.5, 0.5, f"render err\n{type(e).__name__}: {e}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)
            ax.set_axis_off()
            continue
        rms = des["refined_RMS_um"]
        ax.set_title(
            f"{label.upper()}\nRMS = {rms:.1f} um",
            fontsize=10,
            color="C3" if role == "incumbent" else "black",
        )
    title_suffix = {
        "none":   "no rays",
        "lines":  "with 3-field ray bundles",
        "filled": "with 3-field filled bundles",
    }[ray_style]
    fig.suptitle(
        f"Sixlet method-library — unpruned lens cross-sections ({title_suffix})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png}")


def main():
    render_grid(OUT_NO_RAYS,     ray_style="none")
    render_grid(OUT_LINE_RAYS,   ray_style="lines")
    render_grid(OUT_FILLED_RAYS, ray_style="filled")


if __name__ == "__main__":
    main()
