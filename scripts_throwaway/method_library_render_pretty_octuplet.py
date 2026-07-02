#!/usr/bin/env python
"""Polished render of the 7 pruned octuplet designs (3 anchors + 3 near-parents
+ 1 incumbent). Filled glass elements with thin clean outlines.

Two output PNGs:
  figures/method_library_octuplet_pretty.png            (no rays)
  figures/method_library_octuplet_pretty_with_rays.png  (3-field ray bundles)

Color convention:
  - non-incumbent designs: soft blue glass fill (#7BAFD4)
  - INCUMBENT:             soft red glass fill (#E07B7B)
  - outlines:              dark grey, sensor: black
  - rays (3 fields):       blue / green / red

Loads pruned lens JSONs from figures/method_library_data/pruned_lenses/.
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
LENS_DIR = ROOT / "figures" / "method_library_data" / "pruned_lenses"
SUMMARY = ROOT / "figures" / "method_library_data" / "pruned_summary.json"
OUT_NO_RAYS      = ROOT / "figures" / "method_library_octuplet_pretty.png"
OUT_RAYS         = ROOT / "figures" / "method_library_octuplet_pretty_with_rays.png"
OUT_FILLED_RAYS  = ROOT / "figures" / "method_library_octuplet_pretty_with_filled_rays.png"

from deeplens.optics import GeoLens

# ---- styling ----
BLUE_FILL = "#7BAFD4"
RED_FILL  = "#E07B7B"
OUTLINE   = "#1F1F1F"
SENSOR    = "#000000"
LINEWIDTH_LENS = 1.2          # was 1.9 -- thinner & more elegant
LINEWIDTH_SENSOR = 1.6
FILL_NUM_SAMPLES = 192

# ---- ray styling ----
FIELD_COLORS = ["#1565C0", "#2E7D32", "#C62828"]  # blue, green, red
RAY_LW = 0.55
RAY_ALPHA = 0.85
RAY_NUM_PER_FIELD = 11
RAY_WVLN_UM = 0.587
RAY_DEPTH_MM = -20000.0

# ---- filled-bundle styling ----
FILLED_BUNDLE_ALPHA = 0.32
FILLED_BUNDLE_NUM_RAYS = 9   # need >= 2; more samples => smoother envelope


# --------------------------------------------------------------------
# Lens fill polygons -- proper version that respects each surface's
# own semi-aperture (no white gap when back surface is wider than front).
# --------------------------------------------------------------------

def lens_fill_polygons(lens, num_samples=FILL_NUM_SAMPLES):
    """Return [(z_poly, x_poly), ...], one per glass element.

    Polygon is built by:
      1. trace front surface from r = -r0 to +r0
      2. straight line from (z0(+r0), +r0) to (z1(+r1), +r1)  (top side wall)
      3. trace back surface from r = +r1 to -r1
      4. straight line from (z1(-r1), -r1) to (z0(-r0), -r0)  (bottom side wall)
    This handles both r0 < r1 and r0 > r1 without leaving white gaps inside
    the element.
    """
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

        # CCW polygon:
        #   1. front surface bottom-up:  rs0 from -r0 to +r0, z=z0
        #   2. side wall up at r=+r0..+r1: implicit straight via line segment
        #   3. back surface top-down:    rs1 from +r1 to -r1, z=z1
        #   4. side wall back to start:  implicit straight
        z_poly = np.concatenate([z0, z1[::-1]], axis=0)
        x_poly = np.concatenate([rs0, rs1[::-1]], axis=0)
        polys.append((z_poly, x_poly))
    return polys


# --------------------------------------------------------------------
# Clean lens outlines (surfaces, edge connectors, aperture wedges, sensor)
# --------------------------------------------------------------------

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


# --------------------------------------------------------------------
# Ray bundles
# --------------------------------------------------------------------

def hfov_deg_for(lens):
    """Half-FOV in degrees (matches stage-2 fields convention)."""
    if hasattr(lens, "rfov") and lens.rfov is not None:
        try:
            return float(np.rad2deg(float(lens.rfov)))
        except Exception:
            pass
    half_sensor = float(lens.sensor_size[0]) / 2.0 if hasattr(lens, "sensor_size") else float(lens.r_sensor)
    efl = float(getattr(lens, "foclen", 50.0))
    return math.degrees(math.atan(half_sensor / abs(efl)))


def draw_rays_for_lens(lens, ax):
    hfov = hfov_deg_for(lens)
    for fov_frac, color in zip([0.0, 0.5, 1.0], FIELD_COLORS):
        try:
            ray = lens.sample_point_source_2D(
                fov=float(fov_frac * hfov),
                depth=RAY_DEPTH_MM,
                num_rays=RAY_NUM_PER_FIELD,
                wvln=RAY_WVLN_UM,
            )
            ray.prop_to(-1.0)
            _, rec = lens.trace2sensor(ray=ray, record=True)
            n_before = len(ax.lines)
            lens.draw_ray_2d(rec, ax=ax, fig=ax.figure, color=color)
            for line in ax.lines[n_before:]:
                line.set_linewidth(RAY_LW)
                line.set_alpha(RAY_ALPHA)
        except Exception as e:
            print(f"    ray draw failed for fov={fov_frac}: {e}")


# --------------------------------------------------------------------
# Per-cell render
# --------------------------------------------------------------------

def render_one(ax, lens_path, fill_color, with_rays=False):
    lens = GeoLens(filename=str(lens_path), device="cpu")
    # zorder: rays(2) under fill(3)? No -- fill should be at 1, rays at 2,
    # outlines at 3 so the crisp dark surface lines remain on top.
    if with_rays:
        draw_rays_for_lens(lens, ax)  # default zorder ~ 2
    for z_poly, x_poly in lens_fill_polygons(lens):
        ax.fill(z_poly, x_poly, color=fill_color, alpha=0.85 if with_rays else 1.0,
                linewidth=0.0, zorder=3)
    draw_lens_clean(lens, ax)
    ax.set_aspect("equal")
    ax.set_axis_off()


# --------------------------------------------------------------------
# Top-level grid render
# --------------------------------------------------------------------

def render_grid(out_png, with_rays):
    summary = json.loads(SUMMARY.read_text())
    by_key = {(d["role"], d["rank"]): d for d in summary["designs"]}

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
            render_one(ax, des["lens_json"], fill_color=fill_color,
                       with_rays=with_rays)
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

    fig.suptitle(
        "Octuplet method-library — pruned lens cross-sections "
        f"({'with 3-field ray bundles' if with_rays else 'no rays'})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png}")


def main():
    render_grid(OUT_NO_RAYS, with_rays=False)
    render_grid(OUT_RAYS,    with_rays=True)


if __name__ == "__main__":
    main()
