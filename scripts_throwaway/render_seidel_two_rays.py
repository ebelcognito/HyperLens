#!/usr/bin/env python
"""Render ONE diverse-setup octuplet design in the pretty filled style, with
exactly TWO rays drawn: the marginal ray and the chief ray. These are the
two paraxial rays whose intersection heights at every surface fully encode
the third-order Seidel coefficients (S1..S5).

Picks rank 1 from the CHAMPION-diverse manifest by default.

Outputs (two SEPARATE PNGs):
  figures/seidel_two_ray_demo_rank01.png   # lens + 2 rays + sensor, no labels
  figures/seidel_bars_rank01.png           # S1..S5 bar chart for that design
"""
from __future__ import annotations
import json
import os
import sys
import math
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

from scripts_throwaway.fig5_extract import (
    build_lens_template_and_solver, make_geolens, load_bounds, load_pool,
)
from scripts_throwaway.render_spotlight_sixlet_designs import (
    lens_fill_polygons, draw_lens_clean,
)
from deeplens.optics.light.ray import Ray

# --- inputs / outputs ---
CHAMPION_JSON = ROOT / ("results/lensmega/octuplet_REFSWEEP_CHAMPION_seed42_"
                        "20260509_145427/refined_top500_seed42_CHAMPION.json")
CFG_PATH      = ROOT / "configs/ablations/octuplet_REFSWEEP_CHAMPION.yaml"
MANIFEST      = ROOT / "figures/octuplet_CHAMPION_diverse_manifest.json"
OUT_LENS_PNG  = ROOT / "figures/seidel_two_ray_demo_rank01.png"
OUT_BARS_PNG  = ROOT / "figures/seidel_bars_rank01.png"

PICK_RANK = 1   # rank 1 = best RMS in the diverse picks

# --- styling (matches the pretty/filled recipe) ---
GLASS_FILL    = "#7BAFD4"   # soft blue glass
GLASS_ALPHA   = 0.85
LINEWIDTH_LENS = 1.2
LINEWIDTH_SENSOR = 1.6

# Two ray styling — blue + yellow
MARGINAL_COLOR = "#F2C200"  # bright yellow — marginal (axial)
CHIEF_COLOR    = "#1B4965"  # deep blue   — chief (principal)
RAY_LW         = 1.8
RAY_ALPHA      = 1.0

# Bar-chart styling
BAR_POS_COLOR = "#1B4965"   # deep blue for positive S
BAR_NEG_COLOR = "#F2C200"   # yellow     for negative S
BAR_EDGE      = "#1A1A1A"
BAR_LABELS    = ("S₁\nspherical", "S₂\ncoma", "S₃\nastigmatism",
                 "S₄\nPetzval",   "S₅\ndistortion")

WVLN_UM   = 0.587
DEPTH_MM  = -20000.0  # effectively object-at-infinity


def hfov_deg_for(lens):
    if hasattr(lens, "rfov") and lens.rfov is not None:
        try:
            return float(np.rad2deg(float(lens.rfov)))
        except Exception:
            pass
    half_sensor = (float(lens.sensor_size[0]) / 2.0
                   if hasattr(lens, "sensor_size") else float(lens.r_sensor))
    efl = float(getattr(lens, "foclen", 50.0))
    return math.degrees(math.atan(half_sensor / abs(efl)))


def build_marginal_ray(lens):
    """Marginal (axial) paraxial ray:
       on-axis object point  ->  edge of entrance pupil  ->  on-axis image.
    """
    pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
    pupilz = float(pupilz)
    pupilr = float(pupilr) * 0.985  # hair inside the pupil so it doesn't vignette

    # In this codebase the meridional axis stored on the ray is the X-axis
    # (sample_point_source_2D puts fov-offset and pupil sampling on x).
    o = torch.tensor([[0.0, 0.0, DEPTH_MM]], dtype=torch.float32)
    p = torch.tensor([[pupilr, 0.0, pupilz]], dtype=torch.float32)
    d = p - o
    ray = Ray(o, d, WVLN_UM, device=lens.device)
    ray.prop_to(DEPTH_MM)
    return ray


def build_chief_ray(lens, hfov_deg):
    """Chief (principal) paraxial ray:
       field-edge object point  ->  CENTER of entrance pupil  ->  edge of image.
    """
    pupilz, _ = lens.calc_entrance_pupil(paraxial=True)
    pupilz = float(pupilz)

    h_obj = DEPTH_MM * math.tan(math.radians(hfov_deg))
    o = torch.tensor([[h_obj, 0.0, DEPTH_MM]], dtype=torch.float32)
    p = torch.tensor([[0.0,   0.0, pupilz]], dtype=torch.float32)
    d = p - o
    ray = Ray(o, d, WVLN_UM, device=lens.device)
    ray.prop_to(DEPTH_MM)
    return ray


def trace_path(lens, ray):
    """Return (z_path, x_path) numpy arrays for the single ray."""
    _, rec = lens.trace2sensor(ray=ray, record=True)
    arr = torch.stack(rec, dim=-2).detach().cpu().numpy()  # (1, num_path, 3)
    return arr[0, :, 2], arr[0, :, 0]


def draw_ray(ax, z, x, *, color):
    ax.plot(
        z, x,
        color=color, lw=RAY_LW, alpha=RAY_ALPHA,
        solid_capstyle="round",
        zorder=5,
    )


def render_two_ray_diagram(lens, out_png):
    """Lens + two rays + sensor. No title, no legend, no annotations."""
    fig = plt.figure(figsize=(10.0, 4.6), facecolor="white")
    gs = fig.add_gridspec(1, 1, left=0.02, right=0.98, top=0.98, bottom=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("white")

    for z_poly, x_poly in lens_fill_polygons(lens):
        ax.fill(z_poly, x_poly, color=GLASS_FILL, alpha=GLASS_ALPHA,
                linewidth=0.0, zorder=2)

    draw_lens_clean(lens, ax, lw=LINEWIDTH_LENS)

    hfov = hfov_deg_for(lens)
    ray_m = build_marginal_ray(lens)
    ray_c = build_chief_ray(lens, hfov)

    z_m, x_m = trace_path(lens, ray_m)
    z_c, x_c = trace_path(lens, ray_c)

    draw_ray(ax, z_m, x_m, color=MARGINAL_COLOR)
    draw_ray(ax, z_c, x_c, color=CHIEF_COLOR)

    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.savefig(out_png, dpi=220, facecolor="white",
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ----------------------------------------------------------------------
# Seidel S1..S5 — compute via the SeidelDescriptor and render a bar chart
# ----------------------------------------------------------------------

def compute_seidel_S1_to_S5(lens, template, hfov_deg, wvln_um=0.587):
    from optical_rl.qd.lensmega.descriptors.seidel import SeidelDescriptor
    desc = SeidelDescriptor(seidel_terms=("S1", "S2", "S3", "S4", "S5"))
    desc.prepare(template, device="cpu")
    S = desc.compute(
        lens,
        field_y_deg=float(hfov_deg),
        object_depth_mm=DEPTH_MM,
        wavelength_um=wvln_um,
        device="cpu",
    )
    return np.asarray(S.detach().cpu().numpy(), dtype=np.float64).ravel()


def render_seidel_bars(S_vals, out_png):
    """Minimal bar chart of S2 and S3 only. Two bars (blue + dark blue),
    bare axes, single-letter labels under the bars."""
    S2 = float(S_vals[1])
    S3 = float(S_vals[2])

    fig = plt.figure(figsize=(4.2, 4.6), facecolor="white")
    gs = fig.add_gridspec(1, 1, left=0.10, right=0.96, top=0.96, bottom=0.10)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("white")

    x = np.arange(2)
    vals = np.array([abs(S2), abs(S3)])  # magnitude only
    colors = ["#7BAFD4", "#1B4965"]      # blue, dark blue

    ax.bar(x, vals, color=colors, edgecolor="#1A1A1A",
           linewidth=1.0, width=0.55, zorder=3)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#1A1A1A")
    ax.spines["bottom"].set_linewidth(1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(["S₂", "S₃"], fontsize=14, color="#1A1A1A")
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_yticks([])

    vmax = float(np.max(vals)) if np.any(vals) else 1.0
    ax.set_ylim(0, 1.15 * vmax)
    ax.set_xlim(-0.6, 1.6)

    fig.savefig(out_png, dpi=220, facecolor="white",
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    print("=" * 70)
    print(f"Two-ray Seidel render — picking rank {PICK_RANK} from CHAMPION diverse")
    print("=" * 70)

    manifest = json.loads(MANIFEST.read_text())
    pick = next(p for p in manifest["picks"] if p["rank"] == PICK_RANK)
    design_idx = pick["design_idx"]
    rms_um = float(pick["rms_um"])
    print(f"  design_idx = {design_idx}   rms = {rms_um:.3f} µm")

    bounds = load_bounds(CFG_PATH)
    pool   = load_pool(CHAMPION_JSON, bounds)
    record = next(r for r in pool if r["idx"] == design_idx)

    template, solver, prob_cfg, lens_path, p_problem = (
        build_lens_template_and_solver(CFG_PATH)
    )
    lens = make_geolens(prob_cfg, p_problem, lens_path,
                        np.asarray(record["theta"], dtype=np.float64),
                        template, solver)

    OUT_LENS_PNG.parent.mkdir(parents=True, exist_ok=True)
    render_two_ray_diagram(lens, OUT_LENS_PNG)
    print(f"  wrote {OUT_LENS_PNG}")

    hfov = hfov_deg_for(lens)
    S = compute_seidel_S1_to_S5(lens, template, hfov_deg=hfov)
    print(f"  Seidel  S1..S5 = {S}")
    render_seidel_bars(S, OUT_BARS_PNG)
    print(f"  wrote {OUT_BARS_PNG}")


if __name__ == "__main__":
    main()
