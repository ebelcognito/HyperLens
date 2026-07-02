#!/usr/bin/env python
"""Fig 5 sanity render. Throwaway. Default matplotlib styling.

Layout (single figure, ~11 in wide x ~13 in tall):
  Rows 1-4: lens side-views, 6 per row
            Row 1 = sixlet HL
            Row 2 = sixlet SAGA
            Row 3 = octuplet HL
            Row 4 = octuplet SAGA
  Row 5  : (left) sixlet aberration radar HL vs SAGA
           (right) eps-distinct curves for triplet/sixlet/octuplet x HL/SAGA

Reads from figures/fig5_data/. Rebuilds GeoLens objects via
_apply_theta_generic to draw lens cross-sections (the JSON itself
holds geometry, but draw_lens_2d wants a GeoLens).
"""
from __future__ import annotations

import json
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
OUT_PNG = ROOT / "figures" / "fig5_sanity.png"

# Reuse data loading helpers from fig5_extract
from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens, EPS_GRID, EPS_TARGET,
)

SYS_ORDER = ("sixlet", "octuplet")
METHOD_ORDER = ("hl", "saga")


def draw_lens_panel(ax, sys_name, method_name, k, design, builders):
    template, solver, prob_cfg, lens_path, p_problem = builders[sys_name]
    theta = np.array(design["theta"], dtype=np.float64)
    try:
        lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
        lens.draw_lens_2d(ax=ax, fig=ax.figure, color="black")
    except Exception as e:
        ax.text(0.5, 0.5, f"build err\n{type(e).__name__}", ha="center",
                va="center", transform=ax.transAxes, fontsize=6)
    rms = design.get("refined_RMS_um", float("nan"))
    pad_tag = " (pad)" if design.get("padded") else ""
    ax.set_title(f"{sys_name}/{method_name} #{k}\nRMS={rms:.1f}um{pad_tag}", fontsize=6)
    ax.set_aspect("equal")
    ax.set_axis_off()


def draw_radar(ax, radar):
    terms = ["S_1", "S_2", "S_3", "S_4", "S_5"]
    angles = np.linspace(0, 2 * np.pi, len(terms), endpoint=False).tolist()
    angles += angles[:1]

    for method, color in (("hl", "C0"), ("saga", "C3")):
        d = radar[f"sixlet_{method}"]
        norm_const = d["normalisation_constants"]
        p5 = [d[t]["p5"] / norm_const[t] for t in terms]
        p95 = [d[t]["p95"] / norm_const[t] for t in terms]
        # close the polygons
        p5_c = p5 + p5[:1]
        p95_c = p95 + p95[:1]
        # Draw envelope as the band between p5 and p95: fill p95 polygon then
        # cut a hole would need a path; simpler is fill p95 with low alpha and
        # fill p5 (inner) with white-ish over it. For sanity we just fill p95
        # and overlay p5 with no fill. Plus overlay median.
        ax.fill(angles, p95_c, color=color, alpha=0.25, label=f"{method.upper()} p5-p95")
        ax.fill(angles, p5_c, color="white", alpha=1.0, zorder=2)
        ax.plot(angles, p95_c, color=color, lw=1.2)
        ax.plot(angles, p5_c, color=color, lw=0.8, ls=":")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(terms, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_title("sixlet aberration envelope, HL vs SA+GA", fontsize=9)
    ax.legend(loc="upper right", fontsize=6, bbox_to_anchor=(1.25, 1.05))


def draw_eps_curves(ax):
    sys_color = {"triplet": "#a6cee3", "sixlet": "#1f78b4", "octuplet": "#08306b"}
    for sys_name in ("triplet", "sixlet", "octuplet"):
        for method in ("hl", "saga"):
            df = np.loadtxt(DATA / f"eps_curves_{sys_name}_{method}.csv",
                            delimiter=",", skiprows=1)
            ls = "-" if method == "hl" else "--"
            ax.plot(df[:, 0], df[:, 1], ls=ls, color=sys_color[sys_name],
                    lw=1.5, label=f"{sys_name} {method.upper()}")
    # mark eps=0.20 for sixlet HL and SAGA
    for method, color in (("hl", sys_color["sixlet"]), ("saga", sys_color["sixlet"])):
        df = np.loadtxt(DATA / f"eps_curves_sixlet_{method}.csv",
                        delimiter=",", skiprows=1)
        i20 = int(np.argmin(np.abs(df[:, 0] - EPS_TARGET)))
        x = df[i20, 0]
        y = df[i20, 1]
        ax.scatter([x], [y], s=40, edgecolor="black", facecolor=color, zorder=5)
        ax.annotate(f"sixlet {method.upper()}={int(y)}", (x, y),
                    xytext=(6, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("eps-distinct count")
    ax.set_title("eps-distinct count vs epsilon, three systems", fontsize=9)
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)


def main():
    selected = json.loads((DATA / "selected_designs.json").read_text())
    radar = json.loads((DATA / "radar_envelopes.json").read_text())
    builders = {sys_name: build_lens_template_and_solver(SYSTEMS[sys_name]["cfg"])
                for sys_name in SYS_ORDER}

    fig = plt.figure(figsize=(11, 13))
    # 6-row grid: rows 0..3 = lens grids (sixlet HL, sixlet SAGA, oct HL, oct SAGA)
    # rows 4 (and 5) split into radar + eps-curves
    gs = fig.add_gridspec(
        nrows=6, ncols=12, hspace=0.55, wspace=0.30,
        height_ratios=[1, 1, 1, 1, 1.6, 1.6],
    )

    # --- 4 lens-grid rows ---
    grid_specs = [
        ("sixlet", "hl", 0),
        ("sixlet", "saga", 1),
        ("octuplet", "hl", 2),
        ("octuplet", "saga", 3),
    ]
    for sys_name, method, row in grid_specs:
        designs = selected[f"{sys_name}_{method}"]
        for k in range(6):
            ax = fig.add_subplot(gs[row, k * 2:(k + 1) * 2])
            if k < len(designs):
                draw_lens_panel(ax, sys_name, method, k, designs[k], builders)
            else:
                ax.set_axis_off()

    # --- bottom-left: radar (rows 4-5, cols 0-5) ---
    ax_radar = fig.add_subplot(gs[4:6, 0:6], polar=True)
    draw_radar(ax_radar, radar)

    # --- bottom-right: eps curves (rows 4-5, cols 6-11) ---
    ax_eps = fig.add_subplot(gs[4:6, 6:12])
    draw_eps_curves(ax_eps)

    fig.suptitle("Figure 5 sanity (lens forms / aberration radar / eps-distinct)",
                 fontsize=11)
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
