#!/usr/bin/env python
"""Combined Seidel-bar + stability-ribbon panel for each of the 4 storyboard
designs. Paper-grade simplification:

  * y-axis: only `0` and `max` labels (no middle tick)
  * x-axis: only category labels (no middle tick)
  * large bold sans-serif throughout
  * one PNG per design, two panels side-by-side (Seidel left, robust right)
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

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)
from scripts_throwaway.render_spotlight_panels_split import (
    per_field_per_wvln_rms_um, BLUE_FILL, RED_FILL,
)

SPOTS  = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/spotlights_storyboard4.json"
OUT_DIR = ROOT / "figures/storyboard4_combined_panels"

# Bold-font style. Use a chunky sans-serif so labels read well at small size.
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.weight":    "bold",
    "axes.labelweight":"bold",
    "axes.titleweight":"bold",
})


def render_seidel_simple(ax, s2_val, s3_val, fill_color):
    """Two bars: |S2| and |S3|. Bold, endpoint-only y-ticks."""
    vals = [abs(float(s2_val)), abs(float(s3_val))]
    vmax = max(vals) * 1.18
    ax.bar([0, 1], vals, color=fill_color, edgecolor="#222",
           linewidth=1.4, width=0.62)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["S₂", "S₃"], fontsize=32, fontweight="bold")
    ax.set_ylim(0, vmax)
    ax.set_yticks([0, vmax])
    scaled = vmax * 1e3
    top_lbl = f"{scaled:.1f}" if scaled >= 0.1 else f"{scaled:.2g}"
    ax.set_yticklabels(["0", top_lbl], fontsize=22, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#222")
        ax.spines[spine].set_linewidth(2.0)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", length=6, width=2, pad=6)


def render_robust_simple(ax, field_fracs, wvlns, rms_arr, fill_color):
    """Chromatic stability ribbon: per-field min/max across wvlns + median line."""
    rms_arr = np.asarray(rms_arr)
    field_fracs = np.asarray(field_fracs)
    rms_min = rms_arr.min(axis=1)
    rms_max = rms_arr.max(axis=1)
    rms_mid = np.median(rms_arr, axis=1)

    ax.fill_between(field_fracs, rms_min, rms_max,
                    color=fill_color, alpha=0.45, linewidth=0.0)
    ax.plot(field_fracs, rms_mid, color=fill_color, lw=3.4,
            solid_capstyle="round")
    ax.plot(field_fracs, rms_min, color=fill_color, lw=1.0, alpha=0.7)
    ax.plot(field_fracs, rms_max, color=fill_color, lw=1.0, alpha=0.7)

    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["axis", "edge"], fontsize=22, fontweight="bold")
    ymax = float(rms_max.max()) * 1.12
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, ymax])
    ax.set_yticklabels(["0", f"{ymax:.0f}"], fontsize=22, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#222")
        ax.spines[spine].set_linewidth(2.0)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", length=6, width=2, pad=6)
    ax.set_box_aspect(2.0 / 1.7)   # axes box H:W = 2:1.7


def main():
    spots = json.loads(SPOTS.read_text())
    print(f"loaded {len(spots)} picks")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["octuplet"]["cfg"])

    for sp in spots:
        role = sp["role"]
        is_shared = bool(sp.get("in_saga_hull"))
        fill = RED_FILL if is_shared else BLUE_FILL
        theta = np.asarray(sp["theta"], dtype=np.float64)
        print(f"\n[{role}]  {'SHARED' if is_shared else 'HL-only'}  "
              f"S2={sp['S2']:+.5f} S3={sp['S3']:+.5f}  rms={sp['rms_um']:.2f} µm")
        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
        except Exception as e:
            print(f"  FAILED to build lens: {e}")
            continue

        field_fracs, wvlns, rms_arr = per_field_per_wvln_rms_um(
            lens, p_problem, n_fields=7, num_rays=256,
        )

        # 1 row × 2 cols.  Seidel slightly narrower than the ribbon.
        fig = plt.figure(figsize=(8.6, 4.2), facecolor="white")
        gs = fig.add_gridspec(1, 2, left=0.10, right=0.97, top=0.92,
                              bottom=0.18, wspace=0.42, width_ratios=[1.0, 1.4])
        ax_s = fig.add_subplot(gs[0, 0])
        render_seidel_simple(ax_s, sp["S2"], sp["S3"], fill)
        ax_r = fig.add_subplot(gs[0, 1])
        render_robust_simple(ax_r, field_fracs, wvlns, rms_arr, fill)

        out = OUT_DIR / f"storyboard_{role}_seidel_robust.png"
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  -> {out}")

    print(f"\nAll panels in {OUT_DIR}")


if __name__ == "__main__":
    main()
