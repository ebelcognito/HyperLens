#!/usr/bin/env python
"""Render the v2 storyboard (3 HL-exclusive + 1 SA+GA-shared common):
   * 2×2 lens-card grid
   * 4 individual Seidel-bar + chromatic-ribbon panels
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch; torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
from scripts_throwaway.fig5_extract import SYSTEMS, build_lens_template_and_solver, make_geolens
from scripts_throwaway.render_spotlight_sixlet_designs import render_one
from scripts_throwaway.render_storyboard4_seidel_robust_combined import (
    render_seidel_simple, render_robust_simple,
)
from scripts_throwaway.render_spotlight_panels_split import (
    per_field_per_wvln_rms_um, BLUE_FILL, RED_FILL,
)

SPOTS    = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/spotlights_storyboard4_v3.json"
LENS_OUT = ROOT / "figures/octuplet_storyboard4_v3_lenses.png"
PNL_DIR  = ROOT / "figures/storyboard4_v3_combined_panels"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.weight":     "bold",
    "axes.labelweight":"bold",
    "axes.titleweight":"bold",
})

POSITIONS = {"hl1": (0, 0), "hl2": (0, 1), "hl3": (1, 0), "common": (1, 1)}
TITLES = {
    "hl1":    "HL-exclusive #1",
    "hl2":    "HL-exclusive #2",
    "hl3":    "HL-exclusive #3",
    "common": "SA+GA-shared basin",
}


def main():
    spots = json.loads(SPOTS.read_text())
    print(f"loaded {len(spots)} spots from {SPOTS.name}")
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["octuplet"]["cfg"])

    # ---------- 2×2 lens grid ----------
    fig = plt.figure(figsize=(14.0, 5.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.04, right=0.96, top=0.80, bottom=0.05,
                           wspace=0.04, hspace=0.06)
    for sp in spots:
        role = sp["role"]; r, c = POSITIONS[role]
        ax = fig.add_subplot(gs[r, c]); ax.set_facecolor("white")
        fill = RED_FILL if sp["in_saga_hull"] else BLUE_FILL
        theta = np.asarray(sp["theta"], dtype=np.float64)
        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
            render_one(ax, lens, fill_color=fill)
        except Exception as e:
            print(f"  WARN: {role} failed: {e}")
            ax.text(0.5, 0.5, "render failed", ha="center", va="center",
                    transform=ax.transAxes, color="red"); ax.set_axis_off()
        ax.set_title(TITLES[role], fontsize=13, family="serif", style="italic",
                     color="#1A1A1A", pad=6)
        ax.text(0.97, 0.04,
                f"S₂={sp['S2']*1e3:+.3f}×10⁻³\nS₃={sp['S3']*1e3:+.3f}×10⁻³\n"
                f"rms_max(7f×3λ)={sp.get('rms_max_all', float('nan')):.2f} µm",
                ha="right", va="bottom", transform=ax.transAxes,
                fontsize=9, family="serif", color="#1A1A1A",
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor="#888", linewidth=0.6, alpha=0.95))

    fig.text(0.5, 0.93,
             "Octuplet  ·  storyboard v2:  3 HL-exclusive designs that beat the shared SA+GA basin",
             ha="center", va="center", fontsize=17, family="serif",
             style="italic", color="#1A1A1A")
    fig.text(0.5, 0.88,
             "HL picks sit outside the SA+GA convex hull in S₂-S₃ "
             "and have lower worst-case RMS over (7 fields × 3 wavelengths) than the shared basin.",
             ha="center", va="center", fontsize=10.5, family="serif",
             style="italic", color="#3A3A3A")
    box = FancyBboxPatch((0.008, 0.008), 0.984, 0.984,
                         boxstyle="round,pad=0.0,rounding_size=0.008",
                         transform=fig.transFigure,
                         edgecolor="#1A1A1A", facecolor="none", linewidth=1.3,
                         zorder=10, clip_on=False)
    fig.patches.append(box)
    LENS_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(LENS_OUT, dpi=200, facecolor="white"); plt.close(fig)
    print(f"  -> {LENS_OUT}")

    # ---------- 4 Seidel + chromatic-ribbon panels ----------
    PNL_DIR.mkdir(parents=True, exist_ok=True)
    for sp in spots:
        role = sp["role"]
        theta = np.asarray(sp["theta"], dtype=np.float64)
        fill = RED_FILL if sp["in_saga_hull"] else BLUE_FILL
        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
        except Exception as e:
            print(f"  [{role}] FAILED: {e}"); continue
        ff, wv, rms_arr = per_field_per_wvln_rms_um(
            lens, p_problem, n_fields=7, num_rays=256,
        )
        fig = plt.figure(figsize=(8.6, 4.2), facecolor="white")
        gs = fig.add_gridspec(1, 2, left=0.10, right=0.97, top=0.92,
                              bottom=0.18, wspace=0.42, width_ratios=[1.0, 1.4])
        ax_s = fig.add_subplot(gs[0, 0])
        render_seidel_simple(ax_s, sp["S2"], sp["S3"], fill)
        ax_r = fig.add_subplot(gs[0, 1])
        render_robust_simple(ax_r, ff, wv, rms_arr, fill)
        out = PNL_DIR / f"storyboard_{role}_seidel_robust.png"
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  [{role}]  -> {out.name}")

    print(f"\npanels in {PNL_DIR}")


if __name__ == "__main__":
    main()
