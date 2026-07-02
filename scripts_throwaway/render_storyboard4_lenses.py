#!/usr/bin/env python
"""Render lens cross-sections for the 4 storyboard designs:
   R1, R2 — two rightmost (highest S2 / coma) sub-20 µm HL designs
   BOTTOM — lowest S3 (astigmatism) sub-20 µm HL design
   COMMON — shared with SA+GA, RMS ∈ [20, 25] µm
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
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

SPOTS = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/spotlights_storyboard4.json"
OUT   = ROOT / "figures/octuplet_storyboard4_lenses.png"
BLUE_FILL = "#7BAFD4"
RED_FILL  = "#E07B7B"

spots = json.loads(SPOTS.read_text())
print(f"loaded {len(spots)} picks")

template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
    SYSTEMS["octuplet"]["cfg"])

# 2x2 grid: rightmost-1, rightmost-2 on top row; bottom + common on bottom row
fig = plt.figure(figsize=(14.0, 5.6), facecolor="white")
gs = fig.add_gridspec(2, 2, left=0.04, right=0.96, top=0.82, bottom=0.05,
                       wspace=0.04, hspace=0.04)
titles = {
    "right1": "Rightmost #1  (high coma S₂)",
    "right2": "Rightmost #2  (high coma S₂)",
    "bottom": "Bottom-most  (low astig S₃)",
    "common": "Shared basin  (in SA+GA hull, 20–25 µm)",
}
positions = {"right1": (0, 0), "right2": (0, 1), "bottom": (1, 0), "common": (1, 1)}

for sp in spots:
    role = sp["role"]
    r, c = positions[role]
    ax = fig.add_subplot(gs[r, c])
    ax.set_facecolor("white")
    fill = RED_FILL if sp["in_saga_hull"] else BLUE_FILL
    theta = np.asarray(sp["theta"], dtype=np.float64)
    try:
        lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
        render_one(ax, lens, fill_color=fill)
    except Exception as e:
        print(f"  WARN: {role} failed: {e}")
        ax.text(0.5, 0.5, "render failed", ha="center", va="center",
                transform=ax.transAxes, color="red")
        ax.set_axis_off()
    ax.set_title(titles[role], fontsize=12, family="serif", style="italic",
                 color="#1A1A1A", pad=6)
    # RMS + S2/S3 corner label
    ax.text(0.97, 0.04,
            f"RMS  {sp['rms_um']:.2f} µm\nS₂={sp['S2']*1e3:+.2f}×10⁻³\nS₃={sp['S3']*1e3:+.2f}×10⁻³",
            ha="right", va="bottom", transform=ax.transAxes,
            fontsize=8.5, family="serif", color="#1A1A1A",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#888", linewidth=0.6, alpha=0.95))

fig.text(0.5, 0.94,
         "Octuplet  ·  four storyboard designs from the sub-25 µm pool",
         ha="center", va="center", fontsize=18, family="serif",
         style="italic", color="#1A1A1A")
fig.text(0.5, 0.895,
         "Top row: tightest-coma HL-exclusive (outside SA+GA hull).  "
         "Bottom-left: tightest-astig HL-exclusive.  "
         "Bottom-right: shared HL+SA+GA basin (mediocre quality).",
         ha="center", va="center", fontsize=10.5, family="serif",
         style="italic", color="#3A3A3A")

box = FancyBboxPatch((0.008, 0.008), 0.984, 0.984,
                     boxstyle="round,pad=0.0,rounding_size=0.008",
                     transform=fig.transFigure,
                     edgecolor="#1A1A1A", facecolor="none", linewidth=1.3,
                     zorder=10, clip_on=False)
fig.patches.append(box)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor="white")
plt.close(fig)
print(f"-> {OUT}")
