#!/usr/bin/env python
"""Render a 2×5 grid of lens cross-sections for the top 10 ε-distinct
sub-20-µm octuplet designs sampled from EVERY octuplet refined-pool we've
ever produced (fig4 baselines + every ablation variant + BLIND_COMBO +
CHAMPION).
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
from matplotlib.patches import FancyBboxPatch
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)
from scripts_throwaway.render_spotlight_sixlet_designs import render_one

SPOTS  = ROOT / "figures/octuplet_top10_under20um_spotlights.json"
OUT    = ROOT / "figures/octuplet_top10_under20um.png"
BLUE_FILL = "#7BAFD4"
N_ROWS, N_COLS = 2, 5


def main():
    spots = json.loads(SPOTS.read_text())
    print(f"loaded {len(spots)} picks from {SPOTS}")

    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["octuplet"]["cfg"])

    fig = plt.figure(figsize=(20.0, 9.0), facecolor="white")
    gs = fig.add_gridspec(
        N_ROWS, N_COLS,
        left=0.03, right=0.97,
        top=0.80, bottom=0.07,
        wspace=0.04, hspace=0.13,
    )

    min_rms = min(s["rms_um"] for s in spots)
    max_rms = max(s["rms_um"] for s in spots)
    avg_rms = float(np.mean([s["rms_um"] for s in spots]))

    for k, sp in enumerate(spots):
        ax = fig.add_subplot(gs[k // N_COLS, k % N_COLS])
        ax.set_facecolor("white")
        theta = np.asarray(sp["theta"], dtype=np.float64)
        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
            render_one(ax, lens, fill_color=BLUE_FILL)
        except Exception as e:
            print(f"  WARN: design #{k+1} failed: {e}")
            ax.text(0.5, 0.5, f"render failed\n{e}", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=8)
            ax.set_axis_off()

    fig.text(
        0.5, 0.93,
        f"Octuplet  ·  Top 10 ε-distinct designs at refined RMS < 20 µm",
        ha="center", va="center", fontsize=22, family="serif", style="italic",
        color="#1A1A1A",
    )
    fig.text(
        0.5, 0.885,
        f"pooled across 13 octuplet runs (fig4 baselines + every ablation variant)  ·  "
        f"ε=0.30 in normalized 45-D θ-space  ·  RMS range {min_rms:.2f}–{max_rms:.2f} µm, mean {avg_rms:.2f} µm",
        ha="center", va="center", fontsize=12, family="serif", style="italic",
        color="#3A3A3A",
    )

    # Outer frame
    box = FancyBboxPatch(
        (0.008, 0.008), 0.984, 0.984,
        boxstyle="round,pad=0.0,rounding_size=0.008",
        transform=fig.transFigure,
        edgecolor="#1A1A1A", facecolor="none", linewidth=1.4,
        zorder=10, clip_on=False,
    )
    fig.patches.append(box)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
