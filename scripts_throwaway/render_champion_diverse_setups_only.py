#!/usr/bin/env python
"""Render each CHAMPION-diverse design as its own lens-setup-only PNG.

Reuses the picks from figures/octuplet_CHAMPION_diverse_manifest.json so we
render the *same 10 designs* as octuplet_CHAMPION_diverse.png — but each one
in its own file, with no ray bundles, in the same Mathematica-style theme.

Output: figures/octuplet_CHAMPION_diverse_setups/setup_rank{NN}.png
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
    build_lens_template_and_solver, make_geolens, load_bounds, load_pool,
)
from scripts_throwaway.render_spotlight_sixlet_designs import (
    lens_fill_polygons, draw_lens_clean,
)

CHAMPION_JSON = ROOT / "results/lensmega/octuplet_REFSWEEP_CHAMPION_seed42_20260509_145427/refined_top500_seed42_CHAMPION.json"
CFG_PATH = ROOT / "configs/ablations/octuplet_REFSWEEP_CHAMPION.yaml"
MANIFEST = ROOT / "figures/octuplet_CHAMPION_diverse_manifest.json"
OUT_DIR = ROOT / "figures/octuplet_CHAMPION_diverse_setups"

BLUE_FILL = "#7BAFD4"


def render_setup_only(ax, lens, fill_color):
    """Same as render_one but without the ray bundles."""
    for z_poly, x_poly in lens_fill_polygons(lens):
        ax.fill(z_poly, x_poly, color=fill_color, alpha=0.78,
                linewidth=0.0, zorder=3)
    draw_lens_clean(lens, ax)
    ax.set_aspect("equal"); ax.set_axis_off()


def main():
    print("=" * 70)
    print("CHAMPION DIVERSE — setups only (no rays), one PNG per design")
    print("=" * 70)

    manifest = json.loads(MANIFEST.read_text())
    picks = manifest["picks"]
    print(f"  manifest has {len(picks)} picks")

    bounds = load_bounds(CFG_PATH)
    pool = load_pool(CHAMPION_JSON, bounds)
    by_idx = {r["idx"]: r for r in pool}

    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(CFG_PATH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for pick in picks:
        rank = pick["rank"]
        design_idx = pick["design_idx"]
        rms_um = pick["rms_um"]

        record = by_idx.get(design_idx)
        if record is None:
            print(f"  WARN rank {rank}: design_idx {design_idx} not found in pool")
            continue

        # Single-panel figure, same proportions as one cell of the parent grid.
        fig = plt.figure(figsize=(7.0, 4.0), facecolor="white")
        gs = fig.add_gridspec(
            1, 1,
            left=0.04, right=0.96,
            top=0.93, bottom=0.07,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax.set_facecolor("white")

        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path,
                                np.asarray(record["theta"], dtype=np.float64),
                                template, solver)
            render_setup_only(ax, lens, fill_color=BLUE_FILL)
            ax.text(0.97, 0.04, f"{rms_um:.1f} µm",
                    ha="right", va="bottom", transform=ax.transAxes,
                    fontsize=11, family="serif", color="#1A1A1A",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="#888888", linewidth=0.6, alpha=0.9))
        except Exception as e:
            print(f"  WARN rank {rank} (design {design_idx}) render failed: {e}")
            ax.text(0.5, 0.5, "render failed", ha="center", va="center",
                    transform=ax.transAxes, color="red")
            ax.set_axis_off()
            plt.close(fig)
            continue

        # Outer rounded frame (matches the parent figure aesthetic)
        box = FancyBboxPatch(
            (0.012, 0.012), 0.976, 0.976,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            transform=fig.transFigure,
            edgecolor="#1A1A1A", facecolor="none", linewidth=1.6,
            zorder=10, clip_on=False,
        )
        fig.patches.append(box)

        out_png = OUT_DIR / f"setup_rank{rank:02d}.png"
        fig.savefig(out_png, dpi=200, facecolor="white")
        plt.close(fig)
        print(f"  rank {rank:2d}  rms={rms_um:6.2f} µm  ->  {out_png.name}")

    print(f"\n  all setups in: {OUT_DIR}")


if __name__ == "__main__":
    main()
