#!/usr/bin/env python
"""Render the SAGA basin designs at (S2≈-0.36e-3, S3≈+1.6e-3) — the region where
SAGA found designs HL CHAMPION underexplored — alongside the nearest HL CHAMPION
refined design and the best HL CHAMPION design.

Each panel: lens cross-section + 3-field ray bundles + RMS label.
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

from scripts_throwaway.fig5_extract import build_lens_template_and_solver, make_geolens
from scripts_throwaway.render_spotlight_sixlet_designs import render_one

CFG_PATH = ROOT / "configs/ablations/octuplet_REFSWEEP_CHAMPION.yaml"
HL_RUN = ROOT / "results/lensmega/octuplet_REFSWEEP_CHAMPION_seed42_20260509_145427"
HL_REFINED = HL_RUN / "refined_top500_seed42_CHAMPION.json"
SAGA_REFINED = ROOT / "results/steelman_div020_20260506_123801/octuplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json"
DD = ROOT / "results/figure_archive_demo_octuplet_CHAMPION"
OUT = ROOT / "figures/octuplet_SAGA_basin_vs_HL.png"

BLUE_FILL = "#7BAFD4"  # HL designs
ROSE_FILL = "#D49099"  # SAGA designs (the dusty rose from the contour figure)


def get_hl_designs():
    """Return list of (theta, rms, S2, S3, raw_fit) for HL CHAMPION refined."""
    refined = json.load(open(HL_REFINED))["designs"]
    ac = json.load(open(HL_RUN/"archive_cells.json"))
    fit_to_cell = {round(c["fitness"], 8): c["measures"] for c in ac["cells"]}
    out = []
    for d in refined:
        m = fit_to_cell.get(round(d["raw_fitness"], 8))
        if m is None:
            continue
        out.append({
            "theta": np.asarray(d["theta"], dtype=np.float64),
            "rms_um": float(d["rms_um"]),
            "S2": float(m[0]),
            "S3": float(m[1]),
            "raw_fitness": float(d["raw_fitness"]),
        })
    return out


def get_saga_basin_design():
    """The basin design at (S2≈-0.36e-3, S3≈+1.6e-3) with theta available."""
    above = json.load(open(SAGA_REFINED))["designs"]
    # pool_idx=9 (raw_rms 117.56, refined 142.58) — at the basin S2=-3.6e-4, S3=+1.6e-3
    target = above[9]
    return {
        "theta": np.asarray(target["theta"], dtype=np.float64),
        "rms_um": float(target["rms_um"]),
        "raw_rms_um": float(target["raw_rms_um"]),
        "S2": -3.6e-4,  # from saga_descriptor_recompute
        "S3": +1.6e-3,
        "label": "SAGA basin design",
    }


def get_hl_basin_design(hl_designs, ts2=-0.4e-3, ts3=+1.6e-3, r=0.7e-3):
    """Closest HL CHAMPION design to the basin region."""
    candidates = [d for d in hl_designs if abs(d["S2"]-ts2) < r and abs(d["S3"]-ts3) < r]
    if not candidates:
        return None
    return min(candidates, key=lambda d: d["rms_um"])  # best rms in region


def main():
    print("Loading designs…")
    hl_all = get_hl_designs()
    print(f"  HL CHAMPION refined (with valid S2/S3): {len(hl_all)}")

    saga_basin = get_saga_basin_design()
    print(f"  SAGA basin design: rms={saga_basin['rms_um']:.2f} µm  (raw {saga_basin['raw_rms_um']:.2f}) at S2={saga_basin['S2']:+.2e}, S3={saga_basin['S3']:+.2e}")

    hl_basin = get_hl_basin_design(hl_all)
    if hl_basin:
        print(f"  HL CHAMPION best in basin: rms={hl_basin['rms_um']:.2f} µm  at S2={hl_basin['S2']:+.2e}, S3={hl_basin['S3']:+.2e}")
    else:
        print("  no HL CHAMPION design in basin region!")

    hl_best = min(hl_all, key=lambda d: d["rms_um"])
    print(f"  HL CHAMPION pool best (anywhere): rms={hl_best['rms_um']:.2f} µm  at S2={hl_best['S2']:+.2e}, S3={hl_best['S3']:+.2e}")

    # Build solver
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(CFG_PATH)

    panels = []
    panels.append(("SAGA basin design\n(S₂≈−0.36×10⁻³, S₃≈+1.6×10⁻³)",
                   saga_basin["theta"], saga_basin["rms_um"], ROSE_FILL, "SAGA"))
    if hl_basin:
        panels.append((f"HL CHAMPION — same basin region\n(S₂≈{hl_basin['S2']*1e3:+.2f}×10⁻³, S₃≈{hl_basin['S3']*1e3:+.2f}×10⁻³)",
                       hl_basin["theta"], hl_basin["rms_um"], BLUE_FILL, "HL"))
    panels.append((f"HL CHAMPION — global best\n(S₂={hl_best['S2']*1e3:+.2f}×10⁻³, S₃={hl_best['S3']*1e3:+.2f}×10⁻³)",
                   hl_best["theta"], hl_best["rms_um"], BLUE_FILL, "HL"))

    n = len(panels)
    fig = plt.figure(figsize=(6.5*n, 6.0), facecolor="white")
    gs = fig.add_gridspec(1, n, left=0.04, right=0.96, top=0.78, bottom=0.06, wspace=0.08)
    for k, (title, theta, rms, fill, family) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        ax.set_facecolor("white")
        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
            render_one(ax, lens, fill_color=fill)
            # rms label, bottom-right
            ax.text(0.97, 0.04, f"RMS  {rms:.1f}  µm",
                    ha="right", va="bottom", transform=ax.transAxes,
                    fontsize=12, family="serif", color="#1A1A1A",
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                              edgecolor="#888888", linewidth=0.8, alpha=0.95))
        except Exception as e:
            print(f"  WARN: panel {k} failed: {e}")
            ax.text(0.5, 0.5, f"render failed\n{e}", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=9)
            ax.set_axis_off()

        ax.set_title(title, fontsize=13, family="serif",
                     color="#1A1A1A", pad=10)

    fig.text(0.5, 0.94,
             "What's in the SAGA basin HL CHAMPION underexplored?",
             ha="center", va="center", fontsize=20, family="serif",
             style="italic", color="#1A1A1A")
    fig.text(0.5, 0.89,
             "Left: SAGA's best design at this BD location (RMS ~142 µm — usable but not headline-quality).  "
             "Center: HL's best design in the same BD region (RMS ~770 µm — barely refined, basin not polished).  "
             "Right: HL's pool minimum (RMS 4.78 µm — at a different BD location entirely, the warm-band core).",
             ha="center", va="center", fontsize=11, family="serif",
             style="italic", color="#3A3A3A", wrap=True)

    box = FancyBboxPatch(
        (0.008, 0.008), 0.984, 0.984,
        boxstyle="round,pad=0.0,rounding_size=0.008",
        transform=fig.transFigure,
        edgecolor="#1A1A1A", facecolor="none", linewidth=1.4,
        zorder=10, clip_on=False,
    )
    fig.patches.append(box)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
