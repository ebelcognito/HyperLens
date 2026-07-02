#!/usr/bin/env python
"""Render diverse CHAMPION designs as a polished lens-card grid.

Picks N ε-net diverse designs from the CHAMPION refined pool (rms < gate),
renders each as a filled-bundle lens cross-section (soft blue), and arranges
in a clean grid with a serif header.

Output: figures/octuplet_CHAMPION_diverse.png
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
    build_lens_template_and_solver, make_geolens, load_bounds, load_pool, eps_net_select,
)
from scripts_throwaway.render_spotlight_sixlet_designs import render_one

CHAMPION_JSON = ROOT / "results/lensmega/octuplet_REFSWEEP_CHAMPION_seed42_20260509_145427/refined_top500_seed42_CHAMPION.json"
CFG_PATH = ROOT / "configs/ablations/octuplet_REFSWEEP_CHAMPION.yaml"
OUT_PNG = ROOT / "figures/octuplet_CHAMPION_diverse.png"
MANIFEST = ROOT / "figures/octuplet_CHAMPION_diverse_manifest.json"

BLUE_FILL = "#7BAFD4"
N_TARGET = 10
N_ROWS, N_COLS = 2, 5
RMS_GATE_UM = 100.0   # CHAMPION has 273 designs <100 µm; gate filters to that pool
EPS_LADDER = [0.40, 0.35, 0.30, 0.25, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.05, 0.03]


def pick_diverse(pool_clean, dim, n_target):
    """Walk eps ladder; take the loosest eps that yields >= n_target picks."""
    for eps in EPS_LADDER:
        idx = eps_net_select(pool_clean, eps, dim)
        if len(idx) >= n_target:
            chosen = [pool_clean[i] for i in idx[:n_target]]
            print(f"  eps={eps:.2f}: {len(idx)} accepted, taking first {n_target}")
            return chosen, eps
    raise RuntimeError(f"could not fill library (n={n_target}) at any eps in ladder")


def main():
    print("=" * 70)
    print(f"CHAMPION DIVERSE LIBRARY")
    print("=" * 70)

    bounds = load_bounds(CFG_PATH)
    pool = load_pool(CHAMPION_JSON, bounds)
    print(f"  loaded {len(pool)} designs (rms < 1000 um)")

    pool_gated = [r for r in pool if r["rms_um"] < RMS_GATE_UM]
    print(f"  {len(pool_gated)} designs with rms < {RMS_GATE_UM} um")

    if len(pool_gated) < N_TARGET:
        raise RuntimeError(f"only {len(pool_gated)} designs available")

    dim = pool_gated[0]["theta"].size
    chosen, eps_used = pick_diverse(pool_gated, dim, N_TARGET)

    # Pairwise distance summary
    sqrtD = np.sqrt(dim)
    pdists = []
    for a in range(len(chosen)):
        for b in range(a + 1, len(chosen)):
            d = np.linalg.norm(chosen[a]["theta_norm"] - chosen[b]["theta_norm"]) / sqrtD
            pdists.append(d)
    print(f"  pairwise normalized-theta dist: min={min(pdists):.3f} mean={np.mean(pdists):.3f} max={max(pdists):.3f} (eps={eps_used:.2f})")
    for k, r in enumerate(chosen):
        print(f"    #{k+1:2d}  rms={r['rms_um']:6.2f} um  S2={r['S2']:+.2e}  S3={r['S3']:+.2e}")

    # Build solver/template via the matching CHAMPION config (which contains all
    # the topology bits — same problem block as octuplet baseline).
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(CFG_PATH)

    avg_rms = float(np.mean([d["rms_um"] for d in chosen]))
    min_rms = float(min(d["rms_um"] for d in chosen))
    n_more = len(pool_gated) - N_TARGET

    # 5 wide x 2 tall grid → 18 x 8 figure
    fig = plt.figure(figsize=(18.0, 8.0), facecolor="white")
    gs = fig.add_gridspec(
        N_ROWS, N_COLS,
        left=0.04, right=0.96,
        top=0.86, bottom=0.10,
        wspace=0.06, hspace=0.10,
    )
    for k, design in enumerate(chosen):
        ax = fig.add_subplot(gs[k // N_COLS, k % N_COLS])
        ax.set_facecolor("white")
        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path,
                                np.asarray(design["theta"], dtype=np.float64),
                                template, solver)
            render_one(ax, lens, fill_color=BLUE_FILL)
            # Per-panel rms label, bottom-right corner
            ax.text(0.97, 0.04, f"{design['rms_um']:.1f} µm",
                    ha="right", va="bottom", transform=ax.transAxes,
                    fontsize=10, family="serif", color="#1A1A1A",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="#888888", linewidth=0.6, alpha=0.9))
        except Exception as e:
            print(f"  WARN: design #{k+1} render failed: {e}")
            ax.text(0.5, 0.5, "render failed", ha="center", va="center",
                    transform=ax.transAxes, color="red")
            ax.set_axis_off()

    # Mathematica-style serif header
    fig.text(
        0.5, 0.94,
        f"Octuplet CHAMPION  ·  {N_TARGET} ε-distinct designs from full 100k-iter run",
        ha="center", va="center",
        fontsize=22, family="serif", style="italic",
        color="#1A1A1A",
    )
    fig.text(
        0.5, 0.905,
        f"avg RMS  {avg_rms:.1f} µm   ·   best in selection  {min_rms:.2f} µm   ·   pool min RMS  4.78 µm",
        ha="center", va="center",
        fontsize=14, family="serif", style="italic",
        color="#3A3A3A",
    )

    # Footer
    fig.text(
        0.5, 0.045,
        f"… plus {n_more} more diverse designs in the CHAMPION pool with RMS < {RMS_GATE_UM:.0f} µm",
        ha="center", va="center",
        fontsize=12, family="serif", style="italic",
        color="#5A5A5A",
    )

    # Outer frame
    box = FancyBboxPatch(
        (0.012, 0.012), 0.976, 0.976,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        transform=fig.transFigure,
        edgecolor="#1A1A1A", facecolor="none", linewidth=1.6,
        zorder=10, clip_on=False,
    )
    fig.patches.append(box)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"\n  -> {OUT_PNG}")

    # Manifest
    manifest = {
        "source": str(CHAMPION_JSON),
        "n_target": N_TARGET,
        "rms_gate_um": RMS_GATE_UM,
        "eps_used": eps_used,
        "pool_size_after_gate": len(pool_gated),
        "picks": [
            {"rank": k + 1, "design_idx": int(r["idx"]),
             "rms_um": float(r["rms_um"]),
             "S2": float(r["S2"]), "S3": float(r["S3"])}
            for k, r in enumerate(chosen)
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"  -> {MANIFEST}")


if __name__ == "__main__":
    main()
