#!/usr/bin/env python
"""Render an on-axis, single-wavelength spot diagram (white-on-black) for each
of the 10 CHAMPION-diverse picks. One PNG per design.

Output: figures/octuplet_CHAMPION_diverse_spots/spot_rank{NN}.png
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
from matplotlib.patches import FancyBboxPatch
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

from scripts_throwaway.fig5_extract import (
    build_lens_template_and_solver, make_geolens, load_bounds, load_pool,
)

CHAMPION_JSON = ROOT / "results/lensmega/octuplet_REFSWEEP_CHAMPION_seed42_20260509_145427/refined_top500_seed42_CHAMPION.json"
CFG_PATH = ROOT / "configs/ablations/octuplet_REFSWEEP_CHAMPION.yaml"
MANIFEST = ROOT / "figures/octuplet_CHAMPION_diverse_manifest.json"
OUT_DIR = ROOT / "figures/octuplet_CHAMPION_diverse_spots"

WVLN_UM = 0.587            # single wavelength: d-line
NUM_RAYS = 8192            # dense enough to see the aberration shape
DOT_SIZE = 1.6
DOT_ALPHA = 0.55


def trace_on_axis_spot(lens, p_problem, wvln_um, num_rays):
    """Returns (N, 2) centroid-aligned sensor-plane spot in microns."""
    from optical_rl.qd.lensmega.problem.tracing import (
        disk_samples, trace_fields_batch,
    )
    device = "cpu"
    trace_dtype = lens.surfaces[
        p_problem.free_curvature_indices[0]
    ].c.dtype
    aperture = torch.from_numpy(disk_samples(num_rays)).to(
        dtype=trace_dtype, device=device,
    )
    with torch.no_grad():
        pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
    aperture_xy = aperture * float(pupilr)
    pupil_z = float(pupilz)

    fields = ((0.0, 0.0),)
    ray = trace_fields_batch(
        lens, fields, aperture_xy=aperture_xy, pupil_z=pupil_z,
        object_depth_mm=p_problem.object_depth_mm,
        wavelength_um=float(wvln_um), device=device, dtype=trace_dtype,
    )
    valid = ray.is_valid.detach().cpu().numpy()[0].astype(bool)  # (N,)
    xy_mm = ray.o[..., :2].detach().cpu().numpy()[0]              # (N, 2)
    pts = xy_mm[valid]
    if len(pts) == 0:
        return pts
    pts = pts - pts.mean(axis=0)
    return pts * 1000.0  # mm -> um


def main():
    print("=" * 70)
    print("CHAMPION DIVERSE — on-axis spot diagrams (white-on-black)")
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
            print(f"  WARN rank {rank}: design_idx {design_idx} not found")
            continue

        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path,
                                np.asarray(record["theta"], dtype=np.float64),
                                template, solver)
            pts_um = trace_on_axis_spot(lens, p_problem, WVLN_UM, NUM_RAYS)
        except Exception as e:
            print(f"  WARN rank {rank}: trace failed: {e}")
            continue

        if len(pts_um) == 0:
            print(f"  WARN rank {rank}: no valid rays")
            continue

        # Square frame, generous size
        fig = plt.figure(figsize=(6.0, 6.0), facecolor="black")
        ax = fig.add_axes([0.10, 0.10, 0.80, 0.80])
        ax.set_facecolor("black")

        ax.scatter(pts_um[:, 0], pts_um[:, 1],
                   s=DOT_SIZE, alpha=DOT_ALPHA, color="white",
                   linewidths=0.0)

        max_extent = float(np.abs(pts_um).max())
        lim = max_extent * 1.15 if max_extent > 0 else 1.0
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")

        # White ticks + frame on black
        for spine in ax.spines.values():
            spine.set_color("white"); spine.set_linewidth(0.8)
        ax.tick_params(colors="white", labelsize=9, length=4, width=0.6)
        ax.set_xlabel("x  [µm]", fontsize=11, color="white", family="serif")
        ax.set_ylabel("y  [µm]", fontsize=11, color="white", family="serif")
        ax.grid(True, color="#444444", lw=0.4, alpha=0.6)
        ax.set_axisbelow(True)

        # Header / footer chips in white serif
        fig.text(0.5, 0.955,
                 f"Rank {rank}   ·   on-axis spot   ·   λ = {int(round(WVLN_UM*1000))} nm",
                 ha="center", va="center",
                 fontsize=14, family="serif", style="italic", color="white")
        fig.text(0.5, 0.035,
                 f"refined RMS  {rms_um:.2f} µm   ·   {len(pts_um)} rays",
                 ha="center", va="center",
                 fontsize=11, family="serif", style="italic", color="#BBBBBB")

        # Outer rounded frame, white on black
        box = FancyBboxPatch(
            (0.012, 0.012), 0.976, 0.976,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            transform=fig.transFigure,
            edgecolor="white", facecolor="none", linewidth=1.4,
            zorder=10, clip_on=False,
        )
        fig.patches.append(box)

        out_png = OUT_DIR / f"spot_rank{rank:02d}.png"
        fig.savefig(out_png, dpi=200, facecolor="black")
        plt.close(fig)
        print(f"  rank {rank:2d}  rms={rms_um:6.2f} µm  rays={len(pts_um):4d}  "
              f"extent=±{lim:6.1f} µm  ->  {out_png.name}")

    print(f"\n  all spots in: {OUT_DIR}")


if __name__ == "__main__":
    main()
