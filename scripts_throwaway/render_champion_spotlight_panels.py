#!/usr/bin/env python
"""Render lens + Seidel + stability-ribbon panels for the 4 octuplet CHAMPION
spotlights, using the same pretty pipeline as the sixlet/octuplet method
panels (render_spotlight_panels_split.py).

Inputs:
  results/figure_archive_demo_octuplet_CHAMPION/spotlights.json
    — each entry has  S2, S3, rms_um, theta, role, in_saga_hull

Outputs (in figures/champion_spotlight_panels/):
  champion_<role>_lens.png       Lens cross-section (filled glass + ray bundles)
  champion_<role>_seidel.png     Two-bar |S2|/|S3| chart
  champion_<role>_robust.png     Stability ribbon (RMS vs field, ±wvln spread)

Color convention:
  - HL-exclusive spots (in_saga_hull=False) → blue glass (#7BAFD4)
  - Shared spot (in_saga_hull=True)         → red glass (#E07B7B)
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
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)
from scripts_throwaway.render_spotlight_panels_split import (
    render_lens, render_seidel_s2_s3, render_robust,
    per_field_per_wvln_rms_um,
    BLUE_FILL, RED_FILL,
)

SPOTLIGHTS = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/spotlights.json"
OUT_DIR    = ROOT / "figures/champion_spotlight_panels"
CFG        = SYSTEMS["octuplet"]["cfg"]


def main():
    spots = json.loads(SPOTLIGHTS.read_text())
    print(f"loaded {len(spots)} spotlights from {SPOTLIGHTS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(CFG)

    for sp in spots:
        role = sp["role"]
        is_shared = bool(sp.get("in_saga_hull", False))
        fill = RED_FILL if is_shared else BLUE_FILL
        theta = np.asarray(sp["theta"], dtype=np.float64)

        print(f"\n[{role}]  {'SHARED' if is_shared else 'HL-only'}  "
              f"S2={sp['S2']:+.5f}  S3={sp['S3']:+.5f}  rms={sp['rms_um']:.2f} µm")

        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
        except Exception as e:
            print(f"  FAILED to build lens: {e}")
            continue

        prefix = OUT_DIR / f"champion_{role}"
        render_lens(lens, fill, str(prefix) + "_lens.png")
        render_seidel_s2_s3(sp["S2"], sp["S3"], fill, str(prefix) + "_seidel.png")
        field_fracs, wvlns, rms_arr = per_field_per_wvln_rms_um(
            lens, p_problem, n_fields=7, num_rays=256,
        )
        render_robust(field_fracs, wvlns, rms_arr, fill,
                      str(prefix) + "_robust.png")
        print(f"  -> {prefix}_{{lens,seidel,robust}}.png")

    print(f"\nAll panels in {OUT_DIR}")


if __name__ == "__main__":
    main()
