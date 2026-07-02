#!/usr/bin/env python
"""Render the top-4 sixlet designs (by RMS, from sixlet_top9_eps030_spotlights.json)
as pretty lens cross-sections with Style C filled bundles. Fill = #0A2850 (deep navy).

Output: figures/sixlet_top4_navy_pretty/<design_NN>_rms_<x.xx>.png
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DeepLens"))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch; torch.set_num_threads(1)

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)
from scripts_throwaway.render_spotlight_sixlet_designs import (
    render_one,        # uses fill_color you pass
)

BLUE_FILL  = "#7BAFD4"
SPOTS_JSON = ROOT / "figures/sixlet_top9_eps030_spotlights.json"
OUT_DIR    = ROOT / "figures/sixlet_top4_blue_pretty"
N_PICK     = 4


def main():
    import time
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spots = json.loads(SPOTS_JSON.read_text())
    # Sort by RMS ascending and take top N
    def _rms(s): return float(s.get("rms_um", s.get("max_per_wvln_rms_um", 9e9)))
    spots_sorted = sorted(spots, key=_rms)[:N_PICK]
    print(f"Top {N_PICK} by RMS:")
    for i, s in enumerate(spots_sorted, 1):
        print(f"  #{i}  RMS={_rms(s):.3f}µm  S2={s.get('S2',0):+.4e}  S3={s.get('S3',0):+.4e}")

    t0 = time.time()
    print("\nbuilding sixlet template+solver ...", flush=True)
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["sixlet"]["cfg"])
    print(f"  ready in {time.time()-t0:.1f}s", flush=True)

    for i, sp in enumerate(spots_sorted, 1):
        ts = time.time()
        for k in ("theta","refined_theta","theta_refined"):
            if k in sp: theta = np.asarray(sp[k], dtype=np.float64); break
        else:
            raise KeyError(f"no theta in entry #{i}: keys={list(sp.keys())}")
        lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)

        fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.6), facecolor="white")
        ax.set_facecolor("white")
        render_one(ax, lens, fill_color=BLUE_FILL)

        rms = _rms(sp)
        out = OUT_DIR / f"design_{i:02d}_rms_{rms:.2f}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  #{i} -> {out.name}  ({time.time()-ts:.1f}s)", flush=True)

    print(f"\ndone in {time.time()-t0:.1f}s  ->  {OUT_DIR}")


if __name__ == "__main__":
    main()
