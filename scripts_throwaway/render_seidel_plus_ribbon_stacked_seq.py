#!/usr/bin/env python
"""Sequential version of render_seidel_plus_ribbon_stacked.py — no Pool, no
worker re-import tax. Builds the sixlet solver once in the parent and traces
the 4 spotlight designs back-to-back."""
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

from scripts_throwaway.fig5_extract import SYSTEMS, build_lens_template_and_solver, make_geolens
from scripts_throwaway.render_storyboard4_seidel_robust_combined import (
    render_seidel_simple, render_robust_simple,
)
from scripts_throwaway.render_spotlight_panels_split import (
    per_field_per_wvln_rms_um, BLUE_FILL, RED_FILL,
)

SYS    = os.environ.get("SYS", "sixlet")
SPOTS  = Path(os.environ.get("SPOTS",
              str(ROOT / "results/figure_archive_demo/spotlights.json")))
OUTDIR = Path(os.environ.get("OUTDIR",
              str(ROOT / f"figures/{SYS}_seidel_ribbon_stacked")))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

PLOT_W_IN     = 4.0
RIBBON_ASPECT = 2.0 / 1.7
SEIDEL_ASPECT = 0.85
GAP_IN        = 0.55
MARGIN_L_IN   = 0.95
MARGIN_R_IN   = 0.20
MARGIN_B_IN   = 0.55
MARGIN_T_IN   = 0.15

RIBBON_H_IN = PLOT_W_IN * RIBBON_ASPECT
SEIDEL_H_IN = PLOT_W_IN * SEIDEL_ASPECT
FIG_W_IN = MARGIN_L_IN + PLOT_W_IN + MARGIN_R_IN
FIG_H_IN = MARGIN_B_IN + RIBBON_H_IN + GAP_IN + SEIDEL_H_IN + MARGIN_T_IN

LEFT          = MARGIN_L_IN / FIG_W_IN
WIDTH         = PLOT_W_IN / FIG_W_IN
RIBBON_BOTTOM = MARGIN_B_IN / FIG_H_IN
RIBBON_HEIGHT = RIBBON_H_IN / FIG_H_IN
SEIDEL_BOTTOM = (MARGIN_B_IN + RIBBON_H_IN + GAP_IN) / FIG_H_IN
SEIDEL_HEIGHT = SEIDEL_H_IN / FIG_H_IN


def main():
    import time
    spots = json.loads(SPOTS.read_text())
    print(f"system={SYS}  loaded {len(spots)} spots from {SPOTS}", flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("building template + solver ...", flush=True)
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS[SYS]["cfg"])
    print(f"  ready in {time.time()-t0:.1f}s", flush=True)

    def _theta(sp):
        for k in ("theta", "refined_theta", "theta_refined"):
            if k in sp: return sp[k]
        raise KeyError(f"no theta in spotlight: keys={list(sp.keys())}")

    for k, sp in enumerate(spots):
        ts = time.time()
        theta = np.asarray(_theta(sp), dtype=np.float64)
        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
            ff, wv, rms_arr = per_field_per_wvln_rms_um(lens, p_problem, n_fields=7, num_rays=256)
        except Exception as e:
            print(f"  [#{k}] FAIL: {e}", flush=True); continue
        fill = RED_FILL if sp.get("in_saga_hull") else BLUE_FILL

        fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")
        ax_s = fig.add_axes([LEFT, SEIDEL_BOTTOM, WIDTH, SEIDEL_HEIGHT])
        render_seidel_simple(ax_s, sp["S2"], sp["S3"], fill)
        ax_s.set_box_aspect(None)
        ax_r = fig.add_axes([LEFT, RIBBON_BOTTOM, WIDTH, RIBBON_HEIGHT])
        render_robust_simple(ax_r, ff, wv, rms_arr, fill)
        ax_r.set_box_aspect(None)

        role = sp.get("role") or sp.get("label") or f"design_{k+1:02d}"
        out = OUTDIR / f"{role}_seidel_ribbon_stacked.png"
        fig.savefig(out, dpi=220, facecolor="white")
        plt.close(fig)
        print(f"  [{role}] traced+rendered in {time.time()-ts:.1f}s -> {out.name}", flush=True)

    print(f"\noutput dir: {OUTDIR}  (total {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
