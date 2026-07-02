#!/usr/bin/env python
"""Render three user-supplied triplet designs:
   (a) one pretty lens cross-section PNG per design
   (b) one stability-ribbon PNG per design (chromatic min/max + median across wvlns)
Uses the refined theta from each JSON. Triplet system from SYSTEMS["triplet"]."""
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
from scripts_throwaway.render_spotlight_sixlet_designs import render_one
from scripts_throwaway.render_spotlight_panels_split import (
    per_field_per_wvln_rms_um, BLUE_FILL,
)
from scripts_throwaway.render_storyboard4_seidel_robust_combined import (
    render_robust_simple,
)

IN_DIR  = ROOT / "results/user_three_triplets"
OUT_DIR = ROOT / "figures/user_three_triplets"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})


def main():
    import time
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsons = sorted(IN_DIR.glob("design_*.json"))
    print(f"found {len(jsons)} JSONs in {IN_DIR}")

    t0 = time.time()
    print("building triplet template+solver ...", flush=True)
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["triplet"]["cfg"])
    print(f"  ready in {time.time()-t0:.1f}s", flush=True)

    for jp in jsons:
        ts = time.time()
        name = jp.stem
        d = json.loads(jp.read_text())
        theta = np.asarray(d["refined"]["theta"], dtype=np.float64)
        rms_um = float(d["refined"]["rms_um"])
        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
        except Exception as e:
            print(f"  [{name}] make_geolens FAIL: {e}"); continue

        # ---- (a) pretty cross-section ----
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.6), facecolor="white")
        ax.set_facecolor("white")
        render_one(ax, lens, fill_color=BLUE_FILL)
        out_pretty = OUT_DIR / f"{name}_pretty.png"
        fig.tight_layout()
        fig.savefig(out_pretty, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # ---- (b) stability ribbon ----
        try:
            ff, wv, rms_arr = per_field_per_wvln_rms_um(
                lens, p_problem, n_fields=7, num_rays=256)
        except Exception as e:
            print(f"  [{name}] trace FAIL: {e}"); continue
        fig = plt.figure(figsize=(4.8, 5.6), facecolor="white")
        ax = fig.add_axes([0.18, 0.12, 0.78, 0.82])
        render_robust_simple(ax, ff, wv, rms_arr, BLUE_FILL)
        out_ribbon = OUT_DIR / f"{name}_ribbon.png"
        fig.savefig(out_ribbon, dpi=220, facecolor="white")
        plt.close(fig)

        print(f"  [{name}]  rms_um(refined)={rms_um:.2f}  "
              f"ribbon ymax={float(np.asarray(rms_arr).max()):.2f}µm  "
              f"({time.time()-ts:.1f}s)  -> {out_pretty.name}, {out_ribbon.name}",
              flush=True)

    print(f"\ndone in {time.time()-t0:.1f}s  ->  {OUT_DIR}")


if __name__ == "__main__":
    main()
