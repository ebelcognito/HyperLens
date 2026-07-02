#!/usr/bin/env python
"""Render Seidel-bar + chromatic-ribbon as a SINGLE vertically-stacked PNG
per design. The two plot boxes share the same width (in figure coords) so
they visually align in the composite. Output: one PNG per spotlight entry.

Usage (env-driven):
    SPOTS=<json path>  OUTDIR=<dir>  SYS=<sixlet|octuplet>  python ...
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch; torch.set_num_threads(1)
from multiprocessing import Pool

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
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

# --- Layout (everything in inches, converted to figure-relative coords below) ---
PLOT_W_IN     = 4.0           # plot box width (same for both axes)
RIBBON_ASPECT = 2.0 / 1.7     # ribbon box H:W
SEIDEL_ASPECT = 0.85          # seidel box H:W (slightly less than square)
GAP_IN        = 0.55          # vertical gap between the two boxes
MARGIN_L_IN   = 0.95          # space for the y-tick labels
MARGIN_R_IN   = 0.20
MARGIN_B_IN   = 0.55          # space for the x-axis labels under the ribbon
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


_TEMPLATE = _SOLVER = _PC = _LP = _PP = None
def _init_worker():
    global _TEMPLATE, _SOLVER, _PC, _LP, _PP
    import torch; torch.set_num_threads(1)
    _TEMPLATE, _SOLVER, _PC, _LP, _PP = build_lens_template_and_solver(
        SYSTEMS[SYS]["cfg"])


def _trace_worker(args):
    idx, theta_list = args
    import numpy as np
    theta = np.asarray(theta_list, dtype=np.float64)
    try:
        lens = make_geolens(_PC, _PP, _LP, theta, _TEMPLATE, _SOLVER)
        ff, wv, rms = per_field_per_wvln_rms_um(lens, _PP, n_fields=7, num_rays=256)
        return idx, ff.tolist(), list(wv), np.asarray(rms).tolist(), None
    except Exception as e:
        return idx, None, None, None, str(e)


def main():
    spots = json.loads(SPOTS.read_text())
    print(f"system={SYS}  loaded {len(spots)} spots from {SPOTS}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    n_workers = min(len(spots), int(os.environ.get("WORKERS", "8")))
    def _theta(sp):
        for k in ("theta", "refined_theta", "theta_refined"):
            if k in sp: return sp[k]
        raise KeyError(f"no theta in spotlight: keys={list(sp.keys())}")
    jobs = [(k, _theta(sp)) for k, sp in enumerate(spots)]
    traces = {}
    with Pool(processes=n_workers, initializer=_init_worker) as pl:
        for idx, ff, wv, rms, err in pl.imap_unordered(_trace_worker, jobs):
            if err:
                print(f"  trace #{idx+1} FAIL: {err}"); traces[idx] = None
            else:
                traces[idx] = (np.asarray(ff), wv, np.asarray(rms))
    print(f"traces done ({sum(v is not None for v in traces.values())} ok)")

    for k, sp in enumerate(spots):
        t = traces.get(k)
        if t is None: continue
        ff, wv, rms_arr = t
        fill = RED_FILL if sp.get("in_saga_hull") else BLUE_FILL

        fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")
        ax_s = fig.add_axes([LEFT, SEIDEL_BOTTOM, WIDTH, SEIDEL_HEIGHT])
        render_seidel_simple(ax_s, sp["S2"], sp["S3"], fill)
        ax_s.set_box_aspect(None)   # axes rectangle defines the box now

        ax_r = fig.add_axes([LEFT, RIBBON_BOTTOM, WIDTH, RIBBON_HEIGHT])
        render_robust_simple(ax_r, ff, wv, rms_arr, fill)
        ax_r.set_box_aspect(None)

        role = sp.get("role") or sp.get("label") or f"design_{k+1:02d}"
        out = OUTDIR / f"{role}_seidel_ribbon_stacked.png"
        fig.savefig(out, dpi=220, facecolor="white")
        plt.close(fig)
        print(f"  [{role}]  -> {out.name}")

    print(f"\noutput dir: {OUTDIR}")


if __name__ == "__main__":
    main()
