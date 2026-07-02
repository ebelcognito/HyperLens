#!/usr/bin/env python
"""For the top-9 ε-distinct (ε > 0.30, normalised 37-D θ) sub-20µm sixlet
designs pooled across every HL run, produce per-design PNGs:
   * lens-card PNGs   — one cross-section per design
   * ribbon PNGs      — one chromatic-ribbon plot per design (NO Seidel bars)
"""
from __future__ import annotations
import json, os, sys, glob
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
from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
    load_bounds, norm_theta, eps_net_select,
)
from scripts_throwaway.render_spotlight_sixlet_designs import render_one
from scripts_throwaway.render_storyboard4_seidel_robust_combined import render_robust_simple
from scripts_throwaway.render_spotlight_panels_split import (
    per_field_per_wvln_rms_um, BLUE_FILL,
)

RMS_CAP   = 20.0
EPS       = 0.30
N_TOP     = 9
SYS       = "sixlet"
SPOTS_OUT = ROOT / "figures/sixlet_top9_eps030_spotlights.json"
LENS_DIR  = ROOT / "figures/sixlet_top9_eps030_lenses"
RIBBON_DIR= ROOT / "figures/sixlet_top9_eps030_ribbons"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})


def gather_pool():
    bounds = load_bounds(SYSTEMS[SYS]["cfg"])
    D = bounds.shape[0]
    pool = []
    for fp in sorted(glob.glob(str(ROOT / f"results/lensmega/{SYS}*/refined_top*.json"))):
        run = Path(fp).parent.name
        d = json.load(open(fp))
        for i, des in enumerate(d["designs"]):
            r = float(des.get("rms_um", float("nan")))
            if r >= RMS_CAP: continue
            th = np.asarray(des["theta"], dtype=np.float64)
            if len(th) != D: continue
            m = des.get("metrics", {})
            pool.append({
                "run": run, "idx": i, "rms_um": r,
                "theta": list(th),
                "theta_norm": norm_theta(th, bounds),
                "refined_fitness": float(des.get("refined_fitness", 0.0)),
                "S1": float(m.get("seidel_S1", float("nan"))),
                "S2": float(m.get("seidel_S2", float("nan"))),
                "S3": float(m.get("seidel_S3", float("nan"))),
                "S4": float(m.get("seidel_S4", float("nan"))),
                "S5": float(m.get("seidel_S5", float("nan"))),
            })
    pool.sort(key=lambda r: -r["refined_fitness"])
    keep = eps_net_select(pool, EPS, D)
    return [pool[i] for i in keep[:N_TOP]], D


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
    picks, D = gather_pool()
    print(f"selected top-{N_TOP} ε-distinct (ε > {EPS}) sub-{RMS_CAP:.0f}µm sixlet designs")
    serial = [{k: v for k, v in p.items() if k != "theta_norm"} for p in picks]
    SPOTS_OUT.write_text(json.dumps(serial, indent=2))
    print(f"  spotlights -> {SPOTS_OUT.name}")

    LENS_DIR.mkdir(parents=True, exist_ok=True)
    RIBBON_DIR.mkdir(parents=True, exist_ok=True)

    n_workers = min(len(picks), int(os.environ.get("WORKERS", "9")))
    print(f"tracing {len(picks)} designs with {n_workers} workers...")
    jobs = [(k, sp["theta"]) for k, sp in enumerate(picks)]
    traces = {}
    with Pool(processes=n_workers, initializer=_init_worker) as pl:
        for idx, ff, wv, rms, err in pl.imap_unordered(_trace_worker, jobs):
            if err:
                print(f"  trace #{idx+1} FAIL: {err}"); traces[idx] = None
            else:
                traces[idx] = (np.asarray(ff), wv, np.asarray(rms))
    print("traces done.")

    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS[SYS]["cfg"])

    for k, sp in enumerate(picks):
        tag = f"design_{k+1:02d}_rms_{sp['rms_um']:.2f}"

        figL, axL = plt.subplots(figsize=(7.0, 3.4), facecolor="white")
        axL.set_facecolor("white")
        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path,
                                np.asarray(sp["theta"], dtype=np.float64),
                                template, solver)
            render_one(axL, lens, fill_color=BLUE_FILL)
        except Exception as e:
            print(f"  lens #{k+1} fail: {e}")
            axL.text(0.5, 0.5, "render fail", ha="center", va="center",
                     transform=axL.transAxes, color="red"); axL.set_axis_off()
        figL.tight_layout()
        outL = LENS_DIR / f"{tag}.png"
        figL.savefig(outL, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figL)

        t = traces.get(k)
        if t is None: continue
        ff, wv, rms_arr = t
        figR, axR = plt.subplots(figsize=(6.0, 4.2), facecolor="white")
        render_robust_simple(axR, ff, wv, rms_arr, BLUE_FILL)
        outR = RIBBON_DIR / f"{tag}.png"
        figR.savefig(outR, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(figR)
        print(f"  [#{k+1:2d}] {tag}.png  (lens + ribbon)")

    print(f"\nlenses -> {LENS_DIR}\nribbons -> {RIBBON_DIR}")


if __name__ == "__main__":
    main()
