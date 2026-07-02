#!/usr/bin/env python
"""Spot diagrams with lateral colour REMOVED -- each wavelength registered to
its own chief-ray centroid (the residual after the standard per-channel
lateral-CA correction every raw pipeline applies). Shows true monochromatic
sharpness; lateral colour is reported separately, not drawn.

White background, large text. NEW files only:
  figures/sixlet_top4_psf/spot_grid_registered.png
  figures/sixlet_top4_psf/spot_tiles_registered/d{N}_f{0,1,2}.png
"""
from __future__ import annotations
import os, sys, json, math, time
_NTH = os.environ.get("PSF_THREADS", "8")
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, _NTH)
os.environ.setdefault("MPMATH_NOGMPY", "1")
from pathlib import Path
ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "DeepLens"))
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens)
from optical_rl.qd.lensmega.problem.tracing import disk_samples, trace_fields_batch
torch.set_num_threads(int(_NTH))

OUT = ROOT / "figures" / "sixlet_top4_psf"
(OUT / "spot_tiles_registered").mkdir(parents=True, exist_ok=True)
FIELD_LB = ["on-axis", "0.7 field", "full field"]
RMS = [0.89, 1.29, 1.47, 2.09]
WVLN = [0.656, 0.587, 0.486]
COL  = {0.656: "#d62728", 0.587: "#2ca02c", 0.486: "#1f77b4"}
DEPTH = -20000.0; NRAYS = 4000; PCTL_WIN = 99.5; MARGIN = 1.25


def trace_xy(lens, fields, ap_xy, pupil_z, wvln):
    ray = trace_fields_batch(lens, fields, aperture_xy=ap_xy, pupil_z=pupil_z,
        object_depth_mm=DEPTH, wavelength_um=float(wvln), device="cpu", dtype=torch.float64)
    return (ray.o[..., :2].detach().cpu().numpy() * 1e3,
            ray.is_valid.detach().cpu().numpy().astype(bool))


def collect():
    sel = json.loads((OUT.parent / "sixlet_top4_designs_json/selected_top4.json").read_text())
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["sixlet"]["cfg"])
    ap = torch.from_numpy(disk_samples(NRAYS)).to(dtype=torch.float64)
    data = {}; latcol = {}
    for di, entry in enumerate(sel):
        theta = np.asarray(entry["design"]["theta"], dtype=np.float64)
        lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
        with torch.no_grad():
            pz, pr = lens.calc_entrance_pupil(paraxial=True)
        ap_xy = ap * float(pr)
        efl = float(lens.foclen); half = lens.sensor_size[0] / 2.0
        hfov = math.degrees(math.atan(half / abs(efl)))
        fields = ((0.0, 0.0), (0.0, 0.5 * hfov), (0.0, hfov))
        xy = {}; val = {}
        for w in WVLN:
            xy[w], val[w] = trace_xy(lens, fields, ap_xy, float(pz), w)
        for fi in range(3):
            pts = {}; cents = []; allr = []
            for w in WVLN:
                m = val[w][fi]
                c = xy[w][fi][m].mean(axis=0)          # this wavelength's own centroid
                p = xy[w][fi][m] - c                    # register to it -> lateral CA removed
                pts[w] = p; cents.append(c)
                allr.append(np.sqrt((p**2).sum(1)))
            allr = np.concatenate(allr)
            win = max(4.0, MARGIN * np.percentile(allr, PCTL_WIN))
            cg = cents[1]
            lc = max(np.linalg.norm(cents[0] - cg), np.linalg.norm(cents[2] - cg))  # R/B vs G
            data[(di, fi)] = dict(pts=pts, win=win)
            latcol[(di, fi)] = lc
    return data, latcol


def nice_bar(win):
    cands = [b for b in (0.5,1,2,5,10,20,50) if (b / (2*win)) <= 0.42]
    return cands[-1] if cands else 0.5


def render(data, latcol):
    BG = "white"
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(4, 3, figsize=(3*2.7 + 1.3, 4*2.7 + 0.8), facecolor=BG)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.02, wspace=0.07, hspace=0.07)
    for di in range(4):
        for fi in range(3):
            ax = axes[di, fi]; ax.set_facecolor(BG)
            d = data[(di, fi)]; win = d["win"]
            for w in WVLN:
                p = d["pts"][w]
                ax.scatter(p[:, 0], p[:, 1], s=3.2, c=COL[w], alpha=0.5,
                           edgecolors="none", rasterized=True)
            ax.set_xlim(-win, win); ax.set_ylim(-win, win); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#c9c9cf"); s.set_linewidth(1.0)
            if di == 0:
                ax.set_title(FIELD_LB[fi], color="#1a1a1a", fontsize=20, pad=12)
            if fi == 0:
                ax.text(-0.16, 0.5, f"RMS {RMS[di]:.2f} µm", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", color="#1a1a1a", fontsize=18)
            bar = nice_bar(win); frac = bar / (2 * win)
            ax.add_patch(Rectangle((0.06, 0.07), frac, 0.02, transform=ax.transAxes,
                                   facecolor="#222", edgecolor="none"))
            ax.text(0.06 + frac/2, 0.105, f"{bar:g} µm", transform=ax.transAxes,
                    ha="center", va="bottom", color="#222", fontsize=13)
    fig.savefig(OUT / "spot_grid_registered.png", dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "spot_grid_registered.png")
    for di in range(4):
        for fi in range(3):
            d = data[(di, fi)]; win = d["win"]
            f, ax = plt.subplots(figsize=(3, 3), facecolor=BG)
            for w in WVLN:
                p = d["pts"][w]
                ax.scatter(p[:, 0], p[:, 1], s=4.5, c=COL[w], alpha=0.55, edgecolors="none")
            ax.set_xlim(-win, win); ax.set_ylim(-win, win); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#d0d0d4")
            f.savefig(OUT / "spot_tiles_registered" / f"d{di+1}_f{fi}.png", dpi=170,
                      facecolor=BG, bbox_inches="tight", pad_inches=0.02)
            plt.close(f)
    print("wrote 12 tiles ->", OUT / "spot_tiles_registered")
    print("\nlateral colour removed (R/B vs G, um):")
    for di in range(4):
        print(f"  RMS {RMS[di]:.2f}: " + ", ".join(
            f"{FIELD_LB[fi]} {latcol[(di,fi)]:.1f}" for fi in range(3)))


if __name__ == "__main__":
    t0 = time.time()
    render(*collect())
    print(f"done {time.time()-t0:.1f}s")
