#!/usr/bin/env python
"""Uniform-scale spot + PSF grids: every tile is the SAME fixed box (default
20 um), lateral colour removed (each wavelength registered to its own
centroid), Airy diffraction included for the PSF. So all 12 tiles share one
spatial scale and small spots actually read small.

Env: BOX_UM (default 20).  Writes NEW files only:
  figures/sixlet_top4_psf/spot_grid_uniform.png   + spot_tiles_uniform/
  figures/sixlet_top4_psf/psf_grid_uniform.png    + psf_tiles_uniform/
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
import numpy as np, torch
from scipy.special import j1
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens)
from optical_rl.qd.lensmega.problem.tracing import disk_samples, trace_fields_batch
torch.set_num_threads(int(_NTH))

OUT = ROOT / "figures" / "sixlet_top4_psf"
for sub in ("spot_tiles_uniform", "psf_tiles_uniform"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

FIELD_LB = ["on-axis", "0.7 field", "full field"]
RMS = [0.89, 1.29, 1.47, 2.09]
WVLN = [0.656, 0.587, 0.486]
COL  = {0.656: "#d62728", 0.587: "#2ca02c", 0.486: "#1f77b4"}
DEPTH = -20000.0
NRAYS = int(os.environ.get("NRAYS", "400000"))
BOX_UM = float(os.environ.get("BOX_UM", "20"))   # full box width
HALF = BOX_UM / 2.0
PITCH = float(os.environ.get("PITCH", "0.2"))    # um/bin for the PSF image
GAMMA = 0.55
PCTL_NORM = 99.6
SUFFIX = os.environ.get("SUFFIX", "")            # output filename suffix
DPI = int(os.environ.get("DPI", "400"))
SAVE_PDF = os.environ.get("SAVE_PDF", "0") == "1"


def airy_kernel(wvln_um, fnum, pitch_um, rad_factor=4.0):
    r1 = 1.22 * wvln_um * fnum
    half = int(math.ceil(rad_factor * r1 / pitch_um))
    ax = np.arange(-half, half + 1) * pitch_um
    X, Y = np.meshgrid(ax, ax)
    x = math.pi * np.sqrt(X**2 + Y**2) / (wvln_um * fnum)
    x = np.where(x < 1e-9, 1e-9, x)
    k = (2 * j1(x) / x) ** 2
    return k / k.sum()


def trace_xy(lens, fields, ap_xy, pupil_z, wvln):
    ray = trace_fields_batch(lens, fields, aperture_xy=ap_xy, pupil_z=pupil_z,
        object_depth_mm=DEPTH, wavelength_um=float(wvln), device="cpu", dtype=torch.float64)
    return (ray.o[..., :2].detach().cpu().numpy() * 1e3,
            ray.is_valid.detach().cpu().numpy().astype(bool))


def main():
    sel = json.loads((OUT.parent / "sixlet_top4_designs_json/selected_top4.json").read_text())
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["sixlet"]["cfg"])
    ap = torch.from_numpy(disk_samples(NRAYS)).to(dtype=torch.float64)
    spots = {}; psfs = {}; p99 = {}
    edges = np.linspace(-HALF, HALF, int(round(2*HALF/PITCH)) + 1)
    t0 = time.time()
    for di, entry in enumerate(sel):
        theta = np.asarray(entry["design"]["theta"], dtype=np.float64)
        fnum = float(entry["design"]["fnum_actual"])
        lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
        with torch.no_grad():
            pz, pr = lens.calc_entrance_pupil(paraxial=True)
        ap_xy = ap * float(pr)
        efl = float(lens.foclen); half = lens.sensor_size[0] / 2.0
        hfov = math.degrees(math.atan(half / abs(efl)))
        fields = ((0.0, 0.0), (0.0, 0.5*hfov), (0.0, hfov))
        xy = {}; val = {}
        for w in WVLN:
            xy[w], val[w] = trace_xy(lens, fields, ap_xy, float(pz), w)
        for fi in range(3):
            sp = {}; chans = []; rmax = 0.0
            for w in WVLN:
                m = val[w][fi]
                p = xy[w][fi][m]
                p = p - p.mean(axis=0)               # register: lateral colour removed
                sp[w] = p
                rmax = max(rmax, np.percentile(np.sqrt((p**2).sum(1)), 99.0))
                H, _, _ = np.histogram2d(p[:, 1], p[:, 0], bins=[edges, edges])
                H = fftconvolve(H, airy_kernel(w, fnum, PITCH), mode="same")
                chans.append(gaussian_filter(np.clip(H, 0, None), 0.6))
            spots[(di, fi)] = sp
            psfs[(di, fi)] = np.stack(chans, axis=-1)
            p99[(di, fi)] = rmax
    print(f"traced in {time.time()-t0:.1f}s")
    print(f"BOX={BOX_UM}um (half={HALF}).  per-tile 99% spot radius (um) -- clips if > {HALF}:")
    for di in range(4):
        print("  RMS %.2f: " % RMS[di] + ", ".join(
            f"{FIELD_LB[fi]} {p99[(di,fi)]:.1f}" for fi in range(3)))
    render_spot(spots); render_psf(psfs)


def _decorate(ax, di, fi):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#c9c9cf"); s.set_linewidth(1.0)
    if di == 0:
        ax.set_title(FIELD_LB[fi], color="#1a1a1a", fontsize=20, pad=12)
    if fi == 0:
        ax.text(-0.16, 0.5, f"RMS {RMS[di]:.2f} µm", transform=ax.transAxes,
                rotation=90, va="center", ha="center", color="#1a1a1a", fontsize=20)


def _scalebar(ax, dark=True):
    bar = 5.0; frac = bar / BOX_UM
    c = "#222"
    ax.add_patch(Rectangle((0.06, 0.06), frac, 0.024, transform=ax.transAxes,
                           facecolor=c, edgecolor="none"))
    ax.text(0.06 + frac/2, 0.10, f"{bar:g} µm", transform=ax.transAxes,
            ha="center", va="bottom", color=c, fontsize=15)


N_SCATTER = 26000   # points per wavelength (dense -> filled, smooth shape)


def _scatter_mixed(ax, sp, s, alpha):
    """Randomly subsample each wavelength (random avoids the spiral-aliasing
    gaps a strided pick creates), then plot all points in shuffled z-order so
    no colour systematically sits on top."""
    rng = np.random.default_rng(0)
    P = []; C = []
    for w in WVLN:
        p = sp[w]
        if len(p) > N_SCATTER:
            p = p[rng.choice(len(p), N_SCATTER, replace=False)]
        P.append(p); C.append(np.tile(np.array(_rgb(COL[w])), (len(p), 1)))
    P = np.concatenate(P); C = np.concatenate(C)
    order = rng.permutation(len(P))   # deterministic shuffle of z-order
    P = P[order]; C = C[order]
    ax.scatter(P[:, 0], P[:, 1], s=s, c=C, alpha=alpha, edgecolors="none", rasterized=True)


def _rgb(hexc):
    h = hexc.lstrip("#")
    return [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]


def render_spot(spots):
    fig, axes = plt.subplots(4, 3, figsize=(3*2.7 + 1.3, 4*2.7 + 0.8), facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.02, wspace=0.07, hspace=0.07)
    for di in range(4):
        for fi in range(3):
            ax = axes[di, fi]
            _scatter_mixed(ax, spots[(di, fi)], s=2.2, alpha=0.38)
            ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF); ax.set_aspect("equal")
            _decorate(ax, di, fi); _scalebar(ax)
    fig.savefig(OUT / f"spot_grid_uniform{SUFFIX}.png", dpi=DPI, facecolor="white", bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(OUT / f"spot_grid_uniform{SUFFIX}.pdf", dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig); print("wrote", OUT / f"spot_grid_uniform{SUFFIX}.png")
    for di in range(4):
        for fi in range(3):
            f, ax = plt.subplots(figsize=(3, 3), facecolor="white")
            _scatter_mixed(ax, spots[(di, fi)], s=3.0, alpha=0.42)
            ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#d0d0d4")
            f.savefig(OUT / "spot_tiles_uniform" / f"d{di+1}_f{fi}.png", dpi=300,
                      facecolor="white", bbox_inches="tight", pad_inches=0.02)
            plt.close(f)
    print("wrote 12 ->", OUT / "spot_tiles_uniform")


def to_white(img):
    L = img.astype(np.float64)
    norm = np.percentile(L.max(axis=2), PCTL_NORM) or (L.max() or 1.0)
    L = np.clip(L / norm, 0, 1) ** GAMMA
    M = L.max(axis=2, keepdims=True)
    return np.clip(1.0 - (M - L), 0, 1)


def render_psf(psfs):
    fig, axes = plt.subplots(4, 3, figsize=(3*2.7 + 1.3, 4*2.7 + 0.8), facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.02, wspace=0.07, hspace=0.07)
    for di in range(4):
        for fi in range(3):
            ax = axes[di, fi]
            ax.imshow(to_white(psfs[(di, fi)]), interpolation="bilinear",
                      origin="upper", extent=[-HALF, HALF, -HALF, HALF])
            ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF)
            _decorate(ax, di, fi); _scalebar(ax)
    fig.savefig(OUT / f"psf_grid_uniform{SUFFIX}.png", dpi=DPI, facecolor="white", bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(OUT / f"psf_grid_uniform{SUFFIX}.pdf", dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig); print("wrote", OUT / f"psf_grid_uniform{SUFFIX}.png")
    for di in range(4):
        for fi in range(3):
            arr = (np.clip(to_white(psfs[(di, fi)]), 0, 1) * 255).astype(np.uint8)
            Image.fromarray(arr).resize((900, 900), Image.LANCZOS).save(
                OUT / "psf_tiles_uniform" / f"d{di+1}_f{fi}.png")
    print("wrote 12 ->", OUT / "psf_tiles_uniform")


if __name__ == "__main__":
    main()
