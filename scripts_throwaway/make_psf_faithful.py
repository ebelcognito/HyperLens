#!/usr/bin/env python
"""Physically faithful polychromatic PSFs for the 4 selected sixlets, 3 fields.

Faithful in three ways the earlier figure was not:
  1. All wavelengths share ONE frame (referenced to the 587 nm centroid), so
     true lateral colour shows as real separation instead of being recentered
     away.
  2. PSF built by histogramming actual traced ray endpoints (the same trace
     that reproduces each design's stored spot RMS), not the recentering
     psf_rgb path.
  3. Each wavelength convolved with its Airy disk (design f-number) so the
     diffraction-limited on-axis cores are physical, not point-like.

Writes NEW files only:
  figures/sixlet_top4_psf/psf_grid_faithful.png
  figures/sixlet_top4_psf/tiles_faithful/d{N}_f{0,1,2}.png
  figures/sixlet_top4_psf/psf_faithful_raw.npz
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
(OUT / "tiles_faithful").mkdir(parents=True, exist_ok=True)

FIELD_LB = ["on-axis", "0.7 field", "full field"]
RMS = [0.89, 1.29, 1.47, 2.09]
WVLN = [0.656, 0.587, 0.486]   # R, G, B  (G=587 is the reference)
DEPTH = -20000.0
NRAYS = 1_500_000
PITCH = 0.5                    # um/bin
PCTL_WIN = 99.2                # window from this ray-radius percentile
MARGIN = 1.18
GAMMA = 0.55
PCTL_NORM = 99.6


def airy_kernel(wvln_um, fnum, pitch_um, rad_factor=4.0):
    """Normalised Airy intensity kernel on a `pitch_um` grid."""
    r1 = 1.22 * wvln_um * fnum            # first-zero radius [um]
    half = int(math.ceil(rad_factor * r1 / pitch_um))
    ax = (np.arange(-half, half + 1)) * pitch_um
    X, Y = np.meshgrid(ax, ax)
    r = np.sqrt(X**2 + Y**2)
    x = math.pi * r / (wvln_um * fnum)
    x = np.where(x < 1e-9, 1e-9, x)
    k = (2 * j1(x) / x) ** 2
    return k / k.sum()


def trace_xy(lens, fields, ap_xy, pupil_z, wvln):
    ray = trace_fields_batch(lens, fields, aperture_xy=ap_xy, pupil_z=pupil_z,
        object_depth_mm=DEPTH, wavelength_um=float(wvln),
        device="cpu", dtype=torch.float64)
    xy = ray.o[..., :2].detach().cpu().numpy() * 1e3   # mm->um, (F,N,2)
    valid = ray.is_valid.detach().cpu().numpy().astype(bool)  # (F,N)
    return xy, valid


def main():
    sel = json.loads((OUT.parent / "sixlet_top4_designs_json/selected_top4.json").read_text())
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["sixlet"]["cfg"])
    aperture = torch.from_numpy(disk_samples(NRAYS)).to(dtype=torch.float64)

    tiles = {}        # (di,fi) -> dict(img, win_um, pitch)
    t0 = time.time()
    for di, entry in enumerate(sel):
        theta = np.asarray(entry["design"]["theta"], dtype=np.float64)
        fnum = float(entry["design"]["fnum_actual"])
        lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
        with torch.no_grad():
            pz, pr = lens.calc_entrance_pupil(paraxial=True)
        ap_xy = aperture * float(pr)
        efl = float(lens.foclen); half = lens.sensor_size[0] / 2.0
        hfov = math.degrees(math.atan(half / abs(efl)))
        fields = ((0.0, 0.0), (0.0, 0.5 * hfov), (0.0, hfov))

        xy = {}; val = {}
        for w in WVLN:
            xy[w], val[w] = trace_xy(lens, fields, ap_xy, float(pz), w)

        for fi in range(3):
            # reference = green (587) valid-ray centroid at this field
            g = val[0.587][fi]
            cref = xy[0.587][fi][g].mean(axis=0)
            # window from combined ray radius about cref
            allr = []
            for w in WVLN:
                m = val[w][fi]
                d = xy[w][fi][m] - cref
                allr.append(np.sqrt((d**2).sum(1)))
            allr = np.concatenate(allr)
            half_um = max(8.0, MARGIN * np.percentile(allr, PCTL_WIN))
            nb = int(np.ceil(2 * half_um / PITCH))
            edges = np.linspace(-half_um, half_um, nb + 1)
            chans = []
            for w in WVLN:
                m = val[w][fi]
                d = xy[w][fi][m] - cref
                H, _, _ = np.histogram2d(d[:, 1], d[:, 0], bins=[edges, edges])  # row=y
                H = fftconvolve(H, airy_kernel(w, fnum, PITCH), mode="same")
                H = gaussian_filter(H, 0.7)
                chans.append(np.clip(H, 0, None))
            img = np.stack(chans, axis=-1)   # HxWx3 (R,G,B)
            tiles[(di, fi)] = dict(img=img, win_um=2 * half_um)
            print(f"  design {di+1} field {fi}: win={2*half_um:6.1f}um  bins={nb}", flush=True)

    np.savez_compressed(OUT / "psf_faithful_raw.npz",
        **{f"d{di}_f{fi}": tiles[(di, fi)]["img"].astype(np.float32)
           for di in range(4) for fi in range(3)},
        win=np.array([[tiles[(di, fi)]["win_um"] for fi in range(3)] for di in range(4)]))
    print(f"traced+built in {time.time()-t0:.1f}s", flush=True)
    render(sel, tiles)


def to_white(img):
    """percentile-normalise + gamma, then paint colour-ink on white."""
    L = img.astype(np.float64)
    norm = np.percentile(L.max(axis=2), PCTL_NORM) or (L.max() or 1.0)
    L = np.clip(L / norm, 0, 1) ** GAMMA
    M = L.max(axis=2, keepdims=True)
    return np.clip(1.0 - (M - L), 0, 1)


def nice_bar(win_um):
    cands = [b for b in (1,2,5,10,20,50,100,200) if b / win_um <= 0.42]
    return cands[-1] if cands else 1


def render(sel, tiles):
    BG = "white"
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(4, 3, figsize=(3*2.7 + 1.3, 4*2.7 + 0.8), facecolor=BG)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.02, wspace=0.07, hspace=0.07)
    for di in range(4):
        for fi in range(3):
            ax = axes[di, fi]; ax.set_facecolor(BG)
            t = tiles[(di, fi)]
            ax.imshow(to_white(t["img"]), interpolation="bilinear", origin="upper")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#c9c9cf"); s.set_linewidth(1.0)
            if di == 0:
                ax.set_title(FIELD_LB[fi], color="#1a1a1a", fontsize=20, pad=12)
            if fi == 0:
                ax.text(-0.16, 0.5, f"RMS {RMS[di]:.2f} µm", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", color="#1a1a1a", fontsize=18)
            bar = nice_bar(t["win_um"]); frac = bar / t["win_um"]
            ax.add_patch(Rectangle((0.06, 0.07), frac, 0.018, transform=ax.transAxes,
                                   facecolor="#222", edgecolor="none"))
            ax.text(0.06 + frac/2, 0.105, f"{bar:g} µm", transform=ax.transAxes,
                    ha="center", va="bottom", color="#222", fontsize=13)
    fig.savefig(OUT / "psf_grid_faithful.png", dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "psf_grid_faithful.png")
    for di in range(4):
        for fi in range(3):
            arr = (np.clip(to_white(tiles[(di, fi)]["img"]), 0, 1) * 255).astype(np.uint8)
            Image.fromarray(arr).resize((512, 512), Image.LANCZOS).save(
                OUT / "tiles_faithful" / f"d{di+1}_f{fi}.png")
    print("wrote 12 tiles ->", OUT / "tiles_faithful")


if __name__ == "__main__":
    main()
