#!/usr/bin/env python
"""Pretty PSF plots for the 4 selected sixlet designs, 3 fields each.

RGB geometric PSFs (R=656, G=587, B=486 nm) at fields {0, 0.7, 1.0} of the
full diagonal. Clean, Mathematica-style: PSF glow on near-black, per-field
display window so each spot is well-sized, one thin scale bar per column.

Outputs (figures/sixlet_top4_psf/):
  psf_grid.png            4 designs (rows) x 3 fields (cols), labelled
  tiles/d{N}_f{F}.png     12 individual untitled tiles (pure PSF)
  psf_raw.npz             the raw [3,ks,ks] float PSFs
"""
from __future__ import annotations
import os, sys, json, time
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
try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None
from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens)
torch.set_num_threads(int(_NTH))  # fig5_extract pins this to 1 on import; restore

OUT = ROOT / "figures" / "sixlet_top4_psf"
(OUT / "tiles").mkdir(parents=True, exist_ok=True)

FIELDS   = [0.0, 0.7, 1.0]                 # fraction of full diagonal field
FIELD_LB = ["on-axis", "0.7 field", "full field"]
DEPTH    = -20000.0
SPP      = 1_000_000                        # rays/wvln -> smooth PSF
RES_FACTOR = 30                             # fine pixel pitch (~0.4 um)
KS       = 251                              # compute window (~100 um) then crop
GAMMA    = 0.5                              # display stretch (show faint wings)
SMOOTH_PX = 0.8                             # light de-noise


def compute_psf_rgb(lens, f):
    pt = torch.tensor([f, f, DEPTH], dtype=torch.float32)
    with torch.no_grad():
        psf = lens.psf_rgb(points=pt, ks=KS, recenter=True, model="geometric", spp=SPP)
    return psf.cpu().numpy()  # [3, KS, KS]


def r_frac_px(gray, frac):
    """Radius (px from centroid) containing `frac` of energy."""
    ks = gray.shape[0]
    g = gray / (gray.sum() + 1e-12)
    yy, xx = np.mgrid[0:ks, 0:ks]
    cy = (g * yy).sum(); cx = (g * xx).sum()
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    o = np.argsort(r.ravel()); cum = np.cumsum(g.ravel()[o])
    return float(r.ravel()[o][np.searchsorted(cum, frac)])


def tone(rgb, half_px):
    """Crop center +/- half_px, normalise per-tile, gamma, light smooth -> HxWx3 in [0,1]."""
    ks = rgb.shape[1]; c = ks // 2
    h = int(round(half_px))
    crop = rgb[:, c - h:c + h + 1, c - h:c + h + 1]
    img = np.transpose(crop, (1, 2, 0)).astype(np.float64)
    if gaussian_filter is not None and SMOOTH_PX > 0:
        for k in range(3):
            img[..., k] = gaussian_filter(img[..., k], SMOOTH_PX)
    m = img.max()
    if m > 0:
        img = img / m
    img = np.clip(img, 0, 1) ** GAMMA
    return img


def main():
    sel = json.loads((OUT.parent / "sixlet_top4_designs_json/selected_top4.json").read_text())
    t0 = time.time()
    print("building sixlet template+solver ...", flush=True)
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["sixlet"]["cfg"])

    # ---- compute all PSFs ----
    raw = {}          # (di, fi) -> [3,KS,KS]
    pix_um = None
    for di, entry in enumerate(sel):
        theta = np.asarray(entry["design"]["theta"], dtype=np.float64)
        lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
        W, H = lens.sensor_res
        lens.set_sensor_res((W * RES_FACTOR, H * RES_FACTOR))
        pix_um = lens.pixel_size * 1e3
        for fi, f in enumerate(FIELDS):
            ts = time.time()
            raw[(di, fi)] = compute_psf_rgb(lens, f)
            print(f"  design {di+1} (RMS {entry['rms_um']:.2f}) field {f:.1f}: "
                  f"{time.time()-ts:.1f}s", flush=True)

    np.savez_compressed(OUT / "psf_raw.npz", pix_um=pix_um,
                        **{f"d{di}_f{fi}": raw[(di, fi)] for di in range(len(sel)) for fi in range(len(FIELDS))})

    # ---- per-column display window (shared across the 4 designs) ----
    half_px = {}
    win_um = {}
    for fi in range(len(FIELDS)):
        r95 = max(r_frac_px(raw[(di, fi)].sum(0), 0.95) for di in range(len(sel)))
        h = max(8.0, 1.35 * r95)
        half_px[fi] = h
        win_um[fi] = 2 * h * pix_um

    # ---- pretty grid ----
    nD, nF = len(sel), len(FIELDS)
    BG = "#0b0b0d"
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(nD, nF, figsize=(nF * 2.6 + 1.0, nD * 2.6 + 0.6),
                             facecolor=BG)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.95, bottom=0.02,
                        wspace=0.06, hspace=0.06)
    for di in range(nD):
        for fi in range(nF):
            ax = axes[di, fi]; ax.set_facecolor(BG)
            img = tone(raw[(di, fi)], half_px[fi])
            ax.imshow(img, interpolation="bilinear", origin="upper")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#33343a"); s.set_linewidth(1.0)
            # column header
            if di == 0:
                ax.set_title(FIELD_LB[fi], color="#d8dae0", fontsize=12, pad=8)
            # row label (RMS)
            if fi == 0:
                ax.text(-0.12, 0.5, f"RMS {sel[di]['rms_um']:.2f} µm",
                        transform=ax.transAxes, rotation=90, va="center", ha="center",
                        color="#b7bac2", fontsize=11)
            # scale bar (bottom row only, per column), 10 or 20 um
            if di == nD - 1:
                npx = img.shape[0]
                bar_um = 20.0 if win_um[fi] > 70 else (10.0 if win_um[fi] > 30 else 5.0)
                frac = bar_um / win_um[fi]
                x0 = 0.07; y0 = 0.09
                ax.add_patch(Rectangle((x0, y0), frac, 0.018, transform=ax.transAxes,
                                       facecolor="white", edgecolor="none"))
                ax.text(x0 + frac / 2, y0 + 0.05, f"{bar_um:.0f} µm",
                        transform=ax.transAxes, ha="center", va="bottom",
                        color="white", fontsize=8.5)
    fig.savefig(OUT / "psf_grid.png", dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT/'psf_grid.png'}", flush=True)

    # ---- individual untitled tiles ----
    from PIL import Image
    for di in range(nD):
        for fi in range(nF):
            img = tone(raw[(di, fi)], half_px[fi])
            arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(arr).resize((512, 512), Image.LANCZOS).save(
                OUT / "tiles" / f"d{di+1}_f{FIELDS[fi]:.1f}.png")
    print(f"wrote 12 tiles -> {OUT/'tiles'}")
    print(f"pixel pitch = {pix_um:.3f} um | windows(um) = "
          + ", ".join(f"{FIELD_LB[fi]}:{win_um[fi]:.0f}" for fi in range(nF)))
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
