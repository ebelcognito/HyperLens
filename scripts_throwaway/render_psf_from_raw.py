#!/usr/bin/env python
"""Render the pretty PSF grid + tiles from the precomputed psf_raw.npz
(display only -- no torch, fast to iterate)."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from PIL import Image

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
OUT = ROOT / "figures" / "sixlet_top4_psf"
(OUT / "tiles").mkdir(parents=True, exist_ok=True)

FIELD_LB = ["on-axis", "0.7 field", "full field"]
RMS = [0.89, 1.29, 1.47, 2.09]
GAMMA = 0.5
SMOOTH_PX = 0.8
PCTL = 99.6          # normalisation percentile (avoids 1-pixel blowout)
MIN_HALF = 9         # px -> on-axis min window ~7 um
BG = "#0b0b0d"

d = np.load(OUT / "psf_raw.npz")
PIX = float(d["pix_um"])
raw = {(di, fi): d[f"d{di}_f{fi}"] for di in range(4) for fi in range(3)}


def centroid_and_r(gray, frac):
    ks = gray.shape[0]; g = gray / (gray.sum() + 1e-12)
    yy, xx = np.mgrid[0:ks, 0:ks]
    cy = (g * yy).sum(); cx = (g * xx).sum()
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    o = np.argsort(r.ravel()); cum = np.cumsum(g.ravel()[o])
    return cy, cx, float(r.ravel()[o][np.searchsorted(cum, frac)])


# per-tile window: frame each PSF to its own size
half_px = {}
for di in range(4):
    for fi in range(3):
        r90 = centroid_and_r(raw[(di, fi)].sum(0), 0.90)[2]
        half_px[(di, fi)] = max(MIN_HALF, 1.6 * r90)
win_um = {k: 2 * h * PIX for k, h in half_px.items()}


def nice_bar(window_um):
    for b in (1, 2, 5, 10, 20, 50, 100):
        if b / window_um <= 0.42:
            best = b
    # pick the largest bar that is <= ~40% of the window
    cands = [b for b in (1, 2, 5, 10, 20, 50, 100) if b / window_um <= 0.42]
    return cands[-1] if cands else 1


def tile_img(di, fi):
    rgb = raw[(di, fi)]
    gray = rgb.sum(0)
    cy, cx, _ = centroid_and_r(gray, 0.90)
    h = int(round(half_px[(di, fi)]))
    ci, cj = int(round(cy)), int(round(cx))
    img = np.zeros((2 * h + 1, 2 * h + 1, 3), np.float64)
    for k in range(3):
        ch = np.pad(rgb[k], h, mode="constant")
        img[..., k] = ch[ci:ci + 2 * h + 1, cj:cj + 2 * h + 1]
    if SMOOTH_PX > 0:
        for k in range(3):
            img[..., k] = gaussian_filter(img[..., k], SMOOTH_PX)
    norm = np.percentile(img.max(axis=2), PCTL)
    if norm <= 0:
        norm = img.max() or 1.0
    img = np.clip(img / norm, 0, 1) ** GAMMA
    return img


# ---------------- grid ----------------
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, axes = plt.subplots(4, 3, figsize=(3 * 2.6 + 1.0, 4 * 2.6 + 0.6), facecolor=BG)
fig.subplots_adjust(left=0.10, right=0.99, top=0.95, bottom=0.02, wspace=0.06, hspace=0.06)
for di in range(4):
    for fi in range(3):
        ax = axes[di, fi]; ax.set_facecolor(BG)
        ax.imshow(tile_img(di, fi), interpolation="bilinear", origin="upper")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#303137"); s.set_linewidth(1.0)
        if di == 0:
            ax.set_title(FIELD_LB[fi], color="#d8dae0", fontsize=12, pad=8)
        if fi == 0:
            ax.text(-0.14, 0.5, f"RMS {RMS[di]:.2f} µm", transform=ax.transAxes,
                    rotation=90, va="center", ha="center", color="#b7bac2", fontsize=11)
        # small per-tile scale bar (windows differ tile to tile)
        bar_um = nice_bar(win_um[(di, fi)])
        frac = bar_um / win_um[(di, fi)]
        ax.add_patch(Rectangle((0.07, 0.075), frac, 0.014, transform=ax.transAxes,
                               facecolor="white", edgecolor="none", alpha=0.9))
        ax.text(0.07 + frac / 2, 0.105, f"{bar_um:g} µm", transform=ax.transAxes,
                ha="center", va="bottom", color="#e8e8ea", fontsize=7.5)
fig.savefig(OUT / "psf_grid.png", dpi=200, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "psf_grid.png")

# ---------------- individual tiles (no text) ----------------
for di in range(4):
    for fi in range(3):
        arr = (np.clip(tile_img(di, fi), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr).resize((512, 512), Image.LANCZOS).save(
            OUT / "tiles" / f"d{di+1}_f{fi}.png")
print("per-tile windows(um):")
for di in range(4):
    print(f"  RMS {RMS[di]:.2f}: " + ", ".join(
        f"{FIELD_LB[fi]} {win_um[(di,fi)]:.1f}" for fi in range(3)))
