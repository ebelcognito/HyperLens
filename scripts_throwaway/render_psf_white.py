#!/usr/bin/env python
"""White-background, large-text version of the PSF grid (display only).
Reads psf_raw.npz; writes NEW files (does not touch the black-bg outputs)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from PIL import Image

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
OUT = ROOT / "figures" / "sixlet_top4_psf"
(OUT / "tiles_white").mkdir(parents=True, exist_ok=True)

FIELD_LB = ["on-axis", "0.7 field", "full field"]
RMS = [0.89, 1.29, 1.47, 2.09]
GAMMA = 0.55
SMOOTH_PX = 0.8
PCTL = 99.6
MIN_HALF = 9

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


half_px = {}
for di in range(4):
    for fi in range(3):
        r90 = centroid_and_r(raw[(di, fi)].sum(0), 0.90)[2]
        half_px[(di, fi)] = max(MIN_HALF, 1.6 * r90)
win_um = {k: 2 * h * PIX for k, h in half_px.items()}


def nice_bar(window_um):
    cands = [b for b in (1, 2, 5, 10, 20, 50, 100) if b / window_um <= 0.42]
    return cands[-1] if cands else 1


def tile_img(di, fi):
    rgb = raw[(di, fi)]
    cy, cx, _ = centroid_and_r(rgb.sum(0), 0.90)
    h = int(round(half_px[(di, fi)]))
    ci, cj = int(round(cy)), int(round(cx))
    L = np.zeros((2 * h + 1, 2 * h + 1, 3), np.float64)
    for k in range(3):
        ch = np.pad(rgb[k], h, mode="constant")
        L[..., k] = ch[ci:ci + 2 * h + 1, cj:cj + 2 * h + 1]
    if SMOOTH_PX > 0:
        for k in range(3):
            L[..., k] = gaussian_filter(L[..., k], SMOOTH_PX)
    norm = np.percentile(L.max(axis=2), PCTL) or (L.max() or 1.0)
    L = np.clip(L / norm, 0, 1) ** GAMMA
    # paint colored "ink" on white: energy -> colour, background -> white
    M = L.max(axis=2, keepdims=True)
    out = 1.0 - (M - L)              # bg(M=0)->white; saturated colour stays
    return np.clip(out, 0, 1)


# ---------------- grid ----------------
BG = "white"
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, axes = plt.subplots(4, 3, figsize=(3 * 2.7 + 1.3, 4 * 2.7 + 0.8), facecolor=BG)
fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.02, wspace=0.07, hspace=0.07)
for di in range(4):
    for fi in range(3):
        ax = axes[di, fi]; ax.set_facecolor(BG)
        ax.imshow(tile_img(di, fi), interpolation="bilinear", origin="upper")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#c9c9cf"); s.set_linewidth(1.0)
        if di == 0:
            ax.set_title(FIELD_LB[fi], color="#1a1a1a", fontsize=20, pad=12)
        if fi == 0:
            ax.text(-0.16, 0.5, f"RMS {RMS[di]:.2f} µm", transform=ax.transAxes,
                    rotation=90, va="center", ha="center", color="#1a1a1a", fontsize=18)
        bar_um = nice_bar(win_um[(di, fi)])
        frac = bar_um / win_um[(di, fi)]
        ax.add_patch(Rectangle((0.06, 0.07), frac, 0.018, transform=ax.transAxes,
                               facecolor="#222222", edgecolor="none"))
        ax.text(0.06 + frac / 2, 0.105, f"{bar_um:g} µm", transform=ax.transAxes,
                ha="center", va="bottom", color="#222222", fontsize=13)
fig.savefig(OUT / "psf_grid_white.png", dpi=200, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "psf_grid_white.png")

# ---------------- individual tiles (no text, white bg) ----------------
for di in range(4):
    for fi in range(3):
        arr = (np.clip(tile_img(di, fi), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr).resize((512, 512), Image.LANCZOS).save(
            OUT / "tiles_white" / f"d{di+1}_f{fi}.png")
print("wrote 12 tiles ->", OUT / "tiles_white")
