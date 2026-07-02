#!/usr/bin/env python
"""Three separate Mathematica-style PNGs for the sixlet hypothesis figure:

  1. designs_sixlet.png       : 4 lens cross-sections (2×2 grid)
  2. psf_grid_sixlet.png      : 4 designs × 3 fields PSF heatmaps
  3. archive_map_sixlet.png   : (S₂, S₃) scatter with the 4 picks marked

Same data sources as psf_visual_grid_sixlet.py, just split for figure
flexibility in the paper.
"""
from __future__ import annotations
import json, os, sys, glob
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("MPMATH_NOGMPY", "1")

import numpy as np
import torch
torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DeepLens"))

from scripts_throwaway.psf_radial_and_ee_overlay import (
    load_top4, build_lens, compute_psf_at, FIELD_FRACS, FIELD_LABELS,
    DESIGN_COLORS, WVLN_UM, PSF_KS, OVERSAMPLE,
)
from scripts_throwaway.elegant_psf_map import mathematica_style

LAYOUT_DIR = ROOT / "figures" / "zemax_top8_diverse_fnum5"
OUT = ROOT / "figures" / "psf_visual_grid"
OUT.mkdir(parents=True, exist_ok=True)

HL_POOL_GLOB = str(ROOT / "results/lensmega/sixlet*/refined_top*.json")


def load_archive_descriptors():
    pts = []
    for fp in sorted(glob.glob(HL_POOL_GLOB)):
        try: d = json.load(open(fp))
        except: continue
        for des in d.get("designs", []):
            m = des.get("metrics") or {}
            s2 = m.get("seidel_S2"); s3 = m.get("seidel_S3")
            if s2 is None or s3 is None: continue
            pts.append((float(s2), float(s3)))
    return np.array(pts)


def crop_psf_auto(psf: np.ndarray, frac: float = 0.995) -> int:
    H, W = psf.shape
    cy, cx = H // 2, W // 2
    total = psf.sum()
    if total <= 0: return min(cy, cx)
    max_r = min(cy, cx, H - cy - 1, W - cx - 1)
    cum = np.array([psf[cy-r:cy+r+1, cx-r:cx+r+1].sum()
                    for r in range(max_r + 1)]) / total
    return int(np.clip(np.searchsorted(cum, frac), 4, max_r))


def set_mathematica():
    mathematica_style()
    plt.rcParams.update({
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.labelsize": 12, "axes.titlesize": 13,
    })


# ---------------------------------------------------------------
# (1) designs
# ---------------------------------------------------------------

def render_designs(designs: list[dict], out_path: Path):
    set_mathematica()
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 6.4), facecolor="white",
                            gridspec_kw=dict(wspace=0.06, hspace=0.20,
                                              left=0.02, right=0.98,
                                              top=0.94, bottom=0.04))
    for di in range(4):
        ax = axs.flat[di]
        layout_png = LAYOUT_DIR / f"sixlet_{di+1}_layout.png"
        if layout_png.exists():
            img = mpimg.imread(str(layout_png))
            ax.imshow(img, aspect="equal")
        ax.set_axis_off()
        ax.set_title(rf"$\mathrm{{sixlet\#{di+1}}}$",
                     color=DESIGN_COLORS[di], fontsize=14, pad=4,
                     fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


# ---------------------------------------------------------------
# (2) PSF grid
# ---------------------------------------------------------------

def render_psf_grid(designs: list[dict], psfs: dict, pix_um: float,
                     out_path: Path):
    set_mathematica()
    n_d = len(designs); n_f = 3
    # Shared crop extent per field (max across designs), with extra padding.
    PAD_FACTOR = 1.45  # visual breathing room around the 99.5% box
    shared_extent_um = []
    for fi in range(n_f):
        max_h = max(crop_psf_auto(psfs[(di, fi)], frac=0.995) for di in range(n_d))
        max_h = max(8, int(np.ceil(max_h * PAD_FACTOR)))
        shared_extent_um.append(max_h * pix_um)

    # Mathematica-style ContourPlot aesthetic: white background, sequential
    # single-hue fill, thin iso-contour overlay.
    contour_cmap = plt.get_cmap("Blues")
    n_fill_levels = 24
    iso_levels = (0.05, 0.20, 0.50, 0.80)
    iso_color = "#1a3a6b"

    fig, axs = plt.subplots(n_d, n_f, figsize=(8.4, 11.0), facecolor="white",
                            gridspec_kw=dict(wspace=0.16, hspace=0.18,
                                              left=0.10, right=0.98,
                                              top=0.95, bottom=0.05))
    for di in range(n_d):
        for fi in range(n_f):
            ax = axs[di, fi]
            psf = psfs[(di, fi)]
            H, W = psf.shape
            cy, cx = H // 2, W // 2
            h = int(shared_extent_um[fi] / pix_um)
            h = min(h, cy, cx)
            crop = psf[cy-h:cy+h+1, cx-h:cx+h+1].astype(np.float64)
            peak = max(crop.max(), 1e-12)
            crop = crop / peak  # normalize to [0, 1]
            ext = shared_extent_um[fi]
            x = np.linspace(-ext, ext, crop.shape[1])
            y = np.linspace(-ext, ext, crop.shape[0])
            X, Y = np.meshgrid(x, y)
            # Filled contour, sequential blues.
            ax.contourf(X, Y, crop, levels=np.linspace(0, 1, n_fill_levels),
                         cmap=contour_cmap, antialiased=True)
            # Iso-contour outlines at well-known fractions of peak.
            cs = ax.contour(X, Y, crop, levels=iso_levels,
                             colors=iso_color, linewidths=0.7, alpha=0.85)
            # Light label on the outermost contour only.
            try:
                ax.clabel(cs, levels=[iso_levels[0]], inline=True, fontsize=7,
                          fmt={iso_levels[0]: r"5\%"})
            except Exception:
                pass
            ax.set_facecolor("white")
            ax.set_aspect("equal")
            ax.tick_params(width=0.6, length=2.5, color="#777",
                            labelsize=9, pad=1.5)
            for spine in ax.spines.values():
                spine.set_color("#777"); spine.set_linewidth(0.6)
            tick = int(np.round(ext * 0.66))
            if tick > 0:
                ax.set_xticks([-tick, 0, tick])
                ax.set_yticks([-tick, 0, tick])
            if di == 0:
                ax.set_title(FIELD_LABELS[fi], fontsize=12.5, pad=6)
            if fi == 0:
                ax.set_ylabel(rf"$\mathrm{{sixlet\#{di+1}}}$"
                              "\n" + r"$y\ (\mu\mathrm{m})$",
                              color=DESIGN_COLORS[di], fontsize=11,
                              fontweight="bold", labelpad=4)
            if di == n_d - 1:
                ax.set_xlabel(r"$x\ (\mu\mathrm{m})$", fontsize=10, labelpad=2)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


# ---------------------------------------------------------------
# (3) archive map
# ---------------------------------------------------------------

def render_archive_map(designs: list[dict], arch_pts: np.ndarray,
                        out_path: Path):
    set_mathematica()
    SCALE = 1e6
    pick_xy = np.array([
        [(m.get("seidel", {}).get("seidel_S2", float("nan")) or float("nan")) * SCALE,
         (m.get("seidel", {}).get("seidel_S3", float("nan")) or float("nan")) * SCALE]
        for m in designs
    ])
    cx, cy = float(np.mean(pick_xy[:, 0])), float(np.mean(pick_xy[:, 1]))
    span = max(np.ptp(pick_xy[:, 0]), np.ptp(pick_xy[:, 1]), 5.0)
    margin = max(2.0 * span, 30.0)
    xlim = (cx - margin, cx + margin)
    ylim = (cy - margin, cy + margin)

    fig, ax = plt.subplots(figsize=(8.0, 5.4), facecolor="white")
    if len(arch_pts) > 0:
        ax.scatter(arch_pts[:, 0] * SCALE, arch_pts[:, 1] * SCALE,
                   s=10, c="#c8c8c8", alpha=0.55, edgecolors="none",
                   label=rf"$\mathrm{{archive\ elites}}\ (n={len(arch_pts)})$")
    for di in range(4):
        s2, s3 = pick_xy[di]
        ax.scatter([s2], [s3], s=200, c=DESIGN_COLORS[di],
                   edgecolors="black", linewidths=1.0, zorder=5)
        ax.annotate(rf"$\mathrm{{sixlet\#{di+1}}}$", xy=(s2, s3),
                    xytext=(10, 8), textcoords="offset points",
                    fontsize=12.5, color=DESIGN_COLORS[di], fontweight="bold")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$S_2\ \mathrm{(coma,\ \times 10^{-6})}$", labelpad=4)
    ax.set_ylabel(r"$S_3\ \mathrm{(astigmatism,\ \times 10^{-6})}$", labelpad=4)
    ax.grid(True, lw=0.5, color="#dddddd")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10.5, frameon=True, framealpha=0.92,
              edgecolor="#bbb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def main():
    designs = load_top4("sixlet")
    print(f"loaded {len(designs)} sixlet manifests", flush=True)

    # ---- compute PSFs (needed only for #2) ----
    print("building lenses + PSFs", flush=True)
    lenses = [build_lens(m) for m in designs]
    pix_um = float(lenses[0].pixel_size) * 1e3
    ks = PSF_KS * OVERSAMPLE
    psfs = {(di, fi): compute_psf_at(lens, frac, WVLN_UM, ks)
            for fi, frac in enumerate(FIELD_FRACS)
            for di, lens in enumerate(lenses)}

    print("loading archive background")
    arch = load_archive_descriptors()

    print("rendering 3 separate PNGs")
    render_designs(designs, OUT / "designs_sixlet.png")
    render_psf_grid(designs, psfs, pix_um, OUT / "psf_grid_sixlet.png")
    render_archive_map(designs, arch, OUT / "archive_map_sixlet.png")


if __name__ == "__main__":
    main()
