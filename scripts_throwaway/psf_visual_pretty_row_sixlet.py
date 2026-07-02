#!/usr/bin/env python
"""Pretty row-layout renderings of the three sixlet panels, written to
NEW filenames so the prior versions are not clobbered:

  designs_sixlet_row.png       : 1 row × 4 cells, filled-glass lens layouts
                                  (custom call to lens.draw_layout, no
                                  header strip)
  psf_strip_sixlet.png         : 3 rows × 4 cols (rows = fields, cols =
                                  designs); filled Blues contour aesthetic
                                  matches the prior 4×3 grid
  archive_map_sixlet_pretty.png: hexbin density background + colored
                                  marker picks, clean spines

Same data as psf_visual_split_sixlet.py.
"""
from __future__ import annotations
import json, os, sys, glob, tempfile
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
from scripts_throwaway.psf_visual_split_sixlet import (
    load_archive_descriptors, crop_psf_auto, set_mathematica,
)

OUT = ROOT / "figures" / "psf_visual_grid"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# 1. designs row (filled-glass layouts, no FocLen header)
# ---------------------------------------------------------------

def render_designs_row(designs: list[dict], lenses: list, out_path: Path):
    set_mathematica()
    # Render each layout into a temp PNG via DeepLens, no header strip.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        layout_paths = []
        for di, lens in enumerate(lenses):
            p = td / f"layout_{di+1}.png"
            try:
                lens.draw_layout(filename=str(p),
                                  depth=-20000.0,
                                  style="filled",
                                  lens_title="",
                                  glass_facecolor="#dceefc",
                                  glass_edgecolor="#2a3a55",
                                  glass_lw=0.9,
                                  bundle_alpha=0.30)
            except Exception as e:
                print(f"  ! draw_layout failed for design {di+1}: {e}",
                      flush=True)
                continue
            layout_paths.append(p)

        fig, axs = plt.subplots(1, 4, figsize=(14.5, 3.6), facecolor="white",
                                gridspec_kw=dict(wspace=0.04,
                                                  left=0.01, right=0.99,
                                                  top=0.88, bottom=0.02))
        for di, p in enumerate(layout_paths):
            ax = axs[di]
            img = mpimg.imread(str(p))
            ax.imshow(img, aspect="equal")
            ax.set_axis_off()
            ax.set_title(rf"$\mathrm{{sixlet\#{di+1}}}$",
                         color=DESIGN_COLORS[di], fontsize=14, pad=2,
                         fontweight="bold")
        fig.savefig(out_path, dpi=220, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


# ---------------------------------------------------------------
# 2. PSF strip: 3 rows (fields) × 4 cols (designs)
# ---------------------------------------------------------------

def render_psf_strip(designs: list[dict], psfs: dict, pix_um: float,
                      out_path: Path):
    set_mathematica()
    n_d = 4; n_f = 3
    PAD_FACTOR = 1.45
    shared_extent_um = []
    for fi in range(n_f):
        max_h = max(crop_psf_auto(psfs[(di, fi)], frac=0.995) for di in range(n_d))
        max_h = max(8, int(np.ceil(max_h * PAD_FACTOR)))
        shared_extent_um.append(max_h * pix_um)

    contour_cmap = plt.get_cmap("Blues")
    n_fill_levels = 24
    iso_levels = (0.05, 0.20, 0.50, 0.80)
    iso_color = "#1a3a6b"

    fig, axs = plt.subplots(n_f, n_d, figsize=(14.5, 11.0), facecolor="white",
                            gridspec_kw=dict(wspace=0.14, hspace=0.18,
                                              left=0.07, right=0.985,
                                              top=0.95, bottom=0.05))
    for fi in range(n_f):
        for di in range(n_d):
            ax = axs[fi, di]
            psf = psfs[(di, fi)]
            H, W = psf.shape
            cy, cx = H // 2, W // 2
            h = int(shared_extent_um[fi] / pix_um)
            h = min(h, cy, cx)
            crop = psf[cy-h:cy+h+1, cx-h:cx+h+1].astype(np.float64)
            crop = crop / max(crop.max(), 1e-12)
            ext = shared_extent_um[fi]
            x = np.linspace(-ext, ext, crop.shape[1])
            y = np.linspace(-ext, ext, crop.shape[0])
            X, Y = np.meshgrid(x, y)
            ax.contourf(X, Y, crop, levels=np.linspace(0, 1, n_fill_levels),
                         cmap=contour_cmap, antialiased=True)
            cs = ax.contour(X, Y, crop, levels=iso_levels,
                             colors=iso_color, linewidths=0.7, alpha=0.85)
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
            if fi == 0:
                ax.set_title(rf"$\mathrm{{sixlet\#{di+1}}}$",
                             color=DESIGN_COLORS[di], fontsize=13, pad=6,
                             fontweight="bold")
            if di == 0:
                ax.set_ylabel(rf"$\mathrm{{{FIELD_LABELS[fi].strip(chr(36))}}}$"
                              "\n" + r"$y\ (\mu\mathrm{m})$",
                              fontsize=11, labelpad=4)
            if fi == n_f - 1:
                ax.set_xlabel(r"$x\ (\mu\mathrm{m})$", fontsize=10, labelpad=2)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


# ---------------------------------------------------------------
# 3. archive map: hexbin density + colored picks
# ---------------------------------------------------------------

def render_archive_pretty(designs: list[dict], arch_pts: np.ndarray,
                            out_path: Path):
    """Inferno solar-heatmap background (matches aggregate_archive_elegant
    'elegant' scheme), log(1+n) + Gaussian smoothing, with the 4 picks
    overlaid as colored markers with white halo."""
    from scipy.ndimage import gaussian_filter
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

    # 2D density terrain: log(1 + n_per_bin), Gaussian-smoothed.
    arch_xy = arch_pts * SCALE
    NX = NY = 320
    H, xedges, yedges = np.histogram2d(
        arch_xy[:, 0], arch_xy[:, 1], bins=[NX, NY],
        range=[list(xlim), list(ylim)],
    )
    terrain = np.log1p(H).T  # transpose so axes match imshow's (y, x) order
    terrain = gaussian_filter(terrain, sigma=2.0)

    fig, ax = plt.subplots(figsize=(8.4, 5.8), facecolor="white")
    ax.set_facecolor("#000004")  # inferno's floor
    im = ax.imshow(terrain, origin="lower",
                   extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   cmap="inferno", interpolation="bilinear", aspect="auto")
    cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
    cb.set_label(r"$\log(1 + n_{\mathrm{designs}}\ \mathrm{per\ cell})$",
                  fontsize=10, color="#1a0a0a")
    cb.ax.tick_params(labelsize=9, color="#888")

    # Picks: white halo + colored core, label in warm cream for contrast.
    for di in range(4):
        s2, s3 = pick_xy[di]
        ax.scatter([s2], [s3], s=300, c=DESIGN_COLORS[di],
                   edgecolors="white", linewidths=2.0, zorder=6)
        ax.annotate(rf"$\mathrm{{sixlet\#{di+1}}}$", xy=(s2, s3),
                    xytext=(11, 9), textcoords="offset points",
                    fontsize=12.5, color="#FFE6A8", fontweight="bold",
                    path_effects=[
                        plt_path_effect(linewidth=2.5, foreground="#000000",
                                         alpha=0.65),
                    ])

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$S_2\ \mathrm{(coma,\ \times 10^{-6})}$", labelpad=4)
    ax.set_ylabel(r"$S_3\ \mathrm{(astigmatism,\ \times 10^{-6})}$", labelpad=4)
    # Subtle inner grid in warm cream so it reads on the dark fill.
    ax.grid(True, lw=0.3, color="#FFE6A8", alpha=0.18)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#333"); spine.set_linewidth(0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plt_path_effect(linewidth, foreground, alpha):
    """Lightweight stroke-around-text effect for label legibility on dark bg."""
    import matplotlib.patheffects as pe
    return pe.withStroke(linewidth=linewidth, foreground=foreground, alpha=alpha)


def main():
    designs = load_top4("sixlet")
    print(f"loaded {len(designs)} sixlet manifests", flush=True)
    print("building lenses + PSFs", flush=True)
    lenses = [build_lens(m) for m in designs]
    pix_um = float(lenses[0].pixel_size) * 1e3
    ks = PSF_KS * OVERSAMPLE
    psfs = {(di, fi): compute_psf_at(lens, frac, WVLN_UM, ks)
            for fi, frac in enumerate(FIELD_FRACS)
            for di, lens in enumerate(lenses)}
    arch = load_archive_descriptors()
    print(f"  archive: {len(arch)} (S2,S3) points", flush=True)

    render_designs_row(designs, lenses, OUT / "designs_sixlet_row.png")
    render_psf_strip(designs, psfs, pix_um, OUT / "psf_strip_sixlet.png")
    render_archive_pretty(designs, arch, OUT / "archive_map_sixlet_pretty.png")


if __name__ == "__main__":
    main()
