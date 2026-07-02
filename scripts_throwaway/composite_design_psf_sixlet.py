#!/usr/bin/env python
"""Combined design + spot composites for the 4 ε-distinct sixlet picks.
Two layouts (both written; no overrides of prior files):

  composite_sixlet_rowwise.png    : 4 rows × 4 cols
      row = design, cols = [lens layout, on-axis PSF, half-field PSF, corner PSF]
  composite_sixlet_columnwise.png : 4 rows × 4 cols
      col = design, rows = [lens layout, on-axis PSF, half-field PSF, corner PSF]

Lens layouts: rendered fresh via lens.draw_layout(style="filled", no header).
PSFs       : same Blues filled-contour + iso-line aesthetic as psf_strip.
"""
from __future__ import annotations
import os, sys, tempfile
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
    crop_psf_auto, set_mathematica,
)

OUT = ROOT / "figures" / "psf_visual_grid"
OUT.mkdir(parents=True, exist_ok=True)

PAD_FACTOR = 1.45
CONTOUR_CMAP = "Blues"
N_FILL_LEVELS = 24
ISO_LEVELS = (0.05, 0.20, 0.50, 0.80)
ISO_COLOR = "#1a3a6b"


def render_layouts_to_temp(lenses) -> list[Path]:
    """Render each lens via DeepLens to a temp PNG (filled-glass, no header)."""
    td = Path(tempfile.mkdtemp(prefix="layouts_"))
    paths = []
    for di, lens in enumerate(lenses):
        p = td / f"layout_{di+1}.png"
        try:
            lens.draw_layout(filename=str(p), depth=-20000.0,
                              style="filled", lens_title="",
                              glass_facecolor="#dceefc",
                              glass_edgecolor="#2a3a55",
                              glass_lw=0.9, bundle_alpha=0.30)
        except Exception as e:
            print(f"  ! draw_layout failed for design {di+1}: {e}", flush=True)
            continue
        paths.append(p)
    return paths


def psf_extents(psfs: dict, pix_um: float, n_d: int, n_f: int):
    """Shared crop extent per field across designs (max + pad)."""
    shared = []
    for fi in range(n_f):
        max_h = max(crop_psf_auto(psfs[(di, fi)], frac=0.995) for di in range(n_d))
        max_h = max(8, int(np.ceil(max_h * PAD_FACTOR)))
        shared.append(max_h * pix_um)
    return shared


def draw_psf_cell(ax, psf, ext_um, pix_um, show_iso_label=False):
    H, W = psf.shape
    cy, cx = H // 2, W // 2
    h = int(ext_um / pix_um)
    h = min(h, cy, cx)
    crop = psf[cy-h:cy+h+1, cx-h:cx+h+1].astype(np.float64)
    crop = crop / max(crop.max(), 1e-12)
    x = np.linspace(-ext_um, ext_um, crop.shape[1])
    y = np.linspace(-ext_um, ext_um, crop.shape[0])
    X, Y = np.meshgrid(x, y)
    cmap = plt.get_cmap(CONTOUR_CMAP)
    ax.contourf(X, Y, crop, levels=np.linspace(0, 1, N_FILL_LEVELS),
                 cmap=cmap, antialiased=True)
    cs = ax.contour(X, Y, crop, levels=ISO_LEVELS, colors=ISO_COLOR,
                     linewidths=0.7, alpha=0.85)
    if show_iso_label:
        try:
            ax.clabel(cs, levels=[ISO_LEVELS[0]], inline=True, fontsize=7,
                      fmt={ISO_LEVELS[0]: r"5\%"})
        except Exception:
            pass
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.tick_params(width=0.6, length=2.5, color="#777",
                    labelsize=9, pad=1.5)
    for spine in ax.spines.values():
        spine.set_color("#777"); spine.set_linewidth(0.6)
    tick = int(np.round(ext_um * 0.66))
    if tick > 0:
        ax.set_xticks([-tick, 0, tick])
        ax.set_yticks([-tick, 0, tick])


def render_rowwise(designs, lenses, psfs, pix_um, layout_paths, out_path):
    """4 rows × 4 cols. Row = design, cols = [layout, on-axis, half, corner]."""
    set_mathematica()
    n_d = 4; n_f = 3
    exts = psf_extents(psfs, pix_um, n_d, n_f)
    fig, axs = plt.subplots(
        n_d, 1 + n_f,
        figsize=(15.0, 11.0), facecolor="white",
        gridspec_kw=dict(
            width_ratios=[2.2, 1.0, 1.0, 1.0],
            wspace=0.10, hspace=0.20,
            left=0.04, right=0.98, top=0.95, bottom=0.05,
        ),
    )
    for di in range(n_d):
        # Layout cell.
        ax_l = axs[di, 0]
        if di < len(layout_paths):
            img = mpimg.imread(str(layout_paths[di]))
            ax_l.imshow(img, aspect="equal")
        ax_l.set_axis_off()
        # PSF cells.
        for fi in range(n_f):
            ax_p = axs[di, 1 + fi]
            draw_psf_cell(ax_p, psfs[(di, fi)], exts[fi], pix_um,
                          show_iso_label=(di == 0))
            if di == 0:
                ax_p.set_title(FIELD_LABELS[fi], fontsize=13, pad=6)
            if fi == 0:
                # Use empty math placeholder so the design label sits at axis-left
                ax_p.set_ylabel(r"$y\ (\mu\mathrm{m})$",
                                 fontsize=10, labelpad=2)
            if di == n_d - 1:
                ax_p.set_xlabel(r"$x\ (\mu\mathrm{m})$", fontsize=10, labelpad=2)
        # Big colored design label outside the layout, to the left.
        fig.text(0.02, axs[di, 0].get_position().y0 + 0.5 * axs[di, 0].get_position().height,
                 rf"$\mathrm{{sixlet\#{di+1}}}$",
                 color=DESIGN_COLORS[di], fontsize=15, fontweight="bold",
                 rotation=90, va="center", ha="center")
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def render_columnwise(designs, lenses, psfs, pix_um, layout_paths, out_path):
    """4 rows × 4 cols. Col = design, rows = [layout, on-axis, half, corner]."""
    set_mathematica()
    n_d = 4; n_f = 3
    exts = psf_extents(psfs, pix_um, n_d, n_f)
    fig, axs = plt.subplots(
        1 + n_f, n_d,
        figsize=(15.0, 13.0), facecolor="white",
        gridspec_kw=dict(
            height_ratios=[0.95, 1.0, 1.0, 1.0],
            wspace=0.16, hspace=0.18,
            left=0.06, right=0.985, top=0.96, bottom=0.04,
        ),
    )
    for di in range(n_d):
        ax_l = axs[0, di]
        if di < len(layout_paths):
            img = mpimg.imread(str(layout_paths[di]))
            ax_l.imshow(img, aspect="equal")
        ax_l.set_axis_off()
        ax_l.set_title(rf"$\mathrm{{sixlet\#{di+1}}}$",
                       color=DESIGN_COLORS[di], fontsize=14, pad=4,
                       fontweight="bold")
        for fi in range(n_f):
            ax_p = axs[1 + fi, di]
            draw_psf_cell(ax_p, psfs[(di, fi)], exts[fi], pix_um,
                          show_iso_label=(di == 0))
            if di == 0:
                # Row label on left side: which field this row shows.
                clean_label = FIELD_LABELS[fi].strip(chr(36))  # strip outer $...$
                ax_p.set_ylabel(rf"$\mathrm{{{clean_label}}}$"
                                 "\n" + r"$y\ (\mu\mathrm{m})$",
                                 fontsize=11, labelpad=4)
            if fi == n_f - 1:
                ax_p.set_xlabel(r"$x\ (\mu\mathrm{m})$",
                                 fontsize=10, labelpad=2)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def main():
    designs = load_top4("sixlet")
    print(f"loaded {len(designs)} sixlet manifests", flush=True)
    lenses = [build_lens(m) for m in designs]
    pix_um = float(lenses[0].pixel_size) * 1e3
    ks = PSF_KS * OVERSAMPLE
    print("computing PSFs", flush=True)
    psfs = {(di, fi): compute_psf_at(lens, frac, WVLN_UM, ks)
            for fi, frac in enumerate(FIELD_FRACS)
            for di, lens in enumerate(lenses)}
    print("rendering layouts to temp", flush=True)
    layout_paths = render_layouts_to_temp(lenses)
    print("composing rowwise + columnwise figures", flush=True)
    render_rowwise(designs, lenses, psfs, pix_um, layout_paths,
                   OUT / "composite_sixlet_rowwise.png")
    render_columnwise(designs, lenses, psfs, pix_um, layout_paths,
                      OUT / "composite_sixlet_columnwise.png")


if __name__ == "__main__":
    main()
