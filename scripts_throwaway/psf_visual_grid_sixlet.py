#!/usr/bin/env python
"""Visual grid for the sixlet ε-distinct designs:

  top strip   : (S₂, S₃) descriptor landscape with all archive elites as
                a gray cloud + the 4 chosen designs as colored markers
                labeled sixlet#1..#4.
  rows 1..4   : lens cross-section (col 0) + 2D PSF spot at on-axis,
                half-field, and corner (cols 1..3) for each design.

Goal: read the hypothesis off the figure at a glance —
  • top strip: designs scatter over different (S₂, S₃) cells
  • lens column: designs are visibly distinct
  • on-axis PSF column: all four cores look identical
  • corner PSF column: PSF shape diverges by design
"""
from __future__ import annotations
import json, os, sys, glob, time
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

# Background archive pool for the descriptor scatter.
HL_POOL_GLOB = str(ROOT / "results/lensmega/sixlet*/refined_top*.json")
ARCHIVE_BOUNDS_MUL = 0.012  # plot ±12 mrad-ish in (S2, S3) — matches archive cfg


def crop_psf_auto(psf: np.ndarray, frac: float = 0.995) -> tuple[np.ndarray, int]:
    """Return (cropped psf, half-width in pixels)."""
    H, W = psf.shape
    cy, cx = H // 2, W // 2
    total = psf.sum()
    if total <= 0:
        h = min(cy, cx); return psf, h
    max_r = min(cy, cx, H - cy - 1, W - cx - 1)
    cum = np.array([psf[cy-r:cy+r+1, cx-r:cx+r+1].sum()
                    for r in range(max_r + 1)]) / total
    h = int(max(np.searchsorted(cum, frac), 4))
    h = min(h, max_r)
    return psf[cy-h:cy+h+1, cx-h:cx+h+1], h


def load_archive_descriptors():
    """All sixlet HL elites with Seidel (S2, S3); used as gray background."""
    pts = []
    for fp in sorted(glob.glob(HL_POOL_GLOB)):
        try: d = json.load(open(fp))
        except: continue
        for des in d.get("designs", []):
            m = des.get("metrics") or {}
            s2 = m.get("seidel_S2"); s3 = m.get("seidel_S3")
            rms = des.get("rms_um")
            if s2 is None or s3 is None or rms is None: continue
            pts.append((float(s2), float(s3), float(rms)))
    return np.array(pts)


def render(designs: list[dict]):
    n_d = len(designs)
    print(f"=== building {n_d} sixlet lenses ===", flush=True)
    lenses = []
    for k, m in enumerate(designs, 1):
        print(f"  [{k}/{n_d}] {m['source_run']}#{m['source_idx']}", flush=True)
        lenses.append(build_lens(m))
    pix_um = float(lenses[0].pixel_size) * 1e3

    print("computing PSFs at 3 fields each", flush=True)
    ks = PSF_KS * OVERSAMPLE
    psfs = {}
    for fi, frac in enumerate(FIELD_FRACS):
        for di, lens in enumerate(lenses):
            psfs[(di, fi)] = compute_psf_at(lens, frac, WVLN_UM, ks)

    # Pre-crop PSFs and pick a SHARED extent per column so per-field cells are
    # spatially comparable across designs (cores read as tiny, tails read as
    # extended).
    shared_extent_um = []
    crops = {}
    for fi in range(3):
        max_h = 0
        for di in range(n_d):
            _, h = crop_psf_auto(psfs[(di, fi)], frac=0.995)
            max_h = max(max_h, h)
        max_h = max(8, max_h)
        for di in range(n_d):
            H, W = psfs[(di, fi)].shape
            cy, cx = H // 2, W // 2
            h = min(max_h, cy, cx)
            crops[(di, fi)] = psfs[(di, fi)][cy-h:cy+h+1, cx-h:cx+h+1]
        shared_extent_um.append(max_h * pix_um)

    # Load archive background.
    arch = load_archive_descriptors()
    print(f"  archive background: {len(arch)} (S2,S3) points", flush=True)

    # ---------- plot ----------
    mathematica_style()
    plt.rcParams.update({
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.labelsize": 12, "axes.titlesize": 13,
    })
    fig = plt.figure(figsize=(14.5, 13.6), facecolor="white")
    # 5 rows: row 0 = descriptor strip (slim, full width); rows 1..4 = design rows.
    gs = fig.add_gridspec(5, 4, width_ratios=[2.4, 1.0, 1.0, 1.0],
                           height_ratios=[1.1, 1.0, 1.0, 1.0, 1.0],
                           left=0.05, right=0.99, top=0.965, bottom=0.04,
                           wspace=0.10, hspace=0.55)

    # ---- Row 0: descriptor scatter spans all 4 columns ----
    ax_arch = fig.add_subplot(gs[0, :])
    SCALE = 1e6  # plot S2,S3 in units of ×10⁻⁶ (natural scale for this archive)
    pick_xy = np.array([
        [(m.get("seidel", {}).get("seidel_S2", float("nan")) or float("nan")) * SCALE,
         (m.get("seidel", {}).get("seidel_S3", float("nan")) or float("nan")) * SCALE]
        for m in designs
    ])
    # Auto-fit window to picks ± a margin, capped to a sensible max so the
    # archive cloud is still visible around the picks.
    xy = pick_xy
    cx, cy = float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1]))
    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 5.0)
    margin = max(2.0 * span, 30.0)
    xlim = (cx - margin, cx + margin)
    ylim = (cy - margin, cy + margin)
    # Background cloud: restrict to points inside (xlim, ylim) for readability.
    if len(arch) > 0:
        ax_arch.scatter(arch[:, 0] * SCALE, arch[:, 1] * SCALE,
                        s=8, c="#c8c8c8", alpha=0.55, edgecolors="none",
                        label=rf"$\mathrm{{all\ HL\ elites}}\ (n={len(arch)})$")
    for di, m in enumerate(designs):
        s2, s3 = pick_xy[di]
        ax_arch.scatter([s2], [s3], s=180, c=DESIGN_COLORS[di],
                        edgecolors="black", linewidths=0.9, zorder=5)
        ax_arch.annotate(rf"sixlet\#{di+1}", xy=(s2, s3),
                         xytext=(10, 8), textcoords="offset points",
                         fontsize=12, color=DESIGN_COLORS[di],
                         fontweight="bold")
    ax_arch.set_xlabel(r"$S_2\ \mathrm{(coma,\ \times 10^{-6})}$", labelpad=4)
    ax_arch.set_ylabel(r"$S_3\ \mathrm{(astig.,\ \times 10^{-6})}$", labelpad=4)
    ax_arch.set_title(r"$\mathrm{Location\ of\ the\ 4\ }\varepsilon\mathrm{-distinct\ "
                      r"designs\ in\ the\ }(S_2, S_3)\ \mathrm{archive\ landscape}$",
                      pad=6)
    ax_arch.grid(True, lw=0.45, color="#dddddd")
    ax_arch.set_axisbelow(True)
    ax_arch.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.9,
                   edgecolor="#bbb")
    ax_arch.set_xlim(*xlim)
    ax_arch.set_ylim(*ylim)

    # ---- Rows 1..4: lens layout + 3 PSF cells ----
    for di in range(n_d):
        row = 1 + di
        # Col 0: lens layout.
        ax_layout = fig.add_subplot(gs[row, 0])
        layout_png = LAYOUT_DIR / f"sixlet_{di+1}_layout.png"
        if layout_png.exists():
            img = mpimg.imread(str(layout_png))
            ax_layout.imshow(img, aspect="equal")
        ax_layout.set_axis_off()
        ax_layout.set_title(rf"$\mathrm{{sixlet\#{di+1}\ \ layout}}$",
                            color=DESIGN_COLORS[di], fontsize=11, pad=2)

        # Cols 1..3: PSF cells.
        for fi in range(3):
            ax_psf = fig.add_subplot(gs[row, 1 + fi])
            psf = crops[(di, fi)]
            psf_n = psf / max(psf.max(), 1e-12)
            ext = shared_extent_um[fi]
            ax_psf.imshow(psf_n, cmap="turbo",
                          extent=[-ext, ext, -ext, ext], origin="lower",
                          interpolation="bilinear")
            ax_psf.set_facecolor("#0d0d10")
            ax_psf.tick_params(width=0.5, length=2.0, color="#bbb",
                                labelsize=8, pad=1.5)
            for spine in ax_psf.spines.values():
                spine.set_color("#bbb"); spine.set_linewidth(0.5)
            if di == 0:
                ax_psf.set_title(FIELD_LABELS[fi], fontsize=11, pad=4)
            if fi == 0:
                ax_psf.set_ylabel(rf"$y\ (\mu\mathrm{{m}})$", fontsize=9, labelpad=1)
            if di == n_d - 1:
                ax_psf.set_xlabel(rf"$x\ (\mu\mathrm{{m}})$", fontsize=9, labelpad=1)
            # Tight tick set so labels don't clutter.
            tick = int(np.round(ext * 0.66))
            if tick > 0:
                ax_psf.set_xticks([-tick, 0, tick])
                ax_psf.set_yticks([-tick, 0, tick])

    out_path = OUT / "psf_visual_grid_sixlet.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)
    return out_path


def main():
    designs = load_top4("sixlet")
    print(f"loaded {len(designs)} sixlet manifests", flush=True)
    render(designs)


if __name__ == "__main__":
    main()
