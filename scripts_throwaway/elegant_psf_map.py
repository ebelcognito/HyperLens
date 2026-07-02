#!/usr/bin/env python
"""Elegant Mathematica-style PSF field map.

Uses DeepLens compute (lens.psf_map_rgb) but renders entirely with our
own matplotlib layer: serif typography, per-cell µm axes, field-position
labels, RGB-wavelength legend, shared spatial scale across cells.

Inputs: a manifest.json from zemax_reports_top3_poly/.
Output: design_psf_map_elegant.png next to it.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch
torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DeepLens"))

from scripts_throwaway.fig5_extract import (
    build_lens_template_and_solver, make_geolens,
)

WAVE_RGB_NM = (656, 587, 486)        # R, G, B in nm — matches DeepLens WAVE_RGB
WAVE_RGB_UM = (0.656, 0.587, 0.486)
GRID = (5, 5)                         # 5x5 field grid (off-axis radial)
PSF_KS = 128                          # bigger kernel for nicer PSF detail
OVERSAMPLE = 4                        # sensor sub-pixel oversample for PSF analysis


def mathematica_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIX Two Text", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#999999",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def fov_label_deg(lens, frac_h: float, frac_w: float) -> str:
    """Return a Zemax-ish field label '+15.0°' for the centre of grid cell (h, w)."""
    hfov = 0.5 * np.degrees(lens.hfov)
    vfov = 0.5 * np.degrees(lens.vfov)
    # frac in [-1, 1]
    fy, fx = frac_h, frac_w
    if abs(fy) < 1e-6 and abs(fx) < 1e-6:
        return r"$0.0\degree$"
    angle = float(np.hypot(fy * vfov, fx * hfov))
    return rf"${angle:+.1f}\degree$"


def crop_to_energy(psf_rgb: np.ndarray, frac: float = 0.995) -> int:
    """Return half-width (in pixels) that contains `frac` of total intensity
    summed over channels, centered on the kernel midpoint."""
    H, W = psf_rgb.shape[-2:]
    cy, cx = H // 2, W // 2
    img = psf_rgb.sum(axis=0)  # [H, W]
    total = img.sum()
    if total <= 0:
        return min(cy, cx)
    # Concentric squares: radius r covers (2r+1)^2 pixels around (cy, cx)
    max_r = min(cy, cx, H - cy - 1, W - cx - 1)
    cum = np.array([img[cy-r:cy+r+1, cx-r:cx+r+1].sum() for r in range(max_r + 1)])
    cum = cum / total
    return int(np.searchsorted(cum, frac))


def render_elegant_psf_map(manifest_path: Path, out_path: Path | None = None,
                            grid=GRID, ks=PSF_KS, depth_mm=-20000.0,
                            log_scale=False, oversample=OVERSAMPLE):
    m = json.loads(manifest_path.read_text())
    run_dir = ROOT / "results" / "lensmega" / m["source_run"]
    cfg_path = run_dir / "config.yaml"
    theta = np.asarray(m["theta"], dtype=np.float64)

    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(cfg_path)
    lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver,
                        device="cpu")

    # Oversample the sensor by `oversample` (in each axis) so PSFs are sampled
    # finer than the native pixel pitch. set_sensor_res keeps physical sensor
    # size constant and shrinks pixel_size accordingly.
    orig_res = tuple(lens.sensor_res)
    if oversample != 1:
        lens.set_sensor_res(
            (int(orig_res[0] * oversample), int(orig_res[1] * oversample))
        )
        ks = int(ks * oversample)

    # Compute the PSF stack: [grid_h, grid_w, 3, ks, ks].
    with torch.no_grad():
        psf_map = lens.psf_map_rgb(grid=grid, ks=ks, depth=depth_mm).cpu().numpy()

    pix_um = float(lens.pixel_size) * 1e3  # mm → µm per kernel pixel
    grid_w, grid_h = grid
    # Per-cell energy-aware half-width (in pixels); use max across cells as
    # a shared crop so cells are spatially comparable.
    half_widths = np.zeros((grid_h, grid_w), dtype=int)
    for i in range(grid_h):
        for j in range(grid_w):
            half_widths[i, j] = crop_to_energy(psf_map[i, j], frac=0.99)
    half_px = max(int(half_widths.max()), 6)
    half_px = min(half_px, ks // 2 - 1)
    half_um = half_px * pix_um

    # Fractional field offsets per cell (center of cell), in [-1, 1].
    fy = np.linspace(-1.0, 1.0, grid_h) * (grid_h - 1) / grid_h
    fx = np.linspace(-1.0, 1.0, grid_w) * (grid_w - 1) / grid_w

    mathematica_style()
    fig = plt.figure(figsize=(2.0 * grid_w + 0.8, 2.0 * grid_h + 1.1),
                     facecolor="white")
    gs = GridSpec(grid_h, grid_w, figure=fig,
                  left=0.07, right=0.985, top=0.92, bottom=0.10,
                  wspace=0.18, hspace=0.20)

    for i in range(grid_h):
        for j in range(grid_w):
            ax = fig.add_subplot(gs[i, j])
            psf = psf_map[i, j]  # [3, ks, ks]
            H, W = psf.shape[-2:]
            cy, cx = H // 2, W // 2
            crop = psf[:, cy - half_px:cy + half_px + 1,
                          cx - half_px:cx + half_px + 1]   # [3, k, k]
            # Per-cell normalisation: each cell to its own max so structure shows.
            local_max = crop.max()
            if local_max > 0:
                crop = crop / local_max
            if log_scale:
                crop = np.log10(np.clip(crop, 1e-4, 1.0))
                crop = (crop - crop.min()) / (crop.max() - crop.min() + 1e-12)
            # Stack to (k, k, 3) for imshow.
            img = np.transpose(crop, (1, 2, 0))
            ax.imshow(img, extent=[-half_um, half_um, -half_um, half_um],
                      interpolation="bilinear", origin="lower")

            ax.set_xlim(-half_um, half_um)
            ax.set_ylim(-half_um, half_um)
            ax.set_aspect("equal")
            ax.tick_params(width=0.5, length=2.2, color="#bbbbbb", pad=1.5)
            ax.set_facecolor("#0d0d10")
            for spine in ax.spines.values():
                spine.set_color("#888888")
                spine.set_linewidth(0.6)

            # Field-position label, top-right inside the cell.
            ax.text(0.97, 0.965, fov_label_deg(lens, fy[i], fx[j]),
                    transform=ax.transAxes, ha="right", va="top",
                    color="white", fontsize=8,
                    bbox=dict(facecolor=(0, 0, 0, 0.55),
                              edgecolor="none", boxstyle="round,pad=0.18"))

            # Only label outer-edge axes.
            if i != grid_h - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
            if i == grid_h - 1:
                ax.set_xlabel(r"$x\ (\mu\mathrm{m})$", labelpad=2)
            if j == 0:
                ax.set_ylabel(r"$y\ (\mu\mathrm{m})$", labelpad=2)

    # Title + RGB legend strip.
    title = (rf"$\mathrm{{PSF\ field\ map}}\ \mid\ "
             rf"\mathrm{{{m['system']}\ rank\!-\!}}{m['rank']},\ "
             rf"\mathrm{{poly\ RMS}}\!=\!{m['poly_rms_um']:.2f}\,\mu\mathrm{{m}}$")
    fig.suptitle(title, fontsize=11, y=0.985)

    sub = (rf"$\mathrm{{Angular\ Spectrum\ Method}}\ \cdot\ "
           rf"\mathrm{{kernel}}\!=\!{2*half_px+1}\,\mathrm{{px}}\ "
           rf"({2*half_um:.1f}\,\mu\mathrm{{m}}\ \mathrm{{window}})\ \cdot\ "
           rf"\mathrm{{depth}}\!=\!-20\,\mathrm{{m}}$")
    fig.text(0.5, 0.955, sub, ha="center", fontsize=8.5, color="#444444")

    # RGB swatch legend in bottom strip.
    leg_ax = fig.add_axes([0.30, 0.012, 0.40, 0.030])
    leg_ax.set_xlim(0, 3); leg_ax.set_ylim(0, 1); leg_ax.axis("off")
    swatch_colors = ["#e63946", "#2a9d8f", "#1d4ed8"]
    for k, (c, nm, um) in enumerate(zip(swatch_colors, WAVE_RGB_NM, WAVE_RGB_UM)):
        leg_ax.add_patch(Rectangle((k + 0.05, 0.30), 0.30, 0.40,
                                    facecolor=c, edgecolor="#333333", lw=0.4))
        leg_ax.text(k + 0.42, 0.50,
                    rf"$\lambda = {nm}\,\mathrm{{nm}}\ ({um:.3f}\,\mu\mathrm{{m}})$",
                    fontsize=8.5, va="center")

    if out_path is None:
        out_path = manifest_path.parent / "design_psf_map_elegant.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path, (half_um, pix_um)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="manifest.json from a zemax_reports_top3_poly subdir")
    ap.add_argument("--ks", type=int, default=PSF_KS)
    ap.add_argument("--grid-h", type=int, default=GRID[1])
    ap.add_argument("--grid-w", type=int, default=GRID[0])
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    mp = Path(args.manifest).resolve()
    out, info = render_elegant_psf_map(
        mp, grid=(args.grid_w, args.grid_h), ks=args.ks, log_scale=args.log,
    )
    print(f"wrote {out}  (window ±{info[0]:.2f}µm, pixel {info[1]:.3f}µm)",
          flush=True)


if __name__ == "__main__":
    main()
