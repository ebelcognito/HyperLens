#!/usr/bin/env python
"""Render the (S₂, S₃) archive map for the 4 ε-distinct sixlet picks
using the exact 'elegant solar' heatmap format from
plot_aggregate_sixlet_archive.py: inferno terrain with PowerNorm,
log(1+n) density, Gaussian-smoothed best-fitness contours, origin
cross-axes, italic serif title, two side colorbars.

Reads aggregated_sixlet_seidel_archive.json for the per-cell density
and best-fitness grids; overlays the 4 picks from our top4_diverse
manifests at their raw (S₂, S₃) positions, colored by poly fitness.

Output: figures/psf_visual_grid/archive_map_sixlet_solar.png
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("MPMATH_NOGMPY", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.ndimage import gaussian_filter

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

from scripts_throwaway.psf_radial_and_ee_overlay import load_top4

AGG_JSON = ROOT / "aggregated_sixlet_seidel_archive.json"
OUT_FULL = ROOT / "figures" / "psf_visual_grid" / "archive_map_sixlet_solar.png"
OUT_ZOOM = ROOT / "figures" / "psf_visual_grid" / "archive_map_sixlet_solar_zoom.png"
OUT_MICRO = ROOT / "figures" / "psf_visual_grid" / "archive_map_sixlet_solar_microzoom.png"
ZOOM_R_MUL = 1.5e-3  # ±1.5×10⁻³ medium zoom (uses canonical aggregate grid)
MICRO_R_RAW = 50e-6  # ±50×10⁻⁶ micro zoom: re-histograms raw pool at fine grid
MICRO_N_BINS = 160    # high-resolution bins within the microzoom window


# ------------------------- styling (matches 'elegant' scheme) ----------

SCHEME = {
    "cmap_bg":   plt.cm.inferno,
    "cmap_pk":   plt.cm.cividis,
    "ink":       "#1A0A0A",
    "ink_soft":  "#5C3A2E",
    "contour":   "#FFE6A8",
    "contour_halo": "#000000",
    "halo_alpha": 0.5,
    "label_color": "#FFE6A8",
    "page_bg":   "#FFFFFF",
    "ax_bg":     "#000004",
    "pick_edge": "#FFFFFF",
    "pick_size": 320,
}


def main():
    agg = json.loads(AGG_JSON.read_text())
    NI, NJ = agg["target_bins"]
    BOUNDS = np.array(agg["target_bounds"])
    bin_w_s2 = (BOUNDS[0, 1] - BOUNDS[0, 0]) / NI
    bin_w_s3 = (BOUNDS[1, 1] - BOUNDS[1, 0]) / NJ
    full_R = float(BOUNDS[0, 1])  # 0.008 (square canonical bound)

    # Build density + best-fitness grids on the canonical NI×NJ grid.
    n_img   = np.zeros((NI, NJ), dtype=np.float64)
    fit_img = np.full((NI, NJ), np.nan, dtype=np.float64)
    for c in agg["cells"]:
        ii, jj = c["cell_ij"]
        if not (0 <= ii < NI and 0 <= jj < NJ): continue
        n_img[ii, jj] += float(c["n_designs"])
        f = float(c["best_fitness"])
        if np.isnan(fit_img[ii, jj]) or f > fit_img[ii, jj]:
            fit_img[ii, jj] = f

    # Background terrain: log(1+n), Gaussian-smoothed.
    bg = gaussian_filter(np.log1p(n_img), sigma=1.6)
    # Fitness terrain: Gaussian-smoothed; mask out empty cells for contouring.
    fit_filled = np.where(np.isnan(fit_img), -1e9, fit_img)
    fit_smooth = gaussian_filter(fit_filled, sigma=2.2)
    fit_smooth_masked = np.where(n_img > 0, fit_smooth, np.nan)

    # Cell centers in raw S2/S3.
    x_centers_raw = BOUNDS[0, 0] + (np.arange(NI) + 0.5) * bin_w_s2
    y_centers_raw = BOUNDS[1, 0] + (np.arange(NJ) + 0.5) * bin_w_s3
    # Axis units = ×10⁻³ (raw × 1000).
    x_centers = x_centers_raw * 1000
    y_centers = y_centers_raw * 1000
    extent_full = [-full_R * 1000, full_R * 1000, -full_R * 1000, full_R * 1000]

    # Picks: 4 ε-distinct manifests.
    designs = load_top4("sixlet")
    px = np.array([
        d.get("seidel", {}).get("seidel_S2", float("nan")) * 1000
        for d in designs
    ])
    py = np.array([
        d.get("seidel", {}).get("seidel_S3", float("nan")) * 1000
        for d in designs
    ])
    pf = np.array([d.get("poly_refined_fitness",
                          -d.get("poly_rms_um", 0.0) / 1000.0)
                    for d in designs])
    ranks = list(range(1, len(designs) + 1))

    render(bg, fit_smooth_masked, x_centers, y_centers, extent_full,
           px, py, pf, ranks, agg, out_path=OUT_FULL, is_zoom=False)
    # Zoom view: same data, sub-window the heatmap arrays + tighten extent.
    zoom_R = ZOOM_R_MUL * 1000  # in ×10⁻³ units
    cx, cy = float(np.mean(px)), float(np.mean(py))
    xlim = (cx - zoom_R, cx + zoom_R)
    ylim = (cy - zoom_R, cy + zoom_R)
    render(bg, fit_smooth_masked, x_centers, y_centers, extent_full,
           px, py, pf, ranks, agg, out_path=OUT_ZOOM, is_zoom=True,
           xlim=xlim, ylim=ylim)

    # Microzoom: re-histogram the raw HL pool at high resolution within
    # ±MICRO_R_RAW so individual picks resolve.
    print("building microzoom from raw HL pool", flush=True)
    raw_s2, raw_s3, raw_fit = load_raw_pool()
    cx_raw, cy_raw = float(np.mean(px) / 1000), float(np.mean(py) / 1000)
    micro_R = MICRO_R_RAW
    micro_extent_raw = [cx_raw - micro_R, cx_raw + micro_R,
                         cy_raw - micro_R, cy_raw + micro_R]
    H, xe, ye = np.histogram2d(
        raw_s2, raw_s3, bins=[MICRO_N_BINS, MICRO_N_BINS],
        range=[[micro_extent_raw[0], micro_extent_raw[1]],
               [micro_extent_raw[2], micro_extent_raw[3]]],
    )
    n_img_micro = H  # (NX, NY) with S2 along first axis
    fit_img_micro = np.full((MICRO_N_BINS, MICRO_N_BINS), -1e9, dtype=np.float64)
    # Per-cell best fitness.
    ii = np.clip(((raw_s2 - micro_extent_raw[0]) / (2 * micro_R) * MICRO_N_BINS).astype(int),
                  0, MICRO_N_BINS - 1)
    jj = np.clip(((raw_s3 - micro_extent_raw[2]) / (2 * micro_R) * MICRO_N_BINS).astype(int),
                  0, MICRO_N_BINS - 1)
    for k in range(len(raw_s2)):
        if raw_fit[k] > fit_img_micro[ii[k], jj[k]]:
            fit_img_micro[ii[k], jj[k]] = raw_fit[k]
    bg_micro = gaussian_filter(np.log1p(n_img_micro), sigma=2.0)
    fit_smooth_micro = gaussian_filter(fit_img_micro, sigma=2.5)
    fit_micro_masked = np.where(n_img_micro > 0, fit_smooth_micro, np.nan)

    bin_w = (2 * micro_R) / MICRO_N_BINS
    x_centers_micro = (micro_extent_raw[0] + (np.arange(MICRO_N_BINS) + 0.5) * bin_w) * 1000
    y_centers_micro = (micro_extent_raw[2] + (np.arange(MICRO_N_BINS) + 0.5) * bin_w) * 1000
    extent_micro = [micro_extent_raw[0] * 1000, micro_extent_raw[1] * 1000,
                     micro_extent_raw[2] * 1000, micro_extent_raw[3] * 1000]
    xlim_micro = (extent_micro[0], extent_micro[1])
    ylim_micro = (extent_micro[2], extent_micro[3])
    render(bg_micro, fit_micro_masked, x_centers_micro, y_centers_micro,
           extent_micro, px, py, pf, ranks, agg,
           out_path=OUT_MICRO, is_zoom=True,
           xlim=xlim_micro, ylim=ylim_micro,
           is_microzoom=True, clean=True)


def load_raw_pool():
    """Load raw HL sixlet (S2, S3, fitness) triples from refined_top*.json
    (the same pool the aggregate JSON was built from)."""
    import glob
    s2s = []; s3s = []; fits = []
    for fp in sorted(glob.glob(str(ROOT / "results/lensmega/sixlet*/refined_top*.json"))):
        try: d = json.load(open(fp))
        except: continue
        for des in d.get("designs", []):
            m = des.get("metrics") or {}
            s2 = m.get("seidel_S2"); s3 = m.get("seidel_S3")
            f  = des.get("refined_fitness")
            if s2 is None or s3 is None or f is None: continue
            s2s.append(float(s2)); s3s.append(float(s3)); fits.append(float(f))
    return np.array(s2s), np.array(s3s), np.array(fits)


def render(bg, fit_smooth_masked, x_centers, y_centers, extent_full,
            px, py, pf, ranks, agg, out_path, is_zoom=False,
            xlim=None, ylim=None, is_microzoom=False, clean=False):

    # -------- plot --------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIX Two Text", "STIXGeneral"],
        "mathtext.fontset": "stix",
    })
    INK = SCHEME["ink"]; INK_SOFT = SCHEME["ink_soft"]
    CONTOUR = SCHEME["contour"]

    fig = plt.figure(figsize=(11.5, 10.0), facecolor=SCHEME["page_bg"])
    top_margin = 0.965 if clean else 0.84
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        width_ratios=[1.0, 0.032, 0.032],
        height_ratios=[1.0, 1.0],
        wspace=0.06, hspace=0.45,
        left=0.10, right=0.86, top=top_margin, bottom=0.10,
    )
    ax = fig.add_subplot(gs[:, 0])
    cax_bg = fig.add_subplot(gs[0, 1])
    cax_pk = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(SCHEME["ax_bg"])

    bg_vmax = float(np.nanpercentile(bg, 99.5))
    norm_bg = PowerNorm(gamma=0.45, vmin=0.0, vmax=bg_vmax)
    im = ax.imshow(
        bg.T, origin="lower", extent=extent_full,
        cmap=SCHEME["cmap_bg"], alpha=1.0, interpolation="bilinear",
        norm=norm_bg, aspect="equal",
    )
    cb_bg = fig.colorbar(im, cax=cax_bg)
    cb_label_size = 15 if clean else 11
    cb_tick_size  = 13 if clean else 9
    cb_bg.set_label(r"$\mathrm{design\ density}$", fontsize=cb_label_size,
                    color=INK, labelpad=8)
    cb_bg.ax.tick_params(labelsize=cb_tick_size, color="#888")

    # Iso-fitness contours, halo first then bright.
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
    levels = [-10.0, -1.0, -0.1]
    ax.contour(X, Y, fit_smooth_masked, levels=levels,
               colors=SCHEME["contour_halo"], linewidths=4.0, linestyles="-",
               alpha=SCHEME["halo_alpha"], zorder=3.5)
    contour_cs = ax.contour(X, Y, fit_smooth_masked, levels=levels,
                             colors=CONTOUR, linewidths=2.2, linestyles="-",
                             alpha=0.95, zorder=4)
    ax.clabel(contour_cs, inline=True, fontsize=9.5,
              fmt={l: f"{l:g}" for l in levels},
              colors=SCHEME["label_color"])

    # Picks.
    pick_size = 480 if clean else SCHEME["pick_size"]
    sc = ax.scatter(
        px, py, c=pf, cmap=SCHEME["cmap_pk"],
        s=pick_size, edgecolor=SCHEME["pick_edge"], linewidth=2.0,
        zorder=5, alpha=0.98,
    )
    cb_pk = fig.colorbar(sc, cax=cax_pk)
    cb_pk.set_label(r"$\mathrm{pick\ poly\ fitness}$", fontsize=cb_label_size,
                    color=INK, labelpad=8)
    cb_pk.ax.tick_params(labelsize=cb_tick_size, color="#888")

    # Rank annotations.
    rank_size = 15 if clean else 10
    rank_pad = (10, 8) if clean else (7, 6)
    for rk, x, y in zip(ranks, px, py):
        ax.annotate(
            str(rk), xy=(x, y),
            xytext=rank_pad, textcoords="offset points",
            fontsize=rank_size, color=INK, ha="left", va="bottom",
            fontweight="bold", zorder=6,
            bbox=dict(facecolor=SCHEME["page_bg"], edgecolor="none",
                      alpha=0.85, pad=0.6, boxstyle="round,pad=0.25"),
        )

    # Origin guides.
    ax.axhline(0, color=INK, lw=0.5, alpha=0.30, zorder=2)
    ax.axvline(0, color=INK, lw=0.5, alpha=0.30, zorder=2)

    # Axes formatting.
    if xlim is None: xlim = (extent_full[0], extent_full[1])
    if ylim is None: ylim = (extent_full[2], extent_full[3])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(INK)
    tick_size = 17 if clean else 13
    label_size = 22 if clean else 16
    ax.tick_params(colors=INK, labelsize=tick_size, width=1.0, length=5)
    ax.set_xlabel(r"$S_2$  (coma)  $\times 10^{-3}$",
                  fontsize=label_size, color=INK, labelpad=12, fontweight="bold")
    ax.set_ylabel(r"$S_3$  (astigmatism)  $\times 10^{-3}$",
                  fontsize=label_size, color=INK, labelpad=12, fontweight="bold")

    if not clean:
        if is_microzoom:
            half_um = (xlim[1] - xlim[0]) / 2 * 1000
            suffix = rf"   ·   microzoom: $\pm{half_um:.0f}\!\times\!10^{{-6}}$"
        elif is_zoom:
            suffix = rf"   ·   zoom: $\pm{(xlim[1]-xlim[0])/2:.2f}\!\times\!10^{{-3}}$"
        else:
            suffix = ""
        fig.suptitle(
            f"Sixlet APS-C  ·  aggregated Seidel archive  "
            f"({agg['n_eligible_runs']} runs)  ·  4 $\\varepsilon$-distinct picks{suffix}",
            fontsize=16, color=INK, style="italic", y=0.965,
        )
        fig.text(
            0.5, 0.918,
            r"design density $= \log(1+n_{\mathrm{designs}})$ per cell, "
            r"Gaussian-smoothed   ·   contours = best-fitness isolines",
            ha="center", va="top", fontsize=12, color=INK_SOFT, style="italic",
        )
        fig.text(
            0.5, 0.890,
            r"dots = 4 $\varepsilon$-distinct picks at raw $(S_2, S_3)$, "
            r"colored by poly refined fitness",
            ha="center", va="top", fontsize=12, color=INK_SOFT, style="italic",
        )

    fig.savefig(out_path, dpi=220, bbox_inches="tight",
                facecolor=SCHEME["page_bg"])
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
