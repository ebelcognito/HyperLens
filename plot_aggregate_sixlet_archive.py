"""Step 3: Mathematica-style plot of the aggregated sixlet Seidel archive.

Layers:
  1. Background terrain  = log(1 + n_designs per cell), Gaussian-smoothed,
                           warm brown-tan colormap.
  2. Convex hull outline = boundary of cells with best_fitness >= -1.
  3. Scatter picks       = 29 in-bounds top-32 picks at their raw (S2, S3),
                           colored by mono_fitness with a distinct colormap.

Axes shown in units of 10^-3 (S2, S3 are O(1e-3)). Italic serif title.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, PowerNorm
from matplotlib.patches import Polygon as MplPolygon
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
AGG = ROOT / "aggregated_sixlet_seidel_archive.json"
PICKS = ROOT / "mono587_top32_sixlet_picks_raw.json"
OUT_DIR = ROOT / "mono587_top32_sixlet_renders"

# Visual config
SMOOTH_SIGMA = 1.8    # Gaussian sigma in cells
HULL_FIT_TH = -1.0    # convex hull over cells with best_fitness >= this


def vivid_terrain_cmap():
    # Vivid cool ramp, no pale floor.
    colors = ["#1AA8B5", "#1F7AB8", "#2B4FB8", "#6E2E97", "#A91B8F", "#3D0B47"]
    return LinearSegmentedColormap.from_list("vivid_cool", colors, N=256)


SCHEMES = {
    "vivid": {
        "cmap_bg": vivid_terrain_cmap(),
        "cmap_pk": plt.cm.magma,
        "ink": "#0B2545",
        "ink_soft": "#1F4F7E",
        "contour": "#FFFFFF",
        "contour_halo": "#0B2545",
        "halo_alpha": 0.35,
        "label_color": "white",
        "page_bg": "#FFFFFF",
        "ax_bg": "#1AA8B5",
        "pick_edge": "white",
        "pick_size": 110,
        "pick_size_zoom": 130,
        "zoom_R": 3.0,
    },
    "elegant": {
        # Inferno heat palette — black → deep red → orange → yellow
        "cmap_bg": plt.cm.inferno,
        "cmap_pk": plt.cm.cividis,   # cool blue-yellow picks pop against warm inferno
        "ink": "#1A0A0A",
        "ink_soft": "#5C3A2E",
        "contour": "#FFE6A8",        # warm cream — visible on both dark and bright bands
        "contour_halo": "#000000",
        "halo_alpha": 0.5,
        "label_color": "#FFE6A8",
        "page_bg": "#FFFFFF",        # warm cream page
        "ax_bg": "#000004",          # matches inferno's floor
        "pick_edge": "#FFFFFF",
        "pick_size": 110,
        "pick_size_zoom": 240,       # noticeably bigger so picks read as spread out
        "zoom_R": 1.5,               # tighter zoom on the central core
    },
}




def main():
    agg = json.load(open(AGG))
    picks_bundle = json.load(open(PICKS))
    # Drop rank 7 — its S3 ≈ −1.03×10^−2 was the sole pick forcing the view
    # outside the ±8×10^−3 square. Dropping it lets the natural canonical-S2
    # bound define the square window with no padding hack.
    DROP_RANKS = {7}
    picks_all = picks_bundle["picks"]
    picks = [p for p in picks_all if p["rank"] not in DROP_RANKS]
    print(f"dropped ranks {sorted(DROP_RANKS)} (outside ±8×10^-3 square); "
          f"{len(picks)} picks remain")
    NI, NJ = agg["target_bins"]
    BOUNDS = np.array(agg["target_bounds"])

    # === Build the bg at a SQUARE window sized to the canonical S2 bound ===
    # (±0.008 in both axes). The S2 dim covers exactly the archive; the S3
    # dim cuts off canonical data beyond ±0.008 but that region is empty for
    # the kept picks (rank 7 was the only thing out there and we dropped it).
    bin_w_s2_canon = (BOUNDS[0, 1] - BOUNDS[0, 0]) / NI   # 6.25e-5
    bin_w_s3_canon = (BOUNDS[1, 1] - BOUNDS[1, 0]) / NJ   # 9.375e-5
    full_R = float(BOUNDS[0, 1])                          # 0.008  (square radius)
    BW = min(bin_w_s2_canon, bin_w_s3_canon)
    NX = int(np.ceil((2 * full_R) / BW))
    NY = int(np.ceil((2 * full_R) / BW))
    NX += NX % 2; NY += NY % 2
    FULL_EXTENT_RAW = [-full_R, full_R, -full_R, full_R]

    def to_full_idx(s2, s3):
        ii = int(np.clip(np.floor((s2 - FULL_EXTENT_RAW[0]) / BW), 0, NX - 1))
        jj = int(np.clip(np.floor((s3 - FULL_EXTENT_RAW[2]) / BW), 0, NY - 1))
        return ii, jj

    n_img = np.zeros((NX, NY), dtype=np.float64)
    fit_img = np.full((NX, NY), np.nan, dtype=np.float64)
    for c in agg["cells"]:
        # Cell center in data units
        cx = BOUNDS[0, 0] + (c["cell_ij"][0] + 0.5) * bin_w_s2_canon
        cy = BOUNDS[1, 0] + (c["cell_ij"][1] + 0.5) * bin_w_s3_canon
        ii, jj = to_full_idx(cx, cy)
        n_img[ii, jj] += c["n_designs"]
        f = c["best_fitness"]
        if np.isnan(fit_img[ii, jj]) or f > fit_img[ii, jj]:
            fit_img[ii, jj] = f

    # Moderate sigma — enough to soften the boundary but not wash the data out.
    SIGMA_FULL = SMOOTH_SIGMA * 1.8
    bg = gaussian_filter(np.log1p(n_img), sigma=SIGMA_FULL, mode="constant", cval=0.0)

    # Smoothed fitness field via NORMALIZED CONVOLUTION — only filled cells
    # contribute, so each smoothed value is the weighted local average of
    # actual fitness data (no -inf bleed from empty cells).
    mask = np.isfinite(fit_img).astype(np.float64)
    vals = np.where(np.isfinite(fit_img), fit_img, 0.0)
    SIGMA_FIT = SMOOTH_SIGMA * 2.5      # broader so contours look smooth
    num = gaussian_filter(vals * mask, sigma=SIGMA_FIT, mode="constant", cval=0.0)
    den = gaussian_filter(mask,         sigma=SIGMA_FIT, mode="constant", cval=0.0)
    fit_smooth = np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)
    # Show contours wherever the smoothing had enough support (not just at
    # original filled cells — so contours can extend into the gaps).
    fit_smooth_masked = np.where(den > 0.05, fit_smooth, np.nan)

    # Overwrite extent variables used later — heatmap & contour grids now live
    # on the FULL square at the BW resolution.
    extent = FULL_EXTENT_RAW
    NI, NJ = NX, NY
    bin_w_s2 = bin_w_s3 = BW
    # Place the cell-center grid on the new array
    x_centers_full = (FULL_EXTENT_RAW[0] + (np.arange(NX) + 0.5) * BW)
    y_centers_full = (FULL_EXTENT_RAW[2] + (np.arange(NY) + 0.5) * BW)

    # Plot extents in units of 10^-3 — use the FULL square window.
    s2_lo, s2_hi = -full_R * 1000, full_R * 1000
    s3_lo, s3_hi = -full_R * 1000, full_R * 1000
    extent = [s2_lo, s2_hi, s3_lo, s3_hi]

    # Hull of "good" cells (fit >= HULL_FIT_TH), in 10^-3 units.
    # Kept for the zoom radius calc; not drawn anymore (contour line replaces it).
    good_cells_xy = [(BOUNDS[0, 0] + (c["cell_ij"][0] + 0.5) * bin_w_s2_canon,
                      BOUNDS[1, 0] + (c["cell_ij"][1] + 0.5) * bin_w_s3_canon)
                     for c in agg["cells"]
                     if c["best_fitness"] >= HULL_FIT_TH]
    pts = np.array(good_cells_xy) * 1000
    hull = ConvexHull(pts) if len(pts) >= 3 else None
    print(f"Hull built from {len(pts)} cells with fit >= {HULL_FIT_TH}")

    # --- Render helper ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
    })

    px = np.array([p["raw_S2"] for p in picks]) * 1000
    py = np.array([p["raw_S3"] for p in picks]) * 1000
    pf = np.array([p["mono_fitness"] for p in picks])

    # Per-scheme colors are pulled from SCHEMES[scheme] in render()

    def pad_to_window(bg_arr, orig_ext, target_ext, floor=0.0):
        """Resample bg_arr onto a grid that covers target_ext at the same bin
        widths as orig, filling cells outside orig_ext with `floor`. Handles
        target larger OR smaller than orig in any dimension."""
        NI, NJ = bg_arr.shape
        dx = (orig_ext[1] - orig_ext[0]) / NI
        dy = (orig_ext[3] - orig_ext[2]) / NJ
        # Snap target so its corners sit on the orig grid
        x0 = orig_ext[0] + int(np.floor((target_ext[0] - orig_ext[0]) / dx)) * dx
        x1 = orig_ext[0] + int(np.ceil((target_ext[1] - orig_ext[0]) / dx)) * dx
        y0 = orig_ext[2] + int(np.floor((target_ext[2] - orig_ext[2]) / dy)) * dy
        y1 = orig_ext[2] + int(np.ceil((target_ext[3] - orig_ext[2]) / dy)) * dy
        new_NI = int(round((x1 - x0) / dx))
        new_NJ = int(round((y1 - y0) / dy))
        out = np.full((new_NI, new_NJ), floor, dtype=bg_arr.dtype)
        # Where the orig grid lives within the new grid
        i_src_start = max(0, int(round((x0 - orig_ext[0]) / dx)))
        j_src_start = max(0, int(round((y0 - orig_ext[2]) / dy)))
        i_src_end   = min(NI, int(round((x1 - orig_ext[0]) / dx)))
        j_src_end   = min(NJ, int(round((y1 - orig_ext[2]) / dy)))
        i_dst_start = max(0, int(round((orig_ext[0] - x0) / dx)))
        j_dst_start = max(0, int(round((orig_ext[2] - y0) / dy)))
        i_len = i_src_end - i_src_start
        j_len = j_src_end - j_src_start
        if i_len > 0 and j_len > 0:
            out[i_dst_start:i_dst_start+i_len,
                j_dst_start:j_dst_start+j_len] = \
                bg_arr[i_src_start:i_src_end, j_src_start:j_src_end]
        return out, [x0, x1, y0, y1]

    def render(out_path, xlim, ylim, scheme, zoom_label=None, is_zoom=False):
        S = SCHEMES[scheme]
        INK = S["ink"]; INK_SOFT = S["ink_soft"]
        CONTOUR = S["contour"]
        pick_s = S["pick_size_zoom"] if is_zoom else S["pick_size"]
        fig = plt.figure(figsize=(12.0, 11.0), facecolor=S["page_bg"])
        # Square main axes + two narrow colorbar columns
        gs = fig.add_gridspec(
            nrows=2, ncols=3,
            width_ratios=[1.0, 0.028, 0.028],
            height_ratios=[1.0, 1.0],
            wspace=0.06, hspace=0.45,
            left=0.10, right=0.86, top=0.84, bottom=0.09,
        )
        ax = fig.add_subplot(gs[:, 0])
        cax_bg = fig.add_subplot(gs[0, 1])
        cax_pk = fig.add_subplot(gs[1, 1])
        ax.set_facecolor(S["ax_bg"])

        # Terrain — stretch low values up the colormap with PowerNorm
        bg_vmax = float(np.nanpercentile(bg, 99.5))
        norm_bg = PowerNorm(gamma=0.45, vmin=0.0, vmax=bg_vmax)
        im = ax.imshow(
            bg.T, origin="lower", extent=extent,
            cmap=S["cmap_bg"], alpha=1.0, interpolation="bilinear",
            norm=norm_bg,
        )

        # Iso-fitness contour lines (cell centers in 10^-3 units, full grid)
        x_centers = x_centers_full * 1000
        y_centers = y_centers_full * 1000
        X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
        levels = [-10.0, -1.0, -0.1]
        # Halo first (under the main contour)
        ax.contour(
            X, Y, fit_smooth_masked, levels=levels,
            colors=S["contour_halo"], linewidths=4.0, linestyles="-",
            alpha=S["halo_alpha"], zorder=3.5,
        )
        contour_cs = ax.contour(
            X, Y, fit_smooth_masked,
            levels=levels,
            colors=CONTOUR, linewidths=2.4, linestyles="-",
            alpha=0.95, zorder=4,
        )
        ax.clabel(contour_cs, inline=True, fontsize=9.5,
                  fmt={l: f"{l:g}" for l in levels},
                  colors=S["label_color"])

        # Picks
        sc = ax.scatter(
            px, py, c=pf, cmap=S["cmap_pk"],
            s=pick_s, edgecolor=S["pick_edge"], linewidth=1.4,
            zorder=5, alpha=0.98,
        )

        # Rank annotations
        ranks = [p["rank"] for p in picks]
        for rk, x, y in zip(ranks, px, py):
            ax.annotate(
                str(rk), xy=(x, y),
                xytext=(6, 5), textcoords="offset points",
                fontsize=7.5, color=INK,
                ha="left", va="bottom",
                zorder=6,
                bbox=dict(facecolor=S["page_bg"], edgecolor="none",
                          alpha=0.65, pad=0.6, boxstyle="round,pad=0.18"),
            )

        # Origin guides
        ax.axhline(0, color=INK, lw=0.5, alpha=0.30, zorder=2)
        ax.axvline(0, color=INK, lw=0.5, alpha=0.30, zorder=2)

        # Axes
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("bottom", "left"):
            ax.spines[s].set_color(INK)
        ax.tick_params(colors=INK, labelsize=14)
        ax.set_xlabel(r"$S_2$  (coma)  $\times 10^{-3}$",
                      fontsize=18, color=INK, labelpad=12, fontweight="bold")
        ax.set_ylabel(r"$S_3$  (astigmatism)  $\times 10^{-3}$",
                      fontsize=18, color=INK, labelpad=12, fontweight="bold")

        suffix = f"   ·   zoom: $\\pm{zoom_label}$" if zoom_label else ""
        fig.suptitle(
            f"Sixlet APS-C  ·  aggregated Seidel archive  "
            f"({agg['n_eligible_runs']} runs){suffix}",
            fontsize=18, color=INK, style="italic", y=0.965,
        )
        fig.text(
            0.5, 0.918,
            f"design density $= \\log(1+n_{{\\mathrm{{designs}}}})$ per cell, "
            f"Gaussian-smoothed   ·   contours = best-fitness isolines",
            ha="center", va="top", fontsize=13, color=INK_SOFT, style="italic",
        )
        fig.text(
            0.5, 0.890,
            f"dots = {len(picks)} picks at raw $(S_2, S_3)$, colored by mono-fitness",
            ha="center", va="top", fontsize=13, color=INK_SOFT, style="italic",
        )

        # Colorbars
        cb_bg = fig.colorbar(im, cax=cax_bg)
        cb_bg.set_label("design density",
                        fontsize=14, color=INK, fontweight="bold", labelpad=10)
        cb_bg.outline.set_edgecolor(INK); cb_bg.outline.set_linewidth(0.7)
        cb_bg.ax.tick_params(colors=INK, labelsize=12)

        cb_pk = fig.colorbar(sc, cax=cax_pk)
        cb_pk.set_label("pick mono-fitness",
                        fontsize=14, color=INK, fontweight="bold", labelpad=10)
        cb_pk.outline.set_edgecolor(INK); cb_pk.outline.set_linewidth(0.7)
        cb_pk.ax.tick_params(colors=INK, labelsize=12)

        fig.savefig(out_path, dpi=200, facecolor=S["page_bg"])
        plt.close(fig)
        print(f"saved -> {out_path}")

    # Full view: square data window
    full_R_disp = max(abs(s2_lo), abs(s2_hi), abs(s3_lo), abs(s3_hi))
    for scheme in SCHEMES:
        render(OUT_DIR / f"aggregate_archive_{scheme}.png",
               xlim=(-full_R_disp, full_R_disp),
               ylim=(-full_R_disp, full_R_disp),
               scheme=scheme,
               zoom_label=None)

    # Zoom: tight on the central dense core. Each scheme picks its own R.
    for scheme in SCHEMES:
        Z = SCHEMES[scheme]["zoom_R"]
        print(f"zoom ({scheme}) = ±{Z}e-3")
        render(OUT_DIR / f"aggregate_archive_{scheme}_zoom.png",
               xlim=(-Z, Z), ylim=(-Z, Z),
               scheme=scheme, is_zoom=True,
               zoom_label=f"{Z:g}\\times 10^{{-3}}")


if __name__ == "__main__":
    main()
