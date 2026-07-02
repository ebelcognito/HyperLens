"""Unified Mathematica-style aggregate-archive plotter for triplet / sixlet /
octuplet. Produces vivid + elegant variants, full + zoom views.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, SymLogNorm
from scipy.ndimage import gaussian_filter

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")

# Per-system overrides. Most differences are: square radius (must match the
# picks_raw square), zoom radii (one per scheme), and output directory.
SYSTEMS = {
    "triplet": {
        "agg":   "aggregated_triplet_seidel_archive.json",
        "picks": "mono587_top32_triplet_picks_raw.json",
        "out_dir": "mono587_top32_triplet_renders",
        "square_R": 0.001,   # data units
        "zoom_R": {"vivid": 0.4, "elegant": 0.2},
        "label": "Triplet",
    },
    "sixlet": {
        "agg":   "aggregated_sixlet_seidel_archive.json",
        "picks": "mono587_top32_sixlet_picks_raw.json",
        "out_dir": "mono587_top32_sixlet_renders",
        "square_R": 0.008,
        "zoom_R": {"vivid": 3.0, "elegant": 1.5},
        "label": "Sixlet APS-C",
    },
    "octuplet": {
        "agg":   "aggregated_octuplet_seidel_archive.json",
        "picks": "mono587_top32_octuplet_picks_raw.json",
        "out_dir": "mono587_top32_octuplet_renders",
        "square_R": 0.005,
        "zoom_R": {"vivid": 2.0, "elegant": 1.0},
        "label": "Octuplet",
    },
}


def vivid_terrain_cmap():
    colors = ["#1AA8B5", "#1F7AB8", "#2B4FB8", "#6E2E97", "#A91B8F", "#3D0B47"]
    return LinearSegmentedColormap.from_list("vivid_cool", colors, N=256)


SCHEMES = {
    "vivid": {
        "cmap_bg": vivid_terrain_cmap(),
        "cmap_pk": plt.cm.magma,
        "ink": "#0B2545", "ink_soft": "#1F4F7E",
        "contour": "#FFFFFF", "contour_halo": "#0B2545", "halo_alpha": 0.35,
        "label_color": "white",
        "page_bg": "#FFFFFF", "ax_bg": "#1AA8B5", "pick_edge": "white",
        "pick_size": 110, "pick_size_zoom": 130,
    },
    "elegant": {
        "cmap_bg": plt.cm.inferno,
        "cmap_pk": plt.cm.cividis,
        "ink": "#1A0A0A", "ink_soft": "#5C3A2E",
        "contour": "#FFE6A8", "contour_halo": "#000000", "halo_alpha": 0.5,
        "label_color": "#FFE6A8",
        "page_bg": "#FFFFFF", "ax_bg": "#000004", "pick_edge": "#FFFFFF",
        "pick_size": 110, "pick_size_zoom": 240,
    },
}

# Shared visual config
SMOOTH_SIGMA = 1.8
HULL_FIT_TH = -1.0   # for the picks file note; we don't draw the hull anymore
LEVELS = [-10.0, -1.0, -0.1]


def render_system(system):
    cfg = SYSTEMS[system]
    agg = json.load(open(ROOT / cfg["agg"]))
    picks_bundle = json.load(open(ROOT / cfg["picks"]))
    picks = picks_bundle["picks"]
    print(f"\n[{system}] {len(picks)} picks resolved, "
          f"{picks_bundle.get('n_oob_dropped', 0)} OOB dropped, "
          f"{picks_bundle.get('n_unresolved', 0)} unresolved")

    NI, NJ = agg["target_bins"]
    BOUNDS = np.array(agg["target_bounds"])
    bin_w_s2_canon = (BOUNDS[0, 1] - BOUNDS[0, 0]) / NI
    bin_w_s3_canon = (BOUNDS[1, 1] - BOUNDS[1, 0]) / NJ

    # Square window radius (data units) — pulled from system config
    full_R = float(cfg["square_R"])
    BW = min(bin_w_s2_canon, bin_w_s3_canon)
    NX = int(np.ceil((2 * full_R) / BW)); NX += NX % 2
    NY = int(np.ceil((2 * full_R) / BW)); NY += NY % 2

    def to_full_idx(s2, s3):
        ii = int(np.clip(np.floor((s2 - (-full_R)) / BW), 0, NX - 1))
        jj = int(np.clip(np.floor((s3 - (-full_R)) / BW), 0, NY - 1))
        return ii, jj

    n_img = np.zeros((NX, NY), dtype=np.float64)
    fit_img = np.full((NX, NY), np.nan, dtype=np.float64)
    for c in agg["cells"]:
        cx = BOUNDS[0, 0] + (c["cell_ij"][0] + 0.5) * bin_w_s2_canon
        cy = BOUNDS[1, 0] + (c["cell_ij"][1] + 0.5) * bin_w_s3_canon
        if abs(cx) > full_R or abs(cy) > full_R:
            continue
        ii, jj = to_full_idx(cx, cy)
        n_img[ii, jj] += c["n_designs"]
        f = c["best_fitness"]
        if np.isnan(fit_img[ii, jj]) or f > fit_img[ii, jj]:
            fit_img[ii, jj] = f
    print(f"  bg grid {NX}x{NY}, BW={BW:.2e}, full_R=±{full_R:.3e}")

    SIGMA_FULL = SMOOTH_SIGMA * 1.8
    bg = gaussian_filter(np.log1p(n_img), sigma=SIGMA_FULL,
                         mode="constant", cval=0.0)
    # Normalized-convolution smoothing for the fit field
    mask = np.isfinite(fit_img).astype(np.float64)
    vals = np.where(np.isfinite(fit_img), fit_img, 0.0)
    SIGMA_FIT = SMOOTH_SIGMA * 2.5
    num = gaussian_filter(vals * mask, sigma=SIGMA_FIT, mode="constant", cval=0.0)
    den = gaussian_filter(mask,         sigma=SIGMA_FIT, mode="constant", cval=0.0)
    fit_smooth = np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)
    fit_smooth_masked = np.where(den > 0.05, fit_smooth, np.nan)

    extent = [-full_R*1000, full_R*1000, -full_R*1000, full_R*1000]
    x_centers = (-full_R + (np.arange(NX) + 0.5) * BW) * 1000
    y_centers = (-full_R + (np.arange(NY) + 0.5) * BW) * 1000

    px = np.array([p["raw_S2"] for p in picks]) * 1000
    py = np.array([p["raw_S3"] for p in picks]) * 1000
    pf = np.array([p["mono_fitness"] for p in picks])
    ranks = [p["rank"] for p in picks]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
    })

    OUT_DIR = ROOT / cfg["out_dir"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def render(out_path, xlim, ylim, scheme, zoom_label=None, is_zoom=False):
        S = SCHEMES[scheme]
        INK, INK_SOFT, CONTOUR = S["ink"], S["ink_soft"], S["contour"]
        pick_s = S["pick_size_zoom"] if is_zoom else S["pick_size"]

        fig = plt.figure(figsize=(12.0, 11.0), facecolor=S["page_bg"])
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

        # Terrain
        bg_vmax = float(np.nanpercentile(bg, 99.5))
        norm_bg = PowerNorm(gamma=0.45, vmin=0.0, vmax=bg_vmax)
        im = ax.imshow(
            bg.T, origin="lower", extent=extent,
            cmap=S["cmap_bg"], alpha=1.0, interpolation="bilinear",
            norm=norm_bg,
        )

        # Contours
        X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
        ax.contour(
            X, Y, fit_smooth_masked, levels=LEVELS,
            colors=S["contour_halo"], linewidths=4.0, linestyles="-",
            alpha=S["halo_alpha"], zorder=3.5,
        )
        c_cs = ax.contour(
            X, Y, fit_smooth_masked, levels=LEVELS,
            colors=CONTOUR, linewidths=2.4, linestyles="-",
            alpha=0.95, zorder=4,
        )
        ax.clabel(c_cs, inline=True, fontsize=9.5,
                  fmt={l: f"{l:g}" for l in LEVELS},
                  colors=S["label_color"])

        # Picks
        sc = ax.scatter(
            px, py, c=pf, cmap=S["cmap_pk"],
            s=pick_s, edgecolor=S["pick_edge"], linewidth=1.4,
            zorder=5, alpha=0.98,
        )
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

        ax.axhline(0, color=INK, lw=0.5, alpha=0.30, zorder=2)
        ax.axvline(0, color=INK, lw=0.5, alpha=0.30, zorder=2)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("bottom", "left"): ax.spines[s].set_color(INK)
        ax.tick_params(colors=INK, labelsize=14)
        ax.set_xlabel(r"$S_2$  (coma)  $\times 10^{-3}$",
                      fontsize=18, color=INK, labelpad=12, fontweight="bold")
        ax.set_ylabel(r"$S_3$  (astigmatism)  $\times 10^{-3}$",
                      fontsize=18, color=INK, labelpad=12, fontweight="bold")

        suffix = f"   ·   zoom: $\\pm{zoom_label}$" if zoom_label else ""
        fig.suptitle(
            f"{cfg['label']}  ·  aggregated Seidel archive  "
            f"({agg['n_eligible_runs']} runs){suffix}",
            fontsize=18, color=INK, style="italic", y=0.965,
        )
        # Two-line caption so it fits the figure width at larger fonts
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
        print(f"  saved -> {out_path}")

    # Full square (in 10^-3 units)
    R_disp = full_R * 1000
    for scheme in SCHEMES:
        render(OUT_DIR / f"aggregate_archive_{scheme}.png",
               xlim=(-R_disp, R_disp), ylim=(-R_disp, R_disp),
               scheme=scheme, is_zoom=False)
    # Zooms (in 10^-3 units, per-scheme)
    for scheme, Z in cfg["zoom_R"].items():
        render(OUT_DIR / f"aggregate_archive_{scheme}_zoom.png",
               xlim=(-Z, Z), ylim=(-Z, Z),
               scheme=scheme, is_zoom=True,
               zoom_label=f"{Z:g}\\times 10^{{-3}}")


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["triplet", "octuplet"]
    for s in targets:
        render_system(s)


if __name__ == "__main__":
    main()
