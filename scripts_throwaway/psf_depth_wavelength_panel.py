#!/usr/bin/env python
"""PSF panel matching the reference-figure style:
  top row : 3D surface of the on-axis PSF (polychromatic sum at infinity)
  bot row : 1D center-row cross-section overlays for 3 wavelengths
            (B, G, R) at 3 object depths (0.5 m, 1.0 m, infinity)

One panel per topology (sixlet, octuplet), each spanning the 4
ε-distinct designs from figures/topology_cards/.
"""
from __future__ import annotations
import json, os, sys, time
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
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DeepLens"))

from scripts_throwaway.fig5_extract import build_lens_template_and_solver, make_geolens
from scripts_throwaway.elegant_psf_map import mathematica_style

# RGB / depth / style conventions (B-G-R order to match reference legend).
WVLNS_UM = (0.486, 0.587, 0.656)
WAVE_LABELS = ("B", "G", "R")
WAVE_COLORS = ("#1d4ed8", "#2a9d8f", "#e63946")
DEPTHS_MM = (-500.0, -1000.0, -20000.0)
DEPTH_LABELS = (r"-0.5\,\mathrm{m}", r"1.0\,\mathrm{m}", r"\infty")
DEPTH_STYLES = ("--", "-", ":")
DEPTH_ALPHAS = (0.85, 1.0, 0.75)
PSF_KS = 64
OVERSAMPLE = 8
SURFACE_CMAP = "turbo"

CARD_DIRS = {
    "sixlet":   ROOT / "figures/zemax_reports_top4_diverse_fnum5/sixlet",
    "octuplet": ROOT / "figures/zemax_reports_top4_diverse_fnum5/octuplet",
}
OUT_DIR = ROOT / "figures" / "psf_depth_wavelength"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_manifests(sys_name: str):
    base = CARD_DIRS[sys_name]
    out = []
    for rank in (1, 2, 3, 4):
        cand = sorted(base.glob(f"rank{rank}_*/manifest.json"))
        if not cand:
            print(f"  ! no manifest for rank {rank} of {sys_name}")
            continue
        out.append(json.loads(cand[0].read_text()))
    return out


def build_lens(manifest: dict):
    run_dir = ROOT / "results" / "lensmega" / manifest["source_run"]
    cfg_path = run_dir / "config.yaml"
    theta = np.asarray(manifest["theta"], dtype=np.float64)
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(cfg_path)
    lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver,
                        device="cpu")
    if OVERSAMPLE != 1:
        orig_res = tuple(lens.sensor_res)
        lens.set_sensor_res((int(orig_res[0] * OVERSAMPLE),
                              int(orig_res[1] * OVERSAMPLE)))
    return lens


def compute_psf_set(lens, ks: int):
    """Return {(wvln_um, depth_mm): psf [ks,ks] numpy}."""
    out = {}
    for w in WVLNS_UM:
        for d in DEPTHS_MM:
            point = torch.tensor([0.0, 0.0, float(d)],
                                  dtype=torch.float32)
            with torch.no_grad():
                psf = lens.psf(points=point, wvln=float(w), ks=ks)
            out[(float(w), float(d))] = psf.detach().cpu().numpy()
    return out


def crop_psf(psf: np.ndarray, half: int) -> np.ndarray:
    H, W = psf.shape
    cy, cx = H // 2, W // 2
    return psf[cy - half:cy + half + 1, cx - half:cx + half + 1]


def render_panel(sys_name: str, designs: list[dict], out_path: Path,
                  half_pix: int = 12):
    n = len(designs)
    fig = plt.figure(figsize=(2.0 * n + 1.4, 4.4), facecolor="white")
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1.0] * n + [0.32],
                           height_ratios=[1.0, 1.0],
                           left=0.04, right=0.99, top=0.93, bottom=0.10,
                           wspace=0.22, hspace=0.05)

    for i, des in enumerate(designs):
        psfs = des["psfs"]
        # Global max across all 9 PSFs of this design (so defocused PSFs read
        # as the small/wide blobs they actually are, not auto-stretched to 1).
        gmax = max(float(psfs[(w, d)].max()) for w in WVLNS_UM for d in DEPTHS_MM)
        # Top: 3D surface of polychromatic PSF at infinity (in-focus).
        poly_inf = sum(psfs[(w, DEPTHS_MM[2])] for w in WVLNS_UM)
        poly_inf = crop_psf(poly_inf, half_pix)
        poly_inf = poly_inf / (poly_inf.max() + 1e-12)

        ax3 = fig.add_subplot(gs[0, i], projection="3d")
        H, W = poly_inf.shape
        Y, X = np.mgrid[0:H, 0:W]
        ax3.plot_surface(X, Y, poly_inf, cmap=SURFACE_CMAP,
                          rstride=1, cstride=1, edgecolor="none",
                          antialiased=True, linewidth=0)
        ax3.view_init(elev=24, azim=-55)
        ax3.set_box_aspect((1, 1, 0.55))
        ax3.set_axis_off()
        ax3.set_zlim(0, 1.05)
        fnum = des["manifest"].get("fnum_actual", float("nan"))
        ax3.set_title(rf"$\mathrm{{{sys_name}\#{i+1}}},\ f/{fnum:.2f}$",
                      fontsize=10.5, y=0.92)

        # Bottom: 1D cross-section overlays, normalized to the per-design max
        # so defocus actually reads as lower amplitude.
        axb = fig.add_subplot(gs[1, i])
        for wi, w in enumerate(WVLNS_UM):
            for di, d in enumerate(DEPTHS_MM):
                psf = crop_psf(psfs[(w, d)], half_pix) / (gmax + 1e-12)
                line = psf[psf.shape[0] // 2, :]
                x_pix = np.arange(line.size) - line.size // 2
                axb.plot(x_pix, line, color=WAVE_COLORS[wi],
                         linestyle=DEPTH_STYLES[di], lw=1.0,
                         alpha=DEPTH_ALPHAS[di])
        axb.set_xlim(-half_pix, half_pix)
        axb.set_ylim(1e-3, 1.2)
        axb.set_yscale("log")
        axb.set_xticks([-half_pix, 0, half_pix])
        axb.set_xticklabels([rf"${-half_pix}$",
                             r"$\mathrm{pixels}$",
                             rf"${half_pix}$"],
                            fontsize=9)
        axb.set_yticks([])
        axb.grid(True, axis="x", lw=0.3, color="#dddddd", alpha=0.6)
        for spine in ("top", "right", "left"):
            axb.spines[spine].set_visible(False)
        axb.spines["bottom"].set_linewidth(0.7)
        axb.tick_params(width=0.6, length=2.5, color="#666666", pad=2)

    # Legend column (right).
    leg_ax = fig.add_subplot(gs[:, -1])
    leg_ax.axis("off")
    handles = []
    for wi, (col, lbl) in enumerate(zip(WAVE_COLORS, WAVE_LABELS)):
        for di, (st, dlbl) in enumerate(zip(DEPTH_STYLES, DEPTH_LABELS)):
            handles.append(Line2D([0], [0], color=col, linestyle=st, lw=1.5,
                                  label=rf"$\mathrm{{{lbl}}}\ {dlbl}$"))
    leg_ax.legend(handles=handles, loc="center", frameon=False,
                  fontsize=9.5, handlelength=2.2, handletextpad=0.5,
                  labelspacing=0.32)

    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main():
    mathematica_style()
    for sys_name in ("sixlet", "octuplet"):
        print(f"\n=== {sys_name} ===", flush=True)
        manifests = load_manifests(sys_name)
        designs = []
        t0 = time.time()
        for k, mf in enumerate(manifests, 1):
            print(f"  [{k}/{len(manifests)}] building lens...", flush=True)
            ks = PSF_KS * OVERSAMPLE
            lens = build_lens(mf)
            print(f"    computing {len(WVLNS_UM) * len(DEPTHS_MM)} PSFs...",
                  flush=True)
            psfs = compute_psf_set(lens, ks)
            designs.append({"manifest": mf, "psfs": psfs})
            print(f"    done ({time.time() - t0:.1f}s)", flush=True)
        out = OUT_DIR / f"psf_depth_lambda_{sys_name}.png"
        render_panel(sys_name, designs, out)
        print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
