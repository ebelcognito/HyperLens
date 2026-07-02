#!/usr/bin/env python
"""PSF line-plot panel for the top-5 polychromatic sixlet designs.

Layout: 3 rows × 5 cols.
  rows = field height (on-axis / half-field / corner)
  cols = design (#1..#5 by polychromatic refined fitness, no diversity)
  per cell: 1-D PSF center-row cross-section, 9 lines (3 λ × 3 depths)
  legend strip on the right matching the reference (B/G/R × −0.5 m / 1.0 m / ∞)

Matches the figure style the user shared (PSFs in a row, depth+wavelength
legend on the right).
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
from matplotlib.lines import Line2D

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DeepLens"))

from scripts_throwaway.fig5_extract import build_lens_template_and_solver, make_geolens
from scripts_throwaway.elegant_psf_map import mathematica_style

# Same B/G/R convention as the reference legend (R/G/B colors as labels read top-down).
WVLNS_UM = (0.486, 0.587, 0.656)
WAVE_LABELS = ("B", "G", "R")
WAVE_COLORS = ("#1d4ed8", "#2a9d8f", "#e63946")
# Depths relative to the 50mm design depth of -20m. ±~50% defocus shows
# meaningful spread without exploding into ring artifacts.
DEPTHS_MM = (-10000.0, -20000.0, -50000.0)
DEPTH_LABELS = (r"10\,\mathrm{m}", r"20\,\mathrm{m}\ \mathrm{(design)}", r"50\,\mathrm{m}")
DEPTH_STYLES = ("--", "-", ":")
DEPTH_ALPHAS = (0.85, 1.0, 0.80)
FIELD_FRACS = (0.0, 0.5, 1.0)
FIELD_LABEL_TEX = (r"\mathrm{on\!-\!axis}",
                    r"\mathrm{half\!-\!field}",
                    r"\mathrm{corner}")
PSF_KS = 64
OVERSAMPLE = 16   # finer sensor sampling → in-focus PSF spans many pixels
# Visible half-window per field (µm). Hardcoded to keep the in-focus cores
# legible; defocused tails extending off-screen are accepted (they show as
# lower-amplitude curves with their wings cut, which still reads correctly
# as 'broad and weak vs sharp and bright').
HALF_UM_PER_ROW = (25.0, 25.0, 25.0)  # shared window so widths compare visually

OUT = ROOT / "figures" / "psf_lines_top5"
OUT.mkdir(parents=True, exist_ok=True)
FNUM_MAX = 5.0


def top5_polychromatic_sixlet():
    """Top 5 by refined_fitness over the sixlet pool, restricted to f/# < FNUM_MAX."""
    recs = []
    for fp in sorted(glob.glob(str(ROOT / "results/lensmega/sixlet*/refined_top*.json"))):
        try: d = json.load(open(fp))
        except: continue
        run = Path(fp).parent.name
        for i, des in enumerate(d.get("designs", [])):
            f = des.get("refined_fitness"); r = des.get("rms_um")
            th = des.get("theta")
            m = des.get("metrics") or {}
            fn = des.get("fnum_actual") or m.get("fnum_actual")
            if f is None or r is None or th is None or fn is None: continue
            if float(fn) >= FNUM_MAX: continue
            recs.append((float(f), float(r), float(fn), run, i, des, Path(fp).name))
    recs.sort(key=lambda x: -x[0])
    return recs[:5]


def build_one_lens(run_name: str, theta_list):
    run_dir = ROOT / "results/lensmega" / run_name
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        run_dir / "config.yaml"
    )
    theta = np.asarray(theta_list, dtype=np.float64)
    lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver,
                        device="cpu")
    if OVERSAMPLE != 1:
        orig = tuple(lens.sensor_res)
        lens.set_sensor_res((int(orig[0] * OVERSAMPLE),
                              int(orig[1] * OVERSAMPLE)))
    return lens


def compute_psf(lens, field_frac: float, wvln_um: float, depth_mm: float,
                ks: int) -> np.ndarray:
    pt = torch.tensor([0.0, field_frac, depth_mm], dtype=torch.float32)
    with torch.no_grad():
        psf = lens.psf(points=pt, wvln=float(wvln_um), ks=ks, recenter=True)
    return psf.detach().cpu().numpy()


def main():
    picks = top5_polychromatic_sixlet()
    print(f"top-5 polychromatic sixlets (f/# < {FNUM_MAX}):")
    for k, (f, r, fn, run, idx, _, _) in enumerate(picks, 1):
        print(f"  #{k}  poly_RMS={r:5.2f}µm  f/{fn:.2f}  fit={f:+.5f}  src={run}#{idx}",
              flush=True)

    # Build lenses + compute PSFs.
    ks = PSF_KS * OVERSAMPLE
    lenses = []
    pix_um = None
    print("\nbuilding lenses + computing PSFs", flush=True)
    psfs = {}
    for di, (f, r, fn, run, idx, des, _) in enumerate(picks):
        print(f"  [{di+1}/5] building {run}#{idx}", flush=True)
        lens = build_one_lens(run, des["theta"])
        lenses.append(lens)
        if pix_um is None:
            pix_um = float(lens.pixel_size) * 1e3
        t0 = time.time()
        for fi, frac in enumerate(FIELD_FRACS):
            for wi, w in enumerate(WVLNS_UM):
                for dpi, d in enumerate(DEPTHS_MM):
                    psfs[(di, fi, wi, dpi)] = compute_psf(lens, frac, w, d, ks)
        print(f"    27 PSFs in {time.time()-t0:.1f}s", flush=True)

    n_d = 5; n_f = 3
    half_um_per_row = list(HALF_UM_PER_ROW)
    print("per-row visible half-extent (µm):", flush=True)
    for fi in range(n_f):
        print(f"  field {fi}: ±{half_um_per_row[fi]:.1f}", flush=True)

    # ---- plot ----
    mathematica_style()
    plt.rcParams.update({
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.labelsize": 13, "axes.titlesize": 13,
    })

    fig = plt.figure(figsize=(22.0, 10.5), facecolor="white")
    gs = fig.add_gridspec(n_f, n_d + 1,
                          width_ratios=[1.0] * n_d + [0.30],
                          left=0.04, right=0.99, top=0.93, bottom=0.07,
                          wspace=0.18, hspace=0.28)

    def radial_profile(psf: np.ndarray, half_pix: int, pix_um: float):
        """Azimuthal average → (r_um array, intensity array)."""
        H, W = psf.shape
        cy, cx = H // 2, W // 2
        h = min(half_pix, cy, cx, H - cy - 1, W - cx - 1)
        crop = psf[cy - h:cy + h + 1, cx - h:cx + h + 1].astype(np.float64)
        H2, W2 = crop.shape
        y, x = np.mgrid[:H2, :W2]
        cy2, cx2 = H2 // 2, W2 // 2
        r_pix = np.sqrt((y - cy2) ** 2 + (x - cx2) ** 2)
        r_int = np.clip(r_pix.astype(int), 0, h)
        sum_i = np.bincount(r_int.ravel(), weights=crop.ravel(),
                             minlength=h + 1)
        cnt   = np.bincount(r_int.ravel(), minlength=h + 1)
        mean = sum_i / np.maximum(cnt, 1)
        r_um = np.arange(h + 1) * pix_um
        return r_um, mean[:h + 1]

    for fi in range(n_f):
        half_um = half_um_per_row[fi]
        h_pix = int(half_um / pix_um)
        for di in range(n_d):
            ax = fig.add_subplot(gs[fi, di])
            for wi in range(len(WVLNS_UM)):
                for dpi in range(len(DEPTHS_MM)):
                    psf = psfs[(di, fi, wi, dpi)]
                    r_um, prof = radial_profile(psf, h_pix, pix_um)
                    prof = prof / max(prof.max(), 1e-12)
                    # Mirror: render r → −half..+half via reflection so the
                    # plot reads like the reference (peak at 0, symmetric).
                    x_full = np.concatenate([-r_um[::-1][:-1], r_um])
                    y_full = np.concatenate([prof[::-1][:-1], prof])
                    ax.plot(x_full, y_full,
                            color=WAVE_COLORS[wi],
                            linestyle=DEPTH_STYLES[dpi],
                            lw=1.5, alpha=DEPTH_ALPHAS[dpi])

            ax.set_xlim(-half_um, half_um)
            ax.set_ylim(0, 1.05)
            tick = int(np.round(half_um * 0.66))
            ax.set_xticks([-tick, 0, tick])
            ax.set_xticklabels([rf"${-tick}$", r"$0$", rf"${tick}$"], fontsize=11)
            ax.set_yticks([])
            ax.grid(True, axis="x", lw=0.5, color="#dddddd", alpha=0.6)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_linewidth(0.9)
            ax.tick_params(width=0.7, length=3.0, color="#666", pad=2)

            if fi == n_f - 1:
                ax.set_xlabel(r"$x\ (\mu\mathrm{m})$", fontsize=12, labelpad=4)
            if fi == 0:
                rms = picks[di][1]; fnum = picks[di][2]
                ax.set_title(rf"$\mathrm{{sixlet\#{di+1}}}$"
                             "\n"
                             rf"$f/{fnum:.2f},\ \mathrm{{RMS}}={rms:.2f}\,\mu\mathrm{{m}}$",
                             fontsize=13, pad=6)
            if di == 0:
                ax.set_ylabel(rf"${FIELD_LABEL_TEX[fi]}$",
                              fontsize=14, labelpad=10, fontweight="bold")

    # ---- legend strip on the right ----
    leg_ax = fig.add_subplot(gs[:, -1])
    leg_ax.axis("off")
    handles = []
    for wi, (col, lbl) in enumerate(zip(WAVE_COLORS, WAVE_LABELS)):
        for dpi, (st, dlbl) in enumerate(zip(DEPTH_STYLES, DEPTH_LABELS)):
            handles.append(Line2D(
                [0], [0], color=col, linestyle=st, lw=1.8,
                label=rf"$\mathbf{{{lbl}}}\ \ {dlbl}$"))
    leg_ax.legend(handles=handles, loc="center", frameon=False,
                  fontsize=11, handlelength=2.5, handletextpad=0.6,
                  labelspacing=0.40)

    out_path = OUT / "psf_lines_top5_sixlet_fnum5.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
