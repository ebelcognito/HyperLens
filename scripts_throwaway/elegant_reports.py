#!/usr/bin/env python
"""End-to-end Mathematica-style Zemax-like report per design.

Per design:
  design_elegant_psf.png    — RGB PSF field map (5x5), per-cell µm axes
  design_elegant_spot.png   — spot diagram grid (3 fields x 3 wavelengths)
  design_elegant_mtf.png    — MTF (T+S, 3 λ) per field (3 fields)
  manifest_elegant.json     — added metadata about overrides

Reads design provenance from each existing manifest.json under
figures/zemax_reports_top3_poly/.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
# Sidestep a broken sympy/mpmath/gmpy interaction in this env that flares
# when matplotlib mathtext lazily pulls in torch._prims_common.
os.environ.setdefault("MPMATH_NOGMPY", "1")

import numpy as np
import torch
torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Circle

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DeepLens"))

from scripts_throwaway.fig5_extract import (
    build_lens_template_and_solver, make_geolens,
)
from scripts_throwaway.elegant_psf_map import (
    render_elegant_psf_map, mathematica_style, WAVE_RGB_UM, WAVE_RGB_NM,
)

WAVE_COLORS = ("#e63946", "#2a9d8f", "#1d4ed8")  # R, G, B
FIELDS_FRAC = (0.0, 0.5, 1.0)
N_SPOT_RAYS = 512
PSF_KS_MTF = 256
NYQ_OVERSAMPLE = 4  # PSF compute oversample for MTF (sharper line spectra)


# ---------------------------------------------------------------------------
# Spot diagrams
# ---------------------------------------------------------------------------

def compute_spot_endpoints(lens, depth_mm: float, fields_frac=FIELDS_FRAC,
                            num_rays: int = N_SPOT_RAYS):
    """Return (n_fields, n_wvln, num_rays, 2) sensor-plane (x,y) µm,
    chief-ray-centered per (field, wavelength).
    """
    from optical_rl.qd.lensmega.problem.tracing import (
        disk_samples, trace_fields_batch,
    )
    import math

    device = "cpu"
    dtype = torch.float64
    aperture = torch.from_numpy(disk_samples(num_rays)).to(dtype=dtype, device=device)
    with torch.no_grad():
        pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
    pupil_z = float(pupilz)
    pupil_r = float(pupilr)
    aperture_xy = aperture * pupil_r

    half_sensor_diag = float(lens.r_sensor)
    efl = float(getattr(lens, "foclen", float("nan")))
    hfov_deg = math.degrees(math.atan(half_sensor_diag / abs(efl)))
    fields = [(0.0, f * hfov_deg) for f in fields_frac]

    out = np.full((len(fields), len(WAVE_RGB_UM), num_rays, 2), np.nan,
                  dtype=np.float64)
    for wi, w in enumerate(WAVE_RGB_UM):
        ray = trace_fields_batch(
            lens, tuple(fields), aperture_xy=aperture_xy, pupil_z=pupil_z,
            object_depth_mm=depth_mm, wavelength_um=float(w),
            device=device, dtype=dtype,
        )
        # ray.o is sensor-plane (..., 3); take xy
        xy = ray.o[..., :2].detach().cpu().numpy()  # (n_fields, num_rays, 2)
        valid = (ray.is_valid > 0).detach().cpu().numpy()
        for fi in range(len(fields)):
            v = valid[fi]
            out[fi, wi, v] = xy[fi, v]
    return out * 1000.0, fields  # mm → µm


def render_elegant_spot(manifest_path: Path, out_path: Path | None = None,
                         depth_mm: float = -20000.0,
                         fields_frac=FIELDS_FRAC):
    m = json.loads(manifest_path.read_text())
    run_dir = ROOT / "results" / "lensmega" / m["source_run"]
    cfg_path = run_dir / "config.yaml"
    theta = np.asarray(m["theta"], dtype=np.float64)

    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(cfg_path)
    lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver,
                        device="cpu")

    xy_um, fields = compute_spot_endpoints(lens, depth_mm, fields_frac)
    # Center each (field, wvln) cloud on its mean (chief-ray approximation):
    xy_centered = xy_um.copy()
    rms_um = np.full((len(fields), len(WAVE_RGB_UM)), np.nan)
    for fi in range(len(fields)):
        for wi in range(len(WAVE_RGB_UM)):
            pts = xy_um[fi, wi]
            good = ~np.isnan(pts[:, 0])
            if not good.any():
                continue
            cen = pts[good].mean(axis=0)
            xy_centered[fi, wi] = pts - cen
            r2 = (xy_centered[fi, wi, good, 0]**2 + xy_centered[fi, wi, good, 1]**2)
            rms_um[fi, wi] = float(np.sqrt(r2.mean()))

    # Shared box: 95th-percentile radius across all (field, wvln), rounded up.
    all_pts = xy_centered.reshape(-1, 2)
    all_pts = all_pts[~np.isnan(all_pts[:, 0])]
    radial = np.sqrt(all_pts[:, 0]**2 + all_pts[:, 1]**2)
    box_um = max(2.0, float(np.percentile(radial, 99)) * 1.15)
    box_um = float(np.ceil(box_um / 2.0) * 2.0)  # round to even µm

    mathematica_style()
    n_f, n_w = len(fields), len(WAVE_RGB_UM)
    fig, axs = plt.subplots(1, n_f, figsize=(2.8 * n_f + 0.6, 4.0),
                            facecolor="white", squeeze=False)
    axs = axs[0]

    for fi in range(n_f):
        ax = axs[fi]
        for wi in range(n_w):
            pts = xy_centered[fi, wi]
            good = ~np.isnan(pts[:, 0])
            ax.scatter(pts[good, 0], pts[good, 1],
                       s=5.0, c=WAVE_COLORS[wi], alpha=0.55, edgecolors="none")
        ax.set_xlim(-box_um, box_um)
        ax.set_ylim(-box_um, box_um)
        ax.set_aspect("equal")
        ax.grid(True, color="#d8d8d8", lw=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(width=0.5, length=2.2, color="#888888", pad=1.5)
        ax.set_xlabel(r"$x\ (\mu\mathrm{m})$", labelpad=2)
        if fi == 0:
            ax.set_ylabel(r"$y\ (\mu\mathrm{m})$", labelpad=2)
        fov_deg = fields[fi][1]
        ax.set_title(rf"$\mathrm{{FoV}}={fov_deg:.1f}\degree$", fontsize=10.5)
        # Caption under each subplot: average RMS across the three wavelengths.
        avg_rms = float(np.nanmean(rms_um[fi]))
        ax.text(0.5, -0.30,
                rf"$\mathrm{{avg\ RMS}} = {avg_rms:.2f}\,\mu\mathrm{{m}}$",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=13, color="#222222")

    plt.tight_layout(rect=[0.0, 0.06, 1.0, 0.86])
    # Single header line on top: f-num and poly RMS, larger font.
    fnum = m.get("fnum_actual")
    parts = []
    if fnum is not None:
        parts.append(rf"$f/{fnum:.2f}$")
    parts.append(rf"$\mathrm{{poly\ RMS}} = {m['poly_rms_um']:.2f}\,\mu\mathrm{{m}}$")
    fig.text(0.5, 0.965, "   ".join(parts), ha="center",
             fontsize=15.5, color="#111111")
    # Small shared R/G/B legend strip below the header.
    leg_ax = fig.add_axes([0.30, 0.905, 0.40, 0.035])
    leg_ax.set_xlim(0, 3); leg_ax.set_ylim(0, 1); leg_ax.axis("off")
    for k, (c, nm) in enumerate(zip(WAVE_COLORS, WAVE_RGB_NM)):
        leg_ax.scatter([k + 0.18], [0.55], s=22, c=c, edgecolors="none")
        leg_ax.text(k + 0.30, 0.55,
                    rf"$\lambda = {nm}\,\mathrm{{nm}}$",
                    fontsize=9, va="center", color=c)

    if out_path is None:
        out_path = manifest_path.parent / "design_elegant_spot.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# MTF
# ---------------------------------------------------------------------------

def render_elegant_mtf(manifest_path: Path, out_path: Path | None = None,
                        depth_mm: float = -20000.0,
                        fields_frac=FIELDS_FRAC, oversample=NYQ_OVERSAMPLE):
    m = json.loads(manifest_path.read_text())
    run_dir = ROOT / "results" / "lensmega" / m["source_run"]
    cfg_path = run_dir / "config.yaml"
    theta = np.asarray(m["theta"], dtype=np.float64)

    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(cfg_path)
    lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver,
                        device="cpu")

    # Oversample sensor for sharper LSF → MTF spectra.
    orig_res = tuple(lens.sensor_res)
    if oversample != 1:
        lens.set_sensor_res(
            (int(orig_res[0] * oversample), int(orig_res[1] * oversample))
        )
    nyq = 0.5 / lens.pixel_size  # cycles/mm

    mathematica_style()
    n_f = len(fields_frac)
    fig, axs = plt.subplots(1, n_f, figsize=(3.2 * n_f, 3.4),
                            facecolor="white", squeeze=False)
    axs = axs[0]

    max_freq = 0.0
    for fi, fr in enumerate(fields_frac):
        ax = axs[fi]
        fov_rad = fr * lens.rfov
        fov_deg = np.degrees(fov_rad)
        for wi, w in enumerate(WAVE_RGB_UM):
            freq, mt, ms = lens.mtf(fov_rad, wvln=w)
            ax.plot(freq, mt, color=WAVE_COLORS[wi], lw=1.3,
                    label=rf"$\lambda={WAVE_RGB_NM[wi]}\,\mathrm{{nm}}\ \mathrm{{(T)}}$")
            ax.plot(freq, ms, color=WAVE_COLORS[wi], lw=1.3, ls="--",
                    label=rf"$\lambda={WAVE_RGB_NM[wi]}\,\mathrm{{nm}}\ \mathrm{{(S)}}$")
            if freq.size:
                max_freq = max(max_freq, float(freq.max()))
        ax.axvline(nyq, color="#444444", ls=":", lw=0.9,
                   label=rf"$\mathrm{{Nyquist}} = {nyq:.0f}\,\mathrm{{cyc/mm}}$")
        ax.set_xlim(0, min(max_freq, nyq * 1.05))
        ax.set_ylim(0, 1.02)
        ax.grid(True, color="#dddddd", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel(r"$\mathrm{spatial\ frequency}\ (\mathrm{cyc/mm})$",
                      labelpad=2)
        if fi == 0:
            ax.set_ylabel(r"$\mathrm{MTF}$", labelpad=2)
        ax.set_title(rf"$\mathrm{{FoV}}={fov_deg:.1f}\degree$", fontsize=10)
        ax.tick_params(width=0.5, length=2.5)
        if fi == n_f - 1:
            ax.legend(loc="upper right", fontsize=6.5, ncol=1, frameon=True,
                      framealpha=0.85, edgecolor="#bbbbbb")

    plt.tight_layout(rect=[0, 0, 1, 0.86])
    title = (rf"$\mathrm{{MTF}}\ \mid\ "
             rf"\mathrm{{{m['system']}\ rank\!-\!}}{m['rank']},\ "
             rf"\mathrm{{poly\ RMS}}\!=\!{m['poly_rms_um']:.2f}\,\mu\mathrm{{m}}$")
    fig.suptitle(title, fontsize=11, y=0.99)
    sub = (rf"$\mathrm{{tangential\ (solid)}}\ /\ \mathrm{{sagittal\ (dashed)}}\ "
           rf"\cdot\ \mathrm{{Nyquist\ at\ sensor\ pitch}}\ "
           rf"({lens.pixel_size * 1e3:.2f}\,\mu\mathrm{{m/px,\ oversampled\ {oversample}\!\times}})$")
    fig.text(0.5, 0.93, sub, ha="center", fontsize=8.5, color="#444444")

    if out_path is None:
        out_path = manifest_path.parent / "design_elegant_mtf.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one(manifest_path: Path):
    print(f"\n  design: {manifest_path.parent.name}", flush=True)
    t0 = time.time()
    psf_path, _ = render_elegant_psf_map(manifest_path)
    print(f"    psf:  {psf_path.name}  ({time.time()-t0:.1f}s)", flush=True)
    t1 = time.time()
    spot_path = render_elegant_spot(manifest_path)
    print(f"    spot: {spot_path.name}  ({time.time()-t1:.1f}s)", flush=True)
    t2 = time.time()
    mtf_path  = render_elegant_mtf(manifest_path)
    print(f"    mtf:  {mtf_path.name}  ({time.time()-t2:.1f}s)", flush=True)
    return psf_path, spot_path, mtf_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="figures/zemax_reports_top3_poly")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these manifest paths (relative to ROOT)")
    args = ap.parse_args()

    base = ROOT / args.root
    manifests = sorted(base.glob("*/rank*/manifest.json"))
    if args.only:
        manifests = [base / p for p in args.only]
    print(f"found {len(manifests)} manifests", flush=True)
    for mp in manifests:
        try:
            run_one(mp)
        except Exception as e:
            print(f"  ! {mp.parent.name}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
