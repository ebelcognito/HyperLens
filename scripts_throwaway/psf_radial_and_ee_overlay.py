#!/usr/bin/env python
"""Per-topology PSF radial-profile + encircled-energy overlay across the
top-4 ε-distinct designs at three field heights.

Hypothesis being tested: ε-distinct designs that look similar by simple
core metrics (peak, FWHM) on-axis actually diverge in their off-axis
PSF tails because they live at different (S₂, S₃) points on the
aberration manifold.

Layout per topology, single PNG:
  rows = (radial PSF profile log-y, encircled energy)
  cols = (on-axis, half-FoV, corner)
  4 designs overlaid in each cell; per-design color also annotated
  with that design's (S₁, S₂, S₃) Seidel values on the right.
"""
from __future__ import annotations
import json, os, sys, time, glob
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

WVLN_UM = 0.587                   # monochromatic d-line: isolates geometric story
PSF_KS = 64
OVERSAMPLE = 8                    # finer sampling for radial profiles
FIELD_FRACS = (0.0, 0.5, 1.0)
FIELD_LABELS = (r"$\mathrm{on\!-\!axis}$",
                r"$\mathrm{half\!-\!field}$",
                r"$\mathrm{corner}$")
DESIGN_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#a855f7")  # blue/red/green/purple
EE_TARGETS = (0.5, 0.8)
OUT = ROOT / "figures" / "psf_radial_ee_overlay"
OUT.mkdir(parents=True, exist_ok=True)

CARD_DIRS = {
    "sixlet":   ROOT / "figures/zemax_reports_top4_diverse_fnum5/sixlet",
    "octuplet": ROOT / "figures/zemax_reports_top4_diverse_fnum5/octuplet",
}


def load_top4(sys_name: str):
    base = CARD_DIRS[sys_name]
    out = []
    for rank in (1, 2, 3, 4):
        cand = sorted(base.glob(f"rank{rank}_*/manifest.json"))
        if not cand:
            continue
        m = json.loads(cand[0].read_text())
        # Augment with Seidel from source.
        src = ROOT / "results/lensmega" / m["source_run"] / m["source_json"]
        seidel = {}
        try:
            data = json.loads(src.read_text())
            des = data["designs"][m["source_idx"]]
            metrics = des.get("metrics") or {}
            for k in ("seidel_S1", "seidel_S2", "seidel_S3",
                     "seidel_S4", "seidel_S5"):
                if k in metrics:
                    seidel[k] = float(metrics[k])
        except Exception as e:
            print(f"  ! could not load Seidel for {sys_name} rank {rank}: {e}",
                  flush=True)
        m["seidel"] = seidel
        out.append(m)
    return out


def build_lens(manifest: dict):
    run_dir = ROOT / "results" / "lensmega" / manifest["source_run"]
    cfg_path = run_dir / "config.yaml"
    theta = np.asarray(manifest["theta"], dtype=np.float64)
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(cfg_path)
    lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver,
                        device="cpu")
    if OVERSAMPLE != 1:
        orig = tuple(lens.sensor_res)
        lens.set_sensor_res((int(orig[0] * OVERSAMPLE),
                              int(orig[1] * OVERSAMPLE)))
    return lens


def compute_psf_at(lens, field_frac: float, wvln_um: float, ks: int) -> np.ndarray:
    """field_frac in [0,1] of sensor half-diagonal (matches DeepLens chief-ray
    convention used in psf_map_rgb)."""
    # DeepLens uses point=[xf, yf, depth] with xf,yf in [-1,1].
    # Use a y-only positive field point so the radial direction is consistent.
    point = torch.tensor([0.0, field_frac, -20000.0], dtype=torch.float32)
    with torch.no_grad():
        psf = lens.psf(points=point, wvln=float(wvln_um), ks=ks, recenter=True)
    return psf.detach().cpu().numpy()


def radial_profile(psf: np.ndarray, pix_um: float):
    """Azimuthal-average radial intensity profile vs radius (µm)."""
    H, W = psf.shape
    y, x = np.mgrid[:H, :W]
    cy, cx = H // 2, W // 2
    r_pix = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = int(np.floor(min(cy, cx)))
    r_bins = np.arange(0, r_max + 1)
    bin_idx = np.clip(r_pix.astype(int), 0, r_max)
    sum_intensity = np.bincount(bin_idx.ravel(), weights=psf.ravel(),
                                 minlength=r_max + 1)
    count = np.bincount(bin_idx.ravel(), minlength=r_max + 1)
    mean = sum_intensity / np.maximum(count, 1)
    return r_bins * pix_um, mean[:r_max + 1]


def encircled_energy(psf: np.ndarray, pix_um: float):
    """Cumulative fractional energy within radius r (µm)."""
    H, W = psf.shape
    y, x = np.mgrid[:H, :W]
    cy, cx = H // 2, W // 2
    r_pix = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = int(np.floor(min(cy, cx)))
    r_bins = np.arange(0, r_max + 1)
    bin_idx = np.clip(r_pix.astype(int), 0, r_max)
    cum = np.bincount(bin_idx.ravel(), weights=psf.ravel(),
                       minlength=r_max + 1)
    cum = np.cumsum(cum[:r_max + 1])
    total = cum[-1] if cum[-1] > 0 else 1.0
    return r_bins * pix_um, cum / total


def fwhm_from_profile(r_um: np.ndarray, profile: np.ndarray) -> float:
    """Half-power diameter (so radius * 2)."""
    p = profile / max(profile.max(), 1e-12)
    above = np.where(p >= 0.5)[0]
    if len(above) == 0:
        return float("nan")
    # last index where it's at or above 0.5
    r_half = r_um[above[-1]]
    return float(2 * r_half)


def render(sys_name: str, designs: list[dict]):
    n_d = len(designs)
    print(f"\n=== {sys_name}: building {n_d} lenses ===", flush=True)
    lenses = []
    for k, m in enumerate(designs, 1):
        print(f"  [{k}/{n_d}] {m['source_run']}#{m['source_idx']}", flush=True)
        lenses.append(build_lens(m))
    # Each lens has its own pixel_size; we'll use the first lens's px for the
    # plot axis (they're all APS-C 50 mm → effectively identical).
    pix_um = float(lenses[0].pixel_size) * 1e3

    ks = PSF_KS * OVERSAMPLE
    psfs = {}  # (di, fi) -> {design_idx: psf}
    for fi, frac in enumerate(FIELD_FRACS):
        for di, lens in enumerate(lenses):
            t0 = time.time()
            p = compute_psf_at(lens, frac, WVLN_UM, ks)
            psfs[(di, fi)] = p
            print(f"    PSF d={di} f={frac:.2f}: max={p.max():.2e} "
                  f"sum={p.sum():.2e} ({time.time()-t0:.1f}s)", flush=True)

    # ---------- plot ----------
    mathematica_style()
    plt.rcParams.update({
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.labelsize": 13, "axes.titlesize": 14,
    })
    fig = plt.figure(figsize=(14.6, 7.2), facecolor="white")
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.50],
                          height_ratios=[1, 1],
                          left=0.06, right=0.99, top=0.94, bottom=0.10,
                          wspace=0.22, hspace=0.32)

    fwhm_table = []  # (design, field, fwhm_um)
    ee_table   = []  # (design, field, r50, r80) in µm

    for fi in range(3):
        ax_top = fig.add_subplot(gs[0, fi])
        ax_bot = fig.add_subplot(gs[1, fi])
        for di in range(n_d):
            psf = psfs[(di, fi)]
            r_um, prof = radial_profile(psf, pix_um)
            r_um_ee, ee = encircled_energy(psf, pix_um)
            prof_n = prof / max(prof.max(), 1e-12)
            color = DESIGN_COLORS[di]
            ax_top.plot(r_um, prof_n, color=color, lw=1.8,
                        label=rf"$\mathrm{{{sys_name}\#{di+1}}}$")
            ax_bot.plot(r_um_ee, ee, color=color, lw=1.8)
            fwhm_table.append((di, fi, fwhm_from_profile(r_um, prof_n)))
            # Find first r where ee crosses 50% and 80%.
            i50 = int(np.searchsorted(ee, 0.5))
            i80 = int(np.searchsorted(ee, 0.8))
            r50 = r_um_ee[min(i50, len(r_um_ee)-1)] if i50 < len(r_um_ee) else float("nan")
            r80 = r_um_ee[min(i80, len(r_um_ee)-1)] if i80 < len(r_um_ee) else float("nan")
            ee_table.append((di, fi, float(r50), float(r80)))

        # Decoration.
        ax_top.set_xlim(0, 60)
        ax_top.set_ylim(1e-4, 1.2)
        ax_top.set_yscale("log")
        ax_top.set_xlabel(r"$r\ (\mu\mathrm{m})$", labelpad=4)
        if fi == 0:
            ax_top.set_ylabel(r"$\mathrm{normalized\ PSF}\ I(r)$", labelpad=4)
            ax_top.legend(loc="upper right", fontsize=11.5, frameon=True,
                          framealpha=0.92, edgecolor="#bbb",
                          handlelength=2.0, labelspacing=0.3)
        ax_top.set_title(FIELD_LABELS[fi], pad=8)
        ax_top.grid(True, lw=0.45, color="#dddddd", which="both")
        ax_top.set_axisbelow(True)
        ax_top.tick_params(width=0.7, length=3.5)

        ax_bot.set_xlim(0, 60)
        ax_bot.set_ylim(0, 1.02)
        for ee_mark in EE_TARGETS:
            ax_bot.axhline(ee_mark, color="#888", ls=":", lw=0.7)
            ax_bot.text(0.6 * 60, ee_mark + 0.02, f"{int(ee_mark*100)}%",
                        fontsize=10, color="#555")
        ax_bot.set_xlabel(r"$r\ (\mu\mathrm{m})$", labelpad=4)
        if fi == 0:
            ax_bot.set_ylabel(r"$\mathrm{encircled\ energy}\ E(r)$", labelpad=4)
        ax_bot.grid(True, lw=0.45, color="#dddddd")
        ax_bot.set_axisbelow(True)
        ax_bot.tick_params(width=0.7, length=3.5)

    # Right strip: Seidel signature table.
    ax_leg = fig.add_subplot(gs[:, 3])
    ax_leg.axis("off")
    header_y = 0.97
    ax_leg.text(0.0, header_y,
                r"$\mathrm{Seidel\ signature}\ (\times 10^{-3})$",
                fontsize=12.5, transform=ax_leg.transAxes,
                fontweight="bold")
    row_h = 0.21
    for di, m in enumerate(designs):
        y = 0.86 - row_h * di
        col = DESIGN_COLORS[di]
        s = m.get("seidel", {})
        S1 = s.get("seidel_S1", float("nan"))
        S2 = s.get("seidel_S2", float("nan"))
        S3 = s.get("seidel_S3", float("nan"))
        ax_leg.text(0.0, y, rf"$\mathrm{{{sys_name}\#{di+1}}}$",
                    color=col, fontsize=13, transform=ax_leg.transAxes,
                    fontweight="bold")
        ax_leg.text(0.0, y - 0.05, rf"$S_1 = {S1*1e3:+6.3f}$",
                    color="#222", fontsize=11, transform=ax_leg.transAxes,
                    family="monospace")
        ax_leg.text(0.0, y - 0.09, rf"$S_2 = {S2*1e3:+6.3f}$",
                    color="#222", fontsize=11, transform=ax_leg.transAxes,
                    family="monospace")
        ax_leg.text(0.0, y - 0.13, rf"$S_3 = {S3*1e3:+6.3f}$",
                    color="#222", fontsize=11, transform=ax_leg.transAxes,
                    family="monospace")
    out_path = OUT / f"psf_radial_ee_{sys_name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)

    # Save tables for the report.
    return {
        "pixel_size_um": pix_um,
        "fields_frac": list(FIELD_FRACS),
        "designs": [
            {"rank": di + 1, "source_run": designs[di]["source_run"],
             "source_idx": designs[di]["source_idx"],
             "fnum_actual": designs[di].get("fnum_actual"),
             "seidel": designs[di].get("seidel", {})}
            for di in range(n_d)
        ],
        "fwhm_um_by_design_field": [
            {"design": d, "field": f, "fwhm_um": w}
            for (d, f, w) in fwhm_table
        ],
        "ee_radii_um_by_design_field": [
            {"design": d, "field": f, "r50_um": r5, "r80_um": r8}
            for (d, f, r5, r8) in ee_table
        ],
    }


def main():
    results = {}
    for sys_name in ("sixlet", "octuplet"):
        designs = load_top4(sys_name)
        if len(designs) != 4:
            print(f"  ! expected 4 designs for {sys_name}, got {len(designs)}",
                  flush=True)
        results[sys_name] = render(sys_name, designs)
    (OUT / "psf_radial_ee_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT / 'psf_radial_ee_results.json'}")


if __name__ == "__main__":
    main()
