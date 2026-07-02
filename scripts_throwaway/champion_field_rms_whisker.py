#!/usr/bin/env python
"""Per-field RMS distribution across sixlet + octuplet champion archives.
Re-traces every design at 3 fields × 3 wavelengths, averages over wavelengths
for a per-field polychromatic RMS, then draws a Mathematica-style
box-and-whisker (3 fields × 2 topologies).

Step 1: cache per-field RMS to JSON (no recompute on plot iteration).
Step 2: plot from the cache.
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
from matplotlib.patches import Patch

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DeepLens"))

from scripts_throwaway.fig5_extract import build_lens_template_and_solver, make_geolens
from scripts_throwaway.elegant_psf_map import mathematica_style

CHAMPS = {
    "sixlet":   ROOT / "results/lensmega/sixlet_REFSWEEP2_CHAMPION_seed42_20260509_080537/refined_top500_seed42_CHAMPION.json",
    "octuplet": ROOT / "results/lensmega/octuplet_REFSWEEP_CHAMPION_seed42_20260509_145427/refined_top500_seed42_CHAMPION.json",
}
WVLNS_UM = (0.486, 0.587, 0.656)
FIELDS_FRAC = (0.0, 0.5, 1.0)
N_RAYS = 128
DEPTH_MM = -20000.0
CACHE = ROOT / "scripts_throwaway/champion_field_rms_cache.json"
OUT_PNG = ROOT / "figures/champion_field_rms_whisker.png"


def per_field_rms_um(lens, depth_mm: float):
    from optical_rl.qd.lensmega.problem.tracing import disk_samples, trace_fields_batch
    import math
    device, dtype = "cpu", torch.float64
    aperture = torch.from_numpy(disk_samples(N_RAYS)).to(dtype=dtype, device=device)
    with torch.no_grad():
        pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
    pupil_z = float(pupilz); pupil_r = float(pupilr)
    aperture_xy = aperture * pupil_r
    half_sensor_diag = float(lens.r_sensor)
    efl = float(getattr(lens, "foclen", float("nan")))
    hfov_deg = math.degrees(math.atan(half_sensor_diag / abs(efl)))
    fields = tuple((0.0, f * hfov_deg) for f in FIELDS_FRAC)
    rms_per_field = np.zeros(len(fields), dtype=np.float64)
    for fi in range(len(fields)):
        rms_w = []
        for w in WVLNS_UM:
            ray = trace_fields_batch(
                lens, (fields[fi],), aperture_xy=aperture_xy, pupil_z=pupil_z,
                object_depth_mm=depth_mm, wavelength_um=float(w),
                device=device, dtype=dtype,
            )
            xy = ray.o[..., :2].detach().cpu().numpy().reshape(-1, 2) * 1000.0
            valid = (ray.is_valid > 0).detach().cpu().numpy().reshape(-1)
            if valid.sum() == 0:
                rms_w.append(np.nan); continue
            pts = xy[valid]
            cen = pts.mean(axis=0)
            r2 = ((pts - cen) ** 2).sum(axis=1)
            rms_w.append(float(np.sqrt(r2.mean())))
        rms_per_field[fi] = float(np.nanmean(rms_w))
    return rms_per_field


def build_cache():
    cache = {}
    for sys_name, json_path in CHAMPS.items():
        run_dir = json_path.parent
        cfg_path = run_dir / "config.yaml"
        template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(cfg_path)
        designs = json.loads(json_path.read_text())["designs"]
        print(f"\n=== {sys_name}: {len(designs)} designs ===", flush=True)
        records = []
        t0 = time.time()
        n_bad = 0
        for k, des in enumerate(designs):
            theta = np.asarray(des["theta"], dtype=np.float64)
            try:
                lens = make_geolens(prob_cfg, p_problem, lens_path, theta,
                                    template, solver, device="cpu")
                rms = per_field_rms_um(lens, DEPTH_MM)
            except Exception:
                n_bad += 1; continue
            if not np.all(np.isfinite(rms)):
                n_bad += 1; continue
            records.append({
                "idx": int(des.get("idx", k)),
                "rms_um_field0": float(rms[0]),
                "rms_um_field05": float(rms[1]),
                "rms_um_field1": float(rms[2]),
                "rms_um_avg": float(rms.mean()),
                "rms_um_stored": float(des.get("rms_um", np.nan)),
            })
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(designs)} kept={len(records)} bad={n_bad} "
                      f"({time.time()-t0:.1f}s)", flush=True)
        cache[sys_name] = records
        print(f"  done: kept={len(records)} bad={n_bad} in {time.time()-t0:.1f}s",
              flush=True)
    CACHE.write_text(json.dumps(cache, indent=2))
    print(f"\nwrote {CACHE}")
    return cache


def plot_whisker(cache: dict):
    sixlet  = cache["sixlet"]
    octuplet = cache["octuplet"]
    data = {
        "sixlet":   [[d["rms_um_field0"]  for d in sixlet],
                     [d["rms_um_field05"] for d in sixlet],
                     [d["rms_um_field1"]  for d in sixlet]],
        "octuplet": [[d["rms_um_field0"]  for d in octuplet],
                     [d["rms_um_field05"] for d in octuplet],
                     [d["rms_um_field1"]  for d in octuplet]],
    }
    mathematica_style()
    plt.rcParams.update({
        "xtick.labelsize": 13,
        "ytick.labelsize": 12,
        "axes.labelsize": 14,
    })
    fig, ax = plt.subplots(figsize=(8.4, 5.2), facecolor="white")
    pos_centres = np.arange(3)              # one per field
    width = 0.34
    offsets = (-0.5 * width - 0.02, 0.5 * width + 0.02)
    colors = ("#3b5b8a", "#c97a3d")        # cool slate vs warm amber
    edge   = ("#1f3050", "#7a4b25")

    boxprops = dict(linewidth=0.8)
    medianprops = dict(color="#ffffff", linewidth=1.6)
    whiskerprops = dict(linewidth=0.8, color="#444444")
    capprops = dict(linewidth=0.8, color="#444444")
    flierprops = dict(marker="o", markersize=2.0, markerfacecolor="#888888",
                      markeredgecolor="none", alpha=0.55)

    handles = []
    for k, (sys_name, fill, edg) in enumerate(zip(("sixlet", "octuplet"),
                                                   colors, edge)):
        positions = pos_centres + offsets[k]
        bp = ax.boxplot(
            data[sys_name], positions=positions, widths=width,
            patch_artist=True, manage_ticks=False, whis=(5, 95),
            boxprops=boxprops, medianprops=medianprops,
            whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(fill); patch.set_edgecolor(edg); patch.set_alpha(0.85)
        # Mean marker (diamond, white-fill, colored edge) for each field.
        for i, vals in enumerate(data[sys_name]):
            ax.scatter(positions[i], float(np.mean(vals)),
                       marker="D", s=22, facecolor="white",
                       edgecolor=edg, linewidths=1.0, zorder=5)
        handles.append(Patch(facecolor=fill, edgecolor=edg, alpha=0.85,
                             label=rf"$\mathrm{{{sys_name}}}\ (n={len(data[sys_name][0])})$"))

    ax.set_yscale("log")
    ax.set_xticks(pos_centres)
    ax.set_xticklabels([r"$\mathrm{on\!-\!axis}$",
                        r"$\mathrm{half\!-\!field}$",
                        r"$\mathrm{full\!-\!field}$"])
    ax.set_ylabel(r"$\mathrm{spot\ RMS}\ (\mu\mathrm{m})$", labelpad=6)
    ax.grid(True, axis="y", which="both", color="#dcdcdc", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.8, length=4.5)
    for spine in ax.spines.values():
        spine.set_color("#444444"); spine.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=handles, loc="upper left", frameon=True,
              framealpha=0.92, edgecolor="#999999", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


def main():
    if CACHE.exists():
        print(f"using cache {CACHE}")
        cache = json.loads(CACHE.read_text())
    else:
        cache = build_cache()
    plot_whisker(cache)


if __name__ == "__main__":
    main()
