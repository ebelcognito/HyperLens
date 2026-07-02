#!/usr/bin/env python
"""Score all sub-20µm octuplet refined designs across the 13 REFSWEEP runs
by monochromatic field-stability: spread of RMS spot across 7 fields @ 587 nm.

Outputs ranked JSON + a printed leaderboard.
"""
from __future__ import annotations
import json, os, sys, math, glob
from pathlib import Path

for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k,"1")
import numpy as np, torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)
from optical_rl.qd.lensmega.problem.tracing import (
    disk_samples, trace_fields_batch, rms_spot_batch,
)

WVLN_UM   = 0.587
N_FIELDS  = 7
N_RAYS    = 256
RMS_LIMIT = 20.0  # µm
OUT       = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/monochrome_stability_under20.json"


def gather():
    files = sorted(glob.glob(str(
        ROOT / "results/lensmega/octuplet_REFSWEEP_*/refined_top*.json")))
    pool = []
    for fp in files:
        run = Path(fp).parent.name.replace("octuplet_REFSWEEP_", "").split("_seed")[0]
        data = json.load(open(fp))
        for d in data["designs"]:
            r = float(d["rms_um"])
            if r < RMS_LIMIT:
                pool.append({
                    "run": run,
                    "rms_um_stated": r,
                    "theta": d["theta"],
                    "pool_idx": d.get("pool_idx", None),
                })
    return pool


def trace_one(lens, p_problem):
    device = "cpu"
    trace_dtype = lens.surfaces[p_problem.free_curvature_indices[0]].c.dtype
    aperture = torch.from_numpy(disk_samples(N_RAYS)).to(dtype=trace_dtype, device=device)
    with torch.no_grad():
        pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
    aperture_xy = aperture * float(pupilr); pupil_z = float(pupilz)
    sensor_half = (lens.sensor_size[0]/2.0
                   if hasattr(lens,"sensor_size") else float(lens.r_sensor))
    efl = float(getattr(lens,"foclen", float(p_problem.focal_length_target_mm)))
    hfov_deg = math.degrees(math.atan(sensor_half / abs(efl)))
    ff = np.linspace(0.0, 1.0, N_FIELDS)
    fields = [(0.0, float(f * hfov_deg)) for f in ff]
    ray = trace_fields_batch(
        lens, fields, aperture_xy=aperture_xy, pupil_z=pupil_z,
        object_depth_mm=p_problem.object_depth_mm,
        wavelength_um=float(WVLN_UM), device=device, dtype=trace_dtype,
    )
    rms, _ = rms_spot_batch(ray)
    return ff, rms.detach().cpu().numpy() * 1000.0


def main():
    pool = gather()
    print(f"gathered {len(pool)} sub-{RMS_LIMIT}µm designs across all REFSWEEP runs")
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["octuplet"]["cfg"])
    rows = []
    for i, sp in enumerate(pool):
        theta = np.asarray(sp["theta"], dtype=np.float64)
        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
            ff, rms = trace_one(lens, p_problem)
        except Exception as e:
            print(f"  [{i:3d}] FAILED ({sp['run']}): {e}")
            continue
        spread = float(rms.max() - rms.min())
        rows.append({
            "idx": i,
            "run": sp["run"],
            "pool_idx": sp["pool_idx"],
            "rms_um_stated": sp["rms_um_stated"],
            "rms_avg_7f_587": float(rms.mean()),
            "rms_max_7f_587": float(rms.max()),
            "rms_min_7f_587": float(rms.min()),
            "field_spread_um": spread,
            "field_std_um":    float(rms.std()),
            "rms_curve":       rms.tolist(),
            "theta":           sp["theta"],
        })
        if (i+1) % 25 == 0 or i == len(pool)-1:
            print(f"  [{i+1:4d}/{len(pool)}]  last: spread={spread:6.2f} max={rms.max():6.2f}µm")

    rows.sort(key=lambda r: r["field_spread_um"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=2)
    print(f"\nwrote {len(rows)} -> {OUT}")
    print("\nTOP 25 by smallest field spread (587 nm, 7 fields):")
    print(f"  {'rk':>2}  {'spread':>7} {'max':>7} {'avg':>7} {'min':>7} {'stated':>7}  run")
    for k, r in enumerate(rows[:25]):
        print(f"  {k+1:>2d}  {r['field_spread_um']:7.2f} {r['rms_max_7f_587']:7.2f} "
              f"{r['rms_avg_7f_587']:7.2f} {r['rms_min_7f_587']:7.2f} "
              f"{r['rms_um_stated']:7.2f}  {r['run']}")


if __name__ == "__main__":
    main()
