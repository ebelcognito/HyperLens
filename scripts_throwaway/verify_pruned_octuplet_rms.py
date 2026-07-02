#!/usr/bin/env python
"""Verify the reported RMS values on the saved pruned-lens JSONs.

For each of the 7 octuplet designs we (a) reload the pruned lens from disk,
(b) trace 3 fields x 3 wavelengths x 256 rays through it, and (c) compute
the same per-field RMS the runner uses. Compare to the reported value.

Mismatch => pruning or the JSON write/read corrupted the optical behavior.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
SUMMARY = ROOT / "figures" / "method_library_data" / "pruned_summary.json"

from deeplens.optics import GeoLens
from optical_rl.qd.lensmega.problem.tracing import disk_samples, trace_fields_batch

WVLNS = (0.486, 0.587, 0.656)
NUM_RAYS = 256
OBJ_DEPTH_MM = -20000.0


def hfov_deg(lens):
    if hasattr(lens, "rfov") and lens.rfov is not None:
        return float(np.rad2deg(float(lens.rfov)))
    half_sensor = float(lens.sensor_size[0]) / 2.0 if hasattr(lens, "sensor_size") else float(lens.r_sensor)
    efl = float(getattr(lens, "foclen", 50.0))
    return math.degrees(math.atan(half_sensor / abs(efl)))


def rms_of_pruned(lens_path):
    lens = GeoLens(filename=str(lens_path), device="cpu")
    H = hfov_deg(lens)
    fields = ((0.0, 0.0), (0.0, 0.5 * H), (0.0, H))

    with torch.no_grad():
        pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
    pupil_z = float(pupilz)
    pupil_r = float(pupilr)
    aperture_xy = torch.from_numpy(disk_samples(NUM_RAYS)).to(dtype=torch.float64) * pupil_r

    # 3 fields x 3 wvln, average per-field RMS like the runner does
    per_field_rms_um = []
    invalid_frac = []
    for w in WVLNS:
        ray = trace_fields_batch(
            lens, fields, aperture_xy=aperture_xy, pupil_z=pupil_z,
            object_depth_mm=OBJ_DEPTH_MM, wavelength_um=float(w),
            device="cpu", dtype=torch.float64,
        )
        valid = ray.is_valid  # (F, N)
        xy = ray.o[..., :2]   # (F, N, 2)
        denom = valid.sum(dim=-1).clamp_min(1e-8)
        centroid = (xy * valid.unsqueeze(-1)).sum(dim=-2) / denom.unsqueeze(-1)
        err2 = ((xy - centroid.unsqueeze(-2)) ** 2).sum(dim=-1)
        rms_per_field_mm = ((err2 * valid).sum(dim=-1) / denom + 1e-12).sqrt()
        per_field_rms_um.append(rms_per_field_mm.cpu().numpy() * 1000.0)
        invalid_frac.append(1.0 - valid.float().mean(dim=-1).cpu().numpy())

    pf = np.array(per_field_rms_um)  # (W, F) in um
    inv = np.array(invalid_frac)
    # The fitness uses the mean over wavelengths AND fields (no field weights)
    rms_mean_um = float(pf.mean())
    return {
        "rms_mean_um": rms_mean_um,
        "per_field_per_wvln_um": pf.tolist(),
        "invalid_fraction_per_field_per_wvln": inv.tolist(),
        "hfov_deg": H,
        "efl_mm": float(getattr(lens, "foclen", float("nan"))),
        "d_sensor_mm": float(lens.d_sensor.item()) if torch.is_tensor(lens.d_sensor) else float(lens.d_sensor),
        "r_sensor_mm": float(lens.r_sensor),
    }


def main():
    summary = json.loads(SUMMARY.read_text())
    print(f"{'role':12s} {'rank':4s} {'reported RMS':15s} {'pruned-JSON RMS':18s} "
          f"{'EFL':6s} {'d_sensor':10s} {'inv_frac max':14s}")
    print("-" * 95)
    rows = []
    for d in summary["designs"]:
        role = d["role"]
        rank = d["rank"]
        rep = d["refined_RMS_um"]
        try:
            res = rms_of_pruned(d["lens_json"])
        except Exception as e:
            print(f"{role:12s} {rank:4d} {rep:13.2f} um  FAILED: {e}")
            continue
        meas = res["rms_mean_um"]
        delta = meas - rep
        max_inv = float(np.array(res["invalid_fraction_per_field_per_wvln"]).max())
        flag = "" if abs(delta) < 5.0 else "  <-- MISMATCH"
        print(f"{role:12s} {rank:4d} {rep:13.2f} um  {meas:16.2f} um   "
              f"{res['efl_mm']:5.1f}  {res['d_sensor_mm']:9.3f}  "
              f"{max_inv:13.3f}{flag}")
        rows.append({
            "role": role, "rank": rank, "reported_RMS_um": rep,
            "measured_pruned_RMS_um": meas, "delta_um": delta,
            "max_invalid_fraction": max_inv,
            "efl_mm": res["efl_mm"], "d_sensor_mm": res["d_sensor_mm"],
            "per_field_per_wvln_um": res["per_field_per_wvln_um"],
        })
    OUT = ROOT / "figures" / "method_library_data" / "verify_pruned_rms.json"
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
