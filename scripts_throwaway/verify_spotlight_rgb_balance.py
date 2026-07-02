#!/usr/bin/env python
"""Verify that each spotlight is RGB-balanced.

The reported `rms_um` in refined_top500_seed42_hl.json is the *mean* across
3 wavelengths (F=0.486, d=0.587, C=0.656 um). A "spotlight" with mean=15um
might still be 5um at one wavelength and 50um at another. This script
recomputes per-wavelength field-averaged RMS for each of the 8 spotlights
(4 sixlet + 4 octuplet) and reports them so we can decide whether to keep
the picks or re-pick.

Output:
  results/figure_archive_demo/spotlight_rgb_balance.json
  results/figure_archive_demo_octuplet/spotlight_rgb_balance.json
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

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)


def per_wvln_rms(lens, prob_cfg, p_problem, num_rays=256):
    """Recreate the runner's per-wavelength RMS path. Returns dict
    {wvln_um: field_avg_rms_um, ...} plus a list of per-field RMS for each."""
    from optical_rl.qd.lensmega.problem.tracing import (
        disk_samples, trace_fields_batch, rms_spot_batch,
    )

    device = "cpu"
    # Match the runner: dtype is the curvature dtype of the first free surface.
    trace_dtype = lens.surfaces[
        p_problem.free_curvature_indices[0]
    ].c.dtype

    aperture = torch.from_numpy(disk_samples(num_rays)).to(
        dtype=trace_dtype, device=device,
    )

    with torch.no_grad():
        pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
    pupil_z = float(pupilz)
    pupil_r = float(pupilr)
    aperture_xy = aperture * pupil_r

    template_sensor_half = (lens.sensor_size[0] / 2.0
                            if hasattr(lens, "sensor_size")
                            else float(lens.r_sensor))
    efl = float(getattr(lens, "foclen", float(p_problem.focal_length_target_mm)))
    hfov_deg = math.degrees(math.atan(template_sensor_half / abs(efl)))
    fields = ((0.0, 0.0), (0.0, 0.5 * hfov_deg), (0.0, hfov_deg))

    out = {}
    field_rms_per_wvln = {}
    inv_per_wvln = {}
    for w in p_problem.fitness_wavelengths_um:
        ray = trace_fields_batch(
            lens, fields, aperture_xy=aperture_xy, pupil_z=pupil_z,
            object_depth_mm=p_problem.object_depth_mm,
            wavelength_um=float(w), device=device, dtype=trace_dtype,
        )
        rms, inv_frac = rms_spot_batch(ray)
        # rms is in mm; the reported rms_um is mean(rms_per_wvln) * 1000.
        rms_mm = rms.detach().cpu().numpy()
        inv = inv_frac.detach().cpu().numpy()
        field_rms_per_wvln[float(w)] = (rms_mm * 1000.0).tolist()
        inv_per_wvln[float(w)] = inv.tolist()
        # Field-averaged like the runner.
        out[float(w)] = float(rms_mm.mean()) * 1000.0
    return out, field_rms_per_wvln, inv_per_wvln


def match_octuplet_to_theta(spotlights, hl_ref, fit_to_measures,
                            tol_S=1e-7, tol_rms=0.05):
    """Mirror match_spotlights_to_theta from the octuplet renderer."""
    matched = []
    for sp in spotlights:
        best = None
        for d in hl_ref:
            f = round(d["raw_fitness"], 8)
            if f not in fit_to_measures:
                continue
            m = fit_to_measures[f]
            ds2 = abs(m[0] - sp["S2"])
            ds3 = abs(m[1] - sp["S3"])
            drms = abs(d["rms_um"] - sp["rms_um"])
            if ds2 < tol_S and ds3 < tol_S and drms < tol_rms:
                best = d; break
        if best is None:
            cands = []
            for d in hl_ref:
                f = round(d["raw_fitness"], 8)
                if f not in fit_to_measures: continue
                m = fit_to_measures[f]
                if abs(m[0]-sp["S2"]) < tol_S and abs(m[1]-sp["S3"]) < tol_S:
                    cands.append((abs(d["rms_um"]-sp["rms_um"]), d))
            if cands:
                cands.sort(key=lambda t: t[0])
                best = cands[0][1]
        if best is None:
            print(f"  ! no theta match for spotlight {sp['label']}, skipping")
            continue
        matched.append((sp, best))
    return matched


def verify_topology(topology_name, out_dir, theta_supplier):
    """theta_supplier(sp) -> theta np.ndarray. For sixlet, theta is in the
    spotlights file directly; for octuplet, we look it up by raw_fitness."""
    print(f"\n{'='*70}\n{topology_name.upper()}\n{'='*70}")

    spots = json.loads((out_dir / "spotlights.json").read_text())
    template, solver, prob_cfg, lens_path, p_problem = (
        build_lens_template_and_solver(SYSTEMS[topology_name]["cfg"])
    )

    print(f"fitness_wavelengths_um = {tuple(p_problem.fitness_wavelengths_um)}")

    rows = []
    for sp in spots:
        theta = theta_supplier(sp)
        if theta is None:
            print(f"\n[{sp['label']}] SKIPPED (no theta)")
            continue

        lens = make_geolens(prob_cfg, p_problem, lens_path,
                            np.asarray(theta, dtype=np.float64),
                            template, solver)

        wvln_rms, field_rms, inv_frac = per_wvln_rms(
            lens, prob_cfg, p_problem, num_rays=256,
        )
        wvln_keys = sorted(wvln_rms.keys())
        vals = np.array([wvln_rms[k] for k in wvln_keys])
        mean_rms = float(vals.mean())
        max_rms = float(vals.max())
        min_rms = float(vals.min())
        balance_ratio = max_rms / max(min_rms, 1e-9)

        # 'reported' = the stored rms_um. For sixlet it's in spotlights.json;
        # for octuplet it's also in spotlights.json (rms_um at hull cell).
        reported_rms = float(sp.get("rms_um", float("nan")))

        print(f"\n[{sp['label']}]   reported_mean_rms={reported_rms:6.2f} um   "
              f"in_saga_hull={sp.get('in_saga_hull')}")
        for w in wvln_keys:
            print(f"    {w*1000:5.0f} nm   field-avg RMS = "
                  f"{wvln_rms[w]:7.2f} um   "
                  f"per-field=[{', '.join(f'{x:6.2f}' for x in field_rms[w])}]   "
                  f"invalid=[{', '.join(f'{x:.2f}' for x in inv_frac[w])}]")
        print(f"    -> recomputed mean = {mean_rms:6.2f} um   "
              f"max/min = {max_rms:.2f}/{min_rms:.2f} = {balance_ratio:.2f}x   "
              f"{'OK' if balance_ratio < 2.0 and max_rms < 100 else 'WARN'}")

        rows.append({
            "label": sp["label"],
            "S2": sp.get("S2"),
            "S3": sp.get("S3"),
            "in_saga_hull": sp.get("in_saga_hull"),
            "reported_mean_rms_um": reported_rms,
            "per_wvln_rms_um": {f"{int(round(w*1000))}nm": wvln_rms[w]
                                for w in wvln_keys},
            "per_wvln_per_field_rms_um": {
                f"{int(round(w*1000))}nm": field_rms[w] for w in wvln_keys
            },
            "per_wvln_invalid_fraction": {
                f"{int(round(w*1000))}nm": inv_frac[w] for w in wvln_keys
            },
            "recomputed_mean_rms_um": mean_rms,
            "max_per_wvln_rms_um": max_rms,
            "min_per_wvln_rms_um": min_rms,
            "balance_ratio_max_over_min": balance_ratio,
        })

    out_json = out_dir / "spotlight_rgb_balance.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_json}")
    return rows


def main():
    # ---- Sixlet ----
    sixlet_out = ROOT / "results" / "figure_archive_demo"
    def sixlet_theta(sp):
        return np.asarray(sp["refined_theta"], dtype=np.float64)
    verify_topology("sixlet", sixlet_out, sixlet_theta)

    # ---- Octuplet — must look up theta via raw_fitness/measure match ----
    oct_out = ROOT / "results" / "figure_archive_demo_octuplet"
    HL = ROOT / "results" / "lensmega" / "octuplet_apsc_2cement_v2_seed42_20260506_040025"
    spots = json.loads((oct_out / "spotlights.json").read_text())
    hl_ref = json.loads((HL / "refined_top500_seed42_hl.json").read_text())["designs"]
    cells = json.loads((HL / "archive_cells.json").read_text())["cells"]
    fit_to_measures = {round(c["fitness"], 8): c["measures"] for c in cells}
    matched = match_octuplet_to_theta(spots, hl_ref, fit_to_measures)
    print(f"\nOctuplet: matched {len(matched)}/{len(spots)} spotlights to theta")
    label_to_theta = {sp["label"]: np.asarray(d["theta"], dtype=np.float64)
                      for sp, d in matched}

    def oct_theta(sp):
        return label_to_theta.get(sp["label"])
    verify_topology("octuplet", oct_out, oct_theta)


if __name__ == "__main__":
    main()
