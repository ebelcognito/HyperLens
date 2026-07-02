#!/usr/bin/env python
"""Re-pick sixlet spotlights using max-per-wavelength RMS as the eligibility
filter, then greedy farthest-point sampling in (S2, S3) space.

Original picker filtered by mean-across-wvln rms_um (≤30 um) and ranked by
greedy farthest-point sampling of (S2, S3) outside the SA+GA hull. That
admitted out2 — a design with 6 um green / 54 um blue.

New rule:
  1. Filter HL refined designs to those with max-per-wavelength RMS ≤ 30 um.
  2. Filter to those OUTSIDE the SA+GA hull.
  3. Greedy farthest-point in (S2, S3) → 3 HL-only spotlights.
  4. Best-RMS HL design INSIDE SA+GA hull → 1 shared spotlight.

Writes:
  results/figure_archive_demo/spotlights.json    (overwrites)
  results/figure_archive_demo/spotlight_picker_audit.json  (per-candidate diagnostics)
"""
from __future__ import annotations

import functools
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

# Make stdout line-buffered so progress prints flush immediately when
# redirected to a file (Python defaults to block-buffering when not a TTY).
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)


# ---- Geometry: point-in-polygon (simple ray casting) ----
def point_in_polygon(x, y, verts):
    """verts: (N, 2) array of polygon vertices (open polygon, no repeat)."""
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def per_wvln_field_avg_rms(lens, p_problem, num_rays=256):
    """Returns dict {wvln_um: field_avg_rms_um}."""
    from optical_rl.qd.lensmega.problem.tracing import (
        disk_samples, trace_fields_batch, rms_spot_batch,
    )
    device = "cpu"
    trace_dtype = lens.surfaces[
        p_problem.free_curvature_indices[0]
    ].c.dtype
    aperture = torch.from_numpy(disk_samples(num_rays)).to(
        dtype=trace_dtype, device=device,
    )
    with torch.no_grad():
        pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
    aperture_xy = aperture * float(pupilr)
    pupil_z = float(pupilz)

    template_sensor_half = (lens.sensor_size[0] / 2.0
                            if hasattr(lens, "sensor_size")
                            else float(lens.r_sensor))
    efl = float(getattr(lens, "foclen", float(p_problem.focal_length_target_mm)))
    hfov_deg = math.degrees(math.atan(template_sensor_half / abs(efl)))
    fields = ((0.0, 0.0), (0.0, 0.5 * hfov_deg), (0.0, hfov_deg))

    out = {}
    for w in p_problem.fitness_wavelengths_um:
        ray = trace_fields_batch(
            lens, fields, aperture_xy=aperture_xy, pupil_z=pupil_z,
            object_depth_mm=p_problem.object_depth_mm,
            wavelength_um=float(w), device=device, dtype=trace_dtype,
        )
        rms, _ = rms_spot_batch(ray)
        out[float(w)] = float(rms.detach().cpu().numpy().mean()) * 1000.0
    return out


# ---- Multiprocessing worker (must be module-level for spawn) ----
_WORKER_STATE = {}


def _worker_init(topology):
    """Each worker builds its own template + solver + prob_cfg once."""
    sys.stdout.reconfigure(line_buffering=True)
    template, solver, prob_cfg, lens_path, p_problem = (
        build_lens_template_and_solver(SYSTEMS[topology]["cfg"])
    )
    _WORKER_STATE["template"] = template
    _WORKER_STATE["solver"] = solver
    _WORKER_STATE["prob_cfg"] = prob_cfg
    _WORKER_STATE["lens_path"] = lens_path
    _WORKER_STATE["p_problem"] = p_problem


def _worker_eval_theta(theta_list):
    """Build lens from theta and return per-wavelength field-averaged RMS in um.
    Returns None on failure (so caller can mark as ineligible)."""
    try:
        theta = np.asarray(theta_list, dtype=np.float64)
        lens = make_geolens(
            _WORKER_STATE["prob_cfg"], _WORKER_STATE["p_problem"],
            _WORKER_STATE["lens_path"], theta,
            _WORKER_STATE["template"], _WORKER_STATE["solver"],
        )
        return per_wvln_field_avg_rms(lens, _WORKER_STATE["p_problem"])
    except Exception:
        return None


def main():
    OUT = ROOT / "results" / "figure_archive_demo"
    HL_RUN = ROOT / "results" / "lensmega" / "sixlet_apsc_discovery_v2_seed42_20260506_034601"

    # ---- Load HL refined designs, archive cells, and SA+GA hull ----
    refined = json.loads((HL_RUN / "refined_top500_seed42_hl.json").read_text())["designs"]
    arch_cells = json.loads((HL_RUN / "archive_cells.json").read_text())["cells"]
    fit_to_measures = {round(c["fitness"], 8): c["measures"] for c in arch_cells}
    saga_hull = np.loadtxt(OUT / "saga_hull.csv", delimiter=",", skiprows=1)
    print(f"Loaded {len(refined)} HL refined designs, {len(arch_cells)} archive cells, "
          f"{len(saga_hull)} SA+GA hull verts")

    # The figure's (S2, S3) coordinate is the *archive-cell* (pre-refinement)
    # measure, not the post-refinement Seidel value. Look up via raw_fitness.
    # Filter 1: mean rms_um < 50 um (cheap pre-filter, keeps more headroom than 30)
    candidates = []
    n_no_cell = 0
    for d in refined:
        if d["rms_um"] >= 50.0:
            continue
        f = round(d["raw_fitness"], 8)
        if f not in fit_to_measures:
            n_no_cell += 1
            continue
        S2, S3 = fit_to_measures[f]
        candidates.append({
            "design": d, "S2": float(S2), "S3": float(S3),
            "mean_rms_um": float(d["rms_um"]),
            "in_saga": point_in_polygon(S2, S3, saga_hull),
        })
    print(f"After mean-rms < 50 um: {len(candidates)} candidates "
          f"(skipped {n_no_cell} with no archive-cell match)")
    print(f"  outside SA+GA hull: {sum(1 for c in candidates if not c['in_saga'])}")
    print(f"  inside SA+GA hull:  {sum(1 for c in candidates if c['in_saga'])}")

    # ---- Compute per-wvln RMS for all candidates (parallel over workers) ----
    n_workers = int(os.environ.get("REPICK_WORKERS", str(min(16, os.cpu_count() or 1))))
    print(f"Computing per-wvln RMS for {len(candidates)} candidates "
          f"(n_workers={n_workers})...")
    thetas = [np.asarray(c["design"]["theta"], dtype=np.float64).tolist()
              for c in candidates]
    with mp.get_context("spawn").Pool(
        processes=n_workers,
        initializer=_worker_init, initargs=("sixlet",),
    ) as pool:
        results = []
        for i, res in enumerate(pool.imap(_worker_eval_theta, thetas, chunksize=4)):
            results.append(res)
            if (i+1) % 20 == 0 or (i+1) == len(thetas):
                n_ok = sum(1 for r in results if r is not None)
                print(f"  evaluated {i+1}/{len(candidates)}  ({n_ok} successful)")
    for c, res in zip(candidates, results):
        if res is None:
            c["per_wvln_rms_um"] = None
            c["max_rms_um"] = float("inf")
        else:
            c["per_wvln_rms_um"] = res
            c["max_rms_um"] = max(res.values())
            c["min_rms_um"] = min(res.values())
            c["balance_ratio"] = c["max_rms_um"] / max(c["min_rms_um"], 1e-9)
    # We need wvln_keys later; use the first non-null result for that, or
    # fall back to a fresh build_lens to extract.
    template, solver, prob_cfg, lens_path, p_problem = (
        build_lens_template_and_solver(SYSTEMS["sixlet"]["cfg"])
    )
    wvln_keys = sorted(p_problem.fitness_wavelengths_um)

    # ---- Filter 2: max-per-wvln < 30 um ----
    eligible_outside = [c for c in candidates
                        if c["per_wvln_rms_um"] is not None
                        and c["max_rms_um"] < 30.0
                        and not c["in_saga"]]
    eligible_inside = [c for c in candidates
                       if c["per_wvln_rms_um"] is not None
                       and not math.isinf(c["max_rms_um"])
                       and c["in_saga"]]
    print(f"\nAfter max-per-wvln < 30 um:")
    print(f"  HL-only (outside SA+GA):  {len(eligible_outside)} eligible")
    print(f"  shared (inside SA+GA):    {len(eligible_inside)} eligible "
          f"(no max-rms gate; pick best mean)")

    # ---- Greedy farthest-point in (S2, S3) for HL-only spotlights ----
    # Seed with the lowest-max-rms HL-only candidate so the first pick is
    # a high-quality balanced design.
    if not eligible_outside:
        sys.exit("No HL-only candidates passed the max-per-wvln < 30 um filter")
    eligible_outside.sort(key=lambda c: c["max_rms_um"])
    picked_out = [eligible_outside[0]]
    while len(picked_out) < 3:
        best, best_d = None, -1.0
        for c in eligible_outside:
            if c in picked_out: continue
            d_min = min(
                math.hypot(c["S2"] - p["S2"], c["S3"] - p["S3"])
                for p in picked_out
            )
            if d_min > best_d:
                best_d, best = d_min, c
        if best is None: break
        picked_out.append(best)

    # ---- Shared: best mean-rms inside SA+GA hull ----
    if not eligible_inside:
        sys.exit("No HL designs inside SA+GA hull")
    eligible_inside.sort(key=lambda c: c["mean_rms_um"])
    picked_shared = eligible_inside[0]

    # ---- Build new spotlights.json ----
    new_spots = []
    for label, c in zip(["out1", "out2", "out3"], picked_out):
        new_spots.append({
            "label": label,
            "S2": c["S2"],
            "S3": c["S3"],
            "rms_um": c["mean_rms_um"],
            "in_saga_hull": False,
            "refined_theta": list(c["design"]["theta"]),
            "per_wvln_rms_um": {
                f"{int(round(w*1000))}nm": c["per_wvln_rms_um"][w]
                for w in wvln_keys
            },
            "max_per_wvln_rms_um": c["max_rms_um"],
            "balance_ratio": c["balance_ratio"],
        })
    new_spots.append({
        "label": "shared",
        "S2": picked_shared["S2"],
        "S3": picked_shared["S3"],
        "rms_um": picked_shared["mean_rms_um"],
        "in_saga_hull": True,
        "refined_theta": list(picked_shared["design"]["theta"]),
        "per_wvln_rms_um": {
            f"{int(round(w*1000))}nm": picked_shared["per_wvln_rms_um"][w]
            for w in wvln_keys
        },
        "max_per_wvln_rms_um": picked_shared["max_rms_um"],
        "balance_ratio": picked_shared["balance_ratio"],
    })

    # ---- Save ----
    spots_path = OUT / "spotlights.json"
    backup = OUT / "spotlights.before_repick.json"
    if not backup.exists():
        backup.write_text(spots_path.read_text())
        print(f"\nBacked up old spotlights to {backup}")
    spots_path.write_text(json.dumps(new_spots, indent=2))
    print(f"Wrote new spotlights to {spots_path}")

    # ---- Audit log ----
    audit = OUT / "spotlight_picker_audit.json"
    audit_data = {
        "rule": "max-per-wvln <= 30um for HL-only; greedy farthest in (S2,S3) seeded by lowest-max-rms; shared = best mean inside SA+GA hull",
        "n_candidates_after_mean_filter": len(candidates),
        "n_eligible_HL_only": len(eligible_outside),
        "n_eligible_inside": len(eligible_inside),
        "picks": new_spots,
        "all_eligible_outside": [
            {
                "S2": c["S2"], "S3": c["S3"],
                "mean_rms_um": c["mean_rms_um"],
                "max_rms_um": c["max_rms_um"],
                "min_rms_um": c["min_rms_um"],
                "balance_ratio": c["balance_ratio"],
                "per_wvln_rms_um": {
                    f"{int(round(w*1000))}nm": c["per_wvln_rms_um"][w]
                    for w in wvln_keys
                },
            }
            for c in eligible_outside
        ],
    }
    audit.write_text(json.dumps(audit_data, indent=2))
    print(f"Wrote audit log to {audit}")

    print("\nNew spotlights:")
    for sp in new_spots:
        wv_str = ", ".join(f"{k}={v:.1f}" for k, v in sp["per_wvln_rms_um"].items())
        print(f"  {sp['label']:8s}  S2={sp['S2']:+.5f}  S3={sp['S3']:+.5f}  "
              f"mean={sp['rms_um']:.1f}μm  max={sp['max_per_wvln_rms_um']:.1f}μm  "
              f"ratio={sp['balance_ratio']:.2f}x  per-λ=[{wv_str}]")


if __name__ == "__main__":
    main()
