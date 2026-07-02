#!/usr/bin/env python
"""Pick the 3 most diverse HL-only designs with the best stability rings.

Stability score = max RMS over (7 fields × 3 wavelengths). Lower is better —
this single number penalises high spot magnitude AND chromatic spread AND
field-edge degradation simultaneously.

Diversity: greedy farthest-point in (S2, S3) archive-cell space, seeded with
the candidate that has the lowest stability score.

Per topology:
  - load HL refined designs, filter to mean_rms < 50um, outside SA+GA hull
  - parallel eval per-field per-wvln RMS (7 fields × 3 wvlns) for each
  - greedy pick 3 with diversity + low score
  - write spotlights_stable3.json
  - render 3 panels (lens / seidel / robust) per pick

Outputs:
  results/figure_archive_demo[_octuplet]/spotlights_stable3.json
  results/figure_archive_demo[_octuplet]/spotlight_panels_stable3/
      <topology>_pick{1,2,3}_{lens,seidel,robust}.png
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)
from scripts_throwaway.render_spotlight_panels_split import (
    render_lens, render_seidel_s2_s3, render_robust,
    per_field_per_wvln_rms_um,
    BLUE_FILL, RED_FILL,
)


SYSTEM_PATHS = {
    "sixlet": dict(
        out_dir=ROOT / "results" / "figure_archive_demo",
        hl_run=ROOT / "results" / "lensmega"
                    / "sixlet_apsc_discovery_v2_seed42_20260506_034601",
    ),
    "octuplet": dict(
        out_dir=ROOT / "results" / "figure_archive_demo_octuplet",
        hl_run=ROOT / "results" / "lensmega"
                    / "octuplet_apsc_2cement_v2_seed42_20260506_040025",
    ),
}


def point_in_polygon(x, y, verts):
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


# ---- Multiprocessing worker ----
_W = {}


def _worker_init(topology):
    sys.stdout.reconfigure(line_buffering=True)
    template, solver, prob_cfg, lens_path, p_problem = (
        build_lens_template_and_solver(SYSTEMS[topology]["cfg"])
    )
    _W["template"] = template; _W["solver"] = solver
    _W["prob_cfg"] = prob_cfg; _W["lens_path"] = lens_path
    _W["p_problem"] = p_problem


def _worker_eval_theta(theta_list):
    try:
        theta = np.asarray(theta_list, dtype=np.float64)
        lens = make_geolens(_W["prob_cfg"], _W["p_problem"], _W["lens_path"],
                            theta, _W["template"], _W["solver"])
        field_fracs, wvlns, rms_arr = per_field_per_wvln_rms_um(
            lens, _W["p_problem"], n_fields=7, num_rays=256,
        )
        return {
            "field_fracs": field_fracs.tolist(),
            "wvlns_um": [float(w) for w in wvlns],
            "rms_arr_um": rms_arr.tolist(),
            "max_rms_um": float(rms_arr.max()),
            "mean_rms_um": float(rms_arr.mean()),
        }
    except Exception:
        return None


# ---- Pick 3 with greedy farthest-point + stability ----
def pick_three_diverse(eligible, max_score_threshold=None):
    """eligible: list of dicts with 'S2', 'S3', 'score' (lower=better).
    Greedy: seed with lowest score, then add candidates that maximise
    min-distance-to-picked, but only among those with score below threshold."""
    if not eligible:
        return []
    pool = sorted(eligible, key=lambda c: c["score"])
    if max_score_threshold is not None:
        pool = [c for c in pool if c["score"] <= max_score_threshold]
        if len(pool) < 3:
            # fallback: relax filter, take next-best by score
            pool = sorted(eligible, key=lambda c: c["score"])
    picked = [pool[0]]  # seed with lowest score
    while len(picked) < 3 and len(picked) < len(pool):
        best, best_d = None, -1.0
        for c in pool:
            if c in picked: continue
            d = min(math.hypot(c["S2"] - p["S2"], c["S3"] - p["S3"]) for p in picked)
            if d > best_d:
                best_d, best = d, c
        if best is None: break
        picked.append(best)
    return picked


def run_topology(topology):
    print(f"\n{'='*70}\nTOPOLOGY: {topology.upper()}\n{'='*70}")
    paths = SYSTEM_PATHS[topology]
    OUT, HL = paths["out_dir"], paths["hl_run"]

    refined = json.loads((HL / "refined_top500_seed42_hl.json").read_text())["designs"]
    cells = json.loads((HL / "archive_cells.json").read_text())["cells"]
    fit_to_measures = {round(c["fitness"], 8): c["measures"] for c in cells}
    saga_hull = np.loadtxt(OUT / "saga_hull.csv", delimiter=",", skiprows=1)
    print(f"  loaded {len(refined)} HL refined, {len(cells)} archive cells")

    # Filter: mean rms < 50 um AND outside SA+GA hull
    candidates = []
    for d in refined:
        if d["rms_um"] >= 50.0:
            continue
        f = round(d["raw_fitness"], 8)
        if f not in fit_to_measures:
            continue
        S2, S3 = fit_to_measures[f]
        if point_in_polygon(S2, S3, saga_hull):
            continue  # exclude inside hull (we want HL-only diversity)
        candidates.append({
            "design": d, "S2": float(S2), "S3": float(S3),
            "mean_rms_um": float(d["rms_um"]),
        })
    print(f"  HL-only sub-50µm-mean candidates: {len(candidates)}")

    # Parallel eval
    n_workers = int(os.environ.get("PICK_WORKERS",
                                    str(min(16, len(os.sched_getaffinity(0))))))
    print(f"  evaluating per-field per-wvln RMS (n_workers={n_workers})...")
    thetas = [c["design"]["theta"] for c in candidates]
    with mp.get_context("spawn").Pool(
        processes=n_workers,
        initializer=_worker_init, initargs=(topology,),
    ) as pool:
        results = []
        for i, res in enumerate(pool.imap(_worker_eval_theta, thetas, chunksize=4)):
            results.append(res)
            if (i+1) % 25 == 0 or (i+1) == len(thetas):
                n_ok = sum(1 for r in results if r is not None)
                print(f"    {i+1}/{len(thetas)}  ({n_ok} successful)")

    eligible = []
    for c, res in zip(candidates, results):
        if res is None: continue
        c.update(res)
        c["score"] = c["max_rms_um"]
        eligible.append(c)
    print(f"  {len(eligible)} eligible after eval")

    # Threshold: prefer max < 30 um, fall back if fewer than 3
    picks = pick_three_diverse(eligible, max_score_threshold=30.0)
    if len(picks) < 3:
        print(f"  WARN: fewer than 3 with score<30um, falling back")
        picks = pick_three_diverse(eligible, max_score_threshold=None)
    if len(picks) < 3:
        print(f"  ERROR: only {len(picks)} eligible total")
        return

    # Build output records
    out_records = []
    for k, c in enumerate(picks):
        rec = {
            "label": f"pick{k+1}",
            "S2": c["S2"],
            "S3": c["S3"],
            "in_saga_hull": False,
            "stability_score_um": c["score"],
            "max_rms_um": c["max_rms_um"],
            "mean_rms_um": c["mean_rms_um"],
            "field_fracs": c["field_fracs"],
            "wvlns_um": c["wvlns_um"],
            "rms_per_field_per_wvln_um": c["rms_arr_um"],
            "refined_theta": list(c["design"]["theta"]),
            "raw_fitness": float(c["design"]["raw_fitness"]),
            "rms_um": float(c["design"]["rms_um"]),
        }
        out_records.append(rec)

    out_json = OUT / "spotlights_stable3.json"
    out_json.write_text(json.dumps(out_records, indent=2))
    print(f"  wrote {out_json}")

    # Render panels
    panels_dir = OUT / "spotlight_panels_stable3"
    panels_dir.mkdir(exist_ok=True)
    template, solver, prob_cfg, lens_path, p_problem = (
        build_lens_template_and_solver(SYSTEMS[topology]["cfg"])
    )
    for rec in out_records:
        label = rec["label"]
        print(f"  rendering {label}: S2={rec['S2']:+.5f} S3={rec['S3']:+.5f}  "
              f"score={rec['stability_score_um']:.1f}µm")
        lens = make_geolens(prob_cfg, p_problem, lens_path,
                            np.asarray(rec["refined_theta"], dtype=np.float64),
                            template, solver)
        prefix = panels_dir / f"{topology}_{label}"
        render_lens(lens, BLUE_FILL, str(prefix) + "_lens.png")
        render_seidel_s2_s3(rec["S2"], rec["S3"], BLUE_FILL,
                            str(prefix) + "_seidel.png")
        rms_arr = np.asarray(rec["rms_per_field_per_wvln_um"])
        field_fracs = np.asarray(rec["field_fracs"])
        render_robust(field_fracs, rec["wvlns_um"], rms_arr, BLUE_FILL,
                      str(prefix) + "_robust.png")
        print(f"    → {prefix}_{{lens,seidel,robust}}.png")


def main():
    run_topology("sixlet")
    run_topology("octuplet")


if __name__ == "__main__":
    main()
