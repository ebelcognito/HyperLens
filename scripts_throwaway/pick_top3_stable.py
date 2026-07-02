#!/usr/bin/env python
"""Pick 3 HL-only designs ranked by stability, with a diversity floor so the
picks aren't clustered.

Algorithm: sort all eligible candidates by stability score ASC; walk down the
list and accept each one only if it is at least MIN_DIST away (in (S₂, S₃))
from every already-accepted pick. This gives the best stability available
*subject to a minimum spread*. Pure top-N would risk clustering; pure
greedy-farthest-point sacrifices stability for diversity.

Stability score = max RMS over (7 fields × 3 wavelengths). Lower = better.

Per topology:
  - load all HL refined sub-50µm-mean designs outside SA+GA hull
  - parallel eval per-field per-wvln RMS
  - sort by score ASC; take top 3
  - write spotlights_top3stable.json + full ranked list
  - render 3 panels (lens / seidel / robust) per pick

Outputs:
  results/figure_archive_demo[_octuplet]/spotlights_top3stable.json
  results/figure_archive_demo[_octuplet]/stability_full_ranking.json
  results/figure_archive_demo[_octuplet]/spotlight_panels_top3stable/
      <topology>_top{1,2,3}_{lens,seidel,robust}.png
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
    BLUE_FILL,
)
from scripts_throwaway.pick_stable_diverse_3 import (
    SYSTEM_PATHS, point_in_polygon,
    _worker_init, _worker_eval_theta,
)


def run_topology(topology):
    print(f"\n{'='*70}\nTOPOLOGY: {topology.upper()}\n{'='*70}")
    paths = SYSTEM_PATHS[topology]
    OUT, HL = paths["out_dir"], paths["hl_run"]

    refined = json.loads((HL / "refined_top500_seed42_hl.json").read_text())["designs"]
    cells = json.loads((HL / "archive_cells.json").read_text())["cells"]
    fit_to_measures = {round(c["fitness"], 8): c["measures"] for c in cells}
    saga_hull = np.loadtxt(OUT / "saga_hull.csv", delimiter=",", skiprows=1)

    candidates = []
    for d in refined:
        if d["rms_um"] >= 50.0:
            continue
        f = round(d["raw_fitness"], 8)
        if f not in fit_to_measures:
            continue
        S2, S3 = fit_to_measures[f]
        if point_in_polygon(S2, S3, saga_hull):
            continue
        candidates.append({
            "design": d, "S2": float(S2), "S3": float(S3),
            "mean_rms_um": float(d["rms_um"]),
        })
    print(f"  HL-only sub-50µm-mean candidates: {len(candidates)}")

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
    eligible.sort(key=lambda c: c["score"])
    print(f"  {len(eligible)} eligible after eval (sorted by score ASC)")
    top5 = [f"{c['score']:.1f}" for c in eligible[:5]]
    print(f"  top 5 scores (µm): {top5}")

    # ---- Score-cap then greedy farthest-point within the cap ----
    # Cap candidates at SCORE_CAP µm (default 20), then pick 3 with maximum
    # spread in (S₂, S₃) — seeded by the lowest-score candidate.
    score_cap = float(os.environ.get("SCORE_CAP", "20.0"))
    pool = [c for c in eligible if c["score"] <= score_cap]
    if len(pool) < 3:
        print(f"  WARN: only {len(pool)} candidates with score ≤ {score_cap}µm; "
              "relaxing to take the 3 lowest-score candidates regardless of spread")
        pool = eligible[:max(3, len(pool))]
    pool_scores_preview = [f"{c['score']:.1f}" for c in pool[:6]]
    suffix = "..." if len(pool) > 6 else ""
    print(f"  pool: {len(pool)} candidates with score ≤ {score_cap}µm  "
          f"(scores: {pool_scores_preview}{suffix})")

    # Seed with lowest-score; then greedy-farthest-point within the pool.
    pool_sorted = sorted(pool, key=lambda c: c["score"])
    top3 = [pool_sorted[0]]
    while len(top3) < 3 and len(top3) < len(pool):
        best, best_d = None, -1.0
        for c in pool:
            if c in top3: continue
            d = min(math.hypot(c["S2"]-p["S2"], c["S3"]-p["S3"]) for p in top3)
            if d > best_d:
                best_d, best = d, c
        if best is None: break
        top3.append(best)

    # ---- Build output records ----
    out_records = []
    for k, c in enumerate(top3):
        rec = {
            "label": f"top{k+1}",
            "rank": k + 1,
            "S2": c["S2"], "S3": c["S3"],
            "in_saga_hull": False,
            "stability_score_um": c["score"],
            "max_rms_um": c["max_rms_um"],
            "mean_rms_um": c["mean_rms_um"],
            "field_fracs": c["field_fracs"],
            "wvlns_um": c["wvlns_um"],
            "rms_per_field_per_wvln_um": c["rms_arr_um"],
            "refined_theta": list(c["design"]["theta"]),
            "raw_fitness": float(c["design"]["raw_fitness"]),
            "rms_um_reported": float(c["design"]["rms_um"]),
        }
        out_records.append(rec)

    out_json = OUT / "spotlights_top3stable.json"
    out_json.write_text(json.dumps(out_records, indent=2))
    print(f"  wrote {out_json}")

    # Full ranking (lighter — no theta or per-field array)
    full_rank = [
        {
            "rank": i + 1,
            "S2": c["S2"], "S3": c["S3"],
            "score_um": c["score"],
            "mean_rms_um": c["mean_rms_um"],
            "raw_fitness": float(c["design"]["raw_fitness"]),
        }
        for i, c in enumerate(eligible)
    ]
    full_rank_path = OUT / "stability_full_ranking.json"
    full_rank_path.write_text(json.dumps(full_rank, indent=2))
    print(f"  wrote {full_rank_path}")

    # ---- Render panels ----
    panels_dir = OUT / "spotlight_panels_top3stable"
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
    topologies = os.environ.get("TOPOLOGIES", "sixlet").split(",")
    for t in topologies:
        run_topology(t.strip())


if __name__ == "__main__":
    main()
