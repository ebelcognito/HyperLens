"""Step 1: Aggregate the Seidel (S2, S3) archives across the 13 compatible
sixlet runs into a single 256x256 archive over the canonical bounds.

Per-cell channels:
  best_fitness, theta_best, n_designs, n_runs, mean_fitness,
  mean_S2, mean_S3, contributing_runs

Output: aggregated_sixlet_seidel_archive.json
"""
import json, glob, os
from pathlib import Path
import numpy as np

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
OUT_JSON = ROOT / "aggregated_sixlet_seidel_archive.json"

TARGET_BINS = (256, 256)
TARGET_BOUNDS = np.array([[-0.008, 0.008], [-0.012, 0.012]])  # canonical Seidel

# Compatibility rule: descriptor coord names must be ("S2_coma", "S3_astigmatism")
# OR equivalent (the canonical Seidel pair). We've already verified 13 runs
# qualify; we still gate on descriptor_kind == "seidel" + coord names.


def bin_index(val, lo, hi, n):
    return int(np.clip(np.floor((val - lo) / (hi - lo) * n), 0, n - 1))


def main():
    fps = sorted(glob.glob(str(ROOT / "results/lensmega/sixlet*/refined_top*.json")))
    runs = sorted({Path(fp).parent.name for fp in fps})
    print(f"Found {len(runs)} sixlet runs total.")

    eligible = []
    skip_reasons = {}
    for run in runs:
        p = ROOT / f"results/lensmega/{run}/archive_cells.json"
        if not p.exists():
            skip_reasons[run] = "no archive_cells.json"
            continue
        ac = json.load(open(p))
        if ac.get("descriptor_kind") != "seidel":
            skip_reasons[run] = f"descriptor_kind={ac.get('descriptor_kind')}"
            continue
        names = tuple(ac.get("descriptor_coord_names", []))
        if names != ("S2_coma", "S3_astigmatism"):
            skip_reasons[run] = f"coord_names={names}"
            continue
        eligible.append((run, ac))
    print(f"\nEligible Seidel runs: {len(eligible)}")
    for run, ac in eligible:
        print(f"  {run}  bins={ac['archive_bins']}  bounds={ac['archive_bounds']}  "
              f"cells={len(ac['cells'])}")
    if skip_reasons:
        print("\nSkipped:")
        for r, why in skip_reasons.items():
            print(f"  {r}: {why}")

    # Aggregate
    LO_S2, HI_S2 = TARGET_BOUNDS[0]
    LO_S3, HI_S3 = TARGET_BOUNDS[1]
    NI, NJ = TARGET_BINS

    # cell -> dict of channels
    agg = {}

    n_total = 0
    n_oob = 0
    for run, ac in eligible:
        for cell in ac["cells"]:
            n_total += 1
            s2, s3 = cell["measures"]
            if not (LO_S2 <= s2 <= HI_S2 and LO_S3 <= s3 <= HI_S3):
                n_oob += 1
                continue
            i = bin_index(s2, LO_S2, HI_S2, NI)
            j = bin_index(s3, LO_S3, HI_S3, NJ)
            key = (i, j)
            fit = float(cell["fitness"])
            entry = agg.get(key)
            if entry is None:
                agg[key] = {
                    "best_fitness": fit,
                    "theta_best": list(cell["theta"]),
                    "S2_best": float(s2),
                    "S3_best": float(s3),
                    "best_run": run,
                    "metrics_best": cell.get("metrics", {}),
                    "n_designs": 1,
                    "fit_sum": fit,
                    "s2_sum": float(s2),
                    "s3_sum": float(s3),
                    "runs_set": {run},
                }
            else:
                entry["n_designs"] += 1
                entry["fit_sum"] += fit
                entry["s2_sum"] += float(s2)
                entry["s3_sum"] += float(s3)
                entry["runs_set"].add(run)
                if fit > entry["best_fitness"]:
                    entry["best_fitness"] = fit
                    entry["theta_best"] = list(cell["theta"])
                    entry["S2_best"] = float(s2)
                    entry["S3_best"] = float(s3)
                    entry["best_run"] = run
                    entry["metrics_best"] = cell.get("metrics", {})

    n_cells = len(agg)
    n_collisions = sum(1 for v in agg.values() if v["n_designs"] > 1)
    multi_run = sum(1 for v in agg.values() if len(v["runs_set"]) > 1)
    print(f"\nAggregation:")
    print(f"  designs processed: {n_total}")
    print(f"  out of bounds:     {n_oob}")
    print(f"  filled cells:      {n_cells} / {NI*NJ} ({100*n_cells/(NI*NJ):.2f}% coverage)")
    print(f"  cells with >1 design:  {n_collisions}")
    print(f"  cells touched by >1 run: {multi_run}")

    fits = np.array([v["best_fitness"] for v in agg.values()])
    print(f"  best_fitness range: [{fits.min():+.4f}, {fits.max():+.4f}]")
    print(f"  cells with fit >= -1:  {int((fits >= -1.0).sum())}")
    print(f"  cells with fit >= -0.1: {int((fits >= -0.1).sum())}")
    print(f"  cells with fit >= -0.01: {int((fits >= -0.01).sum())}")

    # Serialize
    out_cells = []
    for (i, j), v in sorted(agg.items()):
        out_cells.append({
            "cell_ij": [i, j],
            "best_fitness": v["best_fitness"],
            "mean_fitness": v["fit_sum"] / v["n_designs"],
            "n_designs": v["n_designs"],
            "n_runs": len(v["runs_set"]),
            "S2_best": v["S2_best"],
            "S3_best": v["S3_best"],
            "mean_S2": v["s2_sum"] / v["n_designs"],
            "mean_S3": v["s3_sum"] / v["n_designs"],
            "best_run": v["best_run"],
            "theta_best": v["theta_best"],
            "contributing_runs": sorted(v["runs_set"]),
            "metrics_best": v["metrics_best"],
        })

    out = {
        "system": "sixlet_apsc_v2",
        "descriptor": ["S2_coma", "S3_astigmatism"],
        "target_bins": list(TARGET_BINS),
        "target_bounds": TARGET_BOUNDS.tolist(),
        "n_eligible_runs": len(eligible),
        "eligible_runs": [run for run, _ in eligible],
        "stats": {
            "n_designs_processed": n_total,
            "n_out_of_bounds_dropped": n_oob,
            "n_filled_cells": n_cells,
            "n_collision_cells": n_collisions,
            "n_multirun_cells": multi_run,
        },
        "cells": out_cells,
    }
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT_JSON}  ({os.path.getsize(OUT_JSON)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
