"""Unified Seidel-archive aggregator for triplet / sixlet / octuplet.

Per system, finds all eligible Seidel-descriptor runs, projects each cell
onto a common 256x256 grid over the system's canonical bounds, keeps best
fitness per cell with side-channel stats. Writes one JSON per system.
"""
import json, glob, os, sys
from pathlib import Path
import numpy as np

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")

SYSTEMS = {
    "triplet": {
        "glob": "cooke_triplet*",
        "bounds": np.array([[-0.001, 0.001], [-0.001, 0.001]]),
        "bins": (256, 256),
    },
    "sixlet": {
        "glob": "sixlet*",
        "bounds": np.array([[-0.008, 0.008], [-0.012, 0.012]]),
        "bins": (256, 256),
    },
    "octuplet": {
        "glob": "octuplet*",
        "bounds": np.array([[-0.005, 0.005], [-0.008, 0.008]]),
        "bins": (256, 256),
    },
}


def bin_index(val, lo, hi, n):
    return int(np.clip(np.floor((val - lo) / (hi - lo) * n), 0, n - 1))


def aggregate(system):
    cfg = SYSTEMS[system]
    BOUNDS = cfg["bounds"]
    NI, NJ = cfg["bins"]
    LO_S2, HI_S2 = BOUNDS[0]
    LO_S3, HI_S3 = BOUNDS[1]

    fps = sorted(glob.glob(str(ROOT / f"results/lensmega/{cfg['glob']}/refined_top*.json")))
    runs = sorted({Path(fp).parent.name for fp in fps})
    eligible = []
    skipped = {}
    for run in runs:
        p = ROOT / f"results/lensmega/{run}/archive_cells.json"
        if not p.exists():
            skipped[run] = "no archive_cells.json"; continue
        ac = json.load(open(p))
        if ac.get("descriptor_kind") != "seidel":
            skipped[run] = f"descriptor_kind={ac.get('descriptor_kind')}"; continue
        if tuple(ac.get("descriptor_coord_names", [])) != ("S2_coma", "S3_astigmatism"):
            skipped[run] = f"coord_names={ac.get('descriptor_coord_names')}"; continue
        eligible.append((run, ac))
    print(f"\n[{system}] {len(runs)} runs total, {len(eligible)} eligible Seidel")
    for r, why in skipped.items():
        print(f"  skip {r}: {why}")

    agg = {}
    n_total = n_oob = 0
    for run, ac in eligible:
        for cell in ac["cells"]:
            n_total += 1
            s2, s3 = cell["measures"]
            if not (LO_S2 <= s2 <= HI_S2 and LO_S3 <= s3 <= HI_S3):
                n_oob += 1; continue
            i = bin_index(s2, LO_S2, HI_S2, NI)
            j = bin_index(s3, LO_S3, HI_S3, NJ)
            key = (i, j)
            fit = float(cell["fitness"])
            entry = agg.get(key)
            if entry is None:
                agg[key] = {
                    "best_fitness": fit,
                    "theta_best": list(cell["theta"]),
                    "S2_best": float(s2), "S3_best": float(s3),
                    "best_run": run,
                    "metrics_best": cell.get("metrics", {}),
                    "n_designs": 1, "fit_sum": fit,
                    "s2_sum": float(s2), "s3_sum": float(s3),
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
                    entry["S2_best"] = float(s2); entry["S3_best"] = float(s3)
                    entry["best_run"] = run
                    entry["metrics_best"] = cell.get("metrics", {})

    n_cells = len(agg)
    n_collisions = sum(1 for v in agg.values() if v["n_designs"] > 1)
    multi_run = sum(1 for v in agg.values() if len(v["runs_set"]) > 1)
    fits = np.array([v["best_fitness"] for v in agg.values()])
    print(f"  processed {n_total} designs; {n_oob} OOB; {n_cells} filled cells "
          f"({100*n_cells/(NI*NJ):.2f}%)")
    print(f"  collisions={n_collisions}, multi-run={multi_run}")
    print(f"  best_fitness range: [{fits.min():+.4f}, {fits.max():+.4f}]")
    print(f"  cells >= -1: {int((fits>=-1).sum())}, "
          f">= -0.1: {int((fits>=-0.1).sum())}, "
          f">= -0.01: {int((fits>=-0.01).sum())}")

    out_cells = []
    for (i, j), v in sorted(agg.items()):
        out_cells.append({
            "cell_ij": [i, j],
            "best_fitness": v["best_fitness"],
            "mean_fitness": v["fit_sum"] / v["n_designs"],
            "n_designs": v["n_designs"],
            "n_runs": len(v["runs_set"]),
            "S2_best": v["S2_best"], "S3_best": v["S3_best"],
            "mean_S2": v["s2_sum"] / v["n_designs"],
            "mean_S3": v["s3_sum"] / v["n_designs"],
            "best_run": v["best_run"],
            "theta_best": v["theta_best"],
            "contributing_runs": sorted(v["runs_set"]),
            "metrics_best": v["metrics_best"],
        })

    out = {
        "system": system,
        "descriptor": ["S2_coma", "S3_astigmatism"],
        "target_bins": [NI, NJ],
        "target_bounds": BOUNDS.tolist(),
        "n_eligible_runs": len(eligible),
        "eligible_runs": [r for r, _ in eligible],
        "stats": {
            "n_designs_processed": n_total,
            "n_out_of_bounds_dropped": n_oob,
            "n_filled_cells": n_cells,
            "n_collision_cells": n_collisions,
            "n_multirun_cells": multi_run,
        },
        "cells": out_cells,
    }
    out_path = ROOT / f"aggregated_{system}_seidel_archive.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"  wrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["triplet", "octuplet"]
    for s in targets:
        aggregate(s)


if __name__ == "__main__":
    main()
