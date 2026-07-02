#!/usr/bin/env python
"""Gather ALL geometrically distinct refined designs with refined RMS < 30 um
from BOTH methods (HL + SA+GA) and BOTH seeds (42, 43) where available, for
sixlet and octuplet systems.

Distinctness: greedy epsilon-net at eps=0.20 in unit-normalised theta space
(L2 distance / sqrt(D)). Sorted best-fitness-first across the combined pool
so the kept rep of each cluster is the highest-fitness instance.

Outputs:
  figures/fig5_data/all_distinct_below30um.json     (full geometry per design)
  figures/fig5_data/all_distinct_below30um.csv      (summary)
"""
from __future__ import annotations

import json
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
DATA = ROOT / "figures" / "fig5_data"

from scripts_throwaway.fig5_extract import SYSTEMS, load_bounds

EPS = 0.20
RMS_MAX_UM = 30.0

# Per-system: list of (label, method, seed, json_path)
RUN_SOURCES = {
    "sixlet": [
        ("HL_s42_lensmega",
         "hl", 42,
         ROOT / "results/lensmega/sixlet_apsc_discovery_v2_seed42_20260506_034601/refined_top500_seed42_hl.json"),
        ("HL_s43_lensmega",
         "hl", 43,
         ROOT / "results/lensmega/sixlet_apsc_discovery_v2_seed43_20260506_042806/refined_top500_seed43_hl.json"),
        ("SAGA_s42_steelman",
         "saga", 42,
         ROOT / "results/steelman_div020_20260506_123801/sixlet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json"),
        ("SAGA_s42_divrun",
         "saga", 42,
         ROOT / "results/diversity_run_20260506_033447/sixlet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json"),
        ("SAGA_s43_divrun",
         "saga", 43,
         ROOT / "results/diversity_run_20260506_033447/sixlet/seed43/sa_ga/refined_above_-1.0_seed43_sa_ga.json"),
    ],
    "octuplet": [
        ("HL_s42_lensmega",
         "hl", 42,
         ROOT / "results/lensmega/octuplet_apsc_2cement_v2_seed42_20260506_040025/refined_top500_seed42_hl.json"),
        ("HL_s43_lensmega",
         "hl", 43,
         ROOT / "results/lensmega/octuplet_apsc_2cement_v2_seed43_20260506_044236/refined_top500_seed43_hl.json"),
        ("SAGA_s42_steelman",
         "saga", 42,
         ROOT / "results/steelman_div020_20260506_123801/octuplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json"),
        ("SAGA_s42_divrun",
         "saga", 42,
         ROOT / "results/diversity_run_20260506_033447/octuplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json"),
        ("SAGA_s43_divrun",
         "saga", 43,
         ROOT / "results/diversity_run_20260506_033447/octuplet/seed43/sa_ga/refined_above_-1.0_seed43_sa_ga.json"),
    ],
}


def load_records(json_path: Path, source_label: str, method: str, seed: int,
                 bounds: np.ndarray) -> list[dict]:
    """Load all designs from one refined JSON, filter to RMS<RMS_MAX_UM, attach
    provenance. Returns list of records ready for combining."""
    if not json_path.exists():
        print(f"  MISSING: {json_path}")
        return []
    with json_path.open() as f:
        d = json.load(f)
    designs = d.get("designs", [])
    out = []
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = np.where((hi - lo) > 0.0, hi - lo, 1.0)
    for i, des in enumerate(designs):
        rms = float(des["rms_um"])
        if rms >= RMS_MAX_UM:
            continue
        theta = np.asarray(des["theta"], dtype=np.float64).reshape(-1)
        m = des["metrics"]
        out.append({
            "source_run": source_label,
            "method": method,
            "seed": seed,
            "design_id": int(i),
            "theta": theta,
            "theta_norm": (theta - lo) / span,
            "refined_rms_um": rms,
            "refined_fitness": float(des["refined_fitness"]),
            "raw_rms_um": float(des.get("raw_rms_um", float("nan"))),
            "S1": float(m["seidel_S1"]),
            "S2": float(m["seidel_S2"]),
            "S3": float(m["seidel_S3"]),
            "S4": float(m["seidel_S4"]),
            "S5": float(m["seidel_S5"]),
            "efl_mm": float(des["efl_mm"]),
            "fnum_actual": float(des["fnum_actual"]),
        })
    return out


def greedy_eps_net(records: list[dict], eps: float, dim: int) -> list[int]:
    """Greedy eps-net. Records pre-sorted best-fitness-first.
    Returns indices of accepted records."""
    accepted: list[int] = []
    accepted_norm: list[np.ndarray] = []
    sqrtD = float(np.sqrt(dim))
    for i, r in enumerate(records):
        v = r["theta_norm"]
        ok = True
        for a in accepted_norm:
            if np.linalg.norm(v - a) / sqrtD <= eps:
                ok = False
                break
        if ok:
            accepted.append(i)
            accepted_norm.append(v)
    return accepted


def main():
    output = {}
    csv_rows = []

    for sys_name, sources in RUN_SOURCES.items():
        info = SYSTEMS[sys_name]
        bounds = load_bounds(info["cfg"])
        D = bounds.shape[0]
        print(f"\n=== {sys_name.upper()} (D={D}) ===")

        # Load all sources, concat
        per_source_counts = {}
        all_records = []
        for label, method, seed, path in sources:
            recs = load_records(path, label, method, seed, bounds)
            per_source_counts[label] = len(recs)
            all_records.extend(recs)
            print(f"  source {label:25s}: {len(recs):4d} designs <30um  (path: {path.name})")

        # Sort best-fitness-first across the combined pool
        all_records.sort(key=lambda r: -r["refined_fitness"])

        # Greedy eps-net dedup at eps=0.20
        accepted_idx = greedy_eps_net(all_records, EPS, D)
        accepted = [all_records[i] for i in accepted_idx]
        print(f"  combined pool: {len(all_records)} designs <30um  ->  "
              f"{len(accepted)} distinct at eps={EPS}")

        # Capture only the theta + metadata. Lens geometry is *not* extracted
        # here -- it can be rebuilt from theta when rendering, and reloading
        # GeoLens from JSON for every design is the bottleneck.
        out_designs = []
        for k, r in enumerate(accepted):
            out_designs.append({
                "rank": k,
                "source_run": r["source_run"],
                "method": r["method"],
                "seed": r["seed"],
                "design_id": r["design_id"],
                "theta": [float(x) for x in r["theta"]],
                "S_1": r["S1"], "S_2": r["S2"], "S_3": r["S3"],
                "S_4": r["S4"], "S_5": r["S5"],
                "refined_RMS_um": r["refined_rms_um"],
                "refined_fitness": r["refined_fitness"],
                "raw_RMS_um": r["raw_rms_um"],
                "efl_mm": r["efl_mm"],
                "fnum_actual": r["fnum_actual"],
            })
            csv_rows.append({
                "system": sys_name,
                "rank": k,
                "source_run": r["source_run"],
                "method": r["method"],
                "seed": r["seed"],
                "design_id": r["design_id"],
                "refined_RMS_um": r["refined_rms_um"],
                "refined_fitness": r["refined_fitness"],
                "S_1": r["S1"], "S_2": r["S2"], "S_3": r["S3"],
                "S_4": r["S4"], "S_5": r["S5"],
                "efl_mm": r["efl_mm"],
                "fnum_actual": r["fnum_actual"],
            })

        # Per-method count among accepted
        n_by_method = {"hl": 0, "saga": 0}
        for r in accepted:
            n_by_method[r["method"]] += 1
        print(f"  distinct by method: HL={n_by_method['hl']}, SAGA={n_by_method['saga']}")

        output[sys_name] = {
            "selection_criteria": {
                "rms_max_um": RMS_MAX_UM,
                "eps_distinct": EPS,
                "eps_metric": "L2 / sqrt(D) on unit-normalised theta",
                "dim": D,
            },
            "sources": [
                {
                    "label": label, "method": method, "seed": seed,
                    "path": str(path),
                    "n_designs_below_threshold": per_source_counts[label],
                }
                for (label, method, seed, path) in sources
            ],
            "n_combined_input": len(all_records),
            "n_distinct": len(accepted),
            "n_distinct_by_method": n_by_method,
            "designs": out_designs,
        }

    json_out = DATA / "all_distinct_below30um.json"
    json_out.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {json_out}")

    csv_out = DATA / "all_distinct_below30um.csv"
    cols = ["system", "rank", "source_run", "method", "seed", "design_id",
            "refined_RMS_um", "refined_fitness",
            "S_1", "S_2", "S_3", "S_4", "S_5", "efl_mm", "fnum_actual"]
    with csv_out.open("w") as f:
        f.write(",".join(cols) + "\n")
        for row in csv_rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")
    print(f"wrote {csv_out} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
