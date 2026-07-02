#!/usr/bin/env python
"""Pick 3 maximally diverse refined designs (HL pool) per system, with
quality threshold and an eps-net tier. Save theta, Seidels, RMS, lens
geometry, and spot diagrams for each.

Sixlet:   RMS <= 20 um, eps-net @ eps=0.25 (11 distinct available, pick 3 by best fitness)
Octuplet: RMS <= 30 um, eps-net @ eps=0.25 (exactly 3 distinct available)

Outputs:
  figures/fig5_data/three_diverse_below_thr.json
  figures/fig5_data/three_diverse_spots.npz
  figures/fig5_data/three_diverse_eps_tiers.csv  (eps tier counts table)
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

from scripts_throwaway.fig5_extract import (
    SYSTEMS, load_bounds, load_pool, eps_net_select,
    build_lens_template_and_solver, make_geolens, extract_geometry,
    compute_spot_endpoints,
)

# Per-system selection criteria
SELECTION = {
    "sixlet":   {"rms_max_um": 20.0, "eps": 0.25, "n": 3},
    "octuplet": {"rms_max_um": 30.0, "eps": 0.25, "n": 3},
}

EPS_TIERS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def main():
    output = {}
    spot_data = {}

    # eps-tier counts table
    rows = []
    for sys_name in ("sixlet", "octuplet"):
        info = SYSTEMS[sys_name]
        bounds = load_bounds(info["cfg"])
        D = bounds.shape[0]
        hl = load_pool(info["hl_json"], bounds)
        sa = load_pool(info["saga_json"], bounds)

        for label, pool in [("hl", hl), ("saga", sa)]:
            for T in [20.0, 30.0, 50.0]:
                sub = [r for r in pool if r["rms_um"] <= T]
                row = {"system": sys_name, "method": label, "rms_max_um": T,
                       "pool_size": len(sub)}
                for eps in EPS_TIERS:
                    n = len(eps_net_select(sub, eps, D))
                    row[f"eps_{eps:.2f}"] = n
                rows.append(row)

    csv = DATA / "three_diverse_eps_tiers.csv"
    with csv.open("w") as f:
        cols = ["system", "method", "rms_max_um", "pool_size"] + [f"eps_{e:.2f}" for e in EPS_TIERS]
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")
    print(f"wrote eps-tier table: {csv}")

    # Print readable table
    print("\n=== eps-distinct counts by (system, method, RMS budget) ===")
    print(f"{'sys':9s} {'meth':5s} {'rms<=':6s} {'pool':5s}  " +
          "  ".join(f"e={e:.2f}" for e in EPS_TIERS))
    for row in rows:
        print(f"{row['system']:9s} {row['method']:5s} {int(row['rms_max_um']):3d}    {row['pool_size']:4d}  " +
              "  ".join(f"{row[f'eps_{e:.2f}']:5d}" for e in EPS_TIERS))

    # Selection of 3 diverse designs per system
    for sys_name in ("sixlet", "octuplet"):
        sel = SELECTION[sys_name]
        info = SYSTEMS[sys_name]
        bounds = load_bounds(info["cfg"])
        D = bounds.shape[0]
        hl = load_pool(info["hl_json"], bounds)

        sub = [r for r in hl if r["rms_um"] <= sel["rms_max_um"]]
        idx = eps_net_select(sub, sel["eps"], D)
        chosen_idx = idx[: sel["n"]]
        chosen = [sub[i] for i in chosen_idx]
        if len(chosen) < sel["n"]:
            print(f"WARNING: {sys_name} only has {len(chosen)} eps-distinct "
                  f"designs at eps={sel['eps']}, RMS<={sel['rms_max_um']}")

        print(f"\n=== {sys_name} HL: {len(chosen)} diverse designs picked "
              f"(eps={sel['eps']}, RMS<={sel['rms_max_um']} um) ===")
        for k, r in enumerate(chosen):
            print(f"  #{k+1}  design_id={r['idx']:3d}  RMS={r['rms_um']:6.2f} um  "
                  f"refined_fitness={r['refined_fitness']:.4e}")

        # Pairwise distances among chosen
        if len(chosen) >= 2:
            X = np.array([r["theta_norm"] for r in chosen])
            for a in range(len(chosen)):
                for b in range(a + 1, len(chosen)):
                    d = np.linalg.norm(X[a] - X[b]) / np.sqrt(D)
                    print(f"    pair ({a+1},{b+1}): unit-theta L2/sqrt(D) = {d:.4f}")

        # Build lens, extract geometry, compute spot diagram for each
        template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(info["cfg"])
        entries = []
        for k, r in enumerate(chosen):
            theta = r["theta"]
            try:
                lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
                geom = extract_geometry(lens)
                spot_xy = compute_spot_endpoints(lens, prob_cfg, p_problem, num_rays=256)
            except Exception as e:
                print(f"  WARN: {sys_name} #{k+1} failed: {e}")
                geom = None
                spot_xy = np.full((3, 3, 256, 2), np.nan, dtype=np.float32)
            entries.append({
                "rank": k,
                "design_id": int(r["idx"]),
                "theta": [float(x) for x in r["theta"]],
                "S_1": float(r["S1"]), "S_2": float(r["S2"]), "S_3": float(r["S3"]),
                "S_4": float(r["S4"]), "S_5": float(r["S5"]),
                "refined_RMS_um": float(r["rms_um"]),
                "refined_fitness": float(r["refined_fitness"]),
                "efl_mm": float(r["efl_mm"]),
                "fnum_actual": float(r["fnum_actual"]),
                "lens_geometry": geom,
            })
            spot_data[f"{sys_name}_{k}"] = spot_xy

        output[sys_name] = {
            "selection_criteria": {
                "rms_max_um": sel["rms_max_um"],
                "eps": sel["eps"],
                "method": "hl",
                "from_pool_size": len(hl),
                "from_filtered_pool_size": len(sub),
                "n_distinct_at_eps_in_filtered_pool": len(idx),
            },
            "designs": entries,
        }

    json_out = DATA / "three_diverse_below_thr.json"
    json_out.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {json_out}")
    npz_out = DATA / "three_diverse_spots.npz"
    np.savez(npz_out, **spot_data)
    print(f"wrote {npz_out} ({len(spot_data)} entries)")


if __name__ == "__main__":
    main()
