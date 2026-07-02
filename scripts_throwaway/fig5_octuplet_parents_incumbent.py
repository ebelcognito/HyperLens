#!/usr/bin/env python
"""For octuplet: pick 3 'near-parent' designs (close to one of the 20
distinct refined picks in unit-theta space, but with notably worse RMS)
and 1 'obviously bad incumbent' (high RMS, visually poor).

Output:
  figures/fig5_data/octuplet_parents_and_incumbent.json
  figures/fig5_data/octuplet_parents_and_incumbent.csv
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

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
DATA = ROOT / "figures" / "fig5_data"

from scripts_throwaway.fig5_extract import SYSTEMS, load_bounds, load_pool

# How many anchor pairs to find
N_PARENTS = 3
# RMS band for "near-parent": clearly worse than refined-good but not catastrophic
PARENT_RMS_MIN = 50.0
PARENT_RMS_MAX = 250.0
# Max unit-theta distance for a parent to be considered "visually similar"
PARENT_MAX_DISTANCE = 0.15
# Incumbent target: clearly bad
INCUMBENT_RMS_MIN = 400.0
INCUMBENT_RMS_MAX = 800.0


def main():
    info = SYSTEMS["octuplet"]
    bounds = load_bounds(info["cfg"])
    D = bounds.shape[0]
    sqrtD = float(np.sqrt(D))

    # Search pool: combine octuplet HL (seed42 + seed43) + SAGA divrun (seed42)
    # so we can find the nearest near-parent regardless of which run discovered it.
    POOL_SOURCES = [
        ("HL_s42", "hl", 42, info["hl_json"]),
        ("HL_s43", "hl", 43, ROOT / "results/lensmega/octuplet_apsc_2cement_v2_seed43_20260506_044236/refined_top500_seed43_hl.json"),
        ("SAGA_s42_divrun", "saga", 42, ROOT / "results/diversity_run_20260506_033447/octuplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json"),
    ]
    hl = []  # combined pool with provenance attached
    for tag, method, seed, path in POOL_SOURCES:
        loaded = load_pool(path, bounds)
        for r in loaded:
            r["_source_run"] = tag
            r["_method"] = method
            r["_seed"] = seed
        hl.extend(loaded)
    print(f"octuplet combined pool: {len(hl)} designs (refined RMS<1mm)")

    # Anchors: load distinct refined picks, keep only HL ones (so we can find
    # near-parents in the same HL search neighborhood). Sort by RMS ascending.
    distinct = json.loads((DATA / "all_distinct_below30um.json").read_text())["octuplet"]["designs"]
    hl_anchors = sorted(
        [a for a in distinct if a["method"] == "hl"],
        key=lambda a: a["refined_RMS_um"],
    )
    print(f"distinct refined picks (<30um): {len(distinct)} total, {len(hl_anchors)} HL")

    # Pre-compute pool arrays so we can quickly screen anchors
    pool_arr_pre = np.array([r["theta_norm"] for r in hl])
    pool_rms_pre = np.array([r["rms_um"] for r in hl])
    band_mask = (pool_rms_pre >= PARENT_RMS_MIN) & (pool_rms_pre <= PARENT_RMS_MAX)

    # Iterate HL anchors, keep only those with a near-parent within PARENT_MAX_DISTANCE
    anchors = []
    for a in hl_anchors:
        a_norm = (np.asarray(a["theta"]) - bounds[:, 0]) / np.where(
            (bounds[:, 1] - bounds[:, 0]) > 0, bounds[:, 1] - bounds[:, 0], 1.0
        )
        d_all = np.linalg.norm(pool_arr_pre - a_norm, axis=1) / sqrtD
        d_band = d_all[band_mask]
        if d_band.size and d_band.min() <= PARENT_MAX_DISTANCE:
            anchors.append(a)
            if len(anchors) == N_PARENTS:
                break

    if len(anchors) < N_PARENTS:
        print(f"WARN: only found {len(anchors)} anchors with a near-parent within "
              f"d<={PARENT_MAX_DISTANCE}; relaxing threshold for the rest")
        for a in hl_anchors:
            if any(a["design_id"] == aa["design_id"] and a["source_run"] == aa["source_run"]
                   for aa in anchors):
                continue
            anchors.append(a)
            if len(anchors) == N_PARENTS:
                break

    for k, a in enumerate(anchors):
        print(f"  anchor {k+1}: source={a['source_run']} id={a['design_id']} "
              f"RMS={a['refined_RMS_um']:.2f} um")

    # Pre-compute normalised theta for full HL pool
    pool_arr = np.array([r["theta_norm"] for r in hl])  # (N, D)
    pool_rms = np.array([r["rms_um"] for r in hl])

    out_designs = []

    # ---- 3 near-parents ----
    chosen_pool_indices = set()
    for k, anchor in enumerate(anchors):
        anchor_theta_norm = (np.asarray(anchor["theta"]) - bounds[:, 0]) / np.where(
            (bounds[:, 1] - bounds[:, 0]) > 0, bounds[:, 1] - bounds[:, 0], 1.0
        )
        dists = np.linalg.norm(pool_arr - anchor_theta_norm, axis=1) / sqrtD
        # Mask to RMS band and not yet chosen
        mask = (pool_rms >= PARENT_RMS_MIN) & (pool_rms <= PARENT_RMS_MAX)
        for i in chosen_pool_indices:
            mask[i] = False
        if not mask.any():
            print(f"  WARN: anchor {k+1} -- no candidate in RMS [{PARENT_RMS_MIN}, {PARENT_RMS_MAX}]")
            continue
        candidate_idx = np.where(mask)[0]
        # Print top-3 candidates for sanity
        order = candidate_idx[np.argsort(dists[candidate_idx])][:3]
        print(f"  anchor {k+1} top-3 candidate parents:")
        for j, ci in enumerate(order):
            r_ci = hl[int(ci)]
            print(f"    cand{j+1}: src={r_ci['_source_run']} pool_idx={ci} "
                  f"RMS={pool_rms[ci]:.2f}um  d={dists[ci]:.4f}")
        i_best = int(order[0])
        chosen_pool_indices.add(i_best)
        r = hl[i_best]
        d = float(dists[i_best])
        print(f"  -> picked near-parent {k+1}: src={r['_source_run']} "
              f"design_id={i_best} RMS={r['rms_um']:.2f} um  d={d:.4f}")
        out_designs.append({
            "role": "parent",
            "rank": k,
            "source_run": r["_source_run"],
            "method": r["_method"],
            "seed": r["_seed"],
            "design_id": int(i_best),
            "theta": [float(x) for x in r["theta"]],
            "S_1": float(r["S1"]), "S_2": float(r["S2"]), "S_3": float(r["S3"]),
            "S_4": float(r["S4"]), "S_5": float(r["S5"]),
            "refined_RMS_um": float(r["rms_um"]),
            "refined_fitness": float(r["refined_fitness"]),
            "raw_RMS_um": float(r["raw_rms_um"]),
            "efl_mm": float(r["efl_mm"]),
            "fnum_actual": float(r["fnum_actual"]),
            "anchor_info": {
                "anchor_rank": k,
                "anchor_source_run": anchor["source_run"],
                "anchor_design_id": int(anchor["design_id"]),
                "anchor_refined_RMS_um": float(anchor["refined_RMS_um"]),
                "unit_theta_distance": d,
            },
        })

    # ---- 1 obviously bad incumbent ----
    mask_bad = (pool_rms >= INCUMBENT_RMS_MIN) & (pool_rms <= INCUMBENT_RMS_MAX)
    for i in chosen_pool_indices:
        mask_bad[i] = False
    if mask_bad.any():
        # Pick the absolute worst within the band so the incumbent is unambiguously bad
        bad_indices = np.where(mask_bad)[0]
        i_bad = int(bad_indices[np.argmax(pool_rms[bad_indices])])
        r = hl[i_bad]
        print(f"  incumbent (bad): src={r['_source_run']} design_id={i_bad} RMS={r['rms_um']:.2f} um")
        out_designs.append({
            "role": "incumbent",
            "rank": 0,
            "source_run": r["_source_run"],
            "method": r["_method"],
            "seed": r["_seed"],
            "design_id": int(i_bad),
            "theta": [float(x) for x in r["theta"]],
            "S_1": float(r["S1"]), "S_2": float(r["S2"]), "S_3": float(r["S3"]),
            "S_4": float(r["S4"]), "S_5": float(r["S5"]),
            "refined_RMS_um": float(r["rms_um"]),
            "refined_fitness": float(r["refined_fitness"]),
            "raw_RMS_um": float(r["raw_rms_um"]),
            "efl_mm": float(r["efl_mm"]),
            "fnum_actual": float(r["fnum_actual"]),
        })
    else:
        print(f"  WARN: no design in RMS [{INCUMBENT_RMS_MIN}, {INCUMBENT_RMS_MAX}]")

    output = {
        "system": "octuplet",
        "selection_criteria": {
            "n_parents": N_PARENTS,
            "parent_rms_band_um": [PARENT_RMS_MIN, PARENT_RMS_MAX],
            "incumbent_rms_band_um": [INCUMBENT_RMS_MIN, INCUMBENT_RMS_MAX],
            "anchor_source": "first 3 distinct refined picks from all_distinct_below30um.json",
            "match_metric": "unit-theta L2 / sqrt(D=45), nearest in band",
            "pool_source": "results/lensmega/octuplet_apsc_2cement_v2_seed42_20260506_040025/refined_top500_seed42_hl.json",
        },
        "anchors_used": [
            {
                "rank": k,
                "source_run": a["source_run"],
                "design_id": int(a["design_id"]),
                "refined_RMS_um": float(a["refined_RMS_um"]),
                "theta": [float(x) for x in a["theta"]],
                "S_1": a["S_1"], "S_2": a["S_2"], "S_3": a["S_3"],
                "S_4": a["S_4"], "S_5": a["S_5"],
                "efl_mm": a["efl_mm"], "fnum_actual": a["fnum_actual"],
            }
            for k, a in enumerate(anchors)
        ],
        "designs": out_designs,
    }

    json_out = DATA / "octuplet_parents_and_incumbent.json"
    json_out.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {json_out}")

    csv_out = DATA / "octuplet_parents_and_incumbent.csv"
    cols = ["role", "rank", "design_id", "refined_RMS_um", "refined_fitness",
            "S_1", "S_2", "S_3", "S_4", "S_5", "efl_mm", "fnum_actual"]
    with csv_out.open("w") as f:
        f.write("kind," + ",".join(cols) + ",anchor_id,anchor_RMS_um,distance_to_anchor\n")
        for des in out_designs:
            row = [des["role"]] + [str(des[c]) for c in cols]
            ai = des.get("anchor_info")
            if ai:
                row += [str(ai["anchor_design_id"]), f"{ai['anchor_refined_RMS_um']:.4f}",
                        f"{ai['unit_theta_distance']:.4f}"]
            else:
                row += ["", "", ""]
            f.write(",".join(row) + "\n")
    print(f"wrote {csv_out}")


if __name__ == "__main__":
    main()
