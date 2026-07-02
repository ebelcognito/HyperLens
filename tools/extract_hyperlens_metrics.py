#!/usr/bin/env python3
"""Extract HyperLens reference-run metrics into the SA+GA+Refine handoff
schema, so metrics.json has a directly-comparable hyperlens block.

The HyperLens runner already stores per-elite JSON files in
``<run_dir>/elite_designs/BEST_*.json`` and ``ELITE_*.json`` containing
both raw and refined designs (with stability analysis already attached
in the refined block — same `runner.py` code path that SA+GA+Refine now
calls). All we do here is roll those up into the schema the writing
agent expects.

Usage:
  python tools/extract_hyperlens_metrics.py \\
    --hyperlens-run results/lensmega/sixlet_apsc_discovery_v2_20260505_140231 \\
    --sa-ga-out results/sa_ga_refine_baseline/sixlet_<timestamp>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_elite_designs(run_dir: Path) -> List[Dict[str, Any]]:
    """Load every BEST_* and ELITE_* JSON in elite_designs/.

    HyperLens writes one JSON per refined elite. We treat both as
    candidates for the per-method rollup (BEST is just the
    fitness-best-so-far snapshot).
    """
    elite_dir = run_dir / "elite_designs"
    if not elite_dir.exists():
        return []
    files = sorted(elite_dir.glob("*.json"))
    out = []
    for f in files:
        try:
            d = json.loads(f.read_text())
            if "refined" not in d:
                continue
            d["_source_file"] = f.name
            out.append(d)
        except Exception as e:
            print(f"[skip] {f.name}: {e}")
    return out


def load_total_evals(run_dir: Path) -> int:
    """Read the final eval count from trajectory.csv."""
    tr = run_dir / "trajectory.csv"
    if not tr.exists():
        return -1
    last_row = None
    with tr.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row
    if last_row is None:
        return -1
    return int(last_row.get("evals", -1))


def rollup_metrics(elites: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not elites:
        return {
            "best_raw_fitness": None,
            "refined_rms_best_um": None,
            "refined_rms_p10_um": None,
            "refined_rms_median_um": None,
            "refined_rms_mean_um": None,
            "refined_rms_worst_um": None,
            "count_below_10um": 0,
            "count_below_5um": 0,
            "count_below_3um": 0,
            "improvement_ratio_median": None,
            "improvement_ratio_mean": None,
            "unique_sign_families": 0,
            "tracked_elites": 0,
        }
    raw_fits = np.array([float(e.get("fitness", np.nan)) for e in elites])
    refined_rms_um = np.array([
        float(e["refined"].get("rms_um")
              if isinstance(e["refined"].get("rms_um"), (int, float))
              else float(e["refined"].get("rms_mm", 0)) * 1000)
        for e in elites
    ])
    refined_fits = np.array([
        float(e["refined"].get("fitness", np.nan)) for e in elites
    ])
    sign_families = {e.get("sign_pattern", "?") for e in elites}
    improvement_ratios = []
    for e in elites:
        r = e["refined"].get("improvement_ratio")
        if r is not None:
            improvement_ratios.append(float(r))
    improvement_ratios = (
        np.array(improvement_ratios) if improvement_ratios else np.array([np.nan])
    )

    def safe_pct(arr, p):
        a = arr[np.isfinite(arr)]
        if len(a) == 0:
            return None
        return float(np.percentile(a, p))

    return {
        "best_raw_fitness": float(np.nanmax(raw_fits)),
        "refined_rms_best_um": float(np.nanmin(refined_rms_um)),
        "refined_rms_p10_um": safe_pct(refined_rms_um, 10),
        "refined_rms_median_um": safe_pct(refined_rms_um, 50),
        "refined_rms_mean_um": float(np.nanmean(refined_rms_um)),
        "refined_rms_worst_um": float(np.nanmax(refined_rms_um)),
        "count_below_10um": int(np.sum(refined_rms_um < 10)),
        "count_below_5um": int(np.sum(refined_rms_um < 5)),
        "count_below_3um": int(np.sum(refined_rms_um < 3)),
        "improvement_ratio_median": safe_pct(improvement_ratios, 50),
        "improvement_ratio_mean": float(np.nanmean(improvement_ratios)),
        "unique_sign_families": len(sign_families),
        "tracked_elites": len(elites),
    }


def to_handoff_records(elites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert HyperLens elite JSON to the same record schema
    sa_ga_refine_baseline.py writes for refined_designs_sa_ga.json."""
    out = []
    for e in elites:
        ref = e["refined"]
        out.append({
            "theta": ref.get("theta"),
            "refined_fitness": float(ref.get("fitness", np.nan)),
            "raw_fitness": float(e.get("fitness", np.nan)),
            "rms_um": float(ref.get("rms_um") if isinstance(
                ref.get("rms_um"), (int, float))
                else float(ref.get("rms_mm", 0)) * 1000),
            "rms_mm": float(ref.get("rms_mm", np.nan)),
            "efl_mm": float(ref.get("efl_mm", np.nan) if isinstance(
                ref.get("efl_mm"), (int, float)) else np.nan),
            "fnum_actual": ref.get("metrics", {}).get("fnum_actual"),
            "sign_pattern": e.get("sign_pattern", "?"),
            "lbfgs_steps": ref.get("lbfgs_steps"),
            "converged": ref.get("converged"),
            "final_grad_norm": ref.get("final_grad_norm"),
            "stability": ref.get("stability"),
            "metrics": ref.get("metrics", {}),
            "_source_file": e.get("_source_file"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyperlens-run", required=True,
                    help="Path to existing HyperLens run dir")
    ap.add_argument("--sa-ga-out", required=True,
                    help="Path to SA+GA+Refine run dir to merge into")
    args = ap.parse_args()

    hl_dir = Path(args.hyperlens_run).resolve()
    sg_dir = Path(args.sa_ga_out).resolve()

    elites = load_elite_designs(hl_dir)
    total_evals = load_total_evals(hl_dir)
    print(f"[hyperlens] loaded {len(elites)} elite designs from "
          f"{hl_dir}", flush=True)
    print(f"[hyperlens] trajectory.csv final evals = {total_evals}",
          flush=True)

    hl_metrics = rollup_metrics(elites)
    hl_metrics["wall_clock_seconds"] = None  # not always logged
    hl_records = to_handoff_records(elites)

    # Write refined_designs_hyperlens.json into the SA+GA dir.
    (sg_dir / "refined_designs_hyperlens.json").write_text(
        json.dumps(hl_records, indent=2)
    )
    print(f"[write] {sg_dir/'refined_designs_hyperlens.json'} "
          f"({len(hl_records)} records)", flush=True)

    # Merge into metrics.json (preserving sa_ga_refine block).
    metrics_path = sg_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["hyperlens"] = hl_metrics
    metrics["compute_budget"]["search_evals_actual_hyperlens"] = total_evals
    metrics["hyperlens_run_dir"] = str(hl_dir)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[write] merged hyperlens block into {metrics_path}",
          flush=True)


if __name__ == "__main__":
    main()
