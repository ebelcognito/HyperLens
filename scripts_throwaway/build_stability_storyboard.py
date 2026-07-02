#!/usr/bin/env python
"""Pick top-4 monochromatic-stability designs from the under-20µm pool,
augment with Seidel S1..S5 by re-reading the source refined JSON, and
write spotlights_stability_storyboard4.json for the render scripts."""
from __future__ import annotations
import json, glob
from pathlib import Path

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
LEAD = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/monochrome_stability_under20.json"
OUT  = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/spotlights_stability_storyboard4.json"

# Pick by leaderboard rank, skipping run-duplicates greedily for run diversity.
RANK_PICKS = [1, 2, 3, 5]   # 1-indexed ranks from the leaderboard
ROLES      = ["stable1", "stable2", "stable3", "stable4"]


def load_refined_index():
    """Build {run: [designs...]} dict from all REFSWEEP refined JSONs."""
    idx = {}
    for fp in sorted(glob.glob(str(ROOT / "results/lensmega/octuplet_REFSWEEP_*/refined_top*.json"))):
        run = Path(fp).parent.name.replace("octuplet_REFSWEEP_", "").split("_seed")[0]
        idx[run] = json.load(open(fp))["designs"]
    return idx


def match_seidels(theta, refined_designs):
    """Find the refined-design entry whose theta matches (exact float-equal)."""
    for d in refined_designs:
        if len(d["theta"]) != len(theta):
            continue
        if all(abs(a-b) < 1e-12 for a, b in zip(d["theta"], theta)):
            return d
    return None


def main():
    leaderboard = json.load(open(LEAD))
    print(f"loaded leaderboard with {len(leaderboard)} entries")
    refined_idx = load_refined_index()
    print(f"loaded refined index across {len(refined_idx)} runs")

    picks = []
    for role, rk in zip(ROLES, RANK_PICKS):
        r = leaderboard[rk - 1]
        run = r["run"]
        refined_designs = refined_idx[run]
        match = match_seidels(r["theta"], refined_designs)
        if match is None:
            print(f"  [{role}] !! could not match theta in run={run}")
            continue
        m = match["metrics"]
        sp = {
            "role": role,
            "leaderboard_rank": rk,
            "run": run,
            "rms_um": r["rms_um_stated"],
            "rms_avg_7f_587": r["rms_avg_7f_587"],
            "rms_max_7f_587": r["rms_max_7f_587"],
            "field_spread_um": r["field_spread_um"],
            "rms_curve_587":  r["rms_curve"],
            "S1": float(m["seidel_S1"]),
            "S2": float(m["seidel_S2"]),
            "S3": float(m["seidel_S3"]),
            "S4": float(m["seidel_S4"]),
            "S5": float(m["seidel_S5"]),
            "efl_mm":      float(match["efl_mm"]),
            "fnum_actual": float(match["fnum_actual"]),
            "in_saga_hull": False,   # not relevant here; uniform blue render
            "theta": r["theta"],
        }
        picks.append(sp)
        print(f"  [{role}] rank#{rk}  run={run}  rms={r['rms_um_stated']:.2f}µm  "
              f"spread={r['field_spread_um']:.2f}µm  "
              f"S2={sp['S2']*1e3:+.3f}×10⁻³  S3={sp['S3']*1e3:+.3f}×10⁻³")

    OUT.write_text(json.dumps(picks, indent=2))
    print(f"\nwrote {len(picks)} -> {OUT}")


if __name__ == "__main__":
    main()
