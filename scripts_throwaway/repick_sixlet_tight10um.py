#!/usr/bin/env python
"""Re-pick sixlet spotlights using a stricter max-per-wvln cap of 10 µm
(was 30 µm — let out2@29.4µm and out3 with 5.8x imbalance into the picks).
Reuses per-wvln RMS already cached in spotlight_picker_audit.json and looks
up refined_theta from refined_top500_seed42_hl.json + archive_cells.json by
matching (S2, S3).

Writes spotlights.json (backup → spotlights.before_tight10um.json) and
emits a small log of what was picked.
"""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
OUT  = ROOT / "results/figure_archive_demo"
HL_RUN = ROOT / "results/lensmega/sixlet_apsc_discovery_v2_seed42_20260506_034601"

CAP_MAX_RMS_UM = 10.0
N_HL_PICKS     = 3

audit = json.loads((OUT / "spotlight_picker_audit.json").read_text())
elig  = audit["all_eligible_outside"]                       # 47 candidates

# Match S2/S3 → theta via refined_top500 + archive_cells
refined = json.loads((HL_RUN / "refined_top500_seed42_hl.json").read_text())["designs"]
cells   = json.loads((HL_RUN / "archive_cells.json").read_text())["cells"]
fit_to_meas = {round(c["fitness"], 8): tuple(c["measures"]) for c in cells}

# (S2_rounded, S3_rounded) -> theta from refined designs
key_to_theta = {}
key_to_mean  = {}
for d in refined:
    f = round(d["raw_fitness"], 8)
    if f not in fit_to_meas: continue
    s2, s3 = fit_to_meas[f]
    key = (round(float(s2), 8), round(float(s3), 8))
    key_to_theta[key] = d["theta"]
    key_to_mean[key]  = float(d["rms_um"])

def find_theta(c):
    key = (round(c["S2"], 8), round(c["S3"], 8))
    if key in key_to_theta: return key_to_theta[key]
    # fall back to nearest neighbour
    best, bd = None, 1e9
    for k, th in key_to_theta.items():
        d = math.hypot(k[0]-c["S2"], k[1]-c["S3"])
        if d < bd: bd, best = d, th
    return best

# Filter eligible candidates by stricter cap
pool = [c for c in elig if c["max_rms_um"] <= CAP_MAX_RMS_UM]
pool.sort(key=lambda c: c["max_rms_um"])
print(f"sub-{CAP_MAX_RMS_UM}µm pool: {len(pool)} candidates")
for c in pool:
    print(f"  S2={c['S2']:+.4e}  S3={c['S3']:+.4e}  "
          f"mean={c['mean_rms_um']:5.2f}  max={c['max_rms_um']:5.2f}  "
          f"bal={c['balance_ratio']:.2f}")

# Greedy farthest-point in (S2,S3), seeded by lowest-max
picked = [pool[0]]
while len(picked) < N_HL_PICKS:
    best, bd = None, -1.0
    for c in pool:
        if c in picked: continue
        d_min = min(math.hypot(c["S2"]-p["S2"], c["S3"]-p["S3"]) for p in picked)
        if d_min > bd: bd, best = d_min, c
    if best is None: break
    picked.append(best)

print(f"\nPicked {len(picked)} HL-only spotlights:")
for i, c in enumerate(picked, 1):
    print(f"  out{i}: S2={c['S2']:+.4e}  S3={c['S3']:+.4e}  "
          f"max={c['max_rms_um']:.2f}  mean={c['mean_rms_um']:.2f}  "
          f"bal={c['balance_ratio']:.2f}")

# Keep the existing shared pick from the old spotlights.json
old_spots = json.loads((OUT / "spotlights.json").read_text())
shared = next(s for s in old_spots if s.get("in_saga_hull") is True)
print(f"\nKeeping shared pick: S2={shared['S2']:+.4e}  S3={shared['S3']:+.4e}  "
      f"mean={shared['rms_um']:.2f}  max={shared['max_per_wvln_rms_um']:.2f}")

# Build new spotlights.json
new_spots = []
for i, c in enumerate(picked, 1):
    theta = find_theta(c)
    if theta is None:
        raise RuntimeError(f"could not match theta for S2={c['S2']} S3={c['S3']}")
    new_spots.append({
        "label": f"out{i}",
        "S2": c["S2"],
        "S3": c["S3"],
        "rms_um": c["mean_rms_um"],
        "in_saga_hull": False,
        "refined_theta": theta,
        "per_wvln_rms_um": c["per_wvln_rms_um"],
        "max_per_wvln_rms_um": c["max_rms_um"],
        "balance_ratio": c["balance_ratio"],
    })
new_spots.append(shared)

# Backup and write
backup = OUT / "spotlights.before_tight10um.json"
if not backup.exists():
    backup.write_text((OUT / "spotlights.json").read_text())
    print(f"\nbackup -> {backup.name}")
(OUT / "spotlights.json").write_text(json.dumps(new_spots, indent=2))
print(f"wrote -> {OUT / 'spotlights.json'}")
