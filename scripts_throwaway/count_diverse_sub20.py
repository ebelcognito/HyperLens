#!/usr/bin/env python
"""For all sub-20µm octuplet designs across every HL run, count
geometrically diverse (eps-net @ eps thresholds) designs using the
sqrt(D)-normalized convention from fig5_extract.py."""
from __future__ import annotations
import json, glob, sys, yaml
from pathlib import Path
import numpy as np

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
from scripts_throwaway.fig5_extract import SYSTEMS, load_bounds, norm_theta, eps_net_select

RMS_CAP = 20.0
EPS_LIST = [0.20, 0.25, 0.30]

bounds = load_bounds(SYSTEMS["octuplet"]["cfg"])
D = bounds.shape[0]
print(f"octuplet param dim D = {D}, bounds shape = {bounds.shape}")

# gather every sub-20µm refined design across all runs
pool = []
for fp in sorted(glob.glob(str(ROOT / "results/lensmega/octuplet*/refined_top*.json"))):
    run = Path(fp).parent.name
    d = json.load(open(fp))
    for i, des in enumerate(d["designs"]):
        r = float(des.get("rms_um", float("nan")))
        if r >= RMS_CAP:
            continue
        theta = np.asarray(des["theta"], dtype=np.float64)
        # Pad/clip if dimensions differ (T_tighter_bd / W_wider_bd may use same dim though)
        if len(theta) != D:
            print(f"  SKIP {run}#{i}: theta dim {len(theta)} != {D}")
            continue
        pool.append({
            "run": run, "idx": i, "rms_um": r,
            "theta": theta,
            "theta_norm": norm_theta(theta, bounds),
            "refined_fitness": float(des.get("refined_fitness", 0.0)),
        })

# Sort by best fitness first so the eps-net keeps the strongest representatives
pool.sort(key=lambda r: -r["refined_fitness"])
print(f"\nTotal sub-{RMS_CAP:.0f}µm designs: {len(pool)}\n")

for eps in EPS_LIST:
    keep = eps_net_select(pool, eps, D)
    print(f"  eps = {eps:.2f}  →  {len(keep)} ε-distinct designs")

# Per-eps detail: list runs of accepted reps
print()
for eps in EPS_LIST:
    keep = eps_net_select(pool, eps, D)
    print(f"\n--- eps = {eps:.2f} ({len(keep)} reps) ---")
    for k, idx in enumerate(keep[:30]):
        r = pool[idx]
        print(f"  {k+1:2d}. {r['rms_um']:6.2f}µm  fit={r['refined_fitness']:+.3f}  {r['run']}#{r['idx']}")
    if len(keep) > 30:
        print(f"  ... and {len(keep)-30} more")
