#!/usr/bin/env python
"""Across ALL 13 octuplet REFSWEEP runs (~3837 refined designs),
compute signed distance from each design's S2-S3 to the SA+GA convex hull.
Print the top-30 farthest-outside-hull designs."""
from __future__ import annotations
import json, glob, csv
from pathlib import Path
import numpy as np

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
HULL = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/saga_hull.csv"
OUT  = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/all_designs_hull_distance.json"

RMS_CAP = 30.0  # only consider <30 µm so we don't recommend obvious garbage


def load_hull(path):
    pts = []
    with open(path) as f:
        for row in csv.DictReader(f):
            pts.append((float(row["S2"]), float(row["S3"])))
    pts = np.array(pts)
    cx, cy = pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(pts[:,1]-cy, pts[:,0]-cx))]


def point_in_poly(p, poly):
    x, y = p; n = len(poly); inside = False; j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def dist_point_segment(p, a, b):
    ap = p - a; ab = b - a
    t = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-30), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def signed_hull_distance(p, hull):
    n = len(hull)
    d = min(dist_point_segment(p, hull[i], hull[(i+1)%n]) for i in range(n))
    return -d if point_in_poly(p, hull) else d


def main():
    hull = load_hull(HULL)
    rows = []
    files = sorted(glob.glob(str(
        ROOT / "results/lensmega/octuplet_REFSWEEP_*/refined_top*.json")))
    for fp in files:
        run = Path(fp).parent.name.replace("octuplet_REFSWEEP_", "").split("_seed")[0]
        d = json.load(open(fp))
        for i, des in enumerate(d["designs"]):
            r = float(des.get("rms_um", float("nan")))
            if r >= RMS_CAP:
                continue
            m = des.get("metrics", {})
            S2 = float(m.get("seidel_S2", float("nan")))
            S3 = float(m.get("seidel_S3", float("nan")))
            dist = signed_hull_distance(np.array([S2, S3]), hull)
            rows.append({
                "key": f"{run}#{i}",
                "run": run, "pool_idx_in_file": i,
                "rms_um_stated": r,
                "S2": S2, "S3": S3,
                "saga_dist_signed_S2S3": dist,
                "in_saga_hull": dist < 0,
                "theta": des["theta"],
            })
    print(f"scanned {len(rows)} designs (rms<{RMS_CAP}µm) across {len(files)} runs")

    outside = [r for r in rows if not r["in_saga_hull"]]
    print(f"outside hull: {len(outside)}")
    outside.sort(key=lambda r: -r["saga_dist_signed_S2S3"])

    OUT.write_text(json.dumps(rows, indent=1))
    print(f"saved -> {OUT}")

    print(f"\nTOP 30 farthest-outside-hull (rms<{RMS_CAP}µm):")
    print(f"{'rk':>3}  {'hull_dist×10⁻³':>14}  {'S2×10⁻³':>9} {'S3×10⁻³':>9}  {'stated rms':>10}  run / idx")
    for k, r in enumerate(outside[:30]):
        print(f"{k+1:>3}  {r['saga_dist_signed_S2S3']*1e3:>+14.3f}  "
              f"{r['S2']*1e3:>+9.3f} {r['S3']*1e3:>+9.3f}  "
              f"{r['rms_um_stated']:>10.2f}  {r['run']} #{r['pool_idx_in_file']}")


if __name__ == "__main__":
    main()
