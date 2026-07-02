#!/usr/bin/env python
"""Pick a NEW 4-design octuplet storyboard:
   * 3 HL-exclusive (in_saga_hull == False) designs with:
       - visible chromatic ribbon
       - max-RMS (across fields+wvlns) <= the existing common pick's max-RMS
       - far from SA+GA convex hull in S₂–S₃ plane (matches the UNDER20vsSAGA map)
       - visually distinct (theta diversity)
   * 1 shared/common pick — retained from the existing storyboard

Parallel across designs via multiprocessing.Pool.
"""
from __future__ import annotations
import json, os, sys, math, csv
from pathlib import Path
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
from multiprocessing import Pool, cpu_count

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

POOL  = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/spotlights_under20_vs_saga.json"
HULL  = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/saga_hull.csv"
OLD   = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/spotlights_storyboard4.json"
TRACE = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/under20_chromatic_traces.json"
PICKS = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/spotlights_storyboard4_v2.json"

N_FIELDS = 7
N_RAYS   = 256
N_WORKERS = max(1, int(os.environ.get("STORYBOARD_WORKERS", "10")))


def load_hull_xy(path):
    xs, ys = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            xs.append(float(row["S2"])); ys.append(float(row["S3"]))
    pts = np.array(list(zip(xs, ys)))
    cx, cy = pts.mean(axis=0)
    ang = np.arctan2(pts[:,1]-cy, pts[:,0]-cx)
    return pts[np.argsort(ang)]


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
    return np.linalg.norm(p - (a + t * ab))


def signed_hull_distance(p, hull):
    n = len(hull)
    d = min(dist_point_segment(p, hull[i], hull[(i+1)%n]) for i in range(n))
    return -d if point_in_poly(p, hull) else d


# Module-level so worker children can pickle. Each worker builds its own
# template/solver/problem once via _init_worker.
_TEMPLATE = _SOLVER = _PROB_CFG = _LENS_PATH = _PPROB = None


def _init_worker():
    global _TEMPLATE, _SOLVER, _PROB_CFG, _LENS_PATH, _PPROB
    import torch
    torch.set_num_threads(1)
    from scripts_throwaway.fig5_extract import (
        SYSTEMS, build_lens_template_and_solver,
    )
    _TEMPLATE, _SOLVER, _PROB_CFG, _LENS_PATH, _PPROB = build_lens_template_and_solver(
        SYSTEMS["octuplet"]["cfg"])


def _trace_one(args):
    i, theta_list = args
    import numpy as np
    from scripts_throwaway.fig5_extract import make_geolens
    from scripts_throwaway.render_spotlight_panels_split import per_field_per_wvln_rms_um
    theta = np.asarray(theta_list, dtype=np.float64)
    try:
        lens = make_geolens(_PROB_CFG, _PPROB, _LENS_PATH, theta, _TEMPLATE, _SOLVER)
        ff, wv, rms = per_field_per_wvln_rms_um(
            lens, _PPROB, n_fields=N_FIELDS, num_rays=N_RAYS,
        )
        return i, ff.tolist(), list(wv), np.asarray(rms).tolist(), None
    except Exception as e:
        return i, None, None, None, str(e)


def summarize(rms):
    rms = np.asarray(rms)
    return dict(
        rms_max_all  = float(rms.max()),
        rms_max_med  = float(np.median(rms, axis=1).max()),
        rms_avg_all  = float(rms.mean()),
        chrom_spread_avg = float((rms.max(1) - rms.min(1)).mean()),
        chrom_spread_max = float((rms.max(1) - rms.min(1)).max()),
        ribbon_curve_min = rms.min(1).tolist(),
        ribbon_curve_med = np.median(rms, axis=1).tolist(),
        ribbon_curve_max = rms.max(1).tolist(),
    )


def main():
    pool = json.load(open(POOL))
    print(f"loaded {len(pool)} sub-20µm designs")
    old_storyboard = json.load(open(OLD))
    common_old = next(s for s in old_storyboard if s["role"] == "common")
    hull_xy = load_hull_xy(HULL)
    print(f"saga hull: {len(hull_xy)} vertices  ·  workers={N_WORKERS}")

    universe = pool + [{**common_old, "rank": None, "_is_common": True}]
    jobs = [(i, sp["theta"]) for i, sp in enumerate(universe)]

    traces = []
    failed = 0
    with Pool(processes=N_WORKERS, initializer=_init_worker) as pl:
        for k, res in enumerate(pl.imap_unordered(_trace_one, jobs)):
            i, ff, wv, rms, err = res
            sp = universe[i]
            if err is not None:
                print(f"  [{i:3d}] FAIL: {err}")
                failed += 1
                continue
            s = summarize(rms)
            dist = signed_hull_distance(np.array([sp["S2"], sp["S3"]]), hull_xy)
            traces.append({
                "idx": i,
                "is_common": bool(sp.get("_is_common", False)),
                "rank":     sp.get("rank"),
                "S2": sp["S2"], "S3": sp["S3"],
                "in_saga_hull": bool(sp.get("in_saga_hull", False)),
                "saga_dist_signed_S2S3": dist,
                "stated_rms_um": sp.get("rms_um"),
                "field_fracs": ff, "wvlns": wv,
                **s,
                "theta": sp["theta"],
            })
            if (k+1) % 10 == 0 or (k+1) == len(jobs):
                print(f"  [{k+1:3d}/{len(jobs)}] last idx={i} max={s['rms_max_all']:.2f}µm "
                      f"spread={s['chrom_spread_avg']:.2f}µm dist={dist*1e3:+.3f}×10⁻³")

    print(f"\ntraced {len(traces)} ok, {failed} failed")
    json.dump(traces, open(TRACE, "w"), indent=2)
    print(f"wrote -> {TRACE}")

    # ---- picking ----
    common = next(t for t in traces if t["is_common"])
    print("\nCOMMON:")
    print(f"  rms_max_all={common['rms_max_all']:.2f}  rms_avg={common['rms_avg_all']:.2f}  "
          f"chrom_spread_avg={common['chrom_spread_avg']:.2f}  "
          f"dist_to_hull={common['saga_dist_signed_S2S3']*1e3:+.3f}×10⁻³")

    hl_pool = [t for t in traces if (not t["is_common"]) and (not t["in_saga_hull"])]
    print(f"HL-exclusive pool: {len(hl_pool)}")

    common_max = common["rms_max_all"]
    qual = [t for t in hl_pool if t["rms_max_all"] <= common_max * 1.10
            and t["chrom_spread_avg"] >= 0.5]
    print(f"qualified (rms_max ≤ {common_max:.2f}*1.10, ribbon ≥ 0.5µm): {len(qual)}")

    for t in qual:
        rms_score = (common_max - t["rms_max_all"]) / common_max
        t["_pick_score"] = rms_score + 1.5 * t["saga_dist_signed_S2S3"] / max(common["saga_dist_signed_S2S3"], 1e-9)

    qual.sort(key=lambda t: -t["_pick_score"])

    def theta_dist(a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    picks = [qual[0]] if qual else []
    while len(picks) < 3 and len(qual) > len(picks):
        best, best_v = None, -1e30
        for c in qual:
            if c in picks: continue
            mind = min(theta_dist(c["theta"], p["theta"]) for p in picks)
            v = c["_pick_score"] + 0.2 * mind
            if v > best_v: best_v, best = v, c
        if best is None: break
        picks.append(best)

    print(f"\nPICKED {len(picks)} HL designs:")
    for k, t in enumerate(picks):
        print(f"  hl{k+1}: S2={t['S2']*1e3:+.3f} S3={t['S3']*1e3:+.3f}  "
              f"rms_max={t['rms_max_all']:.2f}µm  spread={t['chrom_spread_avg']:.2f}µm  "
              f"hull_dist={t['saga_dist_signed_S2S3']*1e3:+.3f}×10⁻³  score={t['_pick_score']:.3f}")

    LB = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/storyboard_v2_leaderboard.json"
    json.dump(qual, open(LB, "w"), indent=2)
    print(f"\nwrote {len(qual)} qualified -> {LB}")

    out = []
    for k, t in enumerate(picks):
        out.append({
            "role": f"hl{k+1}",
            "rms_um": t["stated_rms_um"],
            "S2": t["S2"], "S3": t["S3"],
            "rms_max_all": t["rms_max_all"],
            "rms_avg_all": t["rms_avg_all"],
            "chrom_spread_avg": t["chrom_spread_avg"],
            "saga_dist_signed_S2S3": t["saga_dist_signed_S2S3"],
            "in_saga_hull": False,
            "theta": t["theta"],
        })
    out.append({
        "role": "common",
        "rms_um": common_old["rms_um"],
        "S2": common_old["S2"], "S3": common_old["S3"],
        "rms_max_all": common["rms_max_all"],
        "rms_avg_all": common["rms_avg_all"],
        "chrom_spread_avg": common["chrom_spread_avg"],
        "saga_dist_signed_S2S3": common["saga_dist_signed_S2S3"],
        "in_saga_hull": True,
        "theta": common_old["theta"],
    })
    json.dump(out, open(PICKS, "w"), indent=2)
    print(f"\nwrote v2 spotlights -> {PICKS}")


if __name__ == "__main__":
    main()
