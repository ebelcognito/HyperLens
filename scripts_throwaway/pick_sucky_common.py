#!/usr/bin/env python
"""Find an SA+GA-shared (inside saga hull) octuplet design in [20, 30] µm
that has the WORST chromatic + field stability. Use this as the new
storyboard 'common' panel."""
from __future__ import annotations
import json, os, sys, glob, csv
from pathlib import Path
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
from multiprocessing import Pool

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

HULL = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/saga_hull.csv"
SPOT_V2 = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/spotlights_storyboard4_v2.json"
OUT_V3  = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/spotlights_storyboard4_v3.json"

RMS_LO, RMS_HI = 20.0, 30.0
N_WORKERS = int(os.environ.get("WORKERS", "8"))


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


_TEMPLATE = _SOLVER = _PC = _LP = _PP = None
def _init():
    global _TEMPLATE, _SOLVER, _PC, _LP, _PP
    import torch; torch.set_num_threads(1)
    from scripts_throwaway.fig5_extract import SYSTEMS, build_lens_template_and_solver
    _TEMPLATE, _SOLVER, _PC, _LP, _PP = build_lens_template_and_solver(SYSTEMS["octuplet"]["cfg"])


def _trace(args):
    key, theta = args
    import numpy as np
    from scripts_throwaway.fig5_extract import make_geolens
    from scripts_throwaway.render_spotlight_panels_split import per_field_per_wvln_rms_um
    try:
        lens = make_geolens(_PC, _PP, _LP, np.asarray(theta, dtype=np.float64), _TEMPLATE, _SOLVER)
        ff, wv, rms = per_field_per_wvln_rms_um(lens, _PP, n_fields=7, num_rays=256)
        rms = np.asarray(rms)
        return key, dict(
            rms_max_all = float(rms.max()),
            rms_avg_all = float(rms.mean()),
            chrom_spread_avg = float((rms.max(1)-rms.min(1)).mean()),
            ribbon_min = rms.min(1).tolist(),
            ribbon_med = np.median(rms, axis=1).tolist(),
            ribbon_max = rms.max(1).tolist(),
        )
    except Exception as e:
        return key, dict(err=str(e))


def main():
    hull = load_hull(HULL)
    files = sorted(glob.glob(str(ROOT / "results/lensmega/octuplet_REFSWEEP_*/refined_top*.json")))
    candidates = []
    for fp in files:
        run = Path(fp).parent.name.replace("octuplet_REFSWEEP_", "").split("_seed")[0]
        d = json.load(open(fp))
        for i, des in enumerate(d["designs"]):
            r = float(des["rms_um"])
            if not (RMS_LO <= r <= RMS_HI):
                continue
            m = des.get("metrics", {})
            S2 = float(m.get("seidel_S2", float("nan")))
            S3 = float(m.get("seidel_S3", float("nan")))
            if not point_in_poly((S2, S3), hull):
                continue
            candidates.append({
                "key": f"{run}#{i}",
                "run": run, "pool_idx_in_file": i,
                "rms_um_stated": r,
                "S2": S2, "S3": S3,
                "S1": float(m.get("seidel_S1", float("nan"))),
                "S4": float(m.get("seidel_S4", float("nan"))),
                "S5": float(m.get("seidel_S5", float("nan"))),
                "theta": des["theta"],
            })
    print(f"candidates: {len(candidates)} SA+GA-shared in [{RMS_LO},{RMS_HI}] µm")

    jobs = [(c["key"], c["theta"]) for c in candidates]
    by_key = {c["key"]: c for c in candidates}
    with Pool(processes=N_WORKERS, initializer=_init) as pl:
        for k, (key, m) in enumerate(pl.imap_unordered(_trace, jobs, chunksize=4)):
            if "err" in m:
                continue
            by_key[key].update(m)
            if (k+1) % 25 == 0 or (k+1) == len(jobs):
                print(f"  traced {k+1}/{len(jobs)}")

    rows = [c for c in candidates if "rms_max_all" in c]
    # rank by SUCKINESS: high rms_max_all + high chrom spread (visible ribbon)
    rows.sort(key=lambda r: -(r["rms_max_all"] + 0.5 * r["chrom_spread_avg"]))

    print(f"\nTOP 10 suckiest SA+GA-shared in [{RMS_LO},{RMS_HI}] µm:")
    print(f"  {'rk':>2}  {'stated':>6} {'rms_max':>7} {'spread':>7}  S2          S3        run")
    for k, r in enumerate(rows[:10]):
        print(f"  {k+1:>2}  {r['rms_um_stated']:6.2f} {r['rms_max_all']:7.2f} "
              f"{r['chrom_spread_avg']:7.2f}  {r['S2']*1e3:+7.3f}×10⁻³ {r['S3']*1e3:+7.3f}×10⁻³  {r['run']}")

    pick = rows[0]
    print(f"\nPICK: {pick['key']}  rms_max={pick['rms_max_all']:.2f}  spread={pick['chrom_spread_avg']:.2f}")

    # Splice into v2 spotlights → v3 (keep hl1/hl2/hl3, replace common)
    v2 = json.load(open(SPOT_V2))
    out = []
    for sp in v2:
        if sp["role"] != "common":
            out.append(sp)
    out.append({
        "role": "common",
        "rms_um": pick["rms_um_stated"],
        "S2": pick["S2"], "S3": pick["S3"],
        "rms_max_all": pick["rms_max_all"],
        "rms_avg_all": pick["rms_avg_all"],
        "chrom_spread_avg": pick["chrom_spread_avg"],
        "in_saga_hull": True,
        "theta": pick["theta"],
        "_run": pick["run"], "_key": pick["key"],
    })
    json.dump(out, open(OUT_V3, "w"), indent=2)
    print(f"\nwrote v3 spotlights -> {OUT_V3}")


if __name__ == "__main__":
    main()
