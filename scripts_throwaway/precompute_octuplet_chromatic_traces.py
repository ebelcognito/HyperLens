#!/usr/bin/env python
"""Trace EVERY refined octuplet design across all 13 REFSWEEP runs
(3837 designs total) at 3 wavelengths × 7 fields. Cache the per-field
per-wavelength RMS arrays + summary stats so any downstream pick logic
can run instantly against this cache."""
from __future__ import annotations
import json, os, sys, glob
from pathlib import Path
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
from multiprocessing import Pool

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/all_chromatic_traces.json"
N_FIELDS = 7
N_RAYS   = 256
N_WORKERS = int(os.environ.get("PRECOMPUTE_WORKERS", "60"))


_TEMPLATE = _SOLVER = _PROB_CFG = _LENS_PATH = _PPROB = None


def _init_worker():
    global _TEMPLATE, _SOLVER, _PROB_CFG, _LENS_PATH, _PPROB
    import torch; torch.set_num_threads(1)
    from scripts_throwaway.fig5_extract import SYSTEMS, build_lens_template_and_solver
    _TEMPLATE, _SOLVER, _PROB_CFG, _LENS_PATH, _PPROB = build_lens_template_and_solver(
        SYSTEMS["octuplet"]["cfg"])


def _trace_one(args):
    key, theta_list = args
    import numpy as np
    from scripts_throwaway.fig5_extract import make_geolens
    from scripts_throwaway.render_spotlight_panels_split import per_field_per_wvln_rms_um
    theta = np.asarray(theta_list, dtype=np.float64)
    try:
        lens = make_geolens(_PROB_CFG, _PPROB, _LENS_PATH, theta, _TEMPLATE, _SOLVER)
        ff, wv, rms = per_field_per_wvln_rms_um(
            lens, _PPROB, n_fields=N_FIELDS, num_rays=N_RAYS,
        )
        rms = np.asarray(rms)
        return key, dict(
            field_fracs = ff.tolist(),
            wvlns       = list(wv),
            rms_arr     = rms.tolist(),
            rms_max_all = float(rms.max()),
            rms_avg_all = float(rms.mean()),
            rms_max_med_field = float(np.median(rms, axis=1).max()),
            chrom_spread_avg  = float((rms.max(1) - rms.min(1)).mean()),
            chrom_spread_max  = float((rms.max(1) - rms.min(1)).max()),
            ribbon_min = rms.min(1).tolist(),
            ribbon_med = np.median(rms, axis=1).tolist(),
            ribbon_max = rms.max(1).tolist(),
            err = None,
        )
    except Exception as e:
        return key, dict(err=str(e))


def main():
    files = sorted(glob.glob(str(
        ROOT / "results/lensmega/octuplet_REFSWEEP_*/refined_top*.json")))
    print(f"found {len(files)} run files, workers={N_WORKERS}")

    # Build (key, theta, meta) for every design.
    all_meta = []
    jobs = []
    for fp in files:
        run = Path(fp).parent.name.replace("octuplet_REFSWEEP_", "").split("_seed")[0]
        d = json.load(open(fp))
        for i, des in enumerate(d["designs"]):
            key = f"{run}#{i}"
            m = des.get("metrics", {})
            all_meta.append({
                "key": key,
                "run": run,
                "pool_idx_in_file": i,
                "pool_idx": des.get("pool_idx"),
                "rms_um_stated": float(des.get("rms_um", float("nan"))),
                "raw_rms_um":    float(des.get("raw_rms_um", float("nan"))),
                "refined_fitness": float(des.get("refined_fitness", float("nan"))),
                "S1": float(m.get("seidel_S1", float("nan"))),
                "S2": float(m.get("seidel_S2", float("nan"))),
                "S3": float(m.get("seidel_S3", float("nan"))),
                "S4": float(m.get("seidel_S4", float("nan"))),
                "S5": float(m.get("seidel_S5", float("nan"))),
                "efl_mm":      float(des.get("efl_mm", float("nan"))),
                "fnum_actual": float(des.get("fnum_actual", float("nan"))),
                "theta": des["theta"],
            })
            jobs.append((key, des["theta"]))
    print(f"total designs: {len(jobs)}")
    by_key = {m["key"]: m for m in all_meta}

    done = 0
    failed = 0
    with Pool(processes=N_WORKERS, initializer=_init_worker) as pl:
        for k, (key, trace) in enumerate(pl.imap_unordered(_trace_one, jobs, chunksize=8)):
            if trace.get("err"):
                failed += 1
            else:
                by_key[key].update(trace)
                done += 1
            if (k+1) % 200 == 0 or (k+1) == len(jobs):
                print(f"  [{k+1:5d}/{len(jobs)}]  ok={done} failed={failed}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_meta, open(OUT, "w"))
    sz_mb = OUT.stat().st_size / (1024*1024)
    print(f"\nwrote {len(all_meta)} entries ({sz_mb:.1f} MB) -> {OUT}")
    print(f"ok={done}  failed={failed}")


if __name__ == "__main__":
    main()
