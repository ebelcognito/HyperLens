#!/usr/bin/env python
"""Count ε-distinct sixlet & octuplet designs under a mono-λ (0.587 µm)
objective. Same ε-net selector as the polychromatic top9/top17 figures
(normalized-θ greedy farthest-first with dist/√D > ε), uncapped.

Pool: all refined_top*.json files under results/lensmega/{sys}*/.
Each design is re-traced at λ=0.587 µm to get a mono fitness/RMS;
designs whose θ-dim doesn't match the system's canonical config are
skipped.
"""
from __future__ import annotations
import json, glob, os, sys, time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem

SYSTEMS = {
    "sixlet": ROOT / "experiments/gradient_alignment/sixlet_apsc_champion_config.yaml",
    "octuplet": ROOT / "experiments/gradient_alignment/octuplet_apsc_champion_config.yaml",
    "triplet": ROOT / "results/lensmega/cooke_triplet_materials_modernstack_seed42_20260506_033503/config.yaml",
}

# Per-system glob prefix for refined_top*.json discovery (triplet has two
# disjoint families — apsc_triplet has no sub-20µm designs, so the canonical
# Cooke triplet family is the one that gives a meaningful mono-λ pool).
POOL_GLOB_PREFIXES = {
    "sixlet": ("sixlet",),
    "octuplet": ("octuplet",),
    "triplet": ("cooke_triplet",),
}
MONO_WVLN_UM = 0.587
EPS_LIST = [0.30, 0.20]
RMS_CAP_UM = 20.0


def build_mono_problem(cfg_path: Path):
    cfg = load_config(str(cfg_path))
    # Override to single-λ.
    cfg.problem.fitness_wavelengths_um = [MONO_WVLN_UM]
    desc = _build_descriptor(cfg)
    prob = _build_problem(cfg, desc)
    bounds = np.asarray(cfg.problem.theta_bounds, dtype=np.float64)
    return prob, bounds


def gather_pool(system: str) -> list[dict]:
    out = []
    prefixes = POOL_GLOB_PREFIXES.get(system, (system,))
    files = []
    for pref in prefixes:
        files.extend(sorted(glob.glob(str(ROOT / f"results/lensmega/{pref}*/refined_top*.json"))))
    for fp in files:
        run = Path(fp).parent.name
        try:
            data = json.load(open(fp))
        except Exception as e:
            print(f"  skip {fp}: {e}", flush=True)
            continue
        for i, des in enumerate(data.get("designs", [])):
            theta = np.asarray(des["theta"], dtype=np.float64).reshape(-1)
            out.append({
                "run": run,
                "idx": i,
                "theta": theta,
                "poly_rms_um": float(des.get("rms_um", float("nan"))),
                "poly_refined_fitness": float(des.get("refined_fitness", 0.0)),
            })
    return out


def reevaluate_mono(prob, pool: list[dict], D: int) -> list[dict]:
    kept = []
    n_dim_mismatch = n_invalid = 0
    t0 = time.time()
    for k, rec in enumerate(pool):
        if rec["theta"].shape[0] != D:
            n_dim_mismatch += 1
            continue
        try:
            r = prob.evaluate(rec["theta"], with_grads=False)
            fit = float(r.fitness)
            metrics = r.metrics or {}
            rms = float(metrics.get("rms_spot", float("nan"))) * 1000.0  # mm→µm
        except Exception:
            n_invalid += 1
            continue
        # Filter out the catastrophic-rejection sentinel (~-1e9).
        if not np.isfinite(fit) or fit < -1e6:
            n_invalid += 1
            continue
        rec2 = dict(rec)
        rec2["mono_fitness"] = fit
        rec2["mono_rms_um"] = rms
        kept.append(rec2)
        if (k + 1) % 200 == 0:
            print(f"    [{k+1}/{len(pool)}] kept={len(kept)} "
                  f"dim_mm={n_dim_mismatch} invalid={n_invalid} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    print(f"  re-eval done: kept={len(kept)} dim_mm={n_dim_mismatch} "
          f"invalid={n_invalid} in {time.time()-t0:.1f}s", flush=True)
    return kept


def eps_net(records: list[dict], lo: np.ndarray, hi: np.ndarray,
            eps: float) -> int:
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    sqrtD = float(np.sqrt(len(lo)))
    sorted_recs = sorted(records, key=lambda r: -r["mono_fitness"])
    selected_u = []
    count = 0
    for rec in sorted_recs:
        u = (rec["theta"] - lo) / span
        ok = True
        for su in selected_u:
            if np.linalg.norm(u - su) / sqrtD <= eps:
                ok = False; break
        if ok:
            selected_u.append(u)
            count += 1
    return count


def main():
    out_path = ROOT / "scripts_throwaway/count_mono587_diverse_results.json"
    summary = {}
    for sys_name, cfg_path in SYSTEMS.items():
        print(f"\n=== {sys_name} (mono λ={MONO_WVLN_UM} µm) ===", flush=True)
        prob, bounds = build_mono_problem(cfg_path)
        D = bounds.shape[0]
        lo, hi = bounds[:, 0], bounds[:, 1]
        print(f"  canonical θ-dim={D}", flush=True)

        pool = gather_pool(sys_name)
        print(f"  raw pool size: {len(pool)}", flush=True)

        kept = reevaluate_mono(prob, pool, D)
        sub20 = [r for r in kept if r["mono_rms_um"] < RMS_CAP_UM]
        sub10 = [r for r in kept if r["mono_rms_um"] < 10.0]
        print(f"  mono-λ pool: total={len(kept)}  "
              f"sub-{RMS_CAP_UM:.0f}µm={len(sub20)}  "
              f"sub-10µm={len(sub10)}", flush=True)

        sys_out = {"theta_dim": D, "pool_total": len(kept),
                   "pool_sub20um": len(sub20), "pool_sub10um": len(sub10),
                   "counts": {}}
        for eps in EPS_LIST:
            c_all  = eps_net(kept, lo, hi, eps)
            c_s20  = eps_net(sub20, lo, hi, eps)
            c_s10  = eps_net(sub10, lo, hi, eps)
            print(f"  ε={eps:.2f}:  full={c_all:4d}   "
                  f"sub-{RMS_CAP_UM:.0f}µm={c_s20:4d}   sub-10µm={c_s10:4d}",
                  flush=True)
            sys_out["counts"][f"{eps:.2f}"] = {
                "full_pool": c_all, "sub20um": c_s20, "sub10um": c_s10,
            }
        summary[sys_name] = sys_out

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
