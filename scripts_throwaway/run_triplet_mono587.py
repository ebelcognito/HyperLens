#!/usr/bin/env python
"""Triplet mono-λ (0.587 µm) ε-net analysis + top-30 extraction.

Mirrors the sixlet/octuplet pipeline. Pool: cooke_triplet runs only
(apsc_triplet runs have no sub-20µm poly designs).
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

from scripts_throwaway.count_mono587_diverse import (
    build_mono_problem, gather_pool, reevaluate_mono, SYSTEMS, MONO_WVLN_UM,
)
from scripts_throwaway.extract_mono587_top30 import eps_net_pick


def eps_net_count(records, lo, hi, eps):
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    sqrtD = float(np.sqrt(len(lo)))
    sorted_recs = sorted(records, key=lambda r: -r["mono_fitness"])
    selected_u = []
    n = 0
    for rec in sorted_recs:
        u = (rec["theta"] - lo) / span
        ok = True
        for su in selected_u:
            if np.linalg.norm(u - su) / sqrtD <= eps:
                ok = False; break
        if ok:
            selected_u.append(u); n += 1
    return n


def main():
    sys_name = "triplet"
    cfg_path = SYSTEMS[sys_name]
    prob, bounds = build_mono_problem(cfg_path)
    D = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    print(f"=== {sys_name} (mono λ={MONO_WVLN_UM} µm, θ-dim={D}) ===", flush=True)

    pool = gather_pool(sys_name)
    print(f"  raw pool: {len(pool)}", flush=True)
    kept = reevaluate_mono(prob, pool, D)
    sub20 = [r for r in kept if r["mono_rms_um"] < 20.0]
    sub10 = [r for r in kept if r["mono_rms_um"] < 10.0]
    sub5  = [r for r in kept if r["mono_rms_um"] <  5.0]
    print(f"  mono-λ pool: total={len(kept)}  sub-20µm={len(sub20)}  "
          f"sub-10µm={len(sub10)}  sub-5µm={len(sub5)}", flush=True)

    # Counts table.
    print("\n  -- ε-distinct counts --", flush=True)
    print(f"  {'ε':>5}  {'full':>5}  {'sub20':>5}  {'sub10':>5}  {'sub5':>5}",
          flush=True)
    for eps in (0.30, 0.20):
        n_all = eps_net_count(kept,  lo, hi, eps)
        n_20  = eps_net_count(sub20, lo, hi, eps)
        n_10  = eps_net_count(sub10, lo, hi, eps)
        n_5   = eps_net_count(sub5,  lo, hi, eps)
        print(f"  {eps:>5.2f}  {n_all:>5d}  {n_20:>5d}  {n_10:>5d}  {n_5:>5d}",
              flush=True)

    # Top-N ε-distinct at ε=0.30, sub-10µm.
    eps, rms_cap, top_n = 0.30, 10.0, 32
    print(f"\n  -- top-{top_n} ε={eps} sub-{rms_cap:.0f}µm --", flush=True)
    filt = [r for r in kept if r["mono_rms_um"] < rms_cap]
    picks = eps_net_pick(filt, lo, hi, eps, top_n)
    for i, r in enumerate(picks, 1):
        print(f"   #{i:>2}  mono_RMS={r['mono_rms_um']:6.2f}µm  "
              f"mono_fit={r['mono_fitness']:+.5f}  "
              f"poly_RMS={r['poly_rms_um']:6.2f}µm  "
              f"src={r['run']}#{r['idx']}", flush=True)

    out = {
        "system": sys_name,
        "wavelength_um": MONO_WVLN_UM,
        "epsilon": eps,
        "rms_cap_um": rms_cap,
        "top_n_requested": top_n,
        "n_selected": len(picks),
        "theta_dim": D,
        "theta_bounds": [[float(a), float(b)] for a, b in bounds.tolist()],
        "selector": "greedy eps-net in normalized-theta, dist/sqrt(D) > eps, sorted by mono_fitness desc",
        "designs": [
            {
                "rank": i,
                "run": r["run"], "idx": int(r["idx"]),
                "mono_fitness": float(r["mono_fitness"]),
                "mono_rms_um": float(r["mono_rms_um"]),
                "poly_refined_fitness": float(r["poly_refined_fitness"]),
                "poly_rms_um": float(r["poly_rms_um"]),
                "theta": [float(x) for x in r["theta"]],
            }
            for i, r in enumerate(picks, 1)
        ],
    }
    out_path = ROOT / f"scripts_throwaway/mono587_top{top_n}_{sys_name}_eps{int(eps*100):03d}_sub{int(rms_cap)}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
