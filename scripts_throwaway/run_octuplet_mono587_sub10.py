#!/usr/bin/env python
"""Octuplet mono-λ (0.587 µm), ε=0.30 in normalized-θ, sub-10 µm cap,
top-30 ε-distinct designs (or fewer if the bucket saturates).
"""
from __future__ import annotations
import json, os, sys
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


def main():
    sys_name, eps, rms_cap, top_n = "octuplet", 0.30, 10.0, 30
    cfg_path = SYSTEMS[sys_name]
    prob, bounds = build_mono_problem(cfg_path)
    D = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    print(f"=== {sys_name} (mono λ={MONO_WVLN_UM} µm) ε={eps} "
          f"sub-{rms_cap:.0f}µm top-{top_n} ===", flush=True)

    pool = gather_pool(sys_name)
    print(f"  raw pool: {len(pool)}", flush=True)
    kept = reevaluate_mono(prob, pool, D)
    filt = [r for r in kept if r["mono_rms_um"] < rms_cap]
    print(f"  sub-{rms_cap:.0f}µm: {len(filt)}", flush=True)

    picks = eps_net_pick(filt, lo, hi, eps, top_n)
    print(f"  selected {len(picks)} ε-distinct designs:", flush=True)
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
