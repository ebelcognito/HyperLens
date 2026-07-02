#!/usr/bin/env python
"""Same logic as extract_mono587_top30.py but pulls top-40 sixlet and top-32
octuplet at eps=0.30 (same RMS caps).

Outputs:
  scripts_throwaway/mono587_top40_sixlet_eps030_sub10.json
  scripts_throwaway/mono587_top32_octuplet_eps030_sub20.json
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

SPEC = {
    "sixlet":   dict(epsilon=0.30, rms_cap_um=10.0, top_n=40),
    "octuplet": dict(epsilon=0.30, rms_cap_um=20.0, top_n=32),
}


def main():
    for sys_name, spec in SPEC.items():
        print(f"\n=== {sys_name}: ε={spec['epsilon']}  "
              f"RMS < {spec['rms_cap_um']:.0f} µm  top-{spec['top_n']} ===",
              flush=True)
        cfg_path = SYSTEMS[sys_name]
        prob, bounds = build_mono_problem(cfg_path)
        D = bounds.shape[0]
        lo, hi = bounds[:, 0], bounds[:, 1]

        pool = gather_pool(sys_name)
        print(f"  raw pool: {len(pool)}", flush=True)
        kept = reevaluate_mono(prob, pool, D)
        filt = [r for r in kept if r["mono_rms_um"] < spec["rms_cap_um"]]
        print(f"  sub-{spec['rms_cap_um']:.0f}µm: {len(filt)}", flush=True)

        picks = eps_net_pick(filt, lo, hi, spec["epsilon"], spec["top_n"])
        print(f"  selected {len(picks)} ε-distinct designs:", flush=True)

        out = {
            "system": sys_name,
            "wavelength_um": MONO_WVLN_UM,
            "epsilon": spec["epsilon"],
            "rms_cap_um": spec["rms_cap_um"],
            "top_n_requested": spec["top_n"],
            "n_selected": len(picks),
            "theta_dim": D,
            "theta_bounds": [[float(a), float(b)] for a, b in bounds.tolist()],
            "selector": "greedy eps-net in normalized-theta, dist/sqrt(D) > eps, sorted by mono_fitness desc",
            "designs": [
                {
                    "rank": i,
                    "run": r["run"],
                    "idx": int(r["idx"]),
                    "mono_fitness": float(r["mono_fitness"]),
                    "mono_rms_um": float(r["mono_rms_um"]),
                    "poly_refined_fitness": float(r["poly_refined_fitness"]),
                    "poly_rms_um": float(r["poly_rms_um"]),
                    "theta": [float(x) for x in r["theta"]],
                }
                for i, r in enumerate(picks, 1)
            ],
        }
        out_path = ROOT / f"scripts_throwaway/mono587_top{spec['top_n']}_{sys_name}_eps{int(spec['epsilon']*100):03d}_sub{int(spec['rms_cap_um'])}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
