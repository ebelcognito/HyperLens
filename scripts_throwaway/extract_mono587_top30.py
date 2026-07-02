#!/usr/bin/env python
"""Extract the top-30 ε-distinct mono-λ (0.587 µm) designs for sixlet and
octuplet under the same greedy normalized-θ ε-net selector used by
the top9/top17 figures, with these per-system filters:

  sixlet   : ε=0.30, mono RMS < 10 µm
  octuplet : ε=0.30, mono RMS < 20 µm

Outputs one JSON per system with full θ, mono fitness, mono RMS,
poly RMS, and provenance.
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

from scripts_throwaway.count_mono587_diverse import (
    build_mono_problem, gather_pool, reevaluate_mono, SYSTEMS, MONO_WVLN_UM,
)

# (system, epsilon, rms_cap_um, top_n)
SPEC = {
    "sixlet":   dict(epsilon=0.30, rms_cap_um=10.0, top_n=30),
    "octuplet": dict(epsilon=0.30, rms_cap_um=20.0, top_n=30),
}


def eps_net_pick(records, lo, hi, eps, top_n):
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    sqrtD = float(np.sqrt(len(lo)))
    sorted_recs = sorted(records, key=lambda r: -r["mono_fitness"])
    selected = []
    selected_u = []
    for rec in sorted_recs:
        u = (rec["theta"] - lo) / span
        ok = True
        for su in selected_u:
            if np.linalg.norm(u - su) / sqrtD <= eps:
                ok = False
                break
        if ok:
            selected.append(rec)
            selected_u.append(u)
            if len(selected) >= top_n:
                break
    return selected


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
        for i, r in enumerate(picks, 1):
            print(f"   #{i:>2}  mono_RMS={r['mono_rms_um']:6.2f}µm  "
                  f"mono_fit={r['mono_fitness']:+.5f}  "
                  f"poly_RMS={r['poly_rms_um']:6.2f}µm  "
                  f"src={r['run']}#{r['idx']}", flush=True)

        # Serialize.
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
