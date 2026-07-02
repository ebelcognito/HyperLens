#!/usr/bin/env python3
"""Sample stability on HyperLens archive cells (not just refined elites).

The HL archive holds tens of thousands of designs across distinct
behavior-space cells, but only the elite_designs/ directory has stability
data attached. To make the manufacturability comparison fair, we
re-evaluate stability on the top-N archive cells (by fitness) using the
same perturbation procedure as runner.py (rng.randn × 0.005 ×
bounds_range, 20 samples, count fraction with fitness_delta > -|fitness|).

Output: stability_sample.json in the run dir, with per-design
pct_survived and a summary distribution.

Usage:
  python tools/sample_hl_stability.py --hl-run <hl_run_dir> --top 100
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hl-run", required=True)
    ap.add_argument("--top", type=int, default=100,
                    help="Sample top-N archive cells by fitness")
    ap.add_argument("--n-stab", type=int, default=20,
                    help="Stability perturbations per design")
    ap.add_argument("--perturb", type=float, default=0.005,
                    help="Perturbation scale × bounds_range")
    args = ap.parse_args()

    hl_dir = Path(args.hl_run).resolve()

    # Load archive
    with (hl_dir / "archive.pkl").open("rb") as f:
        a = pickle.load(f)
    cells = [(float(v.fitness), np.asarray(v.theta, dtype=np.float64))
             for v in a.cells.values() if v is not None]
    cells.sort(reverse=True, key=lambda x: x[0])
    top = cells[:args.top]
    print(f"[hl-stab] sampling {len(top)} top cells "
          f"(best fit={top[0][0]:.4f}, worst-of-top={top[-1][0]:.4f})",
          flush=True)

    # Load problem (use the run's saved config)
    cfg = load_config(str(hl_dir / "config.yaml"))
    desc = _build_descriptor(cfg)
    prob = _build_problem(cfg, desc)
    prob._ensure_backend()

    bounds = np.asarray(prob.theta_bounds, dtype=np.float64)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    bounds_range = hi - lo

    rng = np.random.RandomState(42)
    results = []
    t0 = time.time()
    for i, (fit, theta) in enumerate(top):
        deltas = []
        for _ in range(args.n_stab):
            noise = rng.randn(len(theta)) * args.perturb * bounds_range
            perturbed = np.clip(theta + noise, lo, hi)
            pr = prob.evaluate(perturbed, with_grads=False)
            deltas.append(pr.fitness - fit)
        deltas = np.array(deltas)
        pct_survived = float((deltas > -abs(fit)).mean()) * 100
        results.append({
            "rank": i,
            "fitness": fit,
            "rms_um": -fit * 1000 if fit > -1 else None,  # rough conversion
            "pct_survived": pct_survived,
            "fitness_delta_mean": float(deltas.mean()),
            "fitness_delta_std": float(deltas.std()),
        })
        if (i + 1) % 25 == 0:
            print(f"[hl-stab] {i+1}/{len(top)} done "
                  f"(wall={time.time()-t0:.1f}s)", flush=True)

    pcts = np.array([r["pct_survived"] for r in results])
    summary = {
        "n_sampled": len(results),
        "stability_distribution": {
            "mean": float(pcts.mean()),
            "median": float(np.median(pcts)),
            "min": float(pcts.min()),
            "max": float(pcts.max()),
            "count_above_50": int((pcts >= 50).sum()),
            "count_above_70": int((pcts >= 70).sum()),
            "count_above_90": int((pcts >= 90).sum()),
        },
        "perturbation_scale": args.perturb,
        "perturbations_per_design": args.n_stab,
        "fitness_range_sampled": [
            float(top[-1][0]), float(top[0][0])
        ],
    }
    out_path = hl_dir / "stability_sample.json"
    out_path.write_text(json.dumps({
        "summary": summary,
        "designs": results,
    }, indent=2))
    print(f"[done] wrote {out_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
