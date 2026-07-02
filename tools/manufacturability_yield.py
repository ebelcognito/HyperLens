#!/usr/bin/env python3
"""Monte-Carlo manufacturability yield: from refined-design pool, count
how many survive realistic manufacturing tolerances.

For each design, sample N tolerance perturbations of the curvatures and
gaps (multiplied by tolerance scale × bounds_range), re-evaluate, and
classify the design as "manufacturable" if its perturbed RMS distribution
satisfies:

   median(perturbed_RMS) <= 1.5 × nominal_RMS

This proxies the question "if a fab makes 100 units of this design with
0.5–1% tolerance, do most of them work?". Designs sitting in needle
basins fail (one perturbation collapses fitness); designs in wide basins
pass.

Workers parallelize across designs; each design is independent.

Usage:
  python tools/manufacturability_yield.py \\
      --refined-json <path> \\
      --problem-config <path> \\
      --tolerance 0.01 \\
      --n-samples 100 \\
      --workers 30 \\
      --rms-mult 1.5 \\
      --out <yield_json>
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_WORKER_STATE: Dict[str, Any] = {}


def _worker_init(cfg_dict: Dict[str, Any]) -> None:
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    from optical_rl.qd.lensmega.config.schema import RunConfig
    from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem
    cfg = RunConfig.model_validate(cfg_dict)
    cfg.io.device = "cpu"
    desc = _build_descriptor(cfg)
    prob = _build_problem(cfg, desc)
    prob._ensure_backend()
    _WORKER_STATE["prob"] = prob
    bounds = np.asarray(cfg.problem.theta_bounds, dtype=np.float64)
    _WORKER_STATE["lo"] = bounds[:, 0]
    _WORKER_STATE["hi"] = bounds[:, 1]
    _WORKER_STATE["bounds_range"] = bounds[:, 1] - bounds[:, 0]


def _worker_yield(args):
    idx, theta, nominal_rms_um, tolerance, n_samples, seed = args
    prob = _WORKER_STATE["prob"]
    lo = _WORKER_STATE["lo"]; hi = _WORKER_STATE["hi"]
    bounds_range = _WORKER_STATE["bounds_range"]
    theta = np.asarray(theta, dtype=np.float64)

    rng = np.random.RandomState(seed)
    rms_vals = []
    for _ in range(n_samples):
        # Tolerance perturbation: 0.5% of bounds_range per parameter, gaussian
        # (mirrors typical surface-tolerance Monte Carlo at 1-2σ).
        noise = rng.randn(len(theta)) * tolerance * bounds_range
        perturbed = np.clip(theta + noise, lo, hi)
        try:
            ev = prob.evaluate(perturbed, with_grads=False)
            rms_um = float(ev.metrics.get("rms_spot", abs(ev.fitness))) * 1000
        except Exception:
            rms_um = float("inf")
        rms_vals.append(rms_um)
    rms_vals = np.array(rms_vals)
    return {
        "idx": idx,
        "nominal_rms_um": nominal_rms_um,
        "perturbed_rms_um": {
            "median": float(np.median(rms_vals)),
            "mean": float(np.mean(rms_vals)),
            "p10": float(np.percentile(rms_vals, 10)),
            "p90": float(np.percentile(rms_vals, 90)),
            "min": float(np.min(rms_vals)),
            "max": float(np.max(rms_vals)),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined-json", required=True,
                    help="Path to refined_above_-X_*.json from one method")
    ap.add_argument("--problem-config", required=True)
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="Perturbation scale (×bounds_range, gaussian)")
    ap.add_argument("--n-samples", type=int, default=100,
                    help="Monte Carlo samples per design")
    ap.add_argument("--rms-mult", type=float, default=1.5,
                    help="Yield threshold: design passes if "
                         "median(perturbed_RMS) <= rms_mult × nominal_RMS")
    ap.add_argument("--rms-cap-um", type=float, default=30.0,
                    help="Only include designs with refined RMS below this "
                         "(default 30μm — focus on high-quality)")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    refined = json.loads(Path(args.refined_json).read_text())
    designs = refined.get("designs", [])
    # Filter sub-RMS-cap
    pool = []
    for i, d in enumerate(designs):
        rms = d.get("rms_um")
        if rms is None or rms > args.rms_cap_um:
            continue
        pool.append((i, list(map(float, d["theta"])), float(rms)))
    print(f"[main] refined pool: {len(designs)}  sub-{args.rms_cap_um}μm: "
          f"{len(pool)}", flush=True)

    if not pool:
        Path(args.out).write_text(json.dumps({
            "n_input": len(designs),
            "n_evaluated": 0,
            "yield_count": 0,
            "yield_pct": 0.0,
            "designs": [],
        }, indent=2))
        print("[main] no designs to evaluate, wrote empty result")
        return

    from optical_rl.qd.lensmega.config.loader import load_config
    cfg = load_config(args.problem_config)
    cfg_dict = cfg.model_dump(mode="json")

    print(f"[main] spawning {args.workers} workers", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(cfg_dict,)) as p:
        tasks = [(idx, theta, nom, args.tolerance, args.n_samples, idx + 1)
                 for idx, theta, nom in pool]
        results = []
        for i, rec in enumerate(p.imap_unordered(_worker_yield, tasks)):
            results.append(rec)
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f"[main] {i+1}/{len(tasks)} (wall={time.time()-t0:.1f}s)",
                      flush=True)
    results.sort(key=lambda r: r["idx"])

    # Yield computation
    n_pass = sum(
        1 for r in results
        if r["perturbed_rms_um"]["median"] <= args.rms_mult * r["nominal_rms_um"]
    )
    yield_pct = 100.0 * n_pass / max(len(results), 1)

    out = {
        "n_input": len(designs),
        "n_evaluated": len(results),
        "tolerance": args.tolerance,
        "n_samples_per_design": args.n_samples,
        "rms_mult_threshold": args.rms_mult,
        "rms_cap_um": args.rms_cap_um,
        "yield_count": n_pass,
        "yield_pct": yield_pct,
        "designs": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[main] yield: {n_pass}/{len(results)} = {yield_pct:.1f}% "
          f"survive median(perturbed_RMS) ≤ {args.rms_mult}× nominal",
          flush=True)
    print(f"[main] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
