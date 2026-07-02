#!/usr/bin/env python3
"""Pick top-N ε-distinct designs from a method's pool and refine each via
the standard lensmega L-BFGS-B pipeline.

Inputs:
  - For HyperLens: --hl-run pointing at archive.pkl (uses cell winners
    by fitness desc, ε-net deduplicated at --epsilon)
  - For SA+GA: --sa-ga-run pointing at unrefined_elites_sa_ga.json

Output: a JSON with refined records (theta, refined_fitness, RMS, stability,
sign_pattern unused) in the run directory.

Usage:
  python tools/refine_top_diverse.py \\
      --pool-from hl --hl-run <run_dir> --top 9 --epsilon 0.20
  python tools/refine_top_diverse.py \\
      --pool-from sa_ga --sa-ga-run <run_dir> --top 9 --epsilon 0.20
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optical_rl.qd.lensmega.config.loader import load_config
from sa_ga_refine_baseline import lbfgs_refine_one, stability_basin, BudgetCounter


def load_hl_pool(run_dir: Path) -> List[Tuple[float, np.ndarray]]:
    arch_path = run_dir / "archive.pkl"
    with arch_path.open("rb") as f:
        a = pickle.load(f)
    out = []
    for v in a.cells.values():
        if v is None: continue
        out.append((float(v.fitness), np.asarray(v.theta, dtype=np.float64)))
    return out


def load_sa_ga_pool(run_dir: Path) -> List[Tuple[float, np.ndarray]]:
    p = run_dir / "unrefined_elites_sa_ga.json"
    if not p.exists():
        p = run_dir / "refined_designs_sa_ga.json"
    designs = json.loads(p.read_text())
    return [
        (float(d["raw_fitness"]), np.asarray(d["theta"], dtype=np.float64))
        for d in designs
    ]


def epsilon_net_pick(
    pool: List[Tuple[float, np.ndarray]],
    n: int, epsilon: float,
    lo: np.ndarray, hi: np.ndarray,
) -> List[Tuple[float, np.ndarray]]:
    rng = hi - lo
    rng = np.where(rng < 1e-12, 1e-12, rng)
    sqrtD = np.sqrt(len(lo))
    pool_sorted = sorted(pool, key=lambda x: -x[0])
    selected = []
    selected_u = []
    for fit, theta in pool_sorted:
        u = (theta - lo) / rng
        if not selected_u:
            selected.append((fit, theta)); selected_u.append(u); continue
        if min(np.linalg.norm(u - su) / sqrtD for su in selected_u) > epsilon:
            selected.append((fit, theta)); selected_u.append(u)
        if len(selected) >= n:
            break
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-from", choices=["hl", "sa_ga"], required=True)
    ap.add_argument("--hl-run", default=None)
    ap.add_argument("--sa-ga-run", default=None)
    ap.add_argument("--problem-config", default=None)
    ap.add_argument("--top", type=int, default=9)
    ap.add_argument("--epsilon", type=float, default=0.20)
    ap.add_argument("--out-tag", type=str, default="refined_top_diverse")
    args = ap.parse_args()

    if args.pool_from == "hl":
        run_dir = Path(args.hl_run).resolve()
        pool = load_hl_pool(run_dir)
        method = "hyperlens"
    else:
        run_dir = Path(args.sa_ga_run).resolve()
        pool = load_sa_ga_pool(run_dir)
        method = "sa_ga"

    cfg_path = (args.problem_config
                or str(run_dir / "config.yaml"))
    cfg = load_config(cfg_path)
    bounds = np.asarray(cfg.problem.theta_bounds, dtype=np.float64)
    lo = bounds[:, 0]; hi = bounds[:, 1]
    io = cfg.io

    print(f"[{method}] pool size: {len(pool)}", flush=True)
    print(f"[{method}] picking top-{args.top} ε={args.epsilon}-distinct designs",
          flush=True)
    selected = epsilon_net_pick(pool, args.top, args.epsilon, lo, hi)
    print(f"[{method}] selected {len(selected)} ε-distinct designs:",
          flush=True)
    for i, (fit, _) in enumerate(selected, 1):
        rms_um = -fit * 1000
        print(f"  #{i}: raw fit={fit:+.5f}  raw RMS≈{rms_um:.1f}μm",
              flush=True)

    # Refine each
    counter = BudgetCounter()
    refined = []
    t0 = time.time()
    for i, (raw_fit, theta) in enumerate(selected, 1):
        print(f"\n[{method}] refining #{i}/{len(selected)} "
              f"raw_fit={raw_fit:+.5f}", flush=True)
        rec = lbfgs_refine_one(
            cfg, theta, lo, hi,
            refine_efl_penalty_weight=io.refine_efl_penalty_weight,
            refine_invalid_penalty_weight=io.refine_invalid_penalty_weight,
            refine_intersect_penalty_weight=io.refine_intersect_penalty_weight,
            refine_fnum_penalty_weight=io.refine_fnum_penalty_weight,
            refine_field_weights=(tuple(io.refine_field_weights)
                                  if io.refine_field_weights else None),
            refine_lbfgs_lr=io.refine_lbfgs_lr,
            refine_lbfgs_max_steps=io.refine_lbfgs_max_steps,
            refine_converge_tol=io.refine_converge_tol,
            counter=counter,
        )
        rec["raw_fitness"] = float(raw_fit)
        rec["raw_rms_um"] = float(-raw_fit * 1000)
        # Stability
        stab = stability_basin(
            cfg, rec["theta"], lo, hi,
            rec["refined_fitness"],
            int(io.refine_stability_samples),
            float(io.refine_stability_perturbation),
            counter,
        )
        rec["stability"] = stab
        # Make theta serialisable
        rec["theta"] = [float(x) for x in rec["theta"]]
        refined.append(rec)
        print(f"  raw={raw_fit:+.5f} → refined={rec['refined_fitness']:+.5f} "
              f"RMS={rec['rms_um']:.1f}μm "
              f"({rec['lbfgs_steps']} steps, "
              f"{'CONVERGED' if rec['converged'] else 'max'}, "
              f"survived={stab['pct_survived']:.0f}%)", flush=True)

    out_path = run_dir / f"{args.out_tag}_{method}.json"
    out_path.write_text(json.dumps({
        "method": method,
        "epsilon": args.epsilon,
        "n_selected": len(selected),
        "designs": refined,
    }, indent=2))
    print(f"\n[{method}] wrote {out_path}", flush=True)
    print(f"[{method}] total wall: {time.time()-t0:.1f}s, "
          f"refine_evals={counter.refine_evals}, "
          f"stability_evals={counter.stability_evals}", flush=True)

    # Compact summary
    print(f"\n=== {method} top-{len(selected)} refined summary ===")
    print(f"  {'#':>2}  {'raw RMS':>8}  {'refined RMS':>12}  "
          f"{'survived':>10}  {'LBFGS':>6}")
    for i, r in enumerate(refined, 1):
        print(f"  {i:>2}  {r['raw_rms_um']:>7.1f}μ  "
              f"{r['rms_um']:>11.1f}μ  "
              f"{r['stability']['pct_survived']:>9.0f}%  "
              f"{r['lbfgs_steps']:>6d}")


if __name__ == "__main__":
    main()
