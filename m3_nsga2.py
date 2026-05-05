"""M3 baseline: NSGA-II with θ-space diversity pressure (OptiFusion-style).

OptiFusion uses simulated annealing + GA with mechanisms to preserve
design diversity.  For our benchmark we use NSGA-II from pymoo with two
objectives:

  (1) spot RMS (minimise)              — the classical fitness
  (2) -avg-dist-to-kNN in theta-space  — a diversity pressure that
       penalises designs clustered near others in the current pop

This is a clean, standard implementation of evolutionary diverse search
that a reader can reproduce without any optics-specific machinery.  If
LensMEGA beats this on (#good elites, #families per budget), that's a
fair baseline comparison.

Usage::

    python m3_nsga2.py --config <yaml> --budget 50000 \
                       --pop-size 50 --out /tmp/m3.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch
torch.set_num_threads(1)

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination import get_termination


def _build_problem(config_path: Path):
    from optical_rl.qd.lensmega.config.loader import load_config
    from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem
    cfg = load_config(config_path)
    descriptor = _build_descriptor(cfg)
    problem = _build_problem(cfg, descriptor)
    problem._ensure_backend()
    return cfg, problem


class LensPymooProblem(Problem):
    """Wrap LensProblem for pymoo.

    Two objectives:
      f1 = -fitness  (i.e. spot RMS + penalties; minimise)
      f2 = -diversity (avg L2 to k-NN in normalised θ; minimise → we use
           a population-dependent placeholder here and re-apply diversity
           via NSGA-II's crowding-distance selection, which already
           operates over objective space.)

    Realistically for lens-design diversity we want θ-space diversity,
    which pymoo supports via setting an additional "decision-space"
    crowding operator.  Simpler: we use the *θ-space k-NN distance* as
    an explicit secondary objective, computed against all individuals
    ever evaluated.
    """

    def __init__(self, lens_problem, theta_bounds, seen_thetas: list,
                 n_workers: int = 1):
        self.lens_problem = lens_problem
        self.seen_thetas = seen_thetas  # grows as we evaluate
        self.n_workers = n_workers
        xl = np.asarray([b[0] for b in theta_bounds], dtype=np.float64)
        xu = np.asarray([b[1] for b in theta_bounds], dtype=np.float64)
        self._range = np.clip(xu - xl, 1e-12, None)
        super().__init__(n_var=xl.size, n_obj=2, xl=xl, xu=xu)

    def _single_eval(self, theta):
        try:
            r = self.lens_problem.evaluate(theta, with_grads=False)
            if not np.isfinite(r.fitness):
                return 1e9
            return -float(r.fitness)
        except Exception:
            return 1e9

    def _kNN_distance(self, theta, k: int = 5) -> float:
        if not self.seen_thetas:
            return 1.0
        arr = np.asarray(self.seen_thetas, dtype=np.float64)
        diff = (arr - theta[None, :]) / self._range[None, :]
        d = np.sqrt((diff ** 2).sum(axis=1))
        if d.size <= k:
            return float(d.mean())
        idx = np.argpartition(d, k)[:k]
        return float(d[idx].mean())

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        f1 = np.empty(n, dtype=np.float64)
        f2 = np.empty(n, dtype=np.float64)
        for i in range(n):
            # Diversity is computed against seen points BEFORE this individual.
            f2[i] = -self._kNN_distance(X[i])
            f1[i] = self._single_eval(X[i])
            self.seen_thetas.append(X[i].tolist())
        out["F"] = np.column_stack([f1, f2])


def _designs_from_history(seen: List[list], evals_fit: List[float],
                          evals_metrics: List[dict]) -> List[dict]:
    out = []
    for theta, fit, metrics in zip(seen, evals_fit, evals_metrics):
        out.append({
            "theta": list(theta),
            "fitness": float(fit),
            "metrics": dict(metrics),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=50000)
    ap.add_argument("--pop-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method-name", default="nsga2_theta_diversity")
    args = ap.parse_args()

    cfg, lens_problem = _build_problem(args.config)
    theta_bounds = [tuple(b) for b in cfg.problem.theta_bounds]

    # Instead of two separate tracking lists, intercept LensProblem to
    # log fitness + metrics per eval.  Simpler: wrap evaluate inline.
    log: List[dict] = []

    orig_evaluate = lens_problem.evaluate

    def evaluate_logged(theta, with_grads=False):
        r = orig_evaluate(theta, with_grads=False)
        log.append({
            "theta": [float(v) for v in theta],
            "fitness": float(r.fitness) if np.isfinite(r.fitness) else -1e9,
            "metrics": {k: float(v) for k, v in r.metrics.items()},
        })
        return r

    lens_problem.evaluate = evaluate_logged  # type: ignore

    pymoo_prob = LensPymooProblem(lens_problem, theta_bounds, seen_thetas=[])

    n_gen = max(1, args.budget // args.pop_size)
    print(f"[m3] pop_size={args.pop_size} n_gen={n_gen} "
          f"budget~{args.pop_size * n_gen}", flush=True)
    algorithm = NSGA2(
        pop_size=args.pop_size,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    t0 = time.perf_counter()
    try:
        minimize(
            pymoo_prob, algorithm,
            get_termination("n_gen", n_gen),
            seed=args.seed, verbose=False,
        )
    except Exception as e:
        print(f"[m3] minimize failed: {type(e).__name__}: {e}", flush=True)
    wall = time.perf_counter() - t0

    # Budget trimming: truncate to budget
    if len(log) > args.budget:
        log = log[: args.budget]

    out = {
        "name": args.method_name,
        "n_evaluations": len(log),
        "wall_seconds": float(wall),
        "config": str(args.config),
        "designs": log,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)

    fits = np.array([d["fitness"] for d in log]) if log else np.array([])
    print(f"[m3] wrote {args.out}  n_evals={len(log)}  wall={wall:.0f}s")
    if fits.size:
        print(f"[m3] best fitness: {fits.max():+.4f}")
        print(f"[m3] fit ≥ -0.2: {int((fits >= -0.2).sum())} / {len(fits)}")
        print(f"[m3] fit ≥ -0.1: {int((fits >= -0.1).sum())} / {len(fits)}")


if __name__ == "__main__":
    main()
