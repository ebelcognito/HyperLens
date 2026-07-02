#!/usr/bin/env python3
"""SA+GA+Refine baseline driver for HyperLens comparison.

Implements the OptiFusion-family algorithm (QGSO, IEEE TCI 2025) at fixed
evaluate()-call budget against a HyperLens reference run. Per-stage
behavior matches QGSO/TCI_Optifusion/gaoptics/gygad.py at the family
level, with two principled deviations (logged in hyperparameters.json):

  - ADAM gradients use autograd, not finite differences. Fairer to the
    baseline since the lensmega problem already exposes autograd.
  - Mutation perturbs num_genes=2 variables per offspring (matching the
    QGSO reference code, not the OptiFusion paper's 0.3 fraction).

The descriptor (Seidel) is computed by LensProblem.evaluate but discarded
— SA+GA+Refine ignores behavior space by construction (spec §2).

Final L-BFGS-B polish uses HyperLens's exact refinement code path
(verbatim from optical_rl/qd/lensmega/core/runner.py:181-329) so any
quality difference comes from the upstream search machinery, not the
polish.

Usage:
  python sa_ga_refine_baseline.py \\
    --problem-config <existing_hyperlens_yaml> \\
    --baseline-config <sa_ga_yaml> \\
    --budget <N_evaluate_calls> \\
    --out <output_dir>
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Make lensmega importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.core.cpu_pool import CPUWorkerPool, auto_worker_count
from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem


# ===========================================================================
# Budget tracking
# ===========================================================================

class BudgetCounter:
    """Counts evaluate()-equivalent calls across all stages.

    1 forward eval = 1 unit
    1 forward+grad eval = 1 unit (matches HyperLens convention; gradients
    come from autograd through the same forward trace, no extra rays).
    1 L-BFGS closure call inside refinement = 1 unit
    """
    def __init__(self) -> None:
        self.search_evals = 0
        self.search_grad_evals = 0
        self.refine_evals = 0
        self.stability_evals = 0
        self.stage_timings: Dict[str, float] = {
            "sa": 0.0, "selection": 0.0, "adam": 0.0,
            "mutation": 0.0, "refine": 0.0, "stability": 0.0,
        }

    @property
    def search_total(self) -> int:
        return self.search_evals + self.search_grad_evals

    @property
    def grand_total(self) -> int:
        return (self.search_evals + self.search_grad_evals
                + self.refine_evals + self.stability_evals)


# ===========================================================================
# Parameter-space helpers (normalized [0,1]^D <-> physical theta)
# ===========================================================================

class ThetaSpace:
    def __init__(self, bounds: np.ndarray) -> None:
        self.lo = bounds[:, 0].astype(np.float64)
        self.hi = bounds[:, 1].astype(np.float64)
        self.range = self.hi - self.lo
        # Guard against degenerate bounds.
        self.range[self.range < 1e-12] = 1e-12
        self.dim = bounds.shape[0]

    def to_unit(self, theta: np.ndarray) -> np.ndarray:
        return (theta - self.lo) / self.range

    def to_physical(self, u: np.ndarray) -> np.ndarray:
        return self.lo + np.clip(u, 0.0, 1.0) * self.range

    def random_unit(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.random((n, self.dim))


# ===========================================================================
# Fitness wrappers (forward only, and forward+grad)
# ===========================================================================

def eval_fitness(problem, theta: np.ndarray, counter: BudgetCounter) -> float:
    """Forward fitness evaluation. Increments search_evals."""
    res = problem.evaluate(theta, with_grads=False)
    counter.search_evals += 1
    return float(res.fitness)


def eval_fitness_and_grad(problem, theta: np.ndarray,
                          counter: BudgetCounter) -> Tuple[float, np.ndarray]:
    """Forward + autograd fitness gradient. Increments search_grad_evals."""
    res = problem.evaluate(theta, with_grads=True)
    counter.search_grad_evals += 1
    return float(res.fitness), res.grads["fitness"].astype(np.float64)


def eval_population(problem, U: np.ndarray, space: ThetaSpace,
                    counter: BudgetCounter,
                    pool: Optional[CPUWorkerPool] = None) -> np.ndarray:
    """Evaluate a (m, D) unit population. Returns fitness array shape (m,).

    Uses the multi-process pool when provided (much faster); otherwise
    serial eval through the main-process problem.
    """
    fits = np.empty(len(U), dtype=np.float64)
    if pool is not None:
        thetas = np.array([space.to_physical(u) for u in U])
        results = pool.evaluate(thetas)
        for i, r in enumerate(results):
            counter.search_evals += 1
            if r is None:
                fits[i] = -1e9  # treat worker failure as terrible fitness
            else:
                fits[i] = float(r.fitness)
    else:
        for i in range(len(U)):
            fits[i] = eval_fitness(problem, space.to_physical(U[i]), counter)
    return fits


# ===========================================================================
# SA inner loop (per outer generation)
# ===========================================================================

def sa_inner_loop(
    problem, U: np.ndarray, space: ThetaSpace, counter: BudgetCounter,
    *, rand_range: float, iter_max: int, plateau_window: int,
    plateau_min_delta: float, rng: np.random.Generator,
    budget_remaining_fn,
    pool: Optional[CPUWorkerPool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """QGSO-family SA: adaptive temperature T = clamp(0.1*|fit|, max=1).

    Mutates a population in unit-space, accepting moves by Metropolis
    criterion. Tracks best-seen per individual. Plateau-terminates when
    mean of best-fits stops improving over `plateau_window` iters by less
    than `plateau_min_delta`.
    """
    t0 = time.time()
    m, D = U.shape
    # Initial fitness.
    fit = eval_population(problem, U, space, counter, pool=pool)
    fit_final = fit.copy()
    U_final = U.copy()
    mean_history = [float(fit_final.mean())]

    for it in range(iter_max):
        if budget_remaining_fn() <= m:
            break
        # Random delta in unit space, scaled by rand_range.
        delta = (rng.random((m, D)) - 0.5) * rand_range
        U_prop = np.clip(U + delta, 0.0, 1.0)
        fit_prop = eval_population(problem, U_prop, space, counter, pool=pool)

        # Update per-individual best-seen.
        improved = fit_prop > fit_final
        U_final[improved] = U_prop[improved]
        fit_final[improved] = fit_prop[improved]

        # Metropolis: fitness is to be MAXIMIZED. delta_fit_neg = fit - fit_prop.
        # Accept with prob exp(-delta_fit_neg / T) when delta_fit_neg > 0.
        delta_fit_neg = fit - fit_prop  # > 0 means proposal is worse
        # QGSO: T = clamp(0.1*|fit_last|, max=1)
        T = np.clip(0.1 * np.abs(fit), a_min=1e-6, a_max=1.0)
        p_accept = np.exp(-np.maximum(delta_fit_neg, 0.0) / T)
        rand_accept = rng.random(m)
        accept = (delta_fit_neg <= 0) | (rand_accept < p_accept)
        U[accept] = U_prop[accept]
        fit[accept] = fit_prop[accept]

        mean_history.append(float(fit_final.mean()))
        if it >= plateau_window:
            improvement = mean_history[-1] - mean_history[-1 - plateau_window]
            if improvement / max(plateau_window - 1, 1) <= plateau_min_delta:
                break

    counter.stage_timings["sa"] += time.time() - t0
    return U_final, fit_final


# ===========================================================================
# Diversity-filtered selection (parents AND elites use same logic)
# ===========================================================================

def diversity_select(U: np.ndarray, fit: np.ndarray, k: int,
                     div_rate: float, fill_with_random: bool,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Greedy: sort by best fitness desc, accept next iff normalized
    Euclidean dist to all already-accepted exceeds div_rate.
    Matches QGSO/gygad.py:434-455 select_parents.
    """
    D = U.shape[1]
    sqrtD = math.sqrt(D)
    order = np.argsort(-fit)  # descending (we maximize)
    selected: List[int] = []
    for idx in order:
        if len(selected) == 0:
            selected.append(idx)
        else:
            min_dist = min(
                np.linalg.norm(U[idx] - U[s]) / sqrtD for s in selected
            )
            if min_dist > div_rate:
                selected.append(idx)
        if len(selected) >= k:
            break
    out = np.empty((k, D), dtype=np.float64)
    fit_out = np.empty(k, dtype=np.float64)
    n_sel = len(selected)
    for i, s in enumerate(selected):
        out[i] = U[s]
        fit_out[i] = fit[s]
    if n_sel < k:
        if fill_with_random and rng is not None:
            out[n_sel:] = rng.random((k - n_sel, D))
            fit_out[n_sel:] = -np.inf
        else:
            # Pad with the worst already-selected (caller doesn't expect random).
            out[n_sel:] = out[n_sel - 1] if n_sel > 0 else 0.0
            fit_out[n_sel:] = fit_out[n_sel - 1] if n_sel > 0 else -np.inf
    return out, fit_out


# ===========================================================================
# ADAM refinement (autograd through lensmega problem)
# ===========================================================================

def adam_refine(
    problem, U_parents: np.ndarray, space: ThetaSpace, counter: BudgetCounter,
    *, iter_num: int, lr_base: float, beta1: float, beta2: float,
    cos_T: float, plateau_window: int, plateau_eps: float,
    budget_remaining_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    """ADAM step in physical theta space using autograd gradients.

    Matches QGSO/gygad.py:157-230 family but uses autograd instead of
    finite differences (logged as a deviation in hyperparameters.json).
    """
    t0 = time.time()
    n_parents, D = U_parents.shape
    theta_phys = np.array([space.to_physical(u) for u in U_parents])

    # ADAM state per parent (in physical units, so step sizes account for
    # bounds-range scaling).
    m_state = np.zeros_like(theta_phys)
    v_state = np.zeros_like(theta_phys)
    t_step = 0
    fit_history: List[np.ndarray] = []
    best_theta = theta_phys.copy()
    best_fit = np.full(n_parents, -np.inf, dtype=np.float64)

    # Per-dimension learning-rate scale = bounds range (mirrors QGSO's
    # `lr_now = lr_rate[vari_flag]`, i.e. step proportional to bounds).
    lr_dim_scale = space.range.copy()

    for it in range(iter_num):
        if budget_remaining_fn() <= n_parents:
            break
        t_step += 1
        cos_factor = ((1 + math.cos(it * math.pi / cos_T)) + 0.2) / 2.2
        lr = lr_base * cos_factor

        # Compute grad per parent via autograd. Each call counts as 1 unit.
        fits_now = np.empty(n_parents)
        grads = np.empty_like(theta_phys)
        for j in range(n_parents):
            f, g = eval_fitness_and_grad(problem, theta_phys[j], counter)
            fits_now[j] = f
            grads[j] = g

        # We MAXIMIZE fitness; ADAM update goes in +gradient direction.
        m_state = beta1 * m_state + (1 - beta1) * grads
        v_state = beta2 * v_state + (1 - beta2) * grads ** 2
        m_hat = m_state / (1 - beta1 ** t_step)
        v_hat = v_state / (1 - beta2 ** t_step)
        step = lr * lr_dim_scale * m_hat / (np.sqrt(v_hat) + 1e-8)
        theta_phys = theta_phys + step
        # Project to bounds.
        theta_phys = np.minimum(np.maximum(theta_phys, space.lo), space.hi)

        # Track best.
        improved = fits_now > best_fit
        best_theta[improved] = theta_phys[improved]
        best_fit[improved] = fits_now[improved]
        fit_history.append(best_fit.copy())

        if len(fit_history) > plateau_window:
            old = fit_history[-plateau_window - 1]
            new = fit_history[-1]
            # Maximization: improvement = new - old, want it positive.
            improvement = new - old
            # Use relative improvement on |best_fit| to mirror QGSO's
            # `(fit_all[-(N+1)] - fit_all[-1]) / fit_all[-(N+1)]` (sign
            # flipped for maximization).
            denom = np.abs(old) + 1e-12
            rel = improvement / denom
            if rel.mean() < plateau_eps and rel.max() < 5 * plateau_eps:
                break

    counter.stage_timings["adam"] += time.time() - t0
    U_out = np.array([space.to_unit(t) for t in best_theta])
    return U_out, best_fit


# ===========================================================================
# Mutation (perturb num_genes random variables per offspring)
# ===========================================================================

def mutate(parents: np.ndarray, n_offspring: int, num_genes: int,
           rng: np.random.Generator) -> np.ndarray:
    """Per-offspring: pick a parent at random, perturb `num_genes` random
    dims to fresh uniform[0,1] values. Matches QGSO/gygad.py:480-587.
    """
    m, D = parents.shape
    parent_idx = rng.integers(0, m, size=n_offspring)
    offspring = parents[parent_idx].copy()
    for i in range(n_offspring):
        dims = rng.choice(D, size=min(num_genes, D), replace=False)
        offspring[i, dims] = rng.random(len(dims))
    return offspring


# ===========================================================================
# L-BFGS-B refinement (verbatim from runner.py:181-329 with adapter)
# ===========================================================================

def lbfgs_refine_one(
    problem_template_cfg, theta: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    *, refine_efl_penalty_weight: Optional[float],
    refine_invalid_penalty_weight: Optional[float],
    refine_intersect_penalty_weight: Optional[float],
    refine_fnum_penalty_weight: Optional[float],
    refine_field_weights: Optional[Tuple[float, ...]],
    refine_lbfgs_lr: float, refine_lbfgs_max_steps: int,
    refine_converge_tol: float, counter: BudgetCounter,
) -> Dict[str, Any]:
    """Runs L-BFGS-B polish on a single design. Mirrors runner.py exactly,
    including the heavier refinement-time penalty weights.
    """
    t0 = time.time()
    refine_cfg = copy.deepcopy(problem_template_cfg)
    if refine_efl_penalty_weight is not None:
        refine_cfg.problem.efl_penalty_weight = refine_efl_penalty_weight
    if refine_invalid_penalty_weight is not None:
        refine_cfg.problem.invalid_penalty_weight = refine_invalid_penalty_weight
    if refine_intersect_penalty_weight is not None:
        refine_cfg.problem.intersect_penalty_weight = refine_intersect_penalty_weight
    if refine_field_weights is not None:
        refine_cfg.problem.field_weights = tuple(refine_field_weights)
    if (refine_cfg.problem.enforce_fnum
            and refine_fnum_penalty_weight is not None):
        refine_cfg.problem.fnum_penalty_weight = refine_fnum_penalty_weight

    desc = _build_descriptor(refine_cfg)
    prob = _build_problem(refine_cfg, desc)
    prob._ensure_backend()

    theta_t = torch.tensor(
        theta.copy(), dtype=torch.float32, device="cpu", requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [theta_t], lr=refine_lbfgs_lr, max_iter=10,
        line_search_fn="strong_wolfe",
    )
    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)

    best_loss = float("inf")
    best_theta = theta.copy()
    lbfgs_steps = 0
    converged = False
    final_gnorm = float("inf")
    no_improve = 0
    closure_calls = [0]

    for step in range(refine_lbfgs_max_steps):
        def closure():
            optimizer.zero_grad()
            with torch.no_grad():
                theta_t.clamp_(lo_t, hi_t)
            fitness, _, _ = prob._fitness_and_metrics(theta_t)
            loss = -fitness
            if loss.requires_grad:
                loss.backward()
            closure_calls[0] += 1
            return loss

        loss_val = optimizer.step(closure)
        lbfgs_steps = step + 1
        if loss_val is not None:
            lv = float(loss_val)
            if lv < best_loss - 1e-8:
                best_loss = lv
                no_improve = 0
                with torch.no_grad():
                    best_theta = theta_t.detach().cpu().numpy().copy()
            else:
                no_improve += 1
        if theta_t.grad is not None:
            final_gnorm = float(theta_t.grad.norm().item())
            if final_gnorm < refine_converge_tol:
                converged = True
                break
        if no_improve >= 5:
            converged = True
            break

    best_theta = np.clip(best_theta, lo, hi)
    refined_fitness = -best_loss
    counter.refine_evals += closure_calls[0]
    counter.stage_timings["refine"] += time.time() - t0
    # Re-evaluate cleanly for metrics.
    refined_eval = prob.evaluate(best_theta, with_grads=False)
    counter.refine_evals += 1
    return {
        "theta": best_theta,
        "refined_fitness": refined_fitness,
        "rms_mm": float(refined_eval.metrics.get("rms_spot",
                                                  abs(refined_fitness))),
        "rms_um": float(refined_eval.metrics.get("rms_spot",
                                                 abs(refined_fitness))) * 1000,
        "efl_mm": float(refined_eval.metrics.get("efl_actual", 0.0)),
        "fnum_actual": refined_eval.metrics.get("fnum_actual"),
        "lbfgs_steps": lbfgs_steps,
        "converged": converged,
        "final_grad_norm": final_gnorm,
        "metrics": {k: float(v) for k, v in refined_eval.metrics.items()},
    }


def stability_basin(
    problem_template_cfg, theta: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    refined_fitness: float, n_samples: int, perturb_scale: float,
    counter: BudgetCounter,
) -> Dict[str, float]:
    """Mirrors runner.py basin stability analysis."""
    t0 = time.time()
    desc = _build_descriptor(problem_template_cfg)
    prob = _build_problem(problem_template_cfg, desc)
    prob._ensure_backend()
    bounds_range = hi - lo
    rng = np.random.RandomState(42)
    deltas = []
    for _ in range(n_samples):
        noise = rng.randn(len(theta)) * perturb_scale * bounds_range
        perturbed = np.clip(theta + noise, lo, hi)
        pr = prob.evaluate(perturbed, with_grads=False)
        counter.stability_evals += 1
        deltas.append(pr.fitness - refined_fitness)
    deltas = np.array(deltas)
    counter.stage_timings["stability"] += time.time() - t0
    return {
        "perturbation_scale": float(perturb_scale),
        "n_samples": int(n_samples),
        "fitness_delta_mean": float(deltas.mean()),
        "fitness_delta_std": float(deltas.std()),
        "fitness_delta_worst": float(deltas.min()),
        "pct_survived": float(
            (deltas > -abs(refined_fitness)).mean()
        ) * 100,
    }


# ===========================================================================
# Main outer loop
# ===========================================================================

def run_sa_ga_refine(args, baseline_cfg: Dict[str, Any], out_dir: Path) -> None:
    # ---- Load problem ----
    problem_template_cfg = load_config(args.problem_config)
    desc = _build_descriptor(problem_template_cfg)
    problem = _build_problem(problem_template_cfg, desc)
    problem._ensure_backend()

    # ---- CPU worker pool for population evals (forward only) ----
    n_workers = auto_worker_count(args.workers)
    print(f"[init] spawning {n_workers} CPU workers for population eval",
          flush=True)
    pool = CPUWorkerPool(
        num_workers=n_workers,
        config_dict=problem_template_cfg.model_dump(mode="json"),
    )

    bounds = np.asarray(problem.theta_bounds, dtype=np.float64)
    space = ThetaSpace(bounds)
    D = space.dim
    print(f"[init] problem theta_dim={D}", flush=True)

    counter = BudgetCounter()
    rng = np.random.default_rng(int(baseline_cfg.get("seed", 42)))

    # ---- Hyperparameters ----
    bc = baseline_cfg
    m = int(bc["population_size"])
    G_max = int(bc.get("generations_max", 1000))
    B_total = int(args.budget)

    # SA
    sa_iter_max = int(bc["sa_iter_max"])
    sa_rand_range = float(bc["sa_rand_range"])
    sa_plateau_window = int(bc["sa_plateau_window"])
    sa_plateau_min_delta = float(bc["sa_plateau_min_delta"])

    # GA
    mu = max(int(round(m * float(bc["ga_selection_ratio"]))), 1)
    nu = max(int(round(m * float(bc["ga_elite_ratio"]))), 1)
    div_rate = float(bc["ga_diversity_min_distance"])

    # ADAM
    adam_iter_num = int(bc["adam_iter_num"])
    adam_lr = float(bc["adam_lr"])
    adam_beta1 = float(bc.get("adam_beta1", 0.9))
    adam_beta2 = float(bc.get("adam_beta2", 0.9))
    adam_cos_T = float(bc.get("adam_cos_T", 10.0))
    adam_plateau_window = int(bc.get("adam_plateau_window", 20))
    adam_plateau_eps = float(bc.get("adam_plateau_eps", 0.001))

    # Mutation
    mut_num_genes = int(bc.get("mutation_num_genes", 2))

    # Refinement (read from problem cfg io block)
    io = problem_template_cfg.io
    refine_threshold = float(io.elite_fitness_threshold)

    # Tracker threshold: drives what gets added to all_time_elites_U.
    # Defaults to elite_fitness_threshold for backwards compatibility, but
    # baseline_cfg can override with a much lower value (e.g. -10) to
    # capture the full search distribution for fair threshold-based
    # comparison against HyperLens's archive (which holds everything).
    tracker_threshold = float(bc.get("tracker_threshold", refine_threshold))
    print(f"[init] tracker_threshold={tracker_threshold} "
          f"(refine_threshold={refine_threshold})", flush=True)

    def budget_remaining() -> int:
        return B_total - counter.search_total

    # ---- Initialize population ----
    U = space.random_unit(m, rng)
    print(f"[init] population m={m}, mu={mu}, nu={nu}, "
          f"div_rate={div_rate}, budget={B_total}", flush=True)

    history = []
    # All-time elite tracker: every individual that ever crossed
    # refine_threshold during search. Diversity-deduplicated by greedy
    # filter in unit space (same div_rate as parent selection, so we
    # don't refine near-duplicates). This mirrors HyperLens's QD archive
    # semantics for "tracked elites" (spec §5) — the count of high-quality
    # diverse designs the method discovered, not just the survivors of the
    # final mutation step.
    all_time_elites_U: List[np.ndarray] = []
    all_time_elites_fit: List[float] = []
    all_time_dedupe_div = float(div_rate)
    sqrtD = math.sqrt(D)
    max_tracked = int(bc.get("max_tracked_elites", 300))

    def maybe_track_elite(u: np.ndarray, fit: float) -> None:
        if fit < tracker_threshold:
            return
        if len(all_time_elites_U) >= max_tracked:
            return
        for ue in all_time_elites_U:
            if np.linalg.norm(u - ue) / sqrtD < all_time_dedupe_div:
                return  # near-duplicate of an already-tracked elite
        all_time_elites_U.append(u.copy())
        all_time_elites_fit.append(float(fit))

    gen = 0
    while gen < G_max and budget_remaining() > m:
        gen_t0 = time.time()
        # Stage 1: SA inner loop.
        U, fit_after_sa = sa_inner_loop(
            problem, U, space, counter,
            rand_range=sa_rand_range, iter_max=sa_iter_max,
            plateau_window=sa_plateau_window,
            plateau_min_delta=sa_plateau_min_delta,
            rng=rng, budget_remaining_fn=budget_remaining,
            pool=pool,
        )
        if budget_remaining() <= 0:
            break

        # Stage 2: parent selection w/ diversity.
        t1 = time.time()
        U_parents, fit_parents = diversity_select(
            U, fit_after_sa, mu, div_rate,
            fill_with_random=True, rng=rng,
        )
        # Re-eval any randomly-filled parents (their fit was -inf).
        needs_eval = ~np.isfinite(fit_parents)
        if needs_eval.any():
            for j in np.where(needs_eval)[0]:
                fit_parents[j] = eval_fitness(
                    problem, space.to_physical(U_parents[j]), counter
                )
        counter.stage_timings["selection"] += time.time() - t1
        if budget_remaining() <= 0:
            break

        # Stage 3: ADAM refinement on parents.
        U_parents, fit_parents_refined = adam_refine(
            problem, U_parents, space, counter,
            iter_num=adam_iter_num, lr_base=adam_lr,
            beta1=adam_beta1, beta2=adam_beta2, cos_T=adam_cos_T,
            plateau_window=adam_plateau_window,
            plateau_eps=adam_plateau_eps,
            budget_remaining_fn=budget_remaining,
        )
        if budget_remaining() <= 0:
            break

        # Track every above-threshold post-ADAM parent in the all-time pool
        # (diversity-deduplicated). These are the discovered candidates
        # we'll refine at the end, regardless of whether they survive
        # subsequent mutation.
        for j in range(len(U_parents)):
            maybe_track_elite(U_parents[j], fit_parents_refined[j])

        # Stage 4: elite selection (over refined parents, w/ diversity).
        U_elites, fit_elites = diversity_select(
            U_parents, fit_parents_refined, nu, div_rate,
            fill_with_random=False, rng=None,
        )

        # Stage 5: mutation to fill out next-gen offspring.
        n_offspring = m - nu
        t1 = time.time()
        U_offspring = mutate(U_parents, n_offspring, mut_num_genes, rng)
        counter.stage_timings["mutation"] += time.time() - t1

        # Stage 6: merge for next generation.
        U = np.vstack([U_elites, U_offspring])

        gen_dt = time.time() - gen_t0
        best = float(fit_parents_refined.max())
        n_above_pop = int((fit_parents_refined >= refine_threshold).sum())
        history.append({
            "gen": gen, "best_fit": best,
            "n_above_threshold_in_parents": n_above_pop,
            "all_time_tracked": len(all_time_elites_U),
            "evals_used": counter.search_total,
            "wall_s": gen_dt,
        })
        print(f"[gen {gen:3d}] best_fit={best:+.6f}  "
              f"above_thresh(parents)={n_above_pop}/{mu}  "
              f"all_time_tracked={len(all_time_elites_U)}  "
              f"evals={counter.search_total}/{B_total}  "
              f"({100*counter.search_total/B_total:.1f}%)  "
              f"wall={gen_dt:.1f}s", flush=True)
        gen += 1

    print(f"[search-done] gens={gen} evals_used={counter.search_total} "
          f"budget={B_total}", flush=True)

    # ---- Always dump UNREFINED elites to disk before refinement ----
    # This captures the search-time diversity: theta + raw_fitness for
    # every above-tracker-threshold design discovered, deduplicated by
    # parameter-space distance. Used directly for diversity comparison
    # against HL's archive (which is also unrefined per-cell winners).
    out_dir.mkdir(parents=True, exist_ok=True)
    unrefined_records = []
    for u, fit in zip(all_time_elites_U, all_time_elites_fit):
        theta_phys = space.to_physical(u)
        unrefined_records.append({
            "theta": [float(x) for x in theta_phys],
            "raw_fitness": float(fit),
        })
    (out_dir / "unrefined_elites_sa_ga.json").write_text(
        json.dumps(unrefined_records, indent=2)
    )
    print(f"[unrefined-dump] {len(unrefined_records)} elites written to "
          f"{out_dir/'unrefined_elites_sa_ga.json'}", flush=True)

    # ---- Skip refinement entirely if --no-refine ----
    if getattr(args, "no_refine", False):
        print("[no-refine] skipping L-BFGS refinement and stability "
              "(per --no-refine flag)", flush=True)
        pool.shutdown()
        # Still write the search-side artifacts for completeness
        write_artifacts(out_dir, args, baseline_cfg, problem_template_cfg,
                        counter, [], history,
                        bounds=bounds, search_budget_target=B_total)
        return

    # ---- Refinement candidates: ALL-TIME tracked elites ----
    # The all-time tracker captured every above-threshold post-ADAM parent
    # discovered during search, deduplicated by parameter-space diversity.
    # We refine every entry. This mirrors HyperLens's "all elites
    # discovered" semantics (HyperLens uses a QD archive for the same
    # purpose); SA+GA+Refine has no archive in its loop, but for the
    # comparison metric "how many high-quality diverse designs did this
    # method find?" we need the same accounting.
    if not all_time_elites_U:
        # Fallback: refine final population's above-threshold entries.
        print("[refine] no all-time elites tracked; falling back to "
              "final population", flush=True)
        fit_final = eval_population(problem, U, space, counter, pool=pool)
        above_idx = np.where(fit_final >= refine_threshold)[0]
        candidates_U = [U[i] for i in above_idx]
        candidates_fit = [float(fit_final[i]) for i in above_idx]
    else:
        candidates_U = all_time_elites_U
        candidates_fit = all_time_elites_fit
    print(f"[refine] {len(candidates_U)} all-time tracked elites "
          f"(threshold {refine_threshold}, "
          f"best raw={max(candidates_fit) if candidates_fit else 'NA'})",
          flush=True)

    refined_records: List[Dict[str, Any]] = []
    for rank, (u_cand, raw_fit_cand) in enumerate(
        zip(candidates_U, candidates_fit)
    ):
        theta_phys = space.to_physical(u_cand)
        print(f"[refine {rank+1}/{len(candidates_U)}] "
              f"raw_fit={raw_fit_cand:+.6f}", flush=True)
        rec = lbfgs_refine_one(
            problem_template_cfg, theta_phys, space.lo, space.hi,
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
        rec["raw_fitness"] = float(raw_fit_cand)
        rec["sign_pattern"] = sign_pattern_from_curvatures(
            rec["theta"], problem_template_cfg
        )
        # Stability (matches runner.py).
        stab = stability_basin(
            problem_template_cfg, rec["theta"], space.lo, space.hi,
            rec["refined_fitness"],
            int(io.refine_stability_samples),
            float(io.refine_stability_perturbation),
            counter,
        )
        rec["stability"] = stab
        refined_records.append(rec)
        print(f"[refine {rank+1}] raw={rec['raw_fitness']:+.6f} "
              f"refined={rec['refined_fitness']:+.6f} "
              f"RMS={rec['rms_um']:.1f}μm "
              f"({rec['lbfgs_steps']} steps, "
              f"{'CONVERGED' if rec['converged'] else 'max_steps'}, "
              f"survived={stab['pct_survived']:.0f}%)", flush=True)

    # ---- Shut down workers cleanly ----
    pool.shutdown()

    # ---- Outputs ----
    write_artifacts(out_dir, args, baseline_cfg, problem_template_cfg,
                    counter, refined_records, history,
                    bounds=bounds, search_budget_target=B_total)


def sign_pattern_from_curvatures(theta: np.ndarray, cfg) -> str:
    """Compact sign-pattern string from the free curvature dimensions.
    Mirrors HyperLens convention (one char per free curvature: +, -, 0).
    """
    n_curv = len(cfg.problem.free_curvature_indices)
    chars = []
    for i in range(n_curv):
        c = theta[i]
        if c > 1e-9:
            chars.append("+")
        elif c < -1e-9:
            chars.append("-")
        else:
            chars.append("0")
    return "".join(chars)


# ===========================================================================
# Output writers (per spec §6)
# ===========================================================================

def write_artifacts(
    out_dir: Path, args, baseline_cfg: Dict[str, Any],
    problem_template_cfg, counter: BudgetCounter,
    refined_records: List[Dict[str, Any]], history: List[Dict[str, Any]],
    bounds: np.ndarray, search_budget_target: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- metrics.json ---
    refined_fits = np.array(
        [r["refined_fitness"] for r in refined_records]
        if refined_records else [np.nan]
    )
    refined_rms_um = np.array(
        [r["rms_um"] for r in refined_records]
        if refined_records else [np.nan]
    )
    raw_fits = np.array(
        [r["raw_fitness"] for r in refined_records]
        if refined_records else [np.nan]
    )

    sign_families: Dict[str, int] = {}
    for r in refined_records:
        sp = r["sign_pattern"]
        sign_families[sp] = sign_families.get(sp, 0) + 1
    # Best refined RMS per family (the failure mode is "many designs but
    # all same family" — show this directly).
    best_rms_per_family: Dict[str, float] = {}
    for r in refined_records:
        sp = r["sign_pattern"]
        rms = float(r["rms_um"])
        if sp not in best_rms_per_family or rms < best_rms_per_family[sp]:
            best_rms_per_family[sp] = rms

    def safe_pct(arr, p):
        if not np.isfinite(arr).any():
            return None
        return float(np.percentile(arr[np.isfinite(arr)], p))

    if refined_records:
        improvement_ratios = np.array(
            [r["raw_fitness"] / r["refined_fitness"]
             if abs(r["refined_fitness"]) > 1e-12 else 0.0
             for r in refined_records]
        )
    else:
        improvement_ratios = np.array([np.nan])

    sa_ga_metrics = {
        "best_raw_fitness": float(np.nanmax(raw_fits)) if refined_records else None,
        "refined_rms_best_um": float(np.nanmin(refined_rms_um)) if refined_records else None,
        "refined_rms_p10_um": safe_pct(refined_rms_um, 10),
        "refined_rms_median_um": safe_pct(refined_rms_um, 50),
        "refined_rms_mean_um": float(np.nanmean(refined_rms_um)) if refined_records else None,
        "refined_rms_worst_um": float(np.nanmax(refined_rms_um)) if refined_records else None,
        "count_below_10um": int(np.sum(refined_rms_um < 10)),
        "count_below_5um": int(np.sum(refined_rms_um < 5)),
        "count_below_3um": int(np.sum(refined_rms_um < 3)),
        "improvement_ratio_median": (
            safe_pct(improvement_ratios, 50) if refined_records else None
        ),
        "improvement_ratio_mean": (
            float(np.nanmean(improvement_ratios)) if refined_records else None
        ),
        "unique_sign_families": len(sign_families),
        "tracked_elites": len(refined_records),
        "designs_per_family": dict(
            sorted(sign_families.items(), key=lambda kv: -kv[1])
        ),
        "best_refined_rms_um_per_family": dict(
            sorted(best_rms_per_family.items(), key=lambda kv: kv[1])
        ),
        "wall_clock_seconds": sum(counter.stage_timings.values()),
    }

    metrics = {
        "experiment": "sa_ga_refine_vs_hyperlens",
        "compute_budget": {
            "search_evals_target": int(search_budget_target),
            "search_evals_actual_sa_ga": int(counter.search_total),
            "refine_evals_sa_ga": int(counter.refine_evals),
            "stability_evals_sa_ga": int(counter.stability_evals),
            "grand_total_sa_ga": int(counter.grand_total),
            "search_evals_actual_hyperlens": "see refined_designs_hyperlens.json or extract via tools/extract_hyperlens_metrics.py",
        },
        "sa_ga_refine": sa_ga_metrics,
        "hyperlens": "to be filled by tools/extract_hyperlens_metrics.py",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- refined_designs_sa_ga.json ---
    designs_out = []
    for r in refined_records:
        designs_out.append({
            "theta": [float(x) for x in r["theta"]],
            "refined_fitness": r["refined_fitness"],
            "raw_fitness": r["raw_fitness"],
            "rms_um": r["rms_um"],
            "rms_mm": r["rms_mm"],
            "efl_mm": r["efl_mm"],
            "fnum_actual": r.get("fnum_actual"),
            "sign_pattern": r["sign_pattern"],
            "lbfgs_steps": r["lbfgs_steps"],
            "converged": r["converged"],
            "final_grad_norm": r["final_grad_norm"],
            "stability": r["stability"],
            "metrics": r["metrics"],
        })
    (out_dir / "refined_designs_sa_ga.json").write_text(
        json.dumps(designs_out, indent=2)
    )

    # --- hyperparameters.json ---
    defaults_applied = {
        "adam_gradients": (
            "autograd (deviation: QGSO uses finite differences). "
            "Strictly fairer to the baseline at fixed budget."
        ),
        "mutation_num_genes": (
            f"{int(baseline_cfg.get('mutation_num_genes', 2))} per offspring "
            "(matches QGSO/gygad.py:495 num_genes=2; not the OptiFusion "
            "paper's claimed p=0.3 mutation fraction)."
        ),
        "sa_temperature": (
            "adaptive T = clamp(0.1*|fit|, max=1) per QGSO/gygad.py:134, "
            "not a pre-scheduled geometric/linear cool."
        ),
    }
    hp_dump = {
        "baseline_config_values": baseline_cfg,
        "deviations_from_spec_or_paper": defaults_applied,
        "problem_config_path": str(args.problem_config),
        "search_budget_target_evals": int(search_budget_target),
    }
    (out_dir / "hyperparameters.json").write_text(json.dumps(hp_dump, indent=2))

    # --- timing.log ---
    timing_lines = ["stage,wall_seconds"]
    for k, v in counter.stage_timings.items():
        timing_lines.append(f"{k},{v:.3f}")
    timing_lines.append(f"total,{sum(counter.stage_timings.values()):.3f}")
    (out_dir / "timing.log").write_text("\n".join(timing_lines) + "\n")

    # --- search_history.csv (extra, for debugging) ---
    if history:
        hist_lines = ["gen,best_fit,evals_used,wall_s"]
        for h in history:
            hist_lines.append(
                f"{h['gen']},{h['best_fit']:.8f},{h['evals_used']},{h['wall_s']:.3f}"
            )
        (out_dir / "search_history.csv").write_text(
            "\n".join(hist_lines) + "\n"
        )

    # --- README.md ---
    readme = f"""# SA+GA+Refine baseline run

This directory contains the artifacts produced by `sa_ga_refine_baseline.py`
for one run of the SA+GA+Refine baseline against a HyperLens reference.

## Files

- `metrics.json` — all SA+GA+Refine metrics per spec §5. The `hyperlens`
  block is a placeholder; populate it by running `tools/extract_hyperlens_metrics.py`
  pointed at the matching HyperLens run directory.
- `refined_designs_sa_ga.json` — per-design records (theta, fitness, RMS,
  sign pattern, stability) for every SA+GA+Refine elite above
  `elite_fitness_threshold` from the problem config.
- `hyperparameters.json` — full hyperparameter dump + deviation log.
- `timing.log` — per-stage wall-clock breakdown.
- `search_history.csv` — per-generation best fitness and budget usage.

## Run-time facts (from this run)

- Search-eval budget target: **{search_budget_target}** evaluate() calls
- Search-eval budget actual: **{counter.search_total}**
- Refinement evals: **{counter.refine_evals}**
- Stability evals: **{counter.stability_evals}**
- Wall-clock total: **{sum(counter.stage_timings.values()):.1f} s**
- SA+GA+Refine elites above threshold: **{len(refined_records)}**

## Caveats

See `hyperparameters.json[deviations_from_spec_or_paper]` for the two
principled deviations (autograd ADAM, mutation num_genes=2). All other
choices match QGSO/TCI_Optifusion's reference implementation at the
family level.
"""
    (out_dir / "README.md").write_text(readme)
    print(f"[done] artifacts written to {out_dir}", flush=True)


# ===========================================================================
# Entrypoint
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem-config", required=True,
                    help="Existing HyperLens problem YAML (sixlet, etc.)")
    ap.add_argument("--baseline-config", required=True,
                    help="SA+GA+Refine hyperparameter YAML")
    ap.add_argument("--budget", type=int, required=True,
                    help="Total evaluate()-call budget for the search phase")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="torch.set_num_threads in main process "
                         "(default: leave alone). Workers always use 1.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of CPU workers for population eval "
                         "(default: auto, ~cpu_count-2 capped at 32)")
    ap.add_argument("--no-refine", action="store_true",
                    help="Skip L-BFGS refinement and stability sampling. "
                         "Search runs as normal; unrefined_elites_sa_ga.json "
                         "is the only design-output artifact. Use this for "
                         "search-time diversity comparison.")
    args = ap.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
        print(f"[init] torch.set_num_threads({args.torch_threads})",
              flush=True)

    with open(args.baseline_config) as f:
        baseline_cfg = yaml.safe_load(f)

    out_dir = Path(args.out).resolve()
    print(f"[init] out_dir={out_dir}", flush=True)

    t0 = time.time()
    run_sa_ga_refine(args, baseline_cfg, out_dir)
    print(f"[total wall-clock] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
