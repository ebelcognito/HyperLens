#!/usr/bin/env python3
"""Refine the un-refined elites from a completed run.

The live runner enqueues elites to a background refiner that may not
drain before shutdown.  This script picks up where it left off: it
loads the elite tracker, finds elites that don't have a refined JSON
yet, and runs L-BFGS offline using the original config.

Usage:
    python refine_unrefined_elites.py <run_dir> [--max N] [--parallel K]
"""

import argparse
import copy as _copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def refine_one(run_cfg, theta: np.ndarray, raw_fitness: float, io_cfg) -> dict:
    """Run L-BFGS refinement on a single design, return the 'refined' dict."""
    from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem

    refine_cfg = _copy.deepcopy(run_cfg)
    p = refine_cfg.problem
    io = io_cfg

    if io.refine_efl_penalty_weight is not None:
        refine_cfg.problem.efl_penalty_weight = io.refine_efl_penalty_weight
    if io.refine_invalid_penalty_weight is not None:
        refine_cfg.problem.invalid_penalty_weight = io.refine_invalid_penalty_weight
    if io.refine_intersect_penalty_weight is not None:
        refine_cfg.problem.intersect_penalty_weight = io.refine_intersect_penalty_weight
    if io.refine_field_weights is not None:
        refine_cfg.problem.field_weights = tuple(io.refine_field_weights)

    desc = _build_descriptor(refine_cfg)
    prob = _build_problem(refine_cfg, desc)
    prob._ensure_backend()

    theta_bounds = list(p.theta_bounds)
    lo = np.array([b[0] for b in theta_bounds], dtype=np.float64)
    hi = np.array([b[1] for b in theta_bounds], dtype=np.float64)
    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)

    theta_t = torch.tensor(
        theta.copy(), dtype=torch.float32, requires_grad=True,
    )

    optimizer = torch.optim.LBFGS(
        [theta_t], lr=io.refine_lbfgs_lr, max_iter=10,
        line_search_fn="strong_wolfe",
    )

    best_loss = float("inf")
    best_theta = theta.copy()
    lbfgs_steps = 0
    converged = False
    final_gnorm = float("inf")
    no_improve = 0

    for step in range(io.refine_lbfgs_max_steps):
        def closure():
            optimizer.zero_grad()
            with torch.no_grad():
                theta_t.clamp_(lo_t, hi_t)
            fitness, _, _ = prob._fitness_and_metrics(theta_t)
            loss = -fitness
            if loss.requires_grad:
                loss.backward()
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
            if final_gnorm < io.refine_converge_tol:
                converged = True
                break

        if no_improve >= 5:
            converged = True
            break

    best_theta = np.clip(best_theta, lo, hi)
    refined_fitness = -best_loss
    improvement = (
        raw_fitness / refined_fitness
        if abs(refined_fitness) > 1e-12 else 0.0
    )

    refined_eval = prob.evaluate(best_theta, with_grads=False)
    ref_rms = refined_eval.metrics.get("rms_spot", abs(refined_fitness))
    ref_efl = refined_eval.metrics.get("efl_actual", 0.0)

    # Basin stability (quick)
    n_stab = io.refine_stability_samples
    perturb_scale = io.refine_stability_perturbation
    bounds_range = hi - lo
    rng = np.random.RandomState(42)
    deltas = []
    for _ in range(n_stab):
        noise = rng.randn(len(best_theta)) * perturb_scale * bounds_range
        perturbed = np.clip(best_theta + noise, lo, hi)
        pr = prob.evaluate(perturbed, with_grads=False)
        deltas.append(pr.fitness - refined_fitness)
    deltas = np.array(deltas)

    stability = {
        "perturbation_scale": perturb_scale,
        "n_samples": n_stab,
        "fitness_delta_mean": round(float(deltas.mean()), 8),
        "fitness_delta_std": round(float(deltas.std()), 8),
        "fitness_delta_worst": round(float(deltas.min()), 8),
        "pct_survived": round(
            float((deltas > -abs(refined_fitness)).mean()) * 100, 1
        ),
        "sensitivity": round(float(deltas.std()), 8),
    }

    return {
        "theta": [float(v) for v in best_theta],
        "fitness": float(refined_fitness),
        "rms_mm": float(ref_rms),
        "rms_um": f"{ref_rms * 1000:.1f}",
        "efl_mm": f"{ref_efl:.2f}",
        "lbfgs_steps": lbfgs_steps,
        "converged": converged,
        "final_grad_norm": round(float(final_gnorm), 8),
        "improvement_ratio": round(float(improvement), 4),
        "raw_fitness": float(raw_fitness),
        "stability": stability,
        "metrics": {k: float(v) for k, v in refined_eval.metrics.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--max", type=int, default=None,
                   help="Refine at most N unrefined elites")
    p.add_argument("--out-suffix", type=str, default="_offline",
                   help="Suffix for JSON filenames to avoid clobbering")
    args = p.parse_args()

    run_dir = args.run_dir
    elite_dir = run_dir / "elite_designs"
    elite_dir.mkdir(exist_ok=True)

    # Load config
    from optical_rl.qd.lensmega.config.loader import load_config
    cfg = load_config(run_dir / "config.yaml")

    # Load elite tracker
    df = pd.read_csv(run_dir / "elite_tracker.csv")
    print(f"[refine] {len(df)} total tracked elites in {run_dir.name}")

    # Find which evals already have refined JSONs
    existing_evals = set()
    for j in elite_dir.glob("*.json"):
        # BEST_0001_1008k.json or elite_0005_3942k.json
        parts = j.stem.split("_")
        try:
            ev = int(parts[-1].rstrip("k"))
            existing_evals.add(ev)
        except (ValueError, IndexError):
            pass

    # Build list of unrefined (eval, theta, fitness, signs) tuples
    # theta is columns c0..c6, g0..g5, d_sensor, and material columns
    theta_cols = [c for c in df.columns
                  if c.startswith("c") or c.startswith("g")
                  or c == "d_sensor" or c.startswith("n") or c.startswith("V")
                  or c.startswith("m") and c[1:].isdigit() is False]

    # The elite tracker has fixed columns: eval,fitness,rms_mm,rms_um,efl_mm,
    # invalid_frac,sign_pattern,sign_match,cosine_sim,c0,c1,c2,c3,c5,c6,g0,g1,
    # g2,g3,g4,g5,d_sensor,m0,m1,[+ materials if present]
    # The theta vector is: [curvatures..., gaps..., flange?, materials?]
    # For cooke_triplet: c0,c1,c2,c3,c5,c6 then g0..g5 then d_sensor
    # But we need the actual theta used (from problem.sample_theta -> theta).
    # The tracker stores each individual component under its name.

    # Rebuild theta from the columns present.
    # We use the ordering from cfg.problem.free_curvature_indices, etc.
    # But tracker only stores curvatures/gaps/d_sensor + measures; not materials!
    # Check if materials columns exist.
    has_materials = any(c.startswith("n") and len(c) <= 2 for c in df.columns) \
                    or "n_0" in df.columns

    print(f"[refine] tracker columns: {list(df.columns)}")
    print(f"[refine] has_materials: {has_materials}")

    # The tracker doesn't always record full theta (e.g., materials).
    # Use the archive as the source of truth.
    print(f"[refine] Loading archive.pkl to recover theta vectors...")
    import pickle
    with open(run_dir / "archive.pkl", "rb") as f:
        archive = pickle.load(f)
    cells = archive.cells if hasattr(archive, "cells") else archive._cells

    # Build eval -> theta map from tracker; but need actual theta.
    # The archive has thetas indexed by (descriptor cell).
    # Match by (fitness, measures) to be safe.
    archive_list = []
    for idx, result in cells.items():
        archive_list.append({
            "fitness": float(result.fitness),
            "measures": np.asarray(result.measures),
            "theta": np.asarray(result.theta),
        })

    def find_theta_for(fitness, m0, m1, tol=1e-6):
        for a in archive_list:
            if abs(a["fitness"] - fitness) < tol \
               and abs(a["measures"][0] - m0) < tol \
               and abs(a["measures"][1] - m1) < tol:
                return a["theta"]
        return None

    # Iterate tracker
    to_refine = []
    for _, row in df.iterrows():
        ev = int(row["eval"])
        if ev in existing_evals:
            continue
        theta = find_theta_for(row["fitness"], row["m0"], row["m1"])
        if theta is None:
            print(f"  skip eval={ev}: no matching archive entry")
            continue
        to_refine.append({
            "eval": ev,
            "theta": theta,
            "fitness": float(row["fitness"]),
            "rms_um": float(row["rms_um"]),
            "efl_mm": float(row["efl_mm"]),
            "sign_pattern": str(row["sign_pattern"]),
            "measures": [float(row["m0"]), float(row["m1"])],
        })

    if args.max is not None:
        to_refine = to_refine[: args.max]

    print(f"[refine] {len(to_refine)} elites to refine "
          f"(already have {len(existing_evals)})")
    print()

    for i, e in enumerate(to_refine, 1):
        tag = f"elite_{i+len(existing_evals):04d}_{e['eval']}k{args.out_suffix}"
        try:
            refined = refine_one(cfg, e["theta"], e["fitness"], cfg.io)
            raw_rms = e["rms_um"]
            ref_rms_str = refined["rms_um"]
            print(f"  [{i}/{len(to_refine)}] eval={e['eval']}k "
                  f"signs=[{e['sign_pattern']}] "
                  f"raw={raw_rms:.1f}μm → refined={ref_rms_str}μm "
                  f"(×{refined['improvement_ratio']:.2f})")

            # Save JSON in same format as the live runner
            info = {
                "eval": e["eval"],
                "elite_index": i + len(existing_evals),
                "fitness": e["fitness"],
                "rms_mm": e["rms_um"] / 1000.0,
                "rms_um": e["rms_um"],
                "efl_mm": e["efl_mm"],
                "sign_pattern": e["sign_pattern"],
                "theta": [float(v) for v in e["theta"]],
                "measures": e["measures"],
                "refined": refined,
            }
            with open(elite_dir / f"{tag}.json", "w") as f:
                json.dump(info, f, indent=2)
        except Exception as exc:
            print(f"  [{i}/{len(to_refine)}] eval={e['eval']}k FAILED: {exc}")

    print(f"\n[refine] Done. Results in {elite_dir}")


if __name__ == "__main__":
    main()
