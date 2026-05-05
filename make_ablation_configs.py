#!/usr/bin/env python3
"""Generate ablation configs (Seidel control + Random BC + Single-objective)
from a base Seidel descriptor config.

Usage:
    python make_ablation_configs.py <base_seidel.yaml> [--iterations N] [--patience P]

Example:
    python make_ablation_configs.py \
        optical_rl/qd/lensmega/config/examples/sixlet_discovery_materials.yaml \
        --iterations 50000 --patience 2

This produces three sibling configs in the same directory:
    ablation_<basename>_seidel_s2s3.yaml      (control - copies base)
    ablation_<basename>_random_bc.yaml         (random projection BCs)
    ablation_<basename>_single_objective.yaml  (1x1 archive, no QD)

For random/single configs, the script computes archive bounds by
sampling random valid designs through the actual problem evaluator.
"""

import argparse
import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


def compute_random_proj_bounds(yaml_dict: dict, seed: int, n_samples: int = 300):
    """Sample random designs, compute random-projection measures,
    return (lo, hi) bounds per axis."""
    # Write a temp config that uses random projection
    tmp_cfg = copy.deepcopy(yaml_dict)
    tmp_cfg["descriptor"] = {
        "kind": "random",
        "params": {"dim": 2, "seed": seed},
    }
    # Remove archive bounds so calibration would run, but we don't actually
    # call the calibration here — we instantiate manually.

    from optical_rl.qd.lensmega.config.schema import RunConfig
    cfg = RunConfig(**tmp_cfg)

    from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem
    descriptor = _build_descriptor(cfg)
    problem = _build_problem(cfg, descriptor)
    problem._ensure_backend()

    thetas = problem.sample_theta(n_samples, seed=42)
    measures = []
    for theta in thetas:
        try:
            r = problem.evaluate(theta, with_grads=False)
            if r.metrics.get("rejected", 0.0) == 0.0 and np.isfinite(r.fitness):
                measures.append(r.measures)
        except Exception:
            pass

    if len(measures) < 30:
        raise RuntimeError(
            f"Only {len(measures)} valid samples — can't compute bounds reliably"
        )

    measures = np.array(measures)
    p2 = np.percentile(measures, 2, axis=0)
    p98 = np.percentile(measures, 98, axis=0)
    span = p98 - p2
    margin = 0.1 * span
    lo = p2 - margin
    hi = p98 + margin
    print(f"  random projection bounds (n={len(measures)} valid): "
          f"[{lo[0]:.2f}, {hi[0]:.2f}], [{lo[1]:.2f}, {hi[1]:.2f}]")
    return lo, hi


def make_seidel_control(base: dict, iterations: int, patience: int) -> dict:
    out = copy.deepcopy(base)
    # Already a Seidel config; just adjust iters/patience and rename.
    out["algorithm"]["iterations"] = iterations
    out["algorithm"]["restart_patience"] = patience
    return out


def make_random_config(base: dict, iterations: int, patience: int,
                       lo: np.ndarray, hi: np.ndarray, seed: int) -> dict:
    out = copy.deepcopy(base)
    out["descriptor"] = {
        "kind": "random",
        "params": {"dim": 2, "seed": seed},
    }
    out["archive"]["bins"] = [256, 256]
    out["archive"]["bounds"] = [
        [float(lo[0]), float(hi[0])],
        [float(lo[1]), float(hi[1])],
    ]
    out["algorithm"]["iterations"] = iterations
    out["algorithm"]["restart_patience"] = patience
    return out


def make_single_obj_config(base: dict, iterations: int, patience: int,
                           lo: np.ndarray, hi: np.ndarray, seed: int) -> dict:
    out = copy.deepcopy(base)
    out["descriptor"] = {
        "kind": "random",
        "params": {"dim": 2, "seed": seed},
    }
    out["archive"]["bins"] = [1, 1]
    out["archive"]["bounds"] = [
        [float(lo[0]), float(hi[0])],
        [float(lo[1]), float(hi[1])],
    ]
    out["algorithm"]["iterations"] = iterations
    out["algorithm"]["restart_patience"] = patience
    out["algorithm"]["new_cell_bonus"] = 0.0
    return out


def write_yaml(cfg: dict, path: Path, header: str) -> None:
    path.write_text(header + "\n" + yaml.dump(cfg, sort_keys=False, default_flow_style=None))
    print(f"  wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base_yaml", type=Path,
                   help="Base Seidel-descriptor config to clone")
    p.add_argument("--iterations", type=int, default=50000,
                   help="Generations to run for all 3 configs (default: 50000)")
    p.add_argument("--patience", type=int, default=2,
                   help="Restart patience (default: 2)")
    p.add_argument("--random-seed", type=int, default=12345,
                   help="Seed for random projection (default: 12345)")
    p.add_argument("--single-seed", type=int, default=99999,
                   help="Seed for single-obj random projection (default: 99999)")
    args = p.parse_args()

    base_path = args.base_yaml
    if not base_path.exists():
        print(f"ERROR: {base_path} not found")
        sys.exit(1)

    base = yaml.safe_load(base_path.read_text())
    if base["descriptor"]["kind"] != "seidel":
        print(f"ERROR: base config must use seidel descriptor, got {base['descriptor']['kind']!r}")
        sys.exit(1)

    base_stem = base_path.stem
    if base_stem.startswith("ablation_"):
        base_stem = base_stem[len("ablation_"):]

    out_dir = base_path.parent
    print(f"Generating ablation configs from {base_path.name}")
    print(f"Iterations: {args.iterations}, Patience: {args.patience}")
    print()

    # Compute bounds for random + single configs
    print("Computing random projection bounds (this samples ~300 random designs)...")
    lo_rand, hi_rand = compute_random_proj_bounds(base, args.random_seed)
    lo_single, hi_single = compute_random_proj_bounds(base, args.single_seed)
    print()

    # Generate 3 configs
    seidel_cfg = make_seidel_control(base, args.iterations, args.patience)
    seidel_cfg["io"]["run_name"] = f"ablation_{base_stem}_seidel_s2s3"

    random_cfg = make_random_config(base, args.iterations, args.patience,
                                    lo_rand, hi_rand, args.random_seed)
    random_cfg["io"]["run_name"] = f"ablation_{base_stem}_random_bc"

    single_cfg = make_single_obj_config(base, args.iterations, args.patience,
                                        lo_single, hi_single, args.single_seed)
    single_cfg["io"]["run_name"] = f"ablation_{base_stem}_single_objective"

    print("Writing configs:")
    write_yaml(
        seidel_cfg,
        out_dir / f"ablation_{base_stem}_seidel_s2s3.yaml",
        f"# ABLATION CONTROL: Seidel S2+S3 BCs (cloned from {base_path.name})\n"
        f"# Iterations: {args.iterations}, restart_patience: {args.patience}",
    )
    write_yaml(
        random_cfg,
        out_dir / f"ablation_{base_stem}_random_bc.yaml",
        f"# ABLATION: Random Projection BCs (control for diversity)\n"
        f"# Same problem/algorithm as seidel; descriptor swapped to random projection\n"
        f"# Iterations: {args.iterations}, restart_patience: {args.patience}",
    )
    write_yaml(
        single_cfg,
        out_dir / f"ablation_{base_stem}_single_objective.yaml",
        f"# ABLATION: Single-Objective (1x1 archive, no QD diversity)\n"
        f"# Random projection used to satisfy descriptor interface but ignored\n"
        f"# Iterations: {args.iterations}, restart_patience: {args.patience}",
    )

    print()
    print("Run with:")
    for cond in ["seidel_s2s3", "random_bc", "single_objective"]:
        print(f"  python3 -m optical_rl.qd.lensmega.run --config "
              f"optical_rl/qd/lensmega/config/examples/ablation_{base_stem}_{cond}.yaml")


if __name__ == "__main__":
    main()
