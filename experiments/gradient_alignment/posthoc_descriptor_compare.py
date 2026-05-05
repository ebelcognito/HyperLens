"""Compare descriptor candidates on three axes: speed × discriminativeness × alignment.

For each candidate descriptor, we replay a random subset of parent θ from a
logged run and measure:

 1. **Speed**: median wall-time per (forward + autograd) descriptor
    evaluation, in ms.
 2. **Discriminativeness**: std of each descriptor component across the
    sampled θ; effective rank of the descriptor matrix
    (participation-ratio of singular values), giving how many independent
    "axes" the descriptor actually uses.
 3. **Alignment**: mean ``|cos(-∇L, ∇M_j)|`` per descriptor dimension,
    using ``∇L`` from the gradient log (re-uses the deterministic stored
    value, no new fitness ray-trace).

This does NOT require a new HyperLens run. It loads existing
``gradient_log.npz`` files and a config, builds the candidate descriptor,
and runs autograd through it on each θ.
"""
from __future__ import annotations

import argparse
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import json
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.descriptors.seidel import SeidelDescriptor
from optical_rl.qd.lensmega.descriptors.zernike import ZernikeDescriptor
from optical_rl.qd.lensmega.problem.full_dof import FullDOFLensProblem
from optical_rl.qd.lensmega.problem.lens_problem import LensProblemConfig


def build_problem(cfg, descriptor):
    p = cfg.problem
    device = "cpu" if cfg.io.device == "auto" else cfg.io.device
    prob_cfg = LensProblemConfig(
        lens_path=p.lens_path,
        wavelength_um=p.wavelength_um,
        fitness_wavelengths_um=tuple(p.fitness_wavelengths_um),
        object_depth_mm=p.object_depth_mm,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        focal_length_target_mm=p.focal_length_target_mm,
        theta_bounds=tuple(tuple(b) for b in p.theta_bounds),
        stage=p.stage,
        num_rays=p.num_rays,
        invalid_tol=p.invalid_tol,
        invalid_penalty_weight=p.invalid_penalty_weight,
        intersect_penalty_weight=p.intersect_penalty_weight,
        air_min_mm=p.air_min_mm,
        thick_min_mm=p.thick_min_mm,
        strict_entrance_pupil=p.strict_entrance_pupil,
        descriptor_field_index=p.descriptor_field_index,
        parameterization=p.parameterization,
        efl_penalty_weight=p.efl_penalty_weight,
        free_gap_indices=tuple(p.free_gap_indices),
        variable_flange=p.variable_flange,
        flange_bounds=tuple(p.flange_bounds) if p.flange_bounds else None,
        variable_materials=p.variable_materials,
        free_material_indices=tuple(p.free_material_indices),
        cemented_interfaces=tuple(p.cemented_interfaces),
        field_weights=tuple(p.field_weights) if p.field_weights else None,
        device=device,
        enforce_fnum=p.enforce_fnum,
        f_number_target=p.f_number_target,
        fnum_penalty_weight=p.fnum_penalty_weight,
    )
    return FullDOFLensProblem(prob_cfg, descriptor)


def cosine_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return np.einsum("ij,ij->i", a, b) / (na * nb + 1e-30)


def effective_rank(values: np.ndarray) -> float:
    """Participation-ratio rank: ((sum s_i)^2 / sum s_i^2) on the SVs of
    the centred descriptor matrix. Tells you how many independent
    directions the descriptor actually uses across the sampled θ.
    For a 1-D descriptor this is 1.0; for K perfectly independent dims
    with equal variance it is K.
    """
    if values.size == 0:
        return 0.0
    X = values - values.mean(axis=0, keepdims=True)
    if X.shape[1] == 1:
        return 1.0
    s = np.linalg.svd(X, compute_uv=False)
    s2 = s ** 2
    if s2.sum() == 0:
        return 0.0
    return float((s2.sum() ** 2) / (s2 ** 2).sum())


def evaluate_candidate(
    name: str,
    descriptor_factory,
    cfg,
    thetas: np.ndarray,
    grad_L: np.ndarray,
    log_every: int = 100,
) -> dict:
    descriptor = descriptor_factory()
    problem = build_problem(cfg, descriptor)
    problem._ensure_backend()
    K = descriptor.dim
    N, P = thetas.shape

    grads_M = np.empty((N, K, P), dtype=np.float64)
    measures = np.empty((N, K), dtype=np.float64)
    times_ms = np.empty(N, dtype=np.float64)

    print(f"[{name}] dim={K}, evaluating {N} θ…", flush=True)
    t_total0 = time.perf_counter()
    for i in range(N):
        t0 = time.perf_counter()
        r = problem.evaluate(thetas[i].astype(np.float64), with_grads=True)
        times_ms[i] = (time.perf_counter() - t0) * 1000.0
        for k in range(K):
            grads_M[i, k] = r.grads[f"m{k}"]
        measures[i] = r.measures
        if (i + 1) % log_every == 0 or i == N - 1:
            elapsed = time.perf_counter() - t_total0
            print(
                f"  {i+1}/{N}  {(i+1)/elapsed:5.1f} eval/s  "
                f"med={np.median(times_ms[:i+1]):5.1f} ms",
                flush=True,
            )
    total_s = time.perf_counter() - t_total0

    # Per-dim alignment with stored ∇L
    abs_cos = np.empty((N, K), dtype=np.float64)
    for k in range(K):
        abs_cos[:, k] = np.abs(cosine_rowwise(grad_L, grads_M[:, k, :]))

    # Joint subspace projection: how much of ∇L lives in span(∇M_1..∇M_K)?
    # For each θ, project grad_L[i] onto the K-dim row-space of grads_M[i]
    # and report ||proj||/||grad_L||.
    proj_frac = np.empty(N, dtype=np.float64)
    for i in range(N):
        M = grads_M[i]                                # (K, P)
        gL = grad_L[i]
        try:
            # Project gL onto rowspace(M). Use lstsq for stability.
            coeff, *_ = np.linalg.lstsq(M.T, gL, rcond=None)
            proj = M.T @ coeff
            num = np.linalg.norm(proj)
            den = np.linalg.norm(gL)
            proj_frac[i] = float(num / (den + 1e-30))
        except np.linalg.LinAlgError:
            proj_frac[i] = float("nan")

    # Discriminativeness
    measure_std = measures.std(axis=0)
    eff_rank = effective_rank(measures)

    return {
        "name": name,
        "dim": int(K),
        "n_theta": int(N),
        "speed_ms_median": float(np.median(times_ms)),
        "speed_ms_p90": float(np.percentile(times_ms, 90)),
        "total_s": float(total_s),
        "abs_cos_per_dim_mean": [float(x) for x in abs_cos.mean(axis=0)],
        "abs_cos_max_over_dims_mean": float(np.max(abs_cos, axis=1).mean()),
        "joint_proj_frac_mean": float(np.mean(proj_frac[np.isfinite(proj_frac)])),
        "joint_proj_frac_median": float(np.median(proj_frac[np.isfinite(proj_frac)])),
        "measure_std": [float(x) for x in measure_std],
        "effective_rank": float(eff_rank),
    }


CANDIDATES = [
    # Seidel baselines (these should match what we already saw)
    ("Seidel[S4,S2]", lambda: SeidelDescriptor(["S4", "S2"])),
    ("Seidel[S2,S3]", lambda: SeidelDescriptor(["S2", "S3"])),
    ("Seidel[S1,S2,S3,S4,S5]",
     lambda: SeidelDescriptor(["S1", "S2", "S3", "S4", "S5"])),
    # Zernike at the original triplet config: 3rd-order coma + spherical
    ("Zernike[7,11]",        lambda: ZernikeDescriptor([7, 11],   num_rays=64)),
    # Higher-order coma + spherical (5th order)
    ("Zernike[9,14]",        lambda: ZernikeDescriptor([9, 14],   num_rays=64)),
    # 4-D combo: 3rd + 5th order
    ("Zernike[7,9,11,14]",   lambda: ZernikeDescriptor([7, 9, 11, 14], num_rays=64)),
    # Wide Zernike sweep covering 3rd → 4th radial order (capped by codebase
    # at Noll 16; 5th-order Zernike Nolls 17+ are not implemented here).
    ("Zernike[4..16]",       lambda: ZernikeDescriptor(
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], num_rays=64
    )),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--n-theta", type=int, default=500,
        help="random subset size of parent θ to evaluate against (0 = all)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    print(f"loading log: {args.log}")
    d = np.load(args.log)
    thetas_full = d["thetas"]
    grad_L_full = d["grad_fitness"]
    N_total = thetas_full.shape[0]

    if args.n_theta > 0 and args.n_theta < N_total:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(N_total, size=args.n_theta, replace=False))
        thetas = thetas_full[idx]
        grad_L = grad_L_full[idx]
        print(
            f"  random subset of {args.n_theta} of {N_total} parent θ "
            f"(seed={args.seed})"
        )
    else:
        thetas = thetas_full
        grad_L = grad_L_full
        print(f"  using all {N_total} parent θ")
    print(f"  theta_dim={thetas.shape[1]}")

    results = []
    for name, factory in CANDIDATES:
        try:
            res = evaluate_candidate(name, factory, cfg, thetas, grad_L)
            results.append(res)
            print(
                f"  ✓ {name:32s}  speed={res['speed_ms_median']:5.1f} ms  "
                f"alignment={res['joint_proj_frac_mean']:.3f}  "
                f"effrank={res['effective_rank']:.2f}",
                flush=True,
            )
        except Exception as e:
            print(f"  ✗ {name:32s}  FAILED: {e}", flush=True)
            results.append({"name": name, "error": str(e)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved: {out}")

    # Pretty summary
    print("\n" + "=" * 78)
    print(
        f"{'descriptor':32s}  {'dim':>3}  {'speed':>8}  "
        f"{'effrk':>5}  {'<|cos|>':>7}  {'joint':>6}"
    )
    print("-" * 78)
    for res in results:
        if "error" in res:
            print(f"{res['name']:32s}  ERROR: {res['error']}")
            continue
        avg_cos = float(np.mean(res["abs_cos_per_dim_mean"]))
        print(
            f"{res['name']:32s}  {res['dim']:3d}  "
            f"{res['speed_ms_median']:6.1f}ms  "
            f"{res['effective_rank']:5.2f}  "
            f"{avg_cos:7.3f}  {res['joint_proj_frac_mean']:6.3f}"
        )
    print("=" * 78)
    print("speed: median (forward+autograd) wall-time per θ")
    print("effrk: effective rank of descriptor matrix (≤ dim)")
    print("<|cos|>: mean per-dim |cos(-∇L, ∇M_j)|")
    print("joint:  ||proj of ∇L onto span(∇M_*)|| / ||∇L|| (∈ [0, 1])")


if __name__ == "__main__":
    main()
