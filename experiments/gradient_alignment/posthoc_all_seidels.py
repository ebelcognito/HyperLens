"""Post-hoc gradient alignment for all five Seidel coefficients.

Replays each parent θ from a logged run and asks: what is
cos(−∇L(θ), ∇M_k(θ)) for k ∈ {S1, S2, S3, S4, S5}?

Only S2 and S3 had gradients computed during the original run (those were
the BC pair). For the other three terms we recompute the descriptor +
gradient on each logged θ. We also recompute ∇M_S2 / ∇M_S3 as a sanity
check against the stored values.

This does NOT re-run the search. It re-evaluates 1× per logged generation
under the same `LensProblem` config but with the descriptor swapped to all
five Seidel terms.
"""
from __future__ import annotations

import argparse
import os

# Match the runner's threading discipline before torch imports.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.descriptors.seidel import SeidelDescriptor
from optical_rl.qd.lensmega.problem.full_dof import FullDOFLensProblem
from optical_rl.qd.lensmega.problem.lens_problem import LensProblemConfig

ALL_FIVE = ("S1", "S2", "S3", "S4", "S5")


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
    if p.parameterization != "full_dof":
        raise NotImplementedError(
            "this script currently assumes parameterization='full_dof'; "
            "extend if you need the other modes"
        )
    return FullDOFLensProblem(prob_cfg, descriptor)


def cosine_rowwise(grad_L: np.ndarray, grad_M: np.ndarray) -> np.ndarray:
    """grad_L: (N, P)   grad_M: (N, P)  →  (N,)"""
    nl = np.linalg.norm(grad_L, axis=1)
    nm = np.linalg.norm(grad_M, axis=1)
    denom = nl * nm + 1e-30
    out = np.einsum("ij,ij->i", grad_L, grad_M) / denom
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to gradient_log.npz")
    ap.add_argument("--config", required=True, help="run config YAML")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument(
        "--limit", type=int, default=0,
        help="max # of θ to process (0 = all). Useful for quick smoke tests.",
    )
    ap.add_argument(
        "--descriptor-field-index", type=int, default=None,
        help=(
            "override problem.descriptor_field_index for this post-hoc. "
            "0=on-axis, 1=mid-FOV, 2=full-field. Default: use the value in "
            "the config."
        ),
    )
    args = ap.parse_args()

    log_path = Path(args.log)
    cfg = load_config(Path(args.config))
    if args.descriptor_field_index is not None:
        original_idx = cfg.problem.descriptor_field_index
        cfg.problem.descriptor_field_index = int(args.descriptor_field_index)
        print(
            f"descriptor_field_index override: "
            f"{original_idx} → {cfg.problem.descriptor_field_index}"
        )

    print(f"loading log: {log_path}")
    d = np.load(log_path)
    thetas = d["thetas"]                 # (N, P)
    grad_L_stored = d["grad_fitness"]    # (N, P)  = +∇fitness = -∇L
    N, P = thetas.shape
    if args.limit > 0:
        N = min(N, int(args.limit))
        thetas = thetas[:N]
        grad_L_stored = grad_L_stored[:N]
    print(f"  {N} parent generations, theta_dim={P}")

    descriptor = SeidelDescriptor(seidel_terms=ALL_FIVE)
    problem = build_problem(cfg, descriptor)
    problem._ensure_backend()
    print("  backend ready, evaluating with all 5 Seidels…")

    K = 5
    grads_M = np.empty((N, K, P), dtype=np.float64)
    grad_L_recomputed = np.empty((N, P), dtype=np.float64)
    measures = np.empty((N, K), dtype=np.float64)

    t0 = time.time()
    log_every = max(1, N // 20)
    for i in range(N):
        r = problem.evaluate(thetas[i].astype(np.float64), with_grads=True)
        assert r.grads is not None
        grad_L_recomputed[i] = r.grads["fitness"]
        for k in range(K):
            grads_M[i, k] = r.grads[f"m{k}"]
        measures[i] = r.measures
        if (i + 1) % log_every == 0 or i == N - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (N - i - 1) / max(rate, 1e-9)
            print(
                f"    {i+1}/{N}  rate={rate:5.1f} evals/s  "
                f"eta={eta/60:4.1f} min",
                flush=True,
            )

    # Sanity: recomputed ∇L should match the stored one. They use the same
    # forward graph + RNG-free deterministic ray sampler, so they should agree
    # to ~float32 precision.
    diff = np.linalg.norm(
        grad_L_recomputed - grad_L_stored, axis=1
    ) / (np.linalg.norm(grad_L_stored, axis=1) + 1e-30)
    print(
        f"\nrecomputed-vs-stored ∇L relative error: "
        f"mean={diff.mean():.3e}  median={np.median(diff):.3e}  "
        f"p99={np.percentile(diff, 99):.3e}"
    )

    # Per-generation cosines using STORED grad_L (matches what the previous
    # analysis used so triplet S2/S3 numbers reproduce exactly).
    cos = np.empty((N, K), dtype=np.float64)
    for k in range(K):
        cos[:, k] = cosine_rowwise(grad_L_stored, grads_M[:, k, :])

    # Sanity vs. the earlier S2/S3 numbers
    print("\nSanity (vs. previously logged S2/S3):")
    for label, idx in [("S2", 1), ("S3", 2)]:
        col = cos[:, idx]
        col = col[np.isfinite(col)]
        print(
            f"  mean cos {label}: {col.mean():+.4f}   "
            f"mean |cos|: {np.abs(col).mean():.4f}"
        )

    # Quick stats
    print("\nMean signed cos / mean |cos| per Seidel term:")
    for k, term in enumerate(ALL_FIVE):
        col = cos[:, k]
        col = col[np.isfinite(col)]
        ac = np.abs(col)
        print(
            f"  {term:5s}  mean={col.mean():+.4f}  median={np.median(col):+.4f}"
            f"   |cos| mean={ac.mean():.4f} median={np.median(ac):.4f}"
            f"   frac |cos|>0.3={float((ac > 0.3).mean()):.3f}"
        )

    # Cross-Seidel similarities (only |cos| since Seidel signs are conventional)
    cross = np.empty((N, K, K), dtype=np.float64)
    for j in range(K):
        for k in range(K):
            if j == k:
                cross[:, j, k] = 1.0
            elif j < k:
                cross[:, j, k] = cosine_rowwise(
                    grads_M[:, j, :], grads_M[:, k, :]
                )
            else:
                cross[:, j, k] = cross[:, k, j]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        cos=cos,
        cross=cross,
        grads_M=grads_M,
        measures=measures,
        grad_L_recomputed=grad_L_recomputed,
        diff_recomputed_vs_stored=diff,
        terms=np.array(ALL_FIVE, dtype=object),
        theta_dim=np.int64(P),
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
