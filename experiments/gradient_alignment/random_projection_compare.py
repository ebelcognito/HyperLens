"""Compare Seidel descriptor gradient alignment against random-projection
descriptors using the existing gradient log.

For random projection, M(θ) = A^T θ with A ∈ ℝ^(theta_dim × dim_out) drawn
from a fixed Gaussian and column-normalized.  ∇_θ M_j = A[:, j] is constant
in θ, so we only need ∇L(t) — already in the gradient log — and a fresh A.

Outputs:
  experiments/gradient_alignment/random_projection_matrix.npy   (single A)
  experiments/gradient_alignment/random_projection_distribution.npz (100 As)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DIM_OUT = 2
PUBLISHED_SEED = 12345  # matches ablation_random_bc.yaml § 5.2


def make_matrix(seed: int, theta_dim: int, dim_out: int = DIM_OUT):
    """Reproduces RandomProjectionDescriptor.prepare(): RandomState seed,
    randn(theta_dim, dim_out), then column-normalise."""
    rng = np.random.RandomState(seed)
    W = rng.randn(theta_dim, dim_out).astype(np.float64)
    for j in range(dim_out):
        W[:, j] /= np.linalg.norm(W[:, j]) + 1e-30
    return W


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """Row-wise cosine between (N, D) and (D,)."""
    nb = np.linalg.norm(b)
    na = np.linalg.norm(a, axis=1)
    denom = na * nb + eps
    return (a @ b) / denom


def find_log(default_dir: Path) -> Path:
    candidates = sorted(default_dir.glob("*/gradient_log.npz"))
    assert candidates, f"no gradient_log.npz under {default_dir}"
    return candidates[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log",
        default=None,
        help="path to gradient_log.npz (auto-detects from --run-dir if omitted)",
    )
    ap.add_argument(
        "--run-dir",
        default="experiments/gradient_alignment/run",
        help="directory containing <run>/gradient_log.npz when --log is omitted",
    )
    ap.add_argument(
        "--matrix-out",
        default="experiments/gradient_alignment/random_projection_matrix.npy",
    )
    ap.add_argument(
        "--out",
        default="experiments/gradient_alignment/random_projection_distribution.npz",
    )
    ap.add_argument(
        "--cosine-path",
        default="experiments/gradient_alignment/cosine_similarities.npz",
        help="precomputed Seidel cosines, used to read mean |cos| reference values",
    )
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else find_log(Path(args.run_dir))
    print(f"loading: {log_path}")
    d = np.load(log_path)
    grad_L = d["grad_fitness"]
    N, theta_dim = grad_L.shape
    print(f"  N = {N}  theta_dim = {theta_dim}")

    # ----- single matrix matching the §5.2 ablation -----
    A_pub = make_matrix(PUBLISHED_SEED, theta_dim=theta_dim)
    pub_path = Path(args.matrix_out)
    pub_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pub_path, A_pub)
    print(f"saved published matrix (seed={PUBLISHED_SEED}): {pub_path}")
    raw = np.random.RandomState(PUBLISHED_SEED).randn(theta_dim, DIM_OUT)
    print(
        f"  raw randn pre-normalisation: mean={raw.mean():+.4f} "
        f"std={raw.std():.4f}  (expect ~0 ± ~1)"
    )
    print(
        f"  column norms after normalisation: "
        f"{[float(np.linalg.norm(A_pub[:, j])) for j in range(DIM_OUT)]} "
        f"(expect 1.0)"
    )

    # ----- per-iteration cosines for the published A -----
    pub_cos = np.empty((N, DIM_OUT), dtype=np.float64)
    for j in range(DIM_OUT):
        pub_cos[:, j] = cosine(grad_L, A_pub[:, j])
    pub_abs = np.abs(pub_cos)

    print()
    print("Random projection (published seed 12345) per-column |cos| with -∇L:")
    for j in range(DIM_OUT):
        col = pub_abs[:, j]
        col = col[np.isfinite(col)]
        print(
            f"  Random_{j+1}: mean |cos|={col.mean():.4f}  "
            f"median={np.median(col):.4f}  "
            f"frac >0.3={np.mean(col > 0.3):.3f}  "
            f"signed mean={pub_cos[:, j].mean():+.4f}"
        )

    # ----- 100 random matrices, distribution of mean |cos| over trajectory -----
    n_matrices = 100
    rng = np.random.default_rng(20260504)
    seeds = rng.integers(low=0, high=2**31 - 1, size=n_matrices)
    seeds[0] = PUBLISHED_SEED  # include the published one in the distribution
    mean_abs_cos = np.empty((n_matrices, DIM_OUT), dtype=np.float64)
    median_abs_cos = np.empty((n_matrices, DIM_OUT), dtype=np.float64)
    frac_strong = np.empty((n_matrices, DIM_OUT), dtype=np.float64)
    matrices = np.empty((n_matrices, theta_dim, DIM_OUT), dtype=np.float64)
    for k, s in enumerate(seeds):
        A = make_matrix(int(s), theta_dim=theta_dim)
        matrices[k] = A
        for j in range(DIM_OUT):
            cj = cosine(grad_L, A[:, j])
            ac = np.abs(cj)
            ac = ac[np.isfinite(ac)]
            mean_abs_cos[k, j] = float(ac.mean())
            median_abs_cos[k, j] = float(np.median(ac))
            frac_strong[k, j] = float((ac > 0.3).mean())

    # Pool both columns together: 200 samples of "mean |cos| of a random
    # 19-D direction with ∇L over the trajectory."
    pooled = mean_abs_cos.flatten()
    print()
    print(f"Distribution of mean |cos| across {n_matrices} random matrices "
          f"× {DIM_OUT} columns = {pooled.size} random descriptor directions:")
    print(f"  mean         : {pooled.mean():.4f}")
    print(f"  std          : {pooled.std():.4f}")
    print(f"  median       : {np.median(pooled):.4f}")
    print(
        f"  95% range    : [{np.percentile(pooled, 2.5):.4f}, "
        f"{np.percentile(pooled, 97.5):.4f}]"
    )
    print(f"  min, max     : {pooled.min():.4f}, {pooled.max():.4f}")
    print()
    cdat = np.load(args.cosine_path, allow_pickle=True)
    seidel_cos = cdat["cos"]
    seidel_labels = (
        list(cdat["descriptor_labels"])
        if "descriptor_labels" in cdat.files
        else ["S2", "S3"]
    )
    seidel_means = np.array(
        [float(np.mean(np.abs(seidel_cos[:, j])))
         for j in range(seidel_cos.shape[1])],
        dtype=np.float64,
    )
    seidel_S2 = float(seidel_means[0])  # back-compat alias for first descriptor
    seidel_S3 = float(seidel_means[1])  # back-compat alias for second descriptor
    print(
        f"Seidel reference (mean |cos| from {args.cosine_path}): "
        + ", ".join(
            f"{seidel_labels[j]}={seidel_means[j]:.4f}"
            for j in range(len(seidel_labels))
        )
    )
    frac_ge_S2 = float(np.mean(pooled >= seidel_means[0]))
    frac_ge_S3 = float(np.mean(pooled >= seidel_means[1]))
    print(f"  frac mean |cos| >= Seidel S2 ({seidel_S2:.4f}) : "
          f"{frac_ge_S2:.4f}  ({int(frac_ge_S2 * pooled.size)}/{pooled.size})")
    print(f"  frac mean |cos| >= Seidel S3 ({seidel_S3:.4f}) : "
          f"{frac_ge_S3:.4f}  ({int(frac_ge_S3 * pooled.size)}/{pooled.size})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        seeds=seeds.astype(np.int64),
        matrices=matrices,
        mean_abs_cos=mean_abs_cos,
        median_abs_cos=median_abs_cos,
        frac_strong=frac_strong,
        pooled_mean_abs_cos=pooled,
        pub_cos=pub_cos,
        pub_abs_cos=pub_abs,
        seidel_S2=np.float64(seidel_S2),    # back-compat: first descriptor
        seidel_S3=np.float64(seidel_S3),    # back-compat: second descriptor
        seidel_means=seidel_means,
        seidel_labels=np.array(seidel_labels, dtype=object),
        random_null=np.float64(np.sqrt(2.0 / (np.pi * theta_dim))),
        theta_dim=np.int64(theta_dim),
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
