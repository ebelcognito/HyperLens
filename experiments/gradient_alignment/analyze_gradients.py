"""Compute cosine similarities between -∇L and ∇M_j from the gradient log.

Sign convention. The runner stores ``grad_fitness`` = ∇_θ fitness, where
``fitness = -L − penalties``. The CMA-MEGA descent direction on L is
``-∇_θ L = +∇_θ fitness``. So ``cos(grad_fitness, grad_m_j)`` is exactly
the quantity we want: cosine between the descent direction and the
descriptor gradient.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if not np.isfinite(na) or not np.isfinite(nb) or na < eps or nb < eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def compute_all(d):
    grad_fit = d["grad_fitness"]                   # (N, P)
    grad_m = d["grad_m"]                           # (N, K, P)
    N, K, P = grad_m.shape
    cos = np.full((N, K), np.nan, dtype=np.float64)
    cross_cos = np.full((N, K * (K - 1) // 2), np.nan, dtype=np.float64)
    norm_l = np.full(N, np.nan, dtype=np.float64)
    norm_m = np.full((N, K), np.nan, dtype=np.float64)

    pair_idx = []
    for j in range(K):
        for k in range(j + 1, K):
            pair_idx.append((j, k))

    for i in range(N):
        gL = grad_fit[i]
        norm_l[i] = np.linalg.norm(gL)
        for j in range(K):
            gM = grad_m[i, j]
            norm_m[i, j] = np.linalg.norm(gM)
            cos[i, j] = cosine_sim(gL, gM)
        for p, (j, k) in enumerate(pair_idx):
            cross_cos[i, p] = cosine_sim(grad_m[i, j], grad_m[i, k])

    return cos, cross_cos, norm_l, norm_m, pair_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log",
        default=None,
        help="path to gradient_log.npz (auto-detects if omitted)",
    )
    ap.add_argument(
        "--out",
        default="experiments/gradient_alignment/cosine_similarities.npz",
    )
    ap.add_argument(
        "--labels", default="S2,S3",
        help="comma-separated descriptor labels (e.g. 'S4,S2'). "
             "Order must match the config's descriptor.params.terms.",
    )
    args = ap.parse_args()
    labels = [s.strip() for s in args.labels.split(",")]

    log_path = args.log
    if log_path is None:
        candidates = sorted(
            Path("experiments/gradient_alignment/run").glob(
                "*/gradient_log.npz"
            )
        )
        assert candidates, "no gradient_log.npz found under run/"
        log_path = str(candidates[-1])
    print(f"loading: {log_path}")
    d = np.load(log_path)
    print(
        f"  N={len(d['evals'])}  theta_dim={d['thetas'].shape[1]}  "
        f"K={d['grad_m'].shape[1]}"
    )

    cos, cross, norm_l, norm_m, pair_idx = compute_all(d)
    print(
        f"cos shape={cos.shape}  cross shape={cross.shape}  "
        f"NaN frames: {int(np.isnan(cos[:, 0]).sum())}/"
        f"{cos.shape[0]}"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        evals=d["evals"],
        generations=d["generations"],
        cos=cos,
        cross_cos=cross,
        norm_l=norm_l,
        norm_m=norm_m,
        pair_idx=np.array(pair_idx, dtype=np.int64),
        descriptor_labels=np.array(labels, dtype=object),
        fitness=d["fitness"],
        theta_dim=np.int64(d["thetas"].shape[1]),
    )
    print(f"saved: {out}")

    # Sanity checks
    self_cos = cosine_sim(d["grad_fitness"][0], d["grad_fitness"][0])
    print(f"sanity: cos(∇L, ∇L) = {self_cos:.6f}  (should be 1.0)")
    # Random-permutation null. Single perm in 19D has std ~ 1/sqrt(19) ≈ 0.23,
    # so we average over many trials to get a clean null.
    rng = np.random.default_rng(0)
    n_trials = 1000
    n_samples = min(50, len(d["grad_fitness"]))
    perm_cos_samples = []
    for i in range(n_samples):
        g = d["grad_fitness"][i].copy()
        for _ in range(n_trials // n_samples):
            perm_cos_samples.append(cosine_sim(g, rng.permutation(g)))
    perm_arr = np.array(perm_cos_samples)
    perm_arr = perm_arr[np.isfinite(perm_arr)]
    theta_dim = d["thetas"].shape[1]
    print(
        f"sanity: cos(∇L, perm(∇L)) over {len(perm_arr)} samples = "
        f"{float(perm_arr.mean()):+.4f} ± {float(perm_arr.std()):.4f} "
        f"(should be ~0 ± ~{1.0 / np.sqrt(theta_dim):.2f} for {theta_dim}D)"
    )
    finite = np.all(np.isfinite(cos))
    print(f"sanity: all finite cosines = {finite}")
    in_range = np.all((cos[~np.isnan(cos)] >= -1.0001) & (cos[~np.isnan(cos)] <= 1.0001))
    print(f"sanity: cos values in [-1, 1] = {in_range}")

    # Snapshot stats
    K = cos.shape[1]
    if len(labels) != K:
        raise ValueError(
            f"--labels has {len(labels)} entries but descriptor_dim={K}"
        )
    stats = {}
    for j in range(K):
        col = cos[:, j]
        col = col[np.isfinite(col)]
        ac = np.abs(col)
        stats[labels[j]] = {
            "n": int(len(col)),
            "mean": float(np.mean(col)),
            "median": float(np.median(col)),
            "p10": float(np.percentile(col, 10)),
            "p90": float(np.percentile(col, 90)),
            "frac_pos": float(np.mean(col > 0)),
            "frac_strong_pos": float(np.mean(col > 0.3)),
            "frac_strong_neg": float(np.mean(col < -0.3)),
            "mean_abs": float(np.mean(ac)),
            "median_abs": float(np.median(ac)),
            "frac_abs_above_0p3": float(np.mean(ac > 0.3)),
        }
    if cross.shape[1] > 0:
        cc = cross[:, 0]
        cc = cc[np.isfinite(cc)]
        stats["cross_S2_S3"] = {
            "mean": float(np.mean(cc)),
            "median": float(np.median(cc)),
            "abs_mean": float(np.mean(np.abs(cc))),
        }
    stats["norms"] = {
        "mean_norm_L": float(np.mean(norm_l[np.isfinite(norm_l)])),
        "mean_norm_M_S2": float(np.mean(norm_m[:, 0][np.isfinite(norm_m[:, 0])])),
        "mean_norm_M_S3": float(np.mean(norm_m[:, 1][np.isfinite(norm_m[:, 1])])),
    }
    print("\nQuick stats:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
