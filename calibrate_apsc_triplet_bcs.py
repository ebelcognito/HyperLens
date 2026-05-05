"""
Pre-search BC calibration for the APS-C 85 mm F/4 cooke triplet.

Sample N theta uniformly within the config's bounds, evaluate each via the
LensProblem (no grads), filter to plausibly-feasible designs, then score
all C(5,2)=10 Seidel pairs by:

  - per-axis std (standardised to z-scores per Seidel across the sample)
  - Pearson |r| between the two axes (orthogonality; lower = more independent info)
  - 2-D effective rank of the (N, 2) BC matrix (closer to 2 = better plane fill)
  - bin coverage at the proposed archive resolution (256 x 256 inside ±2 sigma)
  - Spearman |rho| between |BC| (L2 norm in the 2-D plane) and -log RMS
    (does BC magnitude track quality on this template?)

Output: ranked table printed to stdout. No plots, no artifacts written.

Usage:
    PYTHONPATH=. python3 calibrate_apsc_triplet_bcs.py \\
        --config optical_rl/qd/lensmega/config/examples/cooke_apsc_85mm_f4_discovery.yaml \\
        [--n 1000] [--seed 0] [--workers auto]
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.descriptors.seidel import SeidelDescriptor
from optical_rl.qd.lensmega.run import _build_problem


SEIDELS = ["S1", "S2", "S3", "S4", "S5"]


def effective_rank(M: np.ndarray) -> float:
    """Participation-ratio rank: (sum sigma)^2 / sum sigma^2. Range [1, k]."""
    if M.size == 0:
        return 0.0
    M = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    return float((s.sum() ** 2) / (s ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=1000, help="uniform-theta sample count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--efl-tol", type=float, default=0.20,
                    help="EFL must be within +/- this fraction of target to count as feasible")
    ap.add_argument("--invalid-frac-max", type=float, default=0.20,
                    help="reject samples with invalid_fraction above this")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bounds = np.asarray(cfg.problem.theta_bounds, dtype=np.float64)
    lo, hi = bounds[:, 0], bounds[:, 1]
    P = bounds.shape[0]
    print(f"[calib] theta dim = {P}")
    print(f"[calib] EFL target = {cfg.problem.focal_length_target_mm} mm  "
          f"(feasibility window +/- {100*args.efl_tol:.0f}%)")

    descriptor = SeidelDescriptor(seidel_terms=tuple(SEIDELS))
    problem = _build_problem(cfg, descriptor)
    problem._ensure_backend()

    rng = np.random.default_rng(args.seed)
    thetas = lo + (hi - lo) * rng.random((args.n, P))

    t0 = time.time()
    valid_mask = np.zeros(args.n, dtype=bool)
    seidels = np.full((args.n, 5), np.nan, dtype=np.float64)
    rms = np.full(args.n, np.nan, dtype=np.float64)
    fitness = np.full(args.n, np.nan, dtype=np.float64)
    efl = np.full(args.n, np.nan, dtype=np.float64)
    invf = np.full(args.n, np.nan, dtype=np.float64)
    for i in range(args.n):
        try:
            res = problem.evaluate(thetas[i], with_grads=False)
        except Exception:
            continue
        m = res.metrics or {}
        try:
            seidels[i] = [float(m[f"seidel_{s}"]) for s in SEIDELS]
        except KeyError:
            continue
        rms[i] = float(m.get("rms_spot", np.nan))
        efl[i] = float(m.get("efl_mm", m.get("efl", np.nan)))
        invf[i] = float(m.get("invalid_fraction", 1.0))
        fitness[i] = float(res.fitness)
        valid_mask[i] = (
            np.isfinite(seidels[i]).all()
            and np.isfinite(rms[i]) and rms[i] > 0
            and invf[i] <= args.invalid_frac_max
            and (
                not np.isfinite(efl[i]) or
                abs(efl[i] - cfg.problem.focal_length_target_mm)
                <= args.efl_tol * cfg.problem.focal_length_target_mm
            )
        )
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{args.n} samples evaluated, "
                  f"valid so far = {int(valid_mask[:i+1].sum())} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    n_valid = int(valid_mask.sum())
    print(f"[calib] valid samples: {n_valid}/{args.n}  "
          f"({100*n_valid/args.n:.1f}%)  in {time.time()-t0:.1f}s")
    if n_valid < 50:
        print("[calib] WARNING: <50 valid samples — pair metrics will be noisy.")

    if n_valid == 0:
        sys.exit(2)

    S = seidels[valid_mask]            # (n_valid, 5)
    R = rms[valid_mask]
    log_qual = -np.log(R)              # higher = better-quality design

    # standardise per-Seidel before pair scoring (so the metrics aren't
    # dominated by absolute-magnitude differences across S1..S5)
    S_z = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-30)

    # bins for archive coverage simulation (matches cfg.archive.bins[0]=256)
    bins = int(cfg.archive.bins[0])

    rows = []
    for i, j in itertools.combinations(range(5), 2):
        Mz = S_z[:, [i, j]]
        # per-axis std on RAW Seidel (informative scale-aware)
        std_i = float(S[:, i].std())
        std_j = float(S[:, j].std())
        # Pearson |r| on standardised (same as raw)
        r = float(np.corrcoef(Mz[:, 0], Mz[:, 1])[0, 1])
        # 2-D effective rank on standardised
        erk = effective_rank(Mz)
        # 2-D bin coverage inside [-2, 2] sigma (256x256 grid), matches archive
        clip = np.clip(Mz, -2.0, 2.0)
        ix = np.clip(((clip[:, 0] + 2.0) / 4.0 * bins).astype(int), 0, bins - 1)
        iy = np.clip(((clip[:, 1] + 2.0) / 4.0 * bins).astype(int), 0, bins - 1)
        cov = len(set(zip(ix.tolist(), iy.tolist()))) / (bins * bins)
        # |spearman| between BC magnitude and quality (predicts whether BC
        # magnitude is *informative* about RMS — non-zero in either
        # direction is good; zero means the BC is fitness-orthogonal,
        # which is canonically what MAP-Elites *wants* but is also a
        # signal that the BC may not be tracking the loss landscape)
        bc_mag = np.linalg.norm(Mz, axis=1)
        rho = float(st.spearmanr(bc_mag, log_qual).statistic)
        rows.append({
            "pair": f"{SEIDELS[i]}/{SEIDELS[j]}",
            "std_i_raw": std_i,
            "std_j_raw": std_j,
            "abs_r": abs(r),
            "eff_rank_2d": erk,
            "bin_cov": cov,
            "abs_rho_qual": abs(rho),
        })

    # composite score: high effective rank (toward 2.0), low |corr|, high coverage.
    # |rho_qual| is informational, NOT in the composite — see notes.
    for row in rows:
        row["composite"] = (
            (row["eff_rank_2d"] - 1.0)            # 0 -> bad, 1 -> ideal
            + (1.0 - row["abs_r"])                # 0 -> bad, 1 -> ideal
            + 4.0 * row["bin_cov"]                # 0 -> bad, ~1 -> ideal (cov is small in absolute terms)
        )
    rows.sort(key=lambda r: -r["composite"])

    print()
    print(f"{'rank':<5}{'pair':<8}{'std_i':>10}{'std_j':>10}"
          f"{'|corr|':>8}{'effrk':>8}{'bin_cov':>9}{'|rho_q|':>9}{'score':>8}")
    print("-" * 73)
    for k, row in enumerate(rows, 1):
        print(f"{k:<5}{row['pair']:<8}"
              f"{row['std_i_raw']:>10.3e}{row['std_j_raw']:>10.3e}"
              f"{row['abs_r']:>8.3f}{row['eff_rank_2d']:>8.3f}"
              f"{row['bin_cov']:>9.4f}{row['abs_rho_qual']:>9.3f}"
              f"{row['composite']:>8.3f}")

    print()
    print("Reading guide:")
    print("  effrk near 2.0 = the pair fills its 2-D plane, no collapse to a line.")
    print("  |corr| near 0  = the two axes carry independent information.")
    print("  bin_cov        = fraction of a 256x256 archive grid that is hit by")
    print("                    the uniform-theta sample inside +/-2 sigma.")
    print("  |rho_q|        = |Spearman(|BC|, -log RMS)|. Diagnostic only:")
    print("                    high = BC magnitude tracks design quality (good")
    print("                    for finding the optimum but redundant with the")
    print("                    loss); low = BC is fitness-blind (canonical MAP-")
    print("                    Elites pose). Don't optimise this one direction.")
    print("  score          = (effrk-1) + (1-|corr|) + 4*bin_cov, equal weights.")


if __name__ == "__main__":
    main()
