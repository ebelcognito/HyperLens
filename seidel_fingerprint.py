"""Test whether Seidel acts as a good design-identity fingerprint,
stratified by spot-RMS regime.

Not the same experiment as ``seidel_discrimination.py``
---------------------------------------------------------

``seidel_discrimination.py`` asks: "does Seidel predict spot RMS?"
   R^2(log|S1..S5| -> log RMS)  per tier.

This script asks: "does Seidel fingerprint design identity?"
   Pearson r(d_theta, d_Seidel)  over pairs  per tier.

The thesis is about the second question.  MAP-Elites needs Seidel
cells to correspond to genuinely different designs — i.e. design-space
distance should track Seidel-space distance — for the archive's
diversity pressure to do any real work.  Predicting RMS (question 1)
is orthogonal: two very-different designs with identical RMS still
benefit from being separated in Seidel space because they land in
different aberration regimes.

Method
------

From ``archive_cells.json``:
  - collect each cell's (theta, |S1..S5|, rms_spot).
  - normalize theta to [0, 1] by its empirical range.
  - sample pairs (i, j) — optionally per-tier to equalise sample size.
  - d_theta(i,j)  = L2 distance in normalized-theta space.
  - d_S(i,j)      = L2 distance in log|S| space.
  - report Pearson r(d_theta, d_S), 95% bootstrap CI, per tier.

Thesis predicts:
  r is HIGH in the sub-optimal (large-RMS) tier — Seidel fingerprints
    designs because third-order aberrations dominate and differ between
    designs.
  r DROPS toward 0 in the well-corrected (small-RMS) tier — all |S|
    values collapse toward zero, erasing the design fingerprint.

Usage
-----

    python seidel_fingerprint.py <run_dir> [--tiers 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def _load(archive_cells_path: Path) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, dict
]:
    with archive_cells_path.open() as f:
        d = json.load(f)
    cells = d.get("cells", [])
    thetas: List[List[float]] = []
    seidels: List[List[float]] = []
    rms_vals: List[float] = []
    n_missing = 0
    for c in cells:
        m = c.get("metrics", {})
        theta = c.get("theta")
        rms = m.get("rms_spot")
        s = [m.get(f"seidel_S{k}") for k in (1, 2, 3, 4, 5)]
        if theta is None or rms is None or any(x is None for x in s):
            n_missing += 1
            continue
        if not np.isfinite(rms) or rms <= 0:
            n_missing += 1
            continue
        thetas.append([float(x) for x in theta])
        seidels.append([float(x) for x in s])
        rms_vals.append(float(rms))
    if not seidels:
        raise RuntimeError("no cells with theta + rms_spot + seidel_S1..S5")
    info = {
        "source": str(archive_cells_path),
        "n_total": len(cells),
        "n_kept": len(seidels),
        "n_missing": n_missing,
    }
    return (
        np.asarray(thetas, dtype=np.float64),
        np.abs(np.asarray(seidels, dtype=np.float64)),
        np.asarray(rms_vals, dtype=np.float64),
        info,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = (np.sqrt((xm * xm).sum()) * np.sqrt((ym * ym).sum())) + 1e-300
    return float((xm * ym).sum() / denom)


def _normalize_theta(theta: np.ndarray) -> np.ndarray:
    """Map each theta dim to [0, 1] via its empirical min/max range."""
    lo = theta.min(axis=0, keepdims=True)
    hi = theta.max(axis=0, keepdims=True)
    rng = np.clip(hi - lo, 1e-12, None)
    return (theta - lo) / rng


def _bootstrap_r(
    ds: np.ndarray, dt: np.ndarray, *, n_boot: int = 300, seed: int = 0,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = ds.size
    if n < 5:
        return (float("nan"), float("nan"))
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = _pearson(ds[idx], dt[idx])
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------

def analyze(
    theta_norm: np.ndarray,
    log_S: np.ndarray,
    log_R: np.ndarray,
    *,
    n_tiers: int,
    pairs_per_tier: int,
    seed: int,
) -> dict:
    # ---- Global ----
    n = theta_norm.shape[0]
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, n, size=pairs_per_tier)
    jj = rng.integers(0, n, size=pairs_per_tier)
    mask = ii != jj
    ii, jj = ii[mask], jj[mask]
    d_theta = np.linalg.norm(theta_norm[ii] - theta_norm[jj], axis=1)
    d_S = np.linalg.norm(log_S[ii] - log_S[jj], axis=1)
    r_global = _pearson(d_theta, d_S)
    lo_g, hi_g = _bootstrap_r(d_S, d_theta, seed=seed + 99)

    # ---- Tier edges on log RMS ----
    edges = np.quantile(log_R, np.linspace(0.0, 1.0, n_tiers + 1))
    tiers = []
    for i in range(n_tiers):
        lo, hi = edges[i], edges[i + 1]
        if i == n_tiers - 1:
            mask = (log_R >= lo) & (log_R <= hi)
        else:
            mask = (log_R >= lo) & (log_R < hi)
        idx_in = np.where(mask)[0]
        n_in = idx_in.size
        if n_in < 10:
            tiers.append({
                "lo_rms_mm": float(10 ** lo),
                "hi_rms_mm": float(10 ** hi),
                "n_cells": int(n_in),
                "n_pairs": 0,
                "pearson_r": float("nan"),
                "ci95_lo": float("nan"),
                "ci95_hi": float("nan"),
            })
            continue
        # Sample pairs within this tier.
        rng_t = np.random.default_rng(seed + i + 1)
        pi = rng_t.integers(0, n_in, size=pairs_per_tier)
        pj = rng_t.integers(0, n_in, size=pairs_per_tier)
        diff = pi != pj
        pi, pj = pi[diff], pj[diff]
        ii_t = idx_in[pi]
        jj_t = idx_in[pj]
        d_th = np.linalg.norm(theta_norm[ii_t] - theta_norm[jj_t], axis=1)
        d_sd = np.linalg.norm(log_S[ii_t] - log_S[jj_t], axis=1)
        r = _pearson(d_th, d_sd)
        rlo, rhi = _bootstrap_r(d_sd, d_th, seed=seed + 200 + i)
        tiers.append({
            "lo_rms_mm": float(10 ** lo),
            "hi_rms_mm": float(10 ** hi),
            "n_cells": int(n_in),
            "n_pairs": int(d_th.size),
            "pearson_r": float(r),
            "ci95_lo": float(rlo),
            "ci95_hi": float(rhi),
        })

    return {
        "global": {
            "n_cells": int(n),
            "n_pairs": int(d_S.size),
            "pearson_r": float(r_global),
            "ci95_lo": float(lo_g),
            "ci95_hi": float(hi_g),
        },
        "tiers": tiers,
    }


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot(
    result: dict,
    theta_norm: np.ndarray,
    log_S: np.ndarray,
    log_R: np.ndarray,
    out_path: Path,
    *,
    seed: int = 0,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tiers = result["tiers"]
    n_t = len(tiers)
    fig = plt.figure(figsize=(4.2 * (n_t + 1), 4.8))
    gs = fig.add_gridspec(1, n_t + 1, width_ratios=[1.6] + [1] * n_t)

    # Panel 0: bar chart of r per tier
    ax0 = fig.add_subplot(gs[0, 0])
    xs = np.arange(n_t)
    rs = [t["pearson_r"] for t in tiers]
    lo = np.array([t["ci95_lo"] for t in tiers])
    hi = np.array([t["ci95_hi"] for t in tiers])
    ax0.bar(xs, rs, color="#2CA02C", alpha=0.85)
    ax0.errorbar(
        xs, rs,
        yerr=[np.array(rs) - lo, hi - np.array(rs)],
        fmt="none", color="black", capsize=4,
    )
    labels = [
        f"RMS\n{t['lo_rms_mm']:.2f}–{t['hi_rms_mm']:.2f}mm\n"
        f"n_cells={t['n_cells']}"
        for t in tiers
    ]
    ax0.set_xticks(xs)
    ax0.set_xticklabels(labels, fontsize=8)
    ax0.axhline(0, color="black", lw=0.4)
    ax0.axhline(
        result["global"]["pearson_r"], color="#D62728", ls="--", lw=1,
        label=f"global r = {result['global']['pearson_r']:.3f}",
    )
    ax0.legend(fontsize=8)
    ax0.set_ylabel("Pearson r  (d_θ, d_Seidel)", fontsize=10)
    ax0.set_title(
        "Seidel as design-identity fingerprint\n"
        "(thesis: r is high for sub-optimal tiers, low for well-corrected)",
        fontsize=10,
    )
    ax0.grid(axis="y", alpha=0.3)

    # Per-tier scatter of d_θ vs d_S.
    rng = np.random.default_rng(seed + 7777)
    for i, t in enumerate(tiers):
        ax = fig.add_subplot(gs[0, 1 + i])
        lo_r = np.log10(t["lo_rms_mm"])
        hi_r = np.log10(t["hi_rms_mm"])
        if i == len(tiers) - 1:
            mask = (log_R >= lo_r) & (log_R <= hi_r)
        else:
            mask = (log_R >= lo_r) & (log_R < hi_r)
        idx = np.where(mask)[0]
        if idx.size < 10:
            ax.text(0.5, 0.5, "n too small", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        pairs = min(4000, idx.size * (idx.size - 1) // 2)
        pi = rng.integers(0, idx.size, size=pairs)
        pj = rng.integers(0, idx.size, size=pairs)
        diff = pi != pj
        pi, pj = pi[diff], pj[diff]
        ii = idx[pi]
        jj = idx[pj]
        d_th = np.linalg.norm(theta_norm[ii] - theta_norm[jj], axis=1)
        d_sd = np.linalg.norm(log_S[ii] - log_S[jj], axis=1)
        ax.scatter(d_sd, d_th, s=3, alpha=0.25, color="#444")
        ax.set_title(
            f"tier {i+1}  RMS {t['lo_rms_mm']:.2f}-{t['hi_rms_mm']:.2f}mm\n"
            f"r = {t['pearson_r']:.3f}  "
            f"CI95 [{t['ci95_lo']:.3f}, {t['ci95_hi']:.3f}]",
            fontsize=9,
        )
        ax.set_xlabel("d_Seidel = ||Δ log|S|||", fontsize=8)
        if i == 0:
            ax.set_ylabel("d_θ = ||Δ normalized θ||", fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tiers", type=int, default=4)
    ap.add_argument("--pairs-per-tier", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--max-rms-mm", type=float, default=None,
        help="optionally drop cells with rms_spot > cutoff",
    )
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    archive_cells = run_dir / "archive_cells.json"
    if not archive_cells.exists():
        sys.exit(f"missing {archive_cells}")

    theta, S_abs, rms, info = _load(archive_cells)
    if args.max_rms_mm is not None:
        mask = rms <= args.max_rms_mm
        info["n_dropped_cutoff"] = int((~mask).sum())
        theta, S_abs, rms = theta[mask], S_abs[mask], rms[mask]

    print(f"[fingerprint] loaded n={len(rms)}  info={info}", flush=True)
    print(f"[fingerprint] rms_spot range: "
          f"[{rms.min():.4g}, {rms.max():.4g}] mm  "
          f"median={np.median(rms):.4g} mm", flush=True)

    theta_norm = _normalize_theta(theta)
    log_S = np.log10(np.clip(S_abs, 1e-14, None))
    log_R = np.log10(np.clip(rms, 1e-14, None))

    result = analyze(
        theta_norm, log_S, log_R,
        n_tiers=args.tiers, pairs_per_tier=args.pairs_per_tier,
        seed=args.seed,
    )
    gl = result["global"]
    print(
        f"[fingerprint] global r(d_theta, d_Seidel) = {gl['pearson_r']:+.3f}  "
        f"CI95 [{gl['ci95_lo']:.3f}, {gl['ci95_hi']:.3f}]  "
        f"(n_cells={gl['n_cells']}, n_pairs={gl['n_pairs']})",
        flush=True,
    )
    for i, t in enumerate(result["tiers"]):
        print(
            f"  tier {i+1}  RMS [{t['lo_rms_mm']:.3g}, {t['hi_rms_mm']:.3g}] mm "
            f"n_cells={t['n_cells']}  r={t['pearson_r']:+.3f}  "
            f"CI95 [{t['ci95_lo']:.3f}, {t['ci95_hi']:.3f}]",
            flush=True,
        )

    out_dir: Path = args.out or (run_dir / "lineage_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "seidel_fingerprint.png"
    plot(result, theta_norm, log_S, log_R, out_png, seed=args.seed)
    out_json = out_dir / "seidel_fingerprint.json"
    with out_json.open("w") as f:
        json.dump({"load_info": info, **result}, f, indent=2)
    print(f"[fingerprint] wrote {out_png}", flush=True)
    print(f"[fingerprint] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
