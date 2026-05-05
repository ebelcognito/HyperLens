"""Empirical test of the core Seidel-as-descriptor thesis.

Central claim
-------------
Seidel coefficients are a faithful characterisation of lens behaviour
when third-order aberrations dominate (large spots), and a degraded
one when the lens is well-corrected (small spots) because residual
behaviour is then driven by 5th+ order terms Seidel does not see.

Operational formulation
-----------------------
Given N archive elites each with (rms_spot, |S1|, |S2|, |S3|, |S4|, |S5|):

  1. Global summary:
     - r(log rms_spot, log ||S||_inf)  — one number for the run
     - r(log rms_spot, log |Sk|)       — per-coefficient contribution

  2. Stratified predictability:
     - Sort elites by RMS, partition into Q quantile tiers.
     - Within each tier, fit log(rms_spot) = beta . log|S| + c (OLS).
     - Report R^2 per tier with bootstrap CI.
     - Thesis support: R^2 LOW in the smallest-RMS tier (Seidel can't
       discriminate among well-corrected lenses because |S| is near
       zero for all of them) and HIGHER in the medium/large-RMS tiers.

Source of truth
---------------
``archive_cells.json`` has clean ``metrics.rms_spot`` and
``metrics.seidel_S1..S5`` already stored — no re-evaluation required.

Usage
-----

    python seidel_discrimination.py <run_dir>
    python seidel_discrimination.py <run_dir> --tiers 5 --max-rms-mm 30
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

def _load_archive_cells(archive_cells_path: Path) -> Tuple[
    np.ndarray, np.ndarray, dict
]:
    with archive_cells_path.open() as f:
        d = json.load(f)
    cells = d.get("cells", [])
    seidels: List[List[float]] = []
    rms_vals: List[float] = []
    n_missing = 0
    for c in cells:
        m = c.get("metrics", {})
        rms = m.get("rms_spot")
        s = [m.get(f"seidel_S{k}") for k in (1, 2, 3, 4, 5)]
        if rms is None or any(x is None for x in s):
            n_missing += 1
            continue
        if not np.isfinite(rms) or rms <= 0:
            n_missing += 1
            continue
        seidels.append([float(x) for x in s])
        rms_vals.append(float(rms))
    if not seidels:
        raise RuntimeError("no cells with rms_spot + seidel_S1..S5")
    return (
        np.abs(np.asarray(seidels, dtype=np.float64)),
        np.asarray(rms_vals, dtype=np.float64),
        {"source": "archive_cells.json",
         "n_total": len(cells), "n_kept": len(seidels),
         "n_missing": n_missing},
    )


# ----------------------------------------------------------------------
# OLS + bootstrap
# ----------------------------------------------------------------------

def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """R^2 of OLS regression y = X beta + c (intercept fitted)."""
    if X.shape[0] < X.shape[1] + 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    yhat = Xc @ beta
    ss_res = float(((yc - yhat) ** 2).sum())
    ss_tot = float((yc ** 2).sum()) + 1e-300
    return 1.0 - ss_res / ss_tot


def _bootstrap_r2(
    X: np.ndarray, y: np.ndarray, *, n_boot: int = 500, seed: int = 0
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < X.shape[1] + 5:
        return (float("nan"), float("nan"))
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = _ols_r2(X[idx], y[idx])
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


# ----------------------------------------------------------------------
# Tiered analysis
# ----------------------------------------------------------------------

def analyze(
    S_abs: np.ndarray,
    rms: np.ndarray,
    *,
    n_tiers: int,
    seed: int,
) -> dict:
    log_S = np.log10(np.clip(S_abs, 1e-14, None))
    log_R = np.log10(np.clip(rms, 1e-14, None))
    # ---- global ----
    S_norm = np.linalg.norm(log_S, axis=1)  # proxy for "overall Seidel magnitude"
    r_global = float(np.corrcoef(S_norm, log_R)[0, 1])
    per_coeff = [
        float(np.corrcoef(log_S[:, k], log_R)[0, 1]) for k in range(5)
    ]
    r2_global = _ols_r2(log_S, log_R)
    lo_g, hi_g = _bootstrap_r2(log_S, log_R, seed=seed)

    # ---- tier-wise R^2 ----
    order = np.argsort(log_R)
    sorted_logR = log_R[order]
    edges = np.quantile(sorted_logR, np.linspace(0, 1, n_tiers + 1))
    tiers = []
    for i in range(n_tiers):
        lo, hi = edges[i], edges[i + 1]
        if i == n_tiers - 1:
            mask = (log_R >= lo) & (log_R <= hi)
        else:
            mask = (log_R >= lo) & (log_R < hi)
        if mask.sum() < 10:
            tiers.append({"lo": float(10 ** lo), "hi": float(10 ** hi),
                          "n": int(mask.sum()), "r2": float("nan"),
                          "ci95_lo": float("nan"), "ci95_hi": float("nan")})
            continue
        r2 = _ols_r2(log_S[mask], log_R[mask])
        cilo, cihi = _bootstrap_r2(
            log_S[mask], log_R[mask], seed=seed + i + 1
        )
        tiers.append({
            "lo_rms_mm": float(10 ** lo),
            "hi_rms_mm": float(10 ** hi),
            "n": int(mask.sum()),
            "r2": float(r2),
            "ci95_lo": float(cilo),
            "ci95_hi": float(cihi),
        })

    return {
        "n_total": int(S_abs.shape[0]),
        "global": {
            "r_Snorm_vs_logRMS": r_global,
            "r2_all5_vs_logRMS": float(r2_global),
            "ci95_lo": lo_g,
            "ci95_hi": hi_g,
            "per_coefficient_r": {
                f"S{k+1}": per_coeff[k] for k in range(5)
            },
        },
        "tiers": tiers,
    }


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot(
    result: dict,
    S_abs: np.ndarray,
    rms: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tiers = result["tiers"]
    n_t = len(tiers)
    fig = plt.figure(figsize=(4.2 * (n_t + 1), 4.8))
    gs = fig.add_gridspec(1, n_t + 1, width_ratios=[1.6] + [1] * n_t)

    # ---- Panel 0: bar chart of R^2 per tier ----
    ax0 = fig.add_subplot(gs[0, 0])
    xs = np.arange(n_t)
    r2s = [t["r2"] for t in tiers]
    lo = np.array([t["ci95_lo"] for t in tiers])
    hi = np.array([t["ci95_hi"] for t in tiers])
    ax0.bar(xs, r2s, color="#1F77B4", alpha=0.85)
    ax0.errorbar(
        xs, r2s,
        yerr=[np.array(r2s) - lo, hi - np.array(r2s)],
        fmt="none", color="black", capsize=4,
    )
    labels = []
    for t in tiers:
        ns = t["n"]
        labels.append(
            f"RMS\n{t['lo_rms_mm']:.2f}–{t['hi_rms_mm']:.2f}mm\nn={ns}"
        )
    ax0.set_xticks(xs)
    ax0.set_xticklabels(labels, fontsize=8)
    ax0.axhline(0, color="black", lw=0.4)
    ax0.set_ylabel(r"$R^2$  [log(RMS) ~ log|S|]", fontsize=10)
    r2_glob = result["global"]["r2_all5_vs_logRMS"]
    ax0.axhline(r2_glob, color="#D62728", ls="--", lw=1,
                label=f"global R²={r2_glob:.3f}")
    ax0.legend(fontsize=8, loc="upper right")
    ax0.set_title(
        "Seidel-space predictability of spot-RMS\n(thesis: lower R² in well-corrected tier)",
        fontsize=10,
    )
    ax0.grid(axis="y", alpha=0.3)

    # ---- Per-tier: scatter of predicted-vs-true log RMS ----
    log_S = np.log10(np.clip(S_abs, 1e-14, None))
    log_R = np.log10(np.clip(rms, 1e-14, None))
    for i, t in enumerate(tiers):
        ax = fig.add_subplot(gs[0, 1 + i])
        lo = np.log10(t["lo_rms_mm"])
        hi = np.log10(t["hi_rms_mm"])
        if i == len(tiers) - 1:
            mask = (log_R >= lo) & (log_R <= hi)
        else:
            mask = (log_R >= lo) & (log_R < hi)
        Xm = log_S[mask]
        ym = log_R[mask]
        if Xm.shape[0] < 10:
            ax.text(0.5, 0.5, "n too small", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        Xc = Xm - Xm.mean(axis=0, keepdims=True)
        yc = ym - ym.mean()
        beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        yhat = Xc @ beta + ym.mean()
        ax.scatter(yhat, ym, s=4, alpha=0.35, color="#444")
        mn, mx = min(yhat.min(), ym.min()), max(yhat.max(), ym.max())
        ax.plot([mn, mx], [mn, mx], "--", color="#D62728", lw=1)
        ax.set_title(
            f"tier {i+1}  RMS {t['lo_rms_mm']:.2f}–{t['hi_rms_mm']:.2f}mm\n"
            f"R²={t['r2']:.3f}  CI95 [{t['ci95_lo']:.3f}, {t['ci95_hi']:.3f}]",
            fontsize=9,
        )
        ax.set_xlabel(r"predicted log$_{10}$RMS from |S|", fontsize=8)
        if i == 0:
            ax.set_ylabel(r"actual log$_{10}$RMS", fontsize=8)
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--max-rms-mm", type=float, default=None,
        help="optionally drop cells with rms_spot > cutoff (sanity filter)",
    )
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    archive_cells = run_dir / "archive_cells.json"
    if not archive_cells.exists():
        sys.exit(f"missing archive_cells.json: {archive_cells}")

    S_abs, rms, info = _load_archive_cells(archive_cells)
    if args.max_rms_mm is not None:
        mask = rms <= args.max_rms_mm
        info["n_dropped_cutoff"] = int((~mask).sum())
        S_abs, rms = S_abs[mask], rms[mask]

    print(f"[seidel-disc] loaded {len(rms)} cells {info}", flush=True)
    print(f"[seidel-disc] rms_spot range: "
          f"[{rms.min():.4g}, {rms.max():.4g}] mm  "
          f"median={np.median(rms):.4g} mm", flush=True)

    result = analyze(S_abs, rms, n_tiers=args.tiers, seed=args.seed)
    gl = result["global"]
    print(f"[seidel-disc] global: "
          f"r(||log|S||, log RMS) = {gl['r_Snorm_vs_logRMS']:+.3f}  |  "
          f"R²(log|S1..S5| → log RMS) = {gl['r2_all5_vs_logRMS']:.3f}  "
          f"CI95 [{gl['ci95_lo']:.3f}, {gl['ci95_hi']:.3f}]", flush=True)
    print(f"[seidel-disc] per-coefficient r(log|Sk|, log RMS):", flush=True)
    for k, v in gl["per_coefficient_r"].items():
        print(f"    {k}: {v:+.3f}", flush=True)
    print(f"[seidel-disc] tiered R²:", flush=True)
    for i, t in enumerate(result["tiers"]):
        print(
            f"  tier {i+1}  RMS [{t['lo_rms_mm']:.3g}, {t['hi_rms_mm']:.3g}] mm  "
            f"n={t['n']}  R²={t['r2']:.3f}  "
            f"CI95 [{t['ci95_lo']:.3f}, {t['ci95_hi']:.3f}]",
            flush=True,
        )

    out_dir: Path = args.out or (run_dir / "lineage_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "seidel_discrimination.png"
    plot(result, S_abs, rms, out_png)
    out_json = out_dir / "seidel_discrimination.json"
    with out_json.open("w") as f:
        json.dump({"load_info": info, **result}, f, indent=2)
    print(f"[seidel-disc] wrote {out_png}", flush=True)
    print(f"[seidel-disc] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
