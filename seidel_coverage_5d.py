"""5-D extension of Option A (seidel_coverage.py).

Measures how well each RMS tier covers the full 5-D Seidel space
(|S1|..|S5|).  The archive itself is 2-D (S2, S3), so this is the
*unconstrained* descriptor volume actually realised by the elites —
not the binning MAP-Elites used.

Metrics per tier
----------------
  - bbox_volume_log   : 5-D bounding box in log|S| space (sum of log-range widths)
  - bbox_volume_raw   : 5-D bounding box in |S| space (product of ranges, log₁₀)
  - median_radius_log : median of ||log|S_i||| from the Seidel origin
  - median_radius_raw : median of |||S_i|||  (euclidean, raw units)
  - per_coord_range   : log-range of each |Sₖ|
  - effective_rank    : number of PCA components covering 95% variance
  - anisotropy        : top-eigenvalue / sum of eigenvalues (1.0 = 1-D line)

Thesis prediction: bad-RMS tiers have larger bbox, larger median radius,
and higher effective rank — Seidel spreads across more aberration axes.
Well-corrected tier collapses toward origin with lower effective rank.

Requires ``archive_cells.json`` with ``metrics.seidel_S1..S5`` populated
(run ``augment_archive_seidel.py`` if older run).

Usage
-----

    python seidel_coverage_5d.py <run_dir> [--tiers 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ----------------------------------------------------------------------

def _load(run_dir: Path):
    p = run_dir / "archive_cells.json"
    if not p.exists():
        sys.exit(f"missing {p}")
    with p.open() as f:
        d = json.load(f)
    cells = d.get("cells", [])
    rms, S = [], []
    n_missing = 0
    for c in cells:
        m = c.get("metrics", {})
        r = m.get("rms_spot")
        s = [m.get(f"seidel_S{k}") for k in (1, 2, 3, 4, 5)]
        if r is None or any(x is None for x in s):
            n_missing += 1
            continue
        if not np.isfinite(r) or r <= 0:
            n_missing += 1
            continue
        rms.append(float(r))
        S.append([float(x) for x in s])
    if not S:
        sys.exit("no cells with rms_spot + seidel_S1..S5 (run augment first)")
    return (
        np.asarray(rms, dtype=np.float64),
        np.abs(np.asarray(S, dtype=np.float64)),
        {"n_total": len(cells), "n_kept": len(S), "n_missing": n_missing},
    )


# ----------------------------------------------------------------------

def analyze(rms: np.ndarray, S: np.ndarray, *, n_tiers: int):
    log_S = np.log10(np.clip(S, 1e-14, None))      # (N, 5)
    log_R = np.log10(np.clip(rms, 1e-14, None))    # (N,)
    edges = np.quantile(log_R, np.linspace(0.0, 1.0, n_tiers + 1))

    tiers = []
    for i in range(n_tiers):
        lo, hi = edges[i], edges[i + 1]
        if i == n_tiers - 1:
            mask = (log_R >= lo) & (log_R <= hi)
        else:
            mask = (log_R >= lo) & (log_R < hi)
        if mask.sum() < 5:
            tiers.append({
                "lo_rms_mm": float(10 ** lo),
                "hi_rms_mm": float(10 ** hi),
                "n_cells": int(mask.sum()),
            })
            continue
        LS = log_S[mask]           # (n_in, 5)
        Sraw = S[mask]             # (n_in, 5)
        per_coord_lo = LS.min(axis=0)
        per_coord_hi = LS.max(axis=0)
        per_coord_range = per_coord_hi - per_coord_lo   # length 5
        # bbox volumes
        bbox_log_width_sum = float(per_coord_range.sum())
        # log of product == sum of log-ranges - no double-log
        bbox_volume_log10 = float(per_coord_range.sum())
        # Median radius in log|S| space (from origin of log space — i.e. |S|=1)
        # Equivalent to RMS of log|S| magnitudes per point.
        rad_log = np.linalg.norm(LS, axis=1)
        rad_raw = np.linalg.norm(Sraw, axis=1)
        med_log = float(np.median(rad_log))
        med_raw = float(np.median(rad_raw))

        # PCA on standardised log|S|.
        LSc = LS - LS.mean(axis=0, keepdims=True)
        cov = LSc.T @ LSc / max(1, LSc.shape[0] - 1)
        evals = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
        total = float(evals.sum()) + 1e-300
        cum = np.cumsum(evals) / total
        eff_rank = int(np.searchsorted(cum, 0.95) + 1)
        aniso = float(evals[0] / total)

        tiers.append({
            "lo_rms_mm": float(10 ** lo),
            "hi_rms_mm": float(10 ** hi),
            "n_cells": int(mask.sum()),
            "per_coord_range_log10": per_coord_range.tolist(),
            "bbox_log_volume": bbox_volume_log10,
            "median_radius_log": med_log,
            "median_radius_raw": med_raw,
            "effective_rank_95": eff_rank,
            "anisotropy_top_evalfrac": aniso,
            "cov_eigenvalues": evals.tolist(),
        })

    return tiers


# ----------------------------------------------------------------------

def plot(tiers, rms, S, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_R = np.log10(np.clip(rms, 1e-14, None))
    log_S = np.log10(np.clip(S, 1e-14, None))
    n_t = len(tiers)
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, n_t))

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(3, 3)

    labels = ("S1", "S2", "S3", "S4", "S5")

    # Panel 0: per-coordinate log-range bar groups
    ax = fig.add_subplot(gs[0, 0])
    w = 0.8 / n_t
    xs = np.arange(5)
    for i, t in enumerate(tiers):
        if "per_coord_range_log10" not in t:
            continue
        vals = t["per_coord_range_log10"]
        ax.bar(xs + (i - (n_t - 1) / 2) * w, vals, w, color=colors[i],
               label=f"tier{i+1}", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("log₁₀-range  (decades of |Sₖ|)")
    ax.set_title("Per-coefficient span of |Sₖ| within each RMS tier")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 1: bbox log-volume + median radius bars
    ax = fig.add_subplot(gs[0, 1])
    vols = [t.get("bbox_log_volume", 0.0) for t in tiers]
    ax.bar(np.arange(n_t), vols, color=colors, alpha=0.85)
    ax.set_xticks(np.arange(n_t))
    ax.set_xticklabels([f"tier{i+1}" for i in range(n_t)])
    ax.set_ylabel("Σₖ log₁₀-range |Sₖ|")
    ax.set_title("5-D bbox log-volume (coverage in log|S| space)")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: median radius in log|S|
    ax = fig.add_subplot(gs[0, 2])
    rads = [t.get("median_radius_log", 0.0) for t in tiers]
    ax.bar(np.arange(n_t), rads, color=colors, alpha=0.85)
    for x, v in enumerate(rads):
        ax.text(x, v + max(rads) * 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(np.arange(n_t))
    ax.set_xticklabels([f"tier{i+1}" for i in range(n_t)])
    ax.set_ylabel("median ||log|S|||")
    ax.set_title("Median 5-D Seidel magnitude (distance from ideal optic)")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: effective rank + anisotropy
    ax = fig.add_subplot(gs[1, 0])
    ranks = [t.get("effective_rank_95", 0) for t in tiers]
    anis = [t.get("anisotropy_top_evalfrac", 0.0) for t in tiers]
    x = np.arange(n_t)
    ax.bar(x - 0.2, ranks, 0.35, color=colors, alpha=0.85, label="eff. rank (95%)")
    ax2 = ax.twinx()
    ax2.plot(x, anis, "o--", color="#D62728", label="top-eval fraction")
    ax.set_xticks(x)
    ax.set_xticklabels([f"tier{i+1}" for i in range(n_t)])
    ax.set_ylabel("effective rank (bars)")
    ax2.set_ylabel("top eigenvalue / total (line)")
    ax.set_title("Dimensionality of log|S| covariance per tier")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 4-8: per-coefficient histograms per tier (small multiples)
    for k in range(5):
        ax = fig.add_subplot(gs[1 + (k // 3), (k % 3) + (0 if k < 3 else 0)])
        # careful: remap — use rows 1 and 2 cleanly
        pass

    # Simpler: one row of 5 subplots for per-coefficient histograms spanning
    # rows 2 and 3.  Replot.
    for k in range(5):
        row = 1 + k // 3
        col = k - (3 if row == 2 else 0)
        # Use row 1 for k<3 else row 2
        if k < 3:
            ax = fig.add_subplot(gs[1, k])
        else:
            ax = fig.add_subplot(gs[2, k - 3])
        for i, t in enumerate(tiers):
            lo = np.log10(t["lo_rms_mm"])
            hi = np.log10(t["hi_rms_mm"])
            if i == len(tiers) - 1:
                mask = (log_R >= lo) & (log_R <= hi)
            else:
                mask = (log_R >= lo) & (log_R < hi)
            if mask.sum() < 5:
                continue
            ax.hist(log_S[mask, k], bins=40, alpha=0.5, color=colors[i],
                    density=True, label=f"tier{i+1}")
        ax.set_title(f"log₁₀|{labels[k]}|")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7)

    # Panel 9 (bottom-right): summary text
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    lines = ["tier: RMS range | eff-rank | median ||log|S||", ""]
    for i, t in enumerate(tiers):
        lines.append(
            f"{i+1}: {t['lo_rms_mm']:.2f}-{t['hi_rms_mm']:.2f}mm | "
            f"rank={t.get('effective_rank_95','?')} | "
            f"{t.get('median_radius_log',float('nan')):.2f}"
        )
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            va="top", family="monospace", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=135)
    plt.close(fig)


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tiers", type=int, default=4)
    ap.add_argument("--max-rms-mm", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rms, S, info = _load(args.run_dir)
    if args.max_rms_mm is not None:
        mask = rms <= args.max_rms_mm
        info["n_dropped_cutoff"] = int((~mask).sum())
        rms, S = rms[mask], S[mask]
    print(f"[5d-cov] n={len(rms)} info={info}", flush=True)

    tiers = analyze(rms, S, n_tiers=args.tiers)
    for i, t in enumerate(tiers):
        print(
            f"  tier {i+1}  RMS [{t['lo_rms_mm']:.3g}, {t['hi_rms_mm']:.3g}] mm "
            f"n={t['n_cells']}  "
            f"bbox_log={t.get('bbox_log_volume', float('nan')):.2f}  "
            f"med||log|S|||={t.get('median_radius_log', float('nan')):.2f}  "
            f"eff_rank_95={t.get('effective_rank_95', '?')}  "
            f"aniso={t.get('anisotropy_top_evalfrac', float('nan')):.2f}",
            flush=True,
        )

    out_dir: Path = args.out or (args.run_dir / "lineage_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "seidel_coverage_5d.png"
    plot(tiers, rms, S, out_png)
    out_json = out_dir / "seidel_coverage_5d.json"
    with out_json.open("w") as f:
        json.dump({"load_info": info, "tiers": tiers}, f, indent=2)
    print(f"[5d-cov] wrote {out_png}", flush=True)
    print(f"[5d-cov] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
