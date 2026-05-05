"""Post-hoc: compare 2-D Seidel BC pair choices on the same lineage data.

For every (Si, Sj) pair, simulate what a 2-D MAP-Elites archive would have
looked like under that BC — bin the lineage rows into a 2-D grid, keep
the best-fitness row per cell, then run Option A coverage analysis per
RMS tier.

Produces for each pair:
  hull_area_tier{i}   — fraction of BC bounding box that tier {i} occupies
  median_radius_tier{i} — median distance from BC origin in the tier

Ranks pairs by how strong the "coverage grows with RMS" signal is.

Usage
-----

    python seidel_pair_scan.py <run_dir> [--bins 128] [--tiers 4] \
                               [--max-invalid-frac 0.2] [--min-rms-mm 0.001]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------

def _load_lineage(path: Path, *, max_invalid: float, min_rms_mm: float
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rms, S, fit = [], [], []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("rejected"):
                continue
            r = d.get("rms_spot")
            s = d.get("seidel_all")
            inv = d.get("invalid_fraction", 0.0)
            f_ = d.get("fitness")
            if r is None or s is None or len(s) != 5:
                continue
            if not np.isfinite(r) or r <= 0 or r < min_rms_mm:
                continue
            if inv is not None and inv > max_invalid:
                continue
            rms.append(float(r))
            S.append([float(x) for x in s])
            fit.append(float(f_) if f_ is not None else -np.inf)
    if not rms:
        sys.exit("no valid lineage rows after filtering")
    return (np.asarray(rms, dtype=np.float64),
            np.asarray(S, dtype=np.float64),
            np.asarray(fit, dtype=np.float64))


def _convex_hull_area(pts: np.ndarray) -> float:
    if pts.shape[0] < 3:
        return 0.0
    p = np.unique(pts, axis=0)
    if p.shape[0] < 3:
        return 0.0
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo: List[np.ndarray] = []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    hi: List[np.ndarray] = []
    for q in p[::-1]:
        while len(hi) >= 2 and cross(hi[-2], hi[-1], q) <= 0:
            hi.pop()
        hi.append(q)
    hull = np.asarray(lo[:-1] + hi[:-1])
    if hull.shape[0] < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ---------------------------------------------------------------------------

def analyse_pair(
    rms: np.ndarray,
    S: np.ndarray,
    fit: np.ndarray,
    *,
    ki: int, kj: int,
    bins: int,
    pct_lo: float, pct_hi: float,
    n_tiers: int,
) -> dict:
    # Pick the two BC dims and build archive.
    bc = S[:, [ki, kj]]

    # Percentile bounds.
    xlo, xhi = np.percentile(bc[:, 0], [pct_lo, pct_hi])
    ylo, yhi = np.percentile(bc[:, 1], [pct_lo, pct_hi])
    if xhi <= xlo or yhi <= ylo:
        return {"error": "degenerate bounds"}

    # Build 2-D cell indices, clip to [0, bins-1].
    xi = np.clip(((bc[:, 0] - xlo) / (xhi - xlo) * bins).astype(np.int64),
                 0, bins - 1)
    yi = np.clip(((bc[:, 1] - ylo) / (yhi - ylo) * bins).astype(np.int64),
                 0, bins - 1)
    cell = xi * bins + yi

    # Best fitness per cell → virtual archive indices.
    best_idx: Dict[int, int] = {}
    for n in range(bc.shape[0]):
        c = int(cell[n])
        prev = best_idx.get(c)
        if prev is None or fit[n] > fit[prev]:
            best_idx[c] = n

    arc_idx = np.fromiter(best_idx.values(), dtype=np.int64)
    arc_bc = bc[arc_idx]
    arc_rms = rms[arc_idx]

    # Tier the archive by RMS quartile and compute hull area + median radius.
    log_R = np.log10(arc_rms)
    edges = np.quantile(log_R, np.linspace(0, 1, n_tiers + 1))
    tiers_out = []
    # BC normalised to [0, 1] for hull area.
    bc_norm = np.column_stack([
        (arc_bc[:, 0] - xlo) / (xhi - xlo),
        (arc_bc[:, 1] - ylo) / (yhi - ylo),
    ])
    max_r = np.sqrt(((np.array([xhi, yhi]) - np.array([xlo, ylo])) / 2) ** 2
                    ).sum() ** 0.5
    max_r = float(np.sqrt(((xhi - xlo) / 2) ** 2 + ((yhi - ylo) / 2) ** 2))
    for i in range(n_tiers):
        lo_e, hi_e = edges[i], edges[i + 1]
        if i == n_tiers - 1:
            m = (log_R >= lo_e) & (log_R <= hi_e)
        else:
            m = (log_R >= lo_e) & (log_R < hi_e)
        n_in = int(m.sum())
        if n_in == 0:
            tiers_out.append({"n": 0})
            continue
        hull = _convex_hull_area(bc_norm[m])
        radii = np.linalg.norm(arc_bc[m], axis=1)
        tiers_out.append({
            "lo_rms_mm": float(10 ** lo_e),
            "hi_rms_mm": float(10 ** hi_e),
            "n_cells": n_in,
            "hull_area_norm": hull,
            "median_radius": float(np.median(radii)),
            "median_radius_norm": float(np.median(radii) / max_r),
        })

    # Scalar score: monotonic-hull-growth metric (correlation between tier index
    # and hull area).
    hulls = [t.get("hull_area_norm", 0.0) for t in tiers_out]
    xs = np.arange(len(hulls))
    if len(hulls) > 1 and np.std(hulls) > 0:
        mono_r = float(np.corrcoef(xs, hulls)[0, 1])
    else:
        mono_r = 0.0
    # Ratio of last-to-first hull
    ratio = (hulls[-1] / max(hulls[0], 1e-9)) if len(hulls) > 1 else 0.0

    return {
        "n_archive": int(arc_idx.size),
        "pct_filled": float(arc_idx.size) / (bins * bins),
        "tiers": tiers_out,
        "hull_monotonic_r": mono_r,
        "hull_ratio_last_first": float(ratio),
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--bins", type=int, default=128)
    ap.add_argument("--tiers", type=int, default=4)
    ap.add_argument("--pct-lo", type=float, default=2.0)
    ap.add_argument("--pct-hi", type=float, default=98.0)
    ap.add_argument("--max-invalid-frac", type=float, default=0.2)
    ap.add_argument("--min-rms-mm", type=float, default=0.001)
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="e.g. '2,3' '2,4'; default = all 10 pairs of (1..5)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    lineage_path = args.run_dir / "lineage.jsonl"
    if not lineage_path.exists():
        sys.exit(f"missing {lineage_path}")

    rms, S, fit = _load_lineage(
        lineage_path,
        max_invalid=args.max_invalid_frac,
        min_rms_mm=args.min_rms_mm,
    )
    print(f"[pair-scan] loaded {rms.size} valid lineage rows", flush=True)
    print(f"[pair-scan] RMS: [{rms.min():.3g}, {rms.max():.3g}] mm  "
          f"median={np.median(rms):.3g} mm", flush=True)

    # Pair list.
    if args.pairs:
        pair_list = [tuple(int(x) - 1 for x in p.split(",")) for p in args.pairs]
    else:
        pair_list = list(itertools.combinations(range(5), 2))

    results = {}
    for (ki, kj) in pair_list:
        key = f"S{ki+1},S{kj+1}"
        r = analyse_pair(
            rms, np.abs(S), fit,
            ki=ki, kj=kj,
            bins=args.bins,
            pct_lo=args.pct_lo, pct_hi=args.pct_hi,
            n_tiers=args.tiers,
        )
        results[key] = r
        hulls = [t.get("hull_area_norm", 0.0) for t in r.get("tiers", [])]
        print(
            f"  {key}  archive={r['n_archive']:5d} ({100*r['pct_filled']:.2f}%)  "
            f"hulls=[" + ", ".join(f"{h:.2f}" for h in hulls) + "]  "
            f"mono_r={r['hull_monotonic_r']:+.2f}  "
            f"ratio_last/first={r['hull_ratio_last_first']:.2f}",
            flush=True,
        )

    # Plot bar chart of hull areas per tier for each pair.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_pairs = len(results)
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5),
        gridspec_kw={"width_ratios": [1.5, 1]},
    )
    keys = list(results.keys())
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, args.tiers))
    xs = np.arange(n_pairs)
    w = 0.8 / args.tiers
    ax = axes[0]
    for t in range(args.tiers):
        vals = [results[k]["tiers"][t].get("hull_area_norm", 0.0)
                for k in keys]
        ax.bar(xs + (t - (args.tiers - 1) / 2) * w, vals, w,
               color=colors[t], label=f"tier{t+1}", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(keys, rotation=0, fontsize=9)
    ax.set_ylabel("Hull area (normalized)")
    ax.set_title("Virtual 2-D BC: hull area per RMS tier, for each Seidel pair")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Ranking scatter.
    ax2 = axes[1]
    ratios = [results[k]["hull_ratio_last_first"] for k in keys]
    monos = [results[k]["hull_monotonic_r"] for k in keys]
    ax2.scatter(ratios, monos, s=80, color="#1F77B4")
    for k, x, y in zip(keys, ratios, monos):
        ax2.annotate(k, (x, y), xytext=(4, 4), textcoords="offset points",
                     fontsize=9)
    ax2.set_xlabel("hull(tier4) / hull(tier1)")
    ax2.set_ylabel("Pearson r (tier idx → hull)")
    ax2.set_title("BC-pair ranking")
    ax2.grid(alpha=0.3)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.axvline(1, color="black", lw=0.5)

    fig.tight_layout()
    out_dir = args.out or (args.run_dir / "lineage_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "seidel_pair_scan.png"
    fig.savefig(png, dpi=135)
    plt.close(fig)
    with (out_dir / "seidel_pair_scan.json").open("w") as f:
        json.dump({"bins": args.bins, "tiers": args.tiers,
                   "pct_lo": args.pct_lo, "pct_hi": args.pct_hi,
                   "n_rows": int(rms.size), "pairs": results}, f, indent=2)
    print(f"[pair-scan] wrote {png}", flush=True)


if __name__ == "__main__":
    main()
