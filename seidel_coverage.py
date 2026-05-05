"""Option A: per-tier Seidel-space coverage.

For each RMS tier, measure how much of Seidel (BC) space the archive
fills.  Thesis-support test: the sub-optimal tiers should occupy many
more distinct Seidel cells than the well-corrected tier, because bad
lenses have heterogeneous aberration profiles while well-corrected
lenses collapse toward small |S|.

Metrics reported per tier:
  - n_cells                 : number of filled archive cells
  - frac_cells              : n_cells / total BC cells
  - bbox_area (normalized)  : width * height of the BC bounding box
  - convex_hull_area (norm) : convex-hull area in BC space
  - seidel_radius_median    : median ||measures||₂ from the BC origin
  - entropy_bits            : H of the cell distribution (within-tier, uniform = log2(n_cells))

Also produces a scatter of all archive elites in (S2, S3), colored by
RMS tier, plus per-coordinate histograms.

Usage
-----

    python seidel_coverage.py <run_dir> [--tiers 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _load(run_dir: Path):
    archive_cells = run_dir / "archive_cells.json"
    if not archive_cells.exists():
        sys.exit(f"missing {archive_cells}")
    with archive_cells.open() as f:
        d = json.load(f)
    cells = d.get("cells", [])
    bounds = np.asarray(d.get("archive_bounds"), dtype=np.float64)
    bins = tuple(d.get("archive_bins"))
    names = d.get("descriptor_coord_names", ["m0", "m1"])

    rms = []
    measures = []
    cell_idx = []
    invalid = []
    for c in cells:
        m = c.get("metrics", {})
        r = m.get("rms_spot")
        if r is None or not np.isfinite(r) or r <= 0:
            continue
        rms.append(float(r))
        measures.append([float(x) for x in c["measures"]])
        cell_idx.append(tuple(int(x) for x in c["cell_index"]))
        invalid.append(float(m.get("invalid_fraction", 0.0)))
    return (
        np.asarray(rms, dtype=np.float64),
        np.asarray(measures, dtype=np.float64),
        cell_idx,
        bounds,
        bins,
        names,
        np.asarray(invalid, dtype=np.float64),
    )


def _convex_hull_area(pts: np.ndarray) -> float:
    """Area of the convex hull of 2D points.  Shoelace formula with a
    tiny Andrew's monotone-chain implementation to avoid scipy."""
    if pts.shape[0] < 3:
        return 0.0
    # Remove duplicates.
    p = np.unique(pts, axis=0)
    if p.shape[0] < 3:
        return 0.0
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: List[np.ndarray] = []
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    hull = np.asarray(lower[:-1] + upper[:-1])
    if hull.shape[0] < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _entropy_bits(idxs: List[Tuple[int, ...]]) -> float:
    from collections import Counter
    if not idxs:
        return 0.0
    cnt = Counter(idxs)
    tot = sum(cnt.values())
    p = np.asarray([v / tot for v in cnt.values()], dtype=np.float64)
    return float(-(p * np.log2(p + 1e-300)).sum())


def analyze(rms, measures, cell_idx, bounds, bins, n_tiers):
    # Normalize measures to [0, 1] using archive bounds for comparability.
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    rng = (hi - lo)
    meas_norm = (measures - lo) / rng   # shape (N, 2)

    total_bins = int(np.prod(bins))

    # Tier edges on log10(RMS).
    log_R = np.log10(np.clip(rms, 1e-14, None))
    edges = np.quantile(log_R, np.linspace(0.0, 1.0, n_tiers + 1))
    tiers_out = []
    for i in range(n_tiers):
        lo_log, hi_log = edges[i], edges[i + 1]
        if i == n_tiers - 1:
            mask = (log_R >= lo_log) & (log_R <= hi_log)
        else:
            mask = (log_R >= lo_log) & (log_R < hi_log)
        n_in = int(mask.sum())
        if n_in == 0:
            tiers_out.append({
                "lo_rms_mm": float(10 ** lo_log),
                "hi_rms_mm": float(10 ** hi_log),
                "n_cells": 0,
                "frac_of_grid": 0.0,
                "bbox_area_norm": 0.0,
                "hull_area_norm": 0.0,
                "seidel_radius_median_norm": 0.0,
                "entropy_bits": 0.0,
                "max_entropy_bits": 0.0,
            })
            continue
        pts_norm = meas_norm[mask]
        pts_raw = measures[mask]
        idxs = [cell_idx[k] for k in np.where(mask)[0]]

        # Bounding box in normalized (fraction of archive range).
        bbox_w = float(pts_norm[:, 0].max() - pts_norm[:, 0].min())
        bbox_h = float(pts_norm[:, 1].max() - pts_norm[:, 1].min())
        bbox_area_norm = bbox_w * bbox_h

        hull = _convex_hull_area(pts_norm)

        # "Radius" from origin (the BC origin is "no aberration").
        radius = np.sqrt((pts_raw ** 2).sum(axis=1))
        # normalized by the max possible radius in this archive box
        max_radius = np.sqrt(np.sum(((hi - lo) * 0.5) ** 2))
        rad_med_norm = float(np.median(radius) / max_radius)

        ent = _entropy_bits(idxs)
        max_ent = float(np.log2(n_in)) if n_in > 1 else 0.0

        tiers_out.append({
            "lo_rms_mm": float(10 ** lo_log),
            "hi_rms_mm": float(10 ** hi_log),
            "n_cells": n_in,
            "frac_of_grid": n_in / total_bins,
            "bbox_area_norm": bbox_area_norm,
            "hull_area_norm": hull,
            "seidel_radius_median_norm": rad_med_norm,
            "entropy_bits": ent,
            "max_entropy_bits": max_ent,
        })
    return tiers_out, meas_norm, log_R, edges


def plot(tiers, measures, meas_norm, log_R, edges, names, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])

    # Panel 0: BC-space scatter colored by tier.
    ax0 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(tiers)))
    for i, t in enumerate(tiers):
        lo_log = np.log10(t["lo_rms_mm"])
        hi_log = np.log10(t["hi_rms_mm"])
        if i == len(tiers) - 1:
            m = (log_R >= lo_log) & (log_R <= hi_log)
        else:
            m = (log_R >= lo_log) & (log_R < hi_log)
        ax0.scatter(
            measures[m, 0], measures[m, 1], s=6, alpha=0.5,
            color=colors[i],
            label=f"tier{i+1}  {t['lo_rms_mm']:.2f}-{t['hi_rms_mm']:.2f}mm  "
                  f"(n={t['n_cells']})",
        )
    ax0.set_xlabel(names[0])
    ax0.set_ylabel(names[1])
    ax0.set_title("Archive elites in (S2, S3), coloured by RMS tier")
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=8, loc="best")

    # Panel 1: bar of n_cells per tier.
    ax1 = fig.add_subplot(gs[0, 1])
    xs = np.arange(len(tiers))
    n_vals = [t["n_cells"] for t in tiers]
    ax1.bar(xs, n_vals, color=colors, alpha=0.85)
    for x, v in zip(xs, n_vals):
        ax1.text(x, v + max(n_vals) * 0.01, str(v), ha="center", fontsize=9)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(
        [f"tier{i+1}\n{t['lo_rms_mm']:.2f}-{t['hi_rms_mm']:.2f}mm"
         for i, t in enumerate(tiers)],
        fontsize=8,
    )
    ax1.set_title("Seidel cells filled per tier")
    ax1.set_ylabel("n cells")
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: convex hull area + bbox per tier.
    ax2 = fig.add_subplot(gs[0, 2])
    hull_vals = [t["hull_area_norm"] for t in tiers]
    bbox_vals = [t["bbox_area_norm"] for t in tiers]
    w = 0.38
    ax2.bar(xs - w/2, hull_vals, w, color=colors, alpha=0.85, label="hull area")
    ax2.bar(xs + w/2, bbox_vals, w, color=colors, alpha=0.4, label="bbox area",
            hatch="///")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"tier{i+1}" for i in range(len(tiers))],
                        fontsize=9)
    ax2.set_title("Seidel-space coverage (normalized)")
    ax2.set_ylabel("area fraction of archive box")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: radius-from-origin histogram per tier.
    ax3 = fig.add_subplot(gs[1, 0])
    radii = np.sqrt((measures ** 2).sum(axis=1))
    for i, t in enumerate(tiers):
        lo_log = np.log10(t["lo_rms_mm"])
        hi_log = np.log10(t["hi_rms_mm"])
        if i == len(tiers) - 1:
            m = (log_R >= lo_log) & (log_R <= hi_log)
        else:
            m = (log_R >= lo_log) & (log_R < hi_log)
        if m.sum() == 0:
            continue
        ax3.hist(radii[m], bins=40, alpha=0.55, color=colors[i],
                 label=f"tier{i+1}", density=True)
    ax3.set_xlabel("||(S2, S3)||₂ — distance from BC origin")
    ax3.set_ylabel("density")
    ax3.set_title("How far from 'no-aberration' origin do cells sit, per tier?")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=9)

    # Panel 4: entropy (diversity) per tier.
    ax4 = fig.add_subplot(gs[1, 1])
    ents = [t["entropy_bits"] for t in tiers]
    max_ents = [t["max_entropy_bits"] for t in tiers]
    ax4.bar(xs, ents, color=colors, alpha=0.85, label="entropy (bits)")
    ax4.plot(xs, max_ents, "o--", color="#D62728", label="log₂(n_cells) max")
    ax4.set_xticks(xs)
    ax4.set_xticklabels([f"tier{i+1}" for i in range(len(tiers))], fontsize=9)
    ax4.set_ylabel("entropy (bits)")
    ax4.set_title("Distribution entropy over filled cells per tier")
    ax4.legend(fontsize=9)
    ax4.grid(axis="y", alpha=0.3)

    # Panel 5: frac_of_grid as a headline number
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    lines = ["Headline per tier:", ""]
    for i, t in enumerate(tiers):
        lines.append(
            f"tier {i+1}  RMS {t['lo_rms_mm']:.2f}–{t['hi_rms_mm']:.2f}mm"
        )
        lines.append(
            f"  n_cells={t['n_cells']:4d}  "
            f"{100*t['frac_of_grid']:.3f}% of BC grid"
        )
        lines.append(
            f"  hull area={t['hull_area_norm']:.3f}  "
            f"median |S|={t['seidel_radius_median_norm']:.3f}"
        )
        lines.append("")
    ax5.text(0.02, 0.98, "\n".join(lines), transform=ax5.transAxes,
             va="top", ha="left", family="monospace", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tiers", type=int, default=4)
    ap.add_argument("--max-rms-mm", type=float, default=None)
    ap.add_argument("--min-rms-mm", type=float, default=None,
                    help="drop cells with rms_spot < cutoff (e.g. 0.001) to "
                         "filter clipping artifacts")
    ap.add_argument("--max-invalid-frac", type=float, default=None,
                    help="drop cells with invalid_fraction > cutoff")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rms, measures, cell_idx, bounds, bins, names, invalid = _load(args.run_dir)
    keep = np.ones(rms.size, dtype=bool)
    if args.max_rms_mm is not None:
        keep &= rms <= args.max_rms_mm
    if args.min_rms_mm is not None:
        keep &= rms >= args.min_rms_mm
    if args.max_invalid_frac is not None:
        keep &= invalid <= args.max_invalid_frac
    if not keep.all():
        rms = rms[keep]
        measures = measures[keep]
        cell_idx = [c for k, c in enumerate(cell_idx) if keep[k]]
    print(f"[coverage] n={len(rms)} rms range [{rms.min():.3g}, "
          f"{rms.max():.3g}] mm  median={np.median(rms):.3g} mm  "
          f"grid={bins}", flush=True)

    tiers, meas_norm, log_R, edges = analyze(
        rms, measures, cell_idx, bounds, bins, args.tiers,
    )
    for i, t in enumerate(tiers):
        print(
            f"  tier {i+1}  RMS [{t['lo_rms_mm']:.3g}, {t['hi_rms_mm']:.3g}] mm  "
            f"n_cells={t['n_cells']}  "
            f"{100*t['frac_of_grid']:.3f}% of grid  "
            f"hull={t['hull_area_norm']:.3f}  "
            f"bbox={t['bbox_area_norm']:.3f}  "
            f"H={t['entropy_bits']:.2f}/{t['max_entropy_bits']:.2f} bits  "
            f"median|S|={t['seidel_radius_median_norm']:.3f}",
            flush=True,
        )

    out_dir: Path = args.out or (args.run_dir / "lineage_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "seidel_coverage.png"
    plot(tiers, measures, meas_norm, log_R, edges, names, out_png)
    out_json = out_dir / "seidel_coverage.json"
    with out_json.open("w") as f:
        json.dump({"tiers": tiers, "bins": list(bins),
                   "bounds": bounds.tolist(), "coord_names": names}, f,
                  indent=2)
    print(f"[coverage] wrote {out_png}", flush=True)
    print(f"[coverage] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
