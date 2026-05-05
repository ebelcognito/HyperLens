"""N-D extension of Option A coverage analysis.

Works on archives of any descriptor dimensionality.  Primary use case is
a 5-D Seidel BC archive, but automatically adapts to 2-D or higher.

Per-tier metrics (archive-level, unconstrained by the 2-D cell grid)
------------------------------------------------------------------
  n_cells          : quantile-size — constant by construction
  n_unique_cells   : distinct cell_index tuples occupied by tier
  frac_of_grid     : n_unique_cells / product(bins)
  bbox_log_volume  : sum over k of (log_range |S_k|)
  median_radius_raw : median ||S_raw|| (euclidean)
  median_radius_log : median ||log|S|||
  per_coord_log_range : list of length = descriptor dim
  effective_rank_95  : min number of PCA components covering 95% variance
  anisotropy         : top-eigenvalue / total

Archive source
--------------
``archive_cells.json`` must contain ``metrics.rms_spot`` and all five
``metrics.seidel_S{k}``.  For older runs without the seidel side-channel
fields, run ``augment_archive_seidel.py`` first.

Usage
-----

    python seidel_coverage_nd.py <run_dir> [--tiers 4] [--max-rms-mm N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np


SEIDEL_LABELS = ("S1", "S2", "S3", "S4", "S5")


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------

def _load(run_dir: Path, source: str = "archive"):
    if source == "lineage":
        return _load_lineage(run_dir)
    p = run_dir / "archive_cells.json"
    if not p.exists():
        sys.exit(f"missing {p}")
    with p.open() as f:
        payload = json.load(f)
    cells = payload.get("cells", [])
    bins = tuple(payload.get("archive_bins", ()))
    coord_names = payload.get("descriptor_coord_names", [])
    archive_dim = len(bins)

    rms, S, cell_idx, measures, invalid = [], [], [], [], []
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
        cell_idx.append(tuple(int(x) for x in c.get("cell_index", ())))
        measures.append([float(x) for x in c.get("measures", [])])
        invalid.append(float(m.get("invalid_fraction", 0.0)))
    if not S:
        sys.exit("no cells with rms_spot + seidel_S1..S5 (augment first)")
    info = {
        "archive_dim": archive_dim,
        "bins": list(bins),
        "total_grid_cells": int(np.prod(bins)) if archive_dim > 0 else 0,
        "coord_names": coord_names,
        "n_total": len(cells),
        "n_kept": len(S),
        "n_missing": n_missing,
    }
    return (
        np.asarray(rms, dtype=np.float64),
        np.abs(np.asarray(S, dtype=np.float64)),   # |S1..S5| 5-D regardless
        cell_idx,
        np.asarray(measures, dtype=np.float64) if measures and measures[0] else None,
        info,
        np.asarray(invalid, dtype=np.float64),
    )


def _load_lineage(run_dir: Path):
    """Load every non-rejected evaluation from lineage.jsonl.

    Produces a much richer dataset than archive_cells.json when the
    archive itself is under-populated (e.g. 5-D BCs where CMA-MEGA
    struggles to navigate).  Uses the per-row ``rms_spot`` metric and
    ``seidel_all`` array that modern runs log."""
    p = run_dir / "lineage.jsonl"
    if not p.exists():
        sys.exit(f"missing {p}")
    rms, S, invalid = [], [], []
    bc_cells = []
    with p.open() as f:
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
            if r is None or s is None or len(s) != 5:
                continue
            if not np.isfinite(r) or r <= 0:
                continue
            rms.append(float(r))
            S.append([float(x) for x in s])
            invalid.append(float(d.get("invalid_fraction", 0.0)))
            bc = d.get("bc_cell")
            bc_cells.append(tuple(int(x) for x in bc) if bc else ())
    if not S:
        sys.exit("no valid rows in lineage.jsonl")
    info = {
        "source": "lineage.jsonl",
        "archive_dim": len(bc_cells[0]) if bc_cells and bc_cells[0] else 0,
        "bins": [],
        "total_grid_cells": 0,
        "coord_names": [],
        "n_total": len(S),
        "n_kept": len(S),
        "n_missing": 0,
    }
    return (
        np.asarray(rms, dtype=np.float64),
        np.abs(np.asarray(S, dtype=np.float64)),
        bc_cells,
        None,
        info,
        np.asarray(invalid, dtype=np.float64),
    )


# ----------------------------------------------------------------------
# Per-tier analysis
# ----------------------------------------------------------------------

def analyze(rms, S, cell_idx, *, n_tiers, total_grid_cells):
    log_S = np.log10(np.clip(S, 1e-14, None))
    log_R = np.log10(np.clip(rms, 1e-14, None))
    edges = np.quantile(log_R, np.linspace(0.0, 1.0, n_tiers + 1))

    tiers = []
    for i in range(n_tiers):
        lo, hi = edges[i], edges[i + 1]
        if i == n_tiers - 1:
            mask = (log_R >= lo) & (log_R <= hi)
        else:
            mask = (log_R >= lo) & (log_R < hi)
        idx_in = np.where(mask)[0]
        n_in = int(idx_in.size)
        t: dict = {
            "lo_rms_mm": float(10 ** lo),
            "hi_rms_mm": float(10 ** hi),
            "n_cells": n_in,
        }
        if n_in < 5:
            tiers.append(t)
            continue

        # --- unique archive cells occupied ---
        idxs = [cell_idx[k] for k in idx_in] if cell_idx else []
        if idxs and all(len(x) > 0 for x in idxs):
            unique = len(set(idxs))
            t["n_unique_archive_cells"] = int(unique)
            if total_grid_cells > 0:
                t["frac_of_grid"] = unique / total_grid_cells

        LS = log_S[mask]          # (n_in, 5)
        Sraw = S[mask]
        per_coord_range = (LS.max(axis=0) - LS.min(axis=0)).tolist()
        t["per_coord_log_range"] = per_coord_range
        t["bbox_log_volume"] = float(sum(per_coord_range))

        t["median_radius_raw"] = float(np.median(np.linalg.norm(Sraw, axis=1)))
        t["median_radius_log"] = float(np.median(np.linalg.norm(LS, axis=1)))

        # PCA in log|S| space.
        LSc = LS - LS.mean(axis=0, keepdims=True)
        cov = LSc.T @ LSc / max(1, LSc.shape[0] - 1)
        evals = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
        total = float(evals.sum()) + 1e-300
        t["effective_rank_95"] = int(np.searchsorted(np.cumsum(evals) / total, 0.95) + 1)
        t["anisotropy_top_evalfrac"] = float(evals[0] / total)
        t["cov_eigenvalues"] = evals.tolist()

        # MAD per coord (robust spread).
        t["per_coord_log_mad"] = [
            float(np.median(np.abs(LS[:, k] - np.median(LS[:, k]))))
            for k in range(LS.shape[1])
        ]
        # Median |S_k| per coord (raw) — monotonic "aberration scale" per tier.
        t["per_coord_median_raw"] = [
            float(np.median(Sraw[:, k])) for k in range(Sraw.shape[1])
        ]
        # IQR of log|S_k| per coord — spread of the distribution in log space.
        t["per_coord_log_iqr"] = [
            float(np.quantile(LS[:, k], 0.75) - np.quantile(LS[:, k], 0.25))
            for k in range(LS.shape[1])
        ]
        tiers.append(t)
    return tiers


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def plot(tiers, rms, S, cell_idx, info, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_R = np.log10(np.clip(rms, 1e-14, None))
    log_S = np.log10(np.clip(S, 1e-14, None))
    n_t = len(tiers)
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, n_t))

    fig = plt.figure(figsize=(18, 11))
    # Row 1: three summary panels
    # Rows 2-3: one panel per Seidel coefficient (5 total), in a 2x3 layout
    # Last panel row 3 col 3: summary text
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0])

    # ---- 0: per-tier archive-cell fill -------------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    xs = np.arange(n_t)
    unique = [t.get("n_unique_archive_cells", 0) for t in tiers]
    if any(unique):
        ax0.bar(xs, unique, color=colors, alpha=0.85)
        for x, v in zip(xs, unique):
            ax0.text(x, v + max(unique) * 0.01, str(v), ha="center", fontsize=9)
        ax0.set_ylabel("unique archive cells")
        ax0.set_title(
            f"Archive cells filled per tier  ({info['archive_dim']}-D grid, "
            f"{info['total_grid_cells']:,} possible)"
        )
    else:
        ax0.set_title("no cell_index in archive")
    ax0.set_xticks(xs)
    ax0.set_xticklabels([f"t{i+1}" for i in range(n_t)])
    ax0.grid(axis="y", alpha=0.3)

    # ---- 1: bbox log-volume in |S| space ------------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    vols = [t.get("bbox_log_volume", 0.0) for t in tiers]
    ax1.bar(xs, vols, color=colors, alpha=0.85)
    for x, v in zip(xs, vols):
        ax1.text(x, v + max(vols) * 0.01, f"{v:.1f}", ha="center", fontsize=9)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"t{i+1}" for i in range(n_t)])
    ax1.set_ylabel("Σₖ log₁₀-range |Sₖ|")
    ax1.set_title("5-D |S|-space bbox volume (log₁₀ units)")
    ax1.grid(axis="y", alpha=0.3)

    # ---- 2: effective rank + anisotropy -----------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    ranks = [t.get("effective_rank_95", 0) for t in tiers]
    anis = [t.get("anisotropy_top_evalfrac", 0.0) for t in tiers]
    ax2.bar(xs - 0.2, ranks, 0.35, color=colors, alpha=0.85,
            label="eff. rank (95%)")
    ax2t = ax2.twinx()
    ax2t.plot(xs, anis, "o--", color="#D62728", lw=1.5,
              label="top-eigenvalue frac.")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"t{i+1}" for i in range(n_t)])
    ax2.set_ylabel("effective rank")
    ax2t.set_ylabel("top-eigenvalue / total")
    ax2.set_title("Seidel-space dimensionality per tier")
    ax2.legend(loc="upper left", fontsize=8)
    ax2t.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # ---- 3-7: per-coefficient log|Sₖ| histograms ---------------------
    # 5 histograms arranged: row 1 = S1, S2, S3 (cols 0,1,2); row 2 = S4, S5
    positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    for k, (r, c) in enumerate(positions):
        ax = fig.add_subplot(gs[r, c])
        for i, t in enumerate(tiers):
            lo = np.log10(t["lo_rms_mm"])
            hi = np.log10(t["hi_rms_mm"])
            if i == n_t - 1:
                mask = (log_R >= lo) & (log_R <= hi)
            else:
                mask = (log_R >= lo) & (log_R < hi)
            if mask.sum() < 5:
                continue
            ax.hist(log_S[mask, k], bins=40, alpha=0.5, color=colors[i],
                    density=True, label=f"t{i+1}")
        ax.set_title(f"log₁₀|{SEIDEL_LABELS[k]}|  distribution per tier",
                     fontsize=10)
        ax.set_xlabel(f"log₁₀|{SEIDEL_LABELS[k]}|")
        ax.set_ylabel("density")
        if k == 0:
            ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

    # ---- 8: summary text ---------------------------------------------
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")
    lines = [
        "per-tier summary",
        f"archive: {info['archive_dim']}-D grid, {info['total_grid_cells']:,} cells",
        f"n elites kept: {info['n_kept']:,}",
        "",
    ]
    for i, t in enumerate(tiers):
        parts = [f"tier {i+1}  RMS {t['lo_rms_mm']:.2f}–{t['hi_rms_mm']:.2f}mm"]
        parts.append(f"  n={t['n_cells']}")
        if "n_unique_archive_cells" in t:
            parts.append(
                f"  unique={t['n_unique_archive_cells']}  "
                f"{100 * t.get('frac_of_grid', 0):.3f}% grid"
            )
        if "bbox_log_volume" in t:
            parts.append(
                f"  bbox={t['bbox_log_volume']:.2f} log-dec  "
                f"med‖logS‖={t.get('median_radius_log', float('nan')):.2f}"
            )
        if "effective_rank_95" in t:
            parts.append(
                f"  eff_rank={t['effective_rank_95']}  "
                f"aniso={t['anisotropy_top_evalfrac']:.2f}"
            )
        lines.extend(parts)
        lines.append("")
    ax8.text(0.02, 0.98, "\n".join(lines), transform=ax8.transAxes,
             va="top", ha="left", family="monospace", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=135)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--source", choices=("archive", "lineage"),
                    default="archive",
                    help="read from archive_cells.json (default) or "
                         "lineage.jsonl (every evaluation, useful when "
                         "the archive is sparsely populated)")
    ap.add_argument("--tiers", type=int, default=4)
    ap.add_argument("--max-rms-mm", type=float, default=None)
    ap.add_argument("--min-rms-mm", type=float, default=None,
                    help="drop cells with rms_spot < cutoff (mm) to filter "
                         "out clipping artifacts")
    ap.add_argument("--max-invalid-frac", type=float, default=None,
                    help="drop cells with invalid_fraction > cutoff")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rms, S, cell_idx, measures, info, invalid = _load(
        args.run_dir, source=args.source,
    )
    keep = np.ones(rms.size, dtype=bool)
    if args.max_rms_mm is not None:
        keep &= rms <= args.max_rms_mm
    if args.min_rms_mm is not None:
        keep &= rms >= args.min_rms_mm
    if args.max_invalid_frac is not None and invalid is not None:
        keep &= invalid <= args.max_invalid_frac
    if not keep.all():
        info["n_dropped_filter"] = int((~keep).sum())
        rms = rms[keep]
        S = S[keep]
        cell_idx = [c for k, c in enumerate(cell_idx) if keep[k]]
    print(f"[nd-cov] info={info}", flush=True)
    print(f"[nd-cov] rms: [{rms.min():.3g}, {rms.max():.3g}] mm "
          f"median={np.median(rms):.3g} mm  n={len(rms)}", flush=True)

    tiers = analyze(
        rms, S, cell_idx,
        n_tiers=args.tiers,
        total_grid_cells=info["total_grid_cells"],
    )
    for i, t in enumerate(tiers):
        parts = [f"tier {i+1}  RMS [{t['lo_rms_mm']:.3g}, {t['hi_rms_mm']:.3g}] mm"]
        parts.append(f"n={t['n_cells']}")
        if "n_unique_archive_cells" in t:
            parts.append(
                f"unique={t['n_unique_archive_cells']} "
                f"({100*t.get('frac_of_grid', 0):.3f}%)"
            )
        if "bbox_log_volume" in t:
            parts.append(
                f"bbox={t['bbox_log_volume']:.2f} "
                f"medR_log={t.get('median_radius_log', float('nan')):.2f}"
            )
        if "effective_rank_95" in t:
            parts.append(
                f"rank={t['effective_rank_95']} "
                f"aniso={t['anisotropy_top_evalfrac']:.2f}"
            )
        print("  " + "  ".join(parts), flush=True)

    out_dir: Path = args.out or (args.run_dir / "lineage_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "seidel_coverage_nd.png"
    plot(tiers, rms, S, cell_idx, info, out_png)
    out_json = out_dir / "seidel_coverage_nd.json"
    with out_json.open("w") as f:
        json.dump({"load_info": info, "tiers": tiers}, f, indent=2)
    print(f"[nd-cov] wrote {out_png}", flush=True)
    print(f"[nd-cov] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
