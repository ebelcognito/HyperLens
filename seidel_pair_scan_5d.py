"""
Post-hoc pair-scan on the clean 5-D Seidel archive.

For each of the 10 possible 2-D BC pairs (Si, Sj), i<j, i,j in {1..5}:
  - Coverage: fraction of a standardised 32x32 grid filled by cells projected
              to that pair
  - Hull ratio across fitness tiers: does tier-4 spread more than tier-1?
  - Best fitness achievable in the pair (sanity check)
  - Rank all pairs by a composite "informativeness" score and plot the 10
    projections side by side.

Uses the raw archive (sub-optimal-biased) — the right regime for the paper's
mid-regime argument.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

ARCHIVE = "/home/hisham26/Desktop/RL/results/lensmega/cooke_triplet_seidel5d_hardbounds_20260423_014429"
OUT = "/home/hisham26/Desktop/RL/paper_figures/pair_scan_5d_hardbounds"
os.makedirs(OUT, exist_ok=True)

GRID = 32  # resolution for 2-D coverage measurement
SEIDEL_NAMES = ["S1_spherical", "S2_coma", "S3_astigmatism", "S4_petzval", "S5_distortion"]
SEIDEL_SHORT = ["S1", "S2", "S3", "S4", "S5"]


def load_cells(path):
    with open(os.path.join(path, "archive_cells.json")) as f:
        d = json.load(f)
    cells = d["cells"]
    S = np.array([c["measures"] for c in cells])           # (N, 5)
    fit = np.array([c["fitness"] for c in cells])          # (N,)
    rms = np.array([c["metrics"]["rms_spot"] for c in cells])
    inv = np.array([c["metrics"]["invalid_fraction"] for c in cells])
    efl_err = np.array([abs(c["metrics"]["efl_error_pct"]) for c in cells])
    return S, fit, rms, inv, efl_err


def coverage_2d(x, y, n_bins=GRID):
    """Return fraction of 2-D grid cells that contain >=1 point, on a
    percentile-based [2, 98] range to avoid being dominated by outliers."""
    lo_x, hi_x = np.percentile(x, [2, 98])
    lo_y, hi_y = np.percentile(y, [2, 98])
    xs = ((x - lo_x) / max(hi_x - lo_x, 1e-12) * n_bins).astype(int)
    ys = ((y - lo_y) / max(hi_y - lo_y, 1e-12) * n_bins).astype(int)
    xs = np.clip(xs, 0, n_bins - 1)
    ys = np.clip(ys, 0, n_bins - 1)
    grid = np.zeros((n_bins, n_bins), dtype=bool)
    grid[xs, ys] = True
    return float(grid.sum()) / (n_bins * n_bins)


def convex_hull_area(pts):
    if len(pts) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)  # in 2-D, .volume == area
    except Exception:
        return 0.0


def tier_hull_ratio(x, y, fit, n_tiers=4):
    """Hull(tier4) / Hull(tier1). fit: higher (less negative) = better."""
    qs = np.quantile(fit, np.linspace(0, 1, n_tiers + 1))
    # Tier 1 = best quartile (fitness in top quartile, i.e. largest fit values)
    t1_mask = fit >= qs[-2]
    t4_mask = fit <= qs[1]
    pts_t1 = np.column_stack([x[t1_mask], y[t1_mask]])
    pts_t4 = np.column_stack([x[t4_mask], y[t4_mask]])
    h1 = convex_hull_area(pts_t1)
    h4 = convex_hull_area(pts_t4)
    ratio = h4 / h1 if h1 > 1e-12 else float("inf")
    return h1, h4, ratio


def main():
    S, fit, rms, inv, efl_err = load_cells(ARCHIVE)
    print(f"loaded {len(S)} cells from 5-D run")
    print(f"fitness range: {fit.min():.3f} to {fit.max():.3f}")
    print(f"rms_spot range: {rms.min():.3f} to {rms.max():.3f}")

    # Drop cells with very high invalid_fraction -- they're junk designs
    keep = (inv < 0.3) & np.isfinite(fit)
    S = S[keep]; fit = fit[keep]; rms = rms[keep]; efl_err = efl_err[keep]
    print(f"after invalid-fraction filter (<0.3): {len(S)} cells")
    if len(S) < 50:
        print("too few clean cells -- aborting")
        return

    # Enumerate 10 pairs
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]

    results = []
    for i, j in pairs:
        x = S[:, i]; y = S[:, j]
        cov = coverage_2d(x, y, n_bins=GRID)
        h1, h4, ratio = tier_hull_ratio(x, y, fit)
        best_fit = fit.max()
        results.append({
            "pair": (SEIDEL_SHORT[i], SEIDEL_SHORT[j]),
            "pair_idx": (i, j),
            "coverage_frac": cov,
            "hull_tier1": h1,
            "hull_tier4": h4,
            "hull_ratio_t4_over_t1": ratio,
            "best_fitness": best_fit,
        })

    # Rank by hull_ratio_t4_over_t1 (how much worse-tier spreads vs best-tier)
    results_by_ratio = sorted(results, key=lambda r: r["hull_ratio_t4_over_t1"], reverse=True)
    # Also rank by coverage
    results_by_cov = sorted(results, key=lambda r: r["coverage_frac"], reverse=True)

    print("\n--- RANKED BY TIER-HULL RATIO (tier4 spread / tier1 spread) ---")
    print(f"{'pair':<10} {'cov_32x32':>10} {'hull_t1':>10} {'hull_t4':>10} {'t4/t1':>8}")
    for r in results_by_ratio:
        pair_str = f"({r['pair'][0]}, {r['pair'][1]})"
        print(f"{pair_str:<10} {r['coverage_frac']:>10.3f} "
              f"{r['hull_tier1']:>10.2e} {r['hull_tier4']:>10.2e} "
              f"{r['hull_ratio_t4_over_t1']:>8.2f}")

    print("\n--- RANKED BY COVERAGE (32x32 grid fill fraction) ---")
    print(f"{'pair':<10} {'cov_32x32':>10} {'t4/t1':>8}")
    for r in results_by_cov:
        pair_str = f"({r['pair'][0]}, {r['pair'][1]})"
        print(f"{pair_str:<10} {r['coverage_frac']:>10.3f} "
              f"{r['hull_ratio_t4_over_t1']:>8.2f}")

    # Plot: 10 pair scatters arranged in a 2x5 grid, coloured by fitness
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    vmin, vmax = np.percentile(fit, [2, 98])
    for ax, (i, j), r in zip(axes.flat, [x["pair_idx"] for x in results], results):
        sc = ax.scatter(S[:, i], S[:, j], c=fit, cmap="viridis",
                        vmin=vmin, vmax=vmax, s=6, alpha=0.7)
        ax.set_xlabel(SEIDEL_SHORT[i]);  ax.set_ylabel(SEIDEL_SHORT[j])
        ax.set_title(
            f"({r['pair'][0]}, {r['pair'][1]})\n"
            f"cov={r['coverage_frac']:.2f}  t4/t1={r['hull_ratio_t4_over_t1']:.1f}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
    cbar.set_label("fitness (higher = better)")
    fig.suptitle(
        "5-D Cooke Seidel archive — all 10 BC-pair projections (N={})"
        .format(len(S)),
        fontsize=14, fontweight="bold",
    )
    out_png = os.path.join(OUT, "pair_scan_projections.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nsaved {out_png}")
    plt.close(fig)

    # Second figure: ranked bar chart of hull-ratio and coverage
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [f"({r['pair'][0]},{r['pair'][1]})" for r in results_by_ratio]
    vals_ratio = [r["hull_ratio_t4_over_t1"] for r in results_by_ratio]
    axes[0].barh(labels[::-1], vals_ratio[::-1], color="C0")
    axes[0].set_xlabel("tier-4 / tier-1 hull area ratio")
    axes[0].set_title("Worst-vs-best-tier spread\n(higher = Seidel better resolves\nsub-optimal diversity)")
    axes[0].axvline(1.0, color="k", ls=":", alpha=0.5)
    axes[0].grid(True, axis="x", alpha=0.3)

    labels2 = [f"({r['pair'][0]},{r['pair'][1]})" for r in results_by_cov]
    vals_cov = [r["coverage_frac"] for r in results_by_cov]
    axes[1].barh(labels2[::-1], vals_cov[::-1], color="C2")
    axes[1].set_xlabel(f"coverage (frac of 32x32 grid filled)")
    axes[1].set_title("Archive spread per pair\n(higher = BC pair resolves more\ndistinct design states)")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(OUT, "pair_scan_rankings.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"saved {out_png}")
    plt.close(fig)

    # Save numeric results as JSON
    out_json = os.path.join(OUT, "pair_scan_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
