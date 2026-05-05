"""
Cross-check the hull-ratio ranking with tier-1-focused, outlier-resistant metrics.

Three metrics per BC (all computed on the same cleaned 5-D archive):

  1. hull_ratio = Hull(tier 4) / Hull(tier 1)                    [original]
  2. std_ratio  = ||std(tier 4)||2 / ||std(tier 1)||2            [outlier-resistant]
  3. tier1_compactness = mean pairwise distance within tier 1   [direct tier-1 compactness]
  4. tier1_count = number of distinct BC cells at tier 1
                  (on a standardised grid, measures design-family density at optimum)
  5. best_fitness  = best achievable fitness in tier 1
  6. tier1_density = tier1_count / tier1_hull_volume
                     (family density in the compact tier-1 region)

The intro's claim predicts: good BCs should have LOW tier-1 compactness
(well-corrected designs cluster) AND HIGH tier-1 count/density (many distinct
design families packed into that compact region).
"""
import json
import os
import itertools
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

ARCHIVE = "/home/hisham26/Desktop/RL/results/lensmega/cooke_triplet_seidel5d_hardbounds_20260423_014429"
SEIDEL = ["S1", "S2", "S3", "S4", "S5"]


def load_cells():
    with open(os.path.join(ARCHIVE, "archive_cells.json")) as f:
        d = json.load(f)
    cells = d["cells"]
    S = np.array([c["measures"] for c in cells])
    fit = np.array([c["fitness"] for c in cells])
    inv = np.array([c["metrics"]["invalid_fraction"] for c in cells])
    return S, fit, inv


def hull_volume_safe(pts):
    if len(pts) < pts.shape[1] + 1:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return 0.0


def distinct_cells(coords, n_bins=16):
    lo = np.percentile(coords, 2, axis=0)
    hi = np.percentile(coords, 98, axis=0)
    idx = ((coords - lo) / (hi - lo + 1e-12) * n_bins).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)
    return len(set(map(tuple, idx)))


def mean_pairwise(coords, sample_cap=2000):
    if len(coords) < 2:
        return 0.0
    if len(coords) > sample_cap:
        idx = np.random.default_rng(0).choice(len(coords), sample_cap, replace=False)
        coords = coords[idx]
    return float(pdist(coords, metric="euclidean").mean())


def compute_bc_metrics(coords, fit):
    qs = np.quantile(fit, np.linspace(0, 1, 5))
    t1_mask = fit >= qs[-2]
    t4_mask = fit <= qs[1]
    t1 = coords[t1_mask]; t4 = coords[t4_mask]

    hull_t1 = hull_volume_safe(t1)
    hull_t4 = hull_volume_safe(t4)
    hull_ratio = hull_t4 / hull_t1 if hull_t1 > 1e-24 else float("inf")

    std_t1 = np.linalg.norm(t1.std(axis=0))
    std_t4 = np.linalg.norm(t4.std(axis=0))
    std_ratio = std_t4 / std_t1 if std_t1 > 1e-18 else float("inf")

    tier1_compact = mean_pairwise(t1)
    tier4_compact = mean_pairwise(t4)
    tier4_over_tier1_mpd = tier4_compact / tier1_compact if tier1_compact > 1e-18 else float("inf")

    # Discretised tier-1 cell count (how many distinct well-corrected families we picked up)
    bins_per_dim = {1: 64, 2: 32, 3: 16, 4: 10, 5: 8}.get(coords.shape[1], 16)
    t1_cells = distinct_cells(t1, n_bins=bins_per_dim) if len(t1) > 0 else 0

    # Family density in tier 1
    t1_density = t1_cells / hull_t1 if hull_t1 > 1e-24 else 0.0

    return {
        "hull_ratio": hull_ratio,
        "std_ratio": std_ratio,
        "mpd_ratio": tier4_over_tier1_mpd,
        "tier1_compact": tier1_compact,
        "tier1_cells": t1_cells,
        "tier1_hull": hull_t1,
        "tier1_density": t1_density,
        "best_fit": float(fit[t1_mask].max()) if t1_mask.any() else float("-inf"),
    }


def enumerate_bcs():
    """Return list of (name, index_tuple, dim)."""
    out = []
    for combo in itertools.combinations(range(5), 2):
        out.append(("(" + ",".join(SEIDEL[i] for i in combo) + ")", combo, 2))
    for combo in itertools.combinations(range(5), 3):
        out.append(("(" + ",".join(SEIDEL[i] for i in combo) + ")", combo, 3))
    for combo in itertools.combinations(range(5), 4):
        out.append(("(" + ",".join(SEIDEL[i] for i in combo) + ")", combo, 4))
    out.append(("(S1..S5)", (0, 1, 2, 3, 4), 5))
    return out


def main():
    S, fit, inv = load_cells()
    keep = (inv < 0.3) & np.isfinite(fit)
    S = S[keep]; fit = fit[keep]
    print(f"clean cells: {len(S)}\n")

    rows = []
    for name, combo, dim in enumerate_bcs():
        coords = S[:, list(combo)]
        m = compute_bc_metrics(coords, fit)
        rows.append({"name": name, "dim": dim, **m})

    # Show table, sorted primarily by std_ratio (the new, robust metric)
    hdr = f"{'BC':<22} {'dim':>3} {'hull_R':>7} {'std_R':>7} {'mpd_R':>7} {'t1_cells':>9} {'t1_density':>12} {'best_fit':>10}"
    print("--- ALL BCs SORTED BY std_ratio (outlier-resistant tier4/tier1 spread) ---")
    print(hdr)
    for r in sorted(rows, key=lambda r: r["std_ratio"], reverse=True):
        print(f"{r['name']:<22} {r['dim']:>3} {r['hull_ratio']:>7.2f} {r['std_ratio']:>7.2f} "
              f"{r['mpd_ratio']:>7.2f} {r['tier1_cells']:>9d} {r['tier1_density']:>12.2e} {r['best_fit']:>10.4f}")

    # Now sort by tier1_cells (how many distinct well-corrected DESIGN FAMILIES this BC captured)
    print("\n--- ALL BCs SORTED BY tier1_cells (distinct well-corrected cells) ---")
    print(hdr)
    for r in sorted(rows, key=lambda r: r["tier1_cells"], reverse=True):
        print(f"{r['name']:<22} {r['dim']:>3} {r['hull_ratio']:>7.2f} {r['std_ratio']:>7.2f} "
              f"{r['mpd_ratio']:>7.2f} {r['tier1_cells']:>9d} {r['tier1_density']:>12.2e} {r['best_fit']:>10.4f}")

    # Rank correlation check: does std_ratio agree with hull_ratio?
    hull_order = sorted(range(len(rows)), key=lambda i: rows[i]["hull_ratio"], reverse=True)
    std_order = sorted(range(len(rows)), key=lambda i: rows[i]["std_ratio"], reverse=True)
    # Spearman via rank arrays
    def rank(order):
        r = np.empty(len(order), dtype=float)
        for pos, idx in enumerate(order):
            r[idx] = pos
        return r
    r1 = rank(hull_order); r2 = rank(std_order)
    rho = 1 - (6 * ((r1 - r2) ** 2).sum()) / (len(rows) * (len(rows) ** 2 - 1))
    print(f"\nSpearman rank correlation (hull_ratio vs std_ratio) across all {len(rows)} BCs: rho = {rho:.3f}")

    # Save
    out_json = "/home/hisham26/Desktop/RL/paper_figures/pair_scan_5d_hardbounds/tier1_metrics.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved {out_json}")


if __name__ == "__main__":
    main()
