"""
4-D quadruple scan on the hardcoded-bounds 5-D Seidel archive.
"""
import json
import os
import itertools
import numpy as np
from scipy.spatial import ConvexHull

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


def coverage_nd(coords, n_bins):
    lo = np.percentile(coords, 2, axis=0)
    hi = np.percentile(coords, 98, axis=0)
    idx = ((coords - lo) / (hi - lo + 1e-12) * n_bins).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)
    uniq = set(map(tuple, idx))
    return len(uniq) / (n_bins ** coords.shape[1])


def hull_volume_safe(pts):
    if len(pts) < pts.shape[1] + 1:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return 0.0


def tier_ratio(coords, fit):
    qs = np.quantile(fit, np.linspace(0, 1, 5))
    h1 = hull_volume_safe(coords[fit >= qs[-2]])
    h4 = hull_volume_safe(coords[fit <= qs[1]])
    return h1, h4, (h4 / h1 if h1 > 1e-24 else float("inf"))


def main():
    S, fit, inv = load_cells()
    keep = (inv < 0.3) & np.isfinite(fit)
    S = S[keep]; fit = fit[keep]
    print(f"clean cells: {len(S)}\n")

    # Quadruples C(5,4) = 5
    print(f"{'quadruple':<22} {'cov_8^4':>10} {'cov_10^4':>10} {'hull_t1':>12} {'hull_t4':>12} {'t4/t1':>8}")
    for combo in itertools.combinations(range(5), 4):
        coords = S[:, list(combo)]
        c8 = coverage_nd(coords, 8)
        c10 = coverage_nd(coords, 10)
        h1, h4, ratio = tier_ratio(coords, fit)
        name = "(" + ",".join(SEIDEL[i] for i in combo) + ")"
        print(f"{name:<22} {c8:>10.4f} {c10:>10.4f} {h1:>12.3e} {h4:>12.3e} {ratio:>8.2f}")

    # Full 5-D reference
    print()
    coords5 = S
    c5_8 = coverage_nd(coords5, 8)
    h1, h4, ratio = tier_ratio(coords5, fit)
    print(f"{'(S1..S5) full 5-D':<22} {c5_8:>10.4f} {'-':>10} {h1:>12.3e} {h4:>12.3e} {ratio:>8.2f}")

    # Also reference 3-D (S2,S3,S4) and 2-D (S2,S3)
    print("\n--- references ---")
    for combo, label in [((1,2,3), "(S2,S3,S4) 3-D"),
                         ((1,2), "(S2,S3) 2-D"),
                         ((2,3), "(S3,S4) 2-D")]:
        coords = S[:, list(combo)]
        h1, h4, ratio = tier_ratio(coords, fit)
        print(f"{label:<22} hull t1={h1:>10.3e}  t4={h4:>10.3e}  ratio={ratio:.2f}")


if __name__ == "__main__":
    main()
