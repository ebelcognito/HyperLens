"""
3-D triple-scan on the clean hardcoded-bounds 5-D Seidel archive.

For each of the 10 possible 3-D BC triples (Si, Sj, Sk) from {S1..S5}:
  - 3-D coverage on a standardised 16^3 grid
  - Hull-volume ratio across fitness tiers (tier-4 / tier-1)
  - Comparison against the best 2-D pair (S3, S4) as a reference
"""
import json
import os
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

ARCHIVE = "/home/hisham26/Desktop/RL/results/lensmega/cooke_triplet_seidel5d_hardbounds_20260423_014429"
OUT = "/home/hisham26/Desktop/RL/paper_figures/triple_scan_5d"
os.makedirs(OUT, exist_ok=True)

GRID = 16
SEIDEL_SHORT = ["S1", "S2", "S3", "S4", "S5"]


def load_cells(path):
    with open(os.path.join(path, "archive_cells.json")) as f:
        d = json.load(f)
    cells = d["cells"]
    S = np.array([c["measures"] for c in cells])
    fit = np.array([c["fitness"] for c in cells])
    inv = np.array([c["metrics"]["invalid_fraction"] for c in cells])
    return S, fit, inv


def coverage_nd(coords, n_bins=GRID):
    """coords: (N, D). Return fraction of D-dim standardised grid filled."""
    lo = np.percentile(coords, 2, axis=0)
    hi = np.percentile(coords, 98, axis=0)
    idx = ((coords - lo) / (hi - lo + 1e-12) * n_bins).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)
    uniq = set(map(tuple, idx))
    return len(uniq) / (n_bins ** coords.shape[1])


def hull_volume(pts):
    if len(pts) < pts.shape[1] + 1:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return 0.0


def tier_hull_ratio_nd(coords, fit, n_tiers=4):
    qs = np.quantile(fit, np.linspace(0, 1, n_tiers + 1))
    t1_mask = fit >= qs[-2]   # best quartile
    t4_mask = fit <= qs[1]    # worst quartile
    h1 = hull_volume(coords[t1_mask])
    h4 = hull_volume(coords[t4_mask])
    ratio = h4 / h1 if h1 > 1e-18 else float("inf")
    return h1, h4, ratio


def main():
    S, fit, inv = load_cells(ARCHIVE)
    keep = (inv < 0.3) & np.isfinite(fit)
    S = S[keep]; fit = fit[keep]
    print(f"clean cells: {len(S)}")

    # Pairs (reference)
    pair_results = []
    for i, j in itertools.combinations(range(5), 2):
        coords = S[:, [i, j]]
        cov = coverage_nd(coords, n_bins=32)
        h1, h4, ratio = tier_hull_ratio_nd(coords, fit)
        pair_results.append({
            "kind": "pair",
            "axes": [SEIDEL_SHORT[i], SEIDEL_SHORT[j]],
            "coverage_32": cov,
            "hull_t1": h1,
            "hull_t4": h4,
            "hull_ratio": ratio,
        })

    # Triples
    triple_results = []
    for i, j, k in itertools.combinations(range(5), 3):
        coords = S[:, [i, j, k]]
        cov = coverage_nd(coords, n_bins=GRID)
        h1, h4, ratio = tier_hull_ratio_nd(coords, fit)
        triple_results.append({
            "kind": "triple",
            "axes": [SEIDEL_SHORT[i], SEIDEL_SHORT[j], SEIDEL_SHORT[k]],
            "coverage_16": cov,
            "hull_t1": h1,
            "hull_t4": h4,
            "hull_ratio": ratio,
        })

    print("\n--- 3-D TRIPLE RANKINGS (by hull-volume ratio tier4 / tier1) ---")
    print(f"{'triple':<16} {'cov_16^3':>10} {'hull_t1':>11} {'hull_t4':>11} {'t4/t1':>8}")
    for r in sorted(triple_results, key=lambda r: r["hull_ratio"], reverse=True):
        tri = "(" + ",".join(r["axes"]) + ")"
        print(f"{tri:<16} {r['coverage_16']:>10.4f} {r['hull_t1']:>11.3e} {r['hull_t4']:>11.3e} {r['hull_ratio']:>8.2f}")

    print("\n--- 3-D TRIPLE RANKINGS (by coverage on 16^3 grid) ---")
    for r in sorted(triple_results, key=lambda r: r["coverage_16"], reverse=True):
        tri = "(" + ",".join(r["axes"]) + ")"
        print(f"{tri:<16} cov={r['coverage_16']:>7.4f}  t4/t1={r['hull_ratio']:.2f}")

    # Reference: top pairs
    print("\n--- Top 2-D PAIR references (by hull ratio) ---")
    for r in sorted(pair_results, key=lambda r: r["hull_ratio"], reverse=True)[:4]:
        pr = "(" + ",".join(r["axes"]) + ")"
        print(f"{pr:<12} cov_32^2={r['coverage_32']:.4f}  t4/t1={r['hull_ratio']:.2f}")

    # Paper's current 2-D choice
    p_cur = [r for r in pair_results if r["axes"] == ["S2", "S3"]][0]
    p_top = sorted(pair_results, key=lambda r: r["hull_ratio"], reverse=True)[0]
    t_top = sorted(triple_results, key=lambda r: r["hull_ratio"], reverse=True)[0]
    t_user = [r for r in triple_results if r["axes"] == ["S2", "S3", "S4"]][0]

    print("\n" + "=" * 70)
    print("HEADLINE COMPARISON")
    print("=" * 70)
    print(f"(S2, S3) — paper's current 2-D BC:      hull ratio {p_cur['hull_ratio']:.2f}")
    print(f"(S3, S4) — best 2-D BC:                 hull ratio {p_top['hull_ratio']:.2f}")
    print(f"(S2, S3, S4) — user-asked 3-D BC:       hull ratio {t_user['hull_ratio']:.2f}")
    print(f"Best 3-D BC:  ({','.join(t_top['axes'])})                  hull ratio {t_top['hull_ratio']:.2f}")
    print(f"\nCoverage: (S2,S3) on 32^2 = {p_cur['coverage_32']:.3f}  vs  (S2,S3,S4) on 16^3 = {t_user['coverage_16']:.4f}")

    # Plot: 3D scatter of (S2, S3, S4) coloured by fitness, plus three 2D projections
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    coords3 = S[:, [1, 2, 3]]  # S2, S3, S4
    vmin, vmax = np.percentile(fit, [2, 98])

    fig = plt.figure(figsize=(16, 5))
    ax3d = fig.add_subplot(1, 4, 1, projection="3d")
    p3 = ax3d.scatter(coords3[:, 0] * 1e4, coords3[:, 1] * 1e4, coords3[:, 2] * 1e4,
                      c=fit, cmap="viridis", vmin=vmin, vmax=vmax, s=5, alpha=0.6)
    ax3d.set_xlabel("S2 [×1e-4]"); ax3d.set_ylabel("S3 [×1e-4]"); ax3d.set_zlabel("S4 [×1e-4]")
    ax3d.set_title(f"(S2, S3, S4) 3-D\nt4/t1={t_user['hull_ratio']:.2f}  cov={t_user['coverage_16']:.3f}")

    # three 2-D projections
    proj_axes = [(0, 1, "S2", "S3"), (0, 2, "S2", "S4"), (1, 2, "S3", "S4")]
    for idx, (a, b, la, lb) in enumerate(proj_axes):
        ax = fig.add_subplot(1, 4, idx + 2)
        sc = ax.scatter(coords3[:, a] * 1e4, coords3[:, b] * 1e4, c=fit, cmap="viridis",
                        vmin=vmin, vmax=vmax, s=6, alpha=0.6)
        ax.set_xlabel(f"{la} [×1e-4]"); ax.set_ylabel(f"{lb} [×1e-4]")
        # Find the corresponding 2D pair hull ratio
        pair_r = [r for r in pair_results if set(r["axes"]) == {la, lb}][0]
        ax.set_title(f"proj ({la}, {lb})\nt4/t1={pair_r['hull_ratio']:.2f}")
        ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(sc, ax=fig.get_axes(), shrink=0.8, pad=0.02)
    cbar.set_label("fitness")
    fig.suptitle("3-D Seidel BC: (S2, S3, S4) — 3-D view + three 2-D projections",
                 fontsize=12, fontweight="bold")
    out_png = os.path.join(OUT, "triple_S2S3S4_3d_projections.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nsaved {out_png}")
    plt.close(fig)

    # Save results
    with open(os.path.join(OUT, "triple_scan_results.json"), "w") as f:
        json.dump({"pairs": pair_results, "triples": triple_results}, f, indent=2)


if __name__ == "__main__":
    main()
