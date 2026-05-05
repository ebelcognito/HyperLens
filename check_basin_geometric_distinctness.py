"""
Sanity test: are the refined-theta basins ACTUALLY geometrically distinct?

For the 28 cooke triplet elites with refined RMS ≤ 30 µm:
  1. Re-cluster (same params as descriptor_discriminativeness.py).
  2. Per basin: list curvature sign pattern, member RMS, theta values.
  3. Numerical test: within-basin θ-distance vs between-basin θ-distance.
     If the ratio is small (e.g. < 0.3), basins are well-separated in shape.
  4. Visual: stack one lens-diagram PNG per multi-member basin.
"""
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "cooke_triplet_materials_20260414_173935")
OUT = "/home/hisham26/Desktop/RL/paper_figures/descriptor_discriminativeness_cooke"
os.makedirs(OUT, exist_ok=True)

REFINED_RMS_MAX_UM = 30.0
CLUSTER_FRAC = 0.25


def load_elites():
    paths = sorted(glob.glob(os.path.join(RUN_DIR, "elite_designs", "*.json")))
    out = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        ref = d.get("refined") or {}
        if "theta" not in ref or "rms_mm" not in ref:
            continue
        rms_um = float(ref["rms_mm"]) * 1000.0
        if rms_um > REFINED_RMS_MAX_UM:
            continue
        out.append({
            "name": os.path.basename(p)[:-5],
            "theta_refined": np.array(ref["theta"], dtype=float),
            "rms_um_refined": rms_um,
            "sign_pattern": d.get("sign_pattern", ""),
            "lens_png": os.path.join(RUN_DIR, "elite_designs",
                                     f"{os.path.basename(p)[:-5]}_refined.png"),
        })
    return out


def cluster(elites, frac=CLUSTER_FRAC):
    thetas = np.array([e["theta_refined"] for e in elites])
    mu = thetas.mean(0); sd = thetas.std(0) + 1e-9
    X = (thetas - mu) / sd
    d = pdist(X)
    thresh = max(float(np.median(d) * frac), 0.5)
    Z = linkage(X, method="single", metric="euclidean")
    return fcluster(Z, t=thresh, criterion="distance"), X, thresh


def main():
    elites = load_elites()
    print(f"loaded {len(elites)} elites with refined RMS ≤ {REFINED_RMS_MAX_UM}μm")

    labels, X, thresh = cluster(elites)
    n_basins = len(set(labels))
    print(f"clustered into {n_basins} basins (threshold = {thresh:.3f} std-units)\n")

    # ============== test 1: curvature sign patterns per basin ==============
    print("========== TEST 1: CURVATURE SIGN PATTERN PER BASIN ==========")
    print("(each lens has a 6-character curvature sign pattern; if basins")
    print("share patterns, they may be sub-clusters of the same topology)\n")
    print(f"  {'basin':>6} {'n':>3}  {'curv signs (set)':<22} {'RMS μm range':<18} members")
    basin_to_signs = {}
    for c in sorted(set(labels)):
        mask = np.where(labels == c)[0]
        signs = set(elites[i]["sign_pattern"] for i in mask)
        rms_vals = [elites[i]["rms_um_refined"] for i in mask]
        basin_to_signs[int(c)] = signs
        signs_str = ", ".join(sorted(signs)) if len(signs) <= 3 else f"{len(signs)} distinct"
        rms_str = f"{min(rms_vals):.2f}-{max(rms_vals):.2f}"
        names = ", ".join(elites[i]["name"][:18] for i in mask[:3])
        if len(mask) > 3: names += "..."
        print(f"  {c:>4} {len(mask):>3}  {signs_str:<22} {rms_str:<18} {names}")

    # Check: does any sign pattern appear in MULTIPLE basins?
    print("\n--- sign pattern overlap test ---")
    pattern_to_basins = {}
    for c, signs in basin_to_signs.items():
        for s in signs:
            pattern_to_basins.setdefault(s, []).append(c)
    overlapping = {s: bs for s, bs in pattern_to_basins.items() if len(bs) > 1}
    if overlapping:
        print(f"  ⚠ {len(overlapping)} sign patterns appear in >1 basin:")
        for s, bs in overlapping.items():
            print(f"    '{s}' in basins {bs}")
    else:
        print(f"  ✓ all {len(pattern_to_basins)} sign patterns are basin-unique")
    n_unique_signs = len(pattern_to_basins)
    print(f"  total distinct curvature sign patterns: {n_unique_signs} / {n_basins} basins")

    # ============== test 2: within vs between basin θ-distance ==============
    print("\n========== TEST 2: WITHIN VS BETWEEN BASIN θ-DISTANCE ==========")
    D = squareform(pdist(X, metric="euclidean"))
    within, between = [], []
    for i in range(len(elites)):
        for j in range(i+1, len(elites)):
            (within if labels[i] == labels[j] else between).append(D[i, j])
    if within and between:
        ratio = np.mean(within) / np.mean(between)
        print(f"  mean within-basin   θ-dist: {np.mean(within):.3f}  (n={len(within)} pairs)")
        print(f"  mean between-basin  θ-dist: {np.mean(between):.3f}  (n={len(between)} pairs)")
        print(f"  RATIO (within/between)    : {ratio:.3f}")
        if ratio < 0.3:
            print(f"  → basins are well-separated in θ space (ratio < 0.3)")
        elif ratio < 0.5:
            print(f"  → basins are moderately separated in θ space")
        else:
            print(f"  → basins are poorly separated; clustering may be over-fragmenting")
    else:
        print("  (no multi-member basins for within-basin distance)")

    # ============== test 3: visual gallery ==============
    print("\n========== TEST 3: VISUAL GALLERY ==========")
    multi = [c for c in sorted(set(labels))
             if int(np.sum(labels == c)) >= 2]
    print(f"  {len(multi)} multi-member basins for visual stacking")
    if not multi:
        return

    # Pick best-RMS rep per multi-member basin
    reps = []
    for c in multi:
        mask = np.where(labels == c)[0]
        i = mask[np.argmin([elites[k]["rms_um_refined"] for k in mask])]
        reps.append(elites[i])
        reps[-1]["_basin"] = int(c)

    n = len(reps)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 2.4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, r in zip(axes, reps):
        if os.path.exists(r["lens_png"]):
            ax.imshow(imread(r["lens_png"]))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"basin {r['_basin']}  ·  RMS {r['rms_um_refined']:.2f} μm\n"
            f"signs {r['sign_pattern']}",
            fontsize=9, fontweight="bold",
        )
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "Refined lens diagrams — best-RMS rep per multi-member basin\n"
        "(Cooke triplet, refined RMS ≤ 30 µm)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = os.path.join(OUT, "basins_visual_gallery.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png}")

    print("\n========== VERDICT ==========")
    print(f"  basins:                {n_basins}")
    print(f"  multi-member basins:   {len(multi)}")
    print(f"  distinct curv signs:   {n_unique_signs}")
    if within and between:
        print(f"  within/between ratio:  {ratio:.3f}")


if __name__ == "__main__":
    main()
