"""
Preparation step for the sample-efficiency sub-region experiment.

Load every refined elite from the clean 2-D Cooke archive, cluster them in
standardised theta-space to find distinct basins (known local minima), and
report:
  - how many distinct minima we have
  - their centroids in theta-space
  - the bounding box of all minima (useful for picking a sub-region bound)
  - the PCA directions of the minima cluster (for 2-D visualisation later)
"""
import glob
import json
import os
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

ARCHIVE = (
    "/home/hisham26/Desktop/RL/results/lensmega/"
    "cooke_triplet_materials_20260414_173935"
)

def load_refined_theta(archive_dir):
    paths = sorted(glob.glob(os.path.join(archive_dir, "elite_designs", "*.json")))
    out = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if "refined" not in d or not d["refined"]:
            continue
        ref = d["refined"]
        if "theta" not in ref:
            continue
        out.append({
            "name": os.path.basename(p)[:-5],
            "theta": np.array(ref["theta"], dtype=float),
            "rms_um": float(ref["rms_mm"]) * 1000.0,
        })
    return out


def main():
    elites = load_refined_theta(ARCHIVE)
    print(f"loaded {len(elites)} refined elites")
    thetas = np.array([e["theta"] for e in elites])
    rms = np.array([e["rms_um"] for e in elites])

    # Standardise each dim by its std for fair distance
    mu = thetas.mean(axis=0); sd = thetas.std(axis=0) + 1e-12
    thetas_n = (thetas - mu) / sd

    # Look at pairwise distance distribution to pick a cluster threshold
    d = pdist(thetas_n, metric="euclidean")
    print(f"pairwise theta-distance (standardised):")
    print(f"  min={d.min():.2f}  median={np.median(d):.2f}  max={d.max():.2f}")
    print(f"  percentiles 10/25/50/75/90: "
          f"{np.percentile(d, [10, 25, 50, 75, 90])}")

    # Single-linkage cluster at different thresholds
    print("\nCluster count vs threshold:")
    print(f"{'threshold':>10} {'n_basins':>10} {'largest':>10} {'singletons':>11}")
    for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        Z = linkage(thetas_n, method="single", metric="euclidean")
        labels = fcluster(Z, t=t, criterion="distance")
        sizes = sorted([int((labels == b).sum()) for b in set(labels)], reverse=True)
        singletons = sum(1 for s in sizes if s == 1)
        print(f"{t:>10.2f} {len(sizes):>10d} {sizes[0]:>10d} {singletons:>11d}")

    # Pick threshold = 1.5 (knee heuristic — tune later)
    t_best = 1.5
    Z = linkage(thetas_n, method="single", metric="euclidean")
    labels = fcluster(Z, t=t_best, criterion="distance")
    unique_basins = sorted(set(labels))
    print(f"\nUsing threshold {t_best}: {len(unique_basins)} basins")

    # Report each basin's centroid + member RMS range
    basins = []
    for b in unique_basins:
        mask = labels == b
        if mask.sum() < 1:
            continue
        members = thetas[mask]
        centroid = members.mean(axis=0)
        basins.append({
            "basin_id": int(b),
            "n_members": int(mask.sum()),
            "centroid_theta": centroid.tolist(),
            "rms_range_um": [float(rms[mask].min()), float(rms[mask].max())],
            "best_rms_um": float(rms[mask].min()),
            "member_names": [elites[i]["name"] for i in range(len(elites)) if mask[i]],
        })
    basins.sort(key=lambda b: b["best_rms_um"])

    print(f"\nBasin report (sorted by best refined RMS):")
    print(f"{'basin':>6} {'members':>8} {'best_rms':>10} {'rms_range':>20}")
    for b in basins:
        print(f"{b['basin_id']:>6} {b['n_members']:>8} {b['best_rms_um']:>8.2f} μm "
              f"[{b['rms_range_um'][0]:5.2f}, {b['rms_range_um'][1]:5.2f}] μm")

    # Compute overall bounding box of centroids
    centroids = np.array([b["centroid_theta"] for b in basins])
    bbox_lo = centroids.min(axis=0)
    bbox_hi = centroids.max(axis=0)
    span = bbox_hi - bbox_lo
    print(f"\nCentroid bounding box (min, max, span) per theta dim:")
    theta_labels = (
        [f"c{i}" for i in range(6)] +
        [f"g{i}" for i in range(6)] +
        ["flange"] +
        ["n0", "V0", "n1", "V1", "n2", "V2"]
    )
    for i, lab in enumerate(theta_labels):
        print(f"  {lab:>7} : [{bbox_lo[i]:+8.4f}, {bbox_hi[i]:+8.4f}] "
              f"span={span[i]:+8.4f}")

    # PCA on centroids for 2-D visualisation later
    if len(centroids) >= 3:
        X = (centroids - centroids.mean(axis=0)) / (thetas.std(axis=0) + 1e-12)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        var_explained = (S ** 2) / (S ** 2).sum()
        print(f"\nPCA of centroids (top components, fraction of variance explained):")
        for k in range(min(5, len(S))):
            print(f"  PC{k+1}: {var_explained[k] * 100:.1f}%")
        print(f"Top-2 PCs cumulative: {var_explained[:2].sum() * 100:.1f}%")

    # Save
    out = {
        "archive": ARCHIVE,
        "n_elites": len(elites),
        "cluster_threshold_standardised": t_best,
        "n_basins": len(basins),
        "theta_mean": mu.tolist(),
        "theta_std": sd.tolist(),
        "centroid_bbox": {
            "lo": bbox_lo.tolist(),
            "hi": bbox_hi.tolist(),
            "span": span.tolist(),
        },
        "basins": basins,
    }
    out_dir = "/home/hisham26/Desktop/RL/paper_figures/sample_efficiency_subregion"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "known_minima.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {os.path.join(out_dir, 'known_minima.json')}")


if __name__ == "__main__":
    main()
