"""
How many low-RMS samples per phase, and how geometrically diverse are they?

Uses the same 1000-per-phase + in-plane filter as the landscape figure.
Filters out degenerate near-zero RMS cases (pathological lenses), then:
  - counts samples below various RMS thresholds
  - clusters the low-RMS subset in θ-space to count distinct basins
  - reports mean pairwise θ-distance as a diversity metric
"""
import json
import os
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "sixlet_discovery_20260423_132001")
GRIDS_NPZ = ("/home/hisham26/Desktop/RL/paper_figures/landscape_sixlet/"
             "grids.npz")

TARGET_N = 1000
RESID_KEEP_FRAC = 0.50

# "Legit" filter: drop pathological ~0 RMS + broken designs
MIN_RMS_MM = 1e-3      # >1 μm to exclude degenerate lenses
MAX_INV_FRAC = 0.2     # at least 80% of rays must hit the sensor


def load_lineage_full():
    seeds, offs = [], []
    with open(os.path.join(RUN_DIR, "lineage.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            th = r.get("theta")
            if th is None:
                continue
            row = {
                "theta": np.asarray(th, dtype=float),
                "fitness": float(r.get("fitness", np.nan)),
                "rms_spot": float(r.get("rms_spot", np.nan)),
                "invalid_fraction": float(r.get("invalid_fraction", 1.0)),
            }
            (seeds if r["source"] == "seed" else offs).append(row)
    return seeds, offs


def cluster_basins(thetas, rms_cap_um=None):
    """Standardise θ, single-linkage cluster. Threshold chosen so that
    extremely similar lenses (< ~10% of mean pairwise) are merged."""
    if len(thetas) < 2:
        return np.array([0] * len(thetas)), 1
    mu = thetas.mean(axis=0); sd = thetas.std(axis=0) + 1e-9
    X = (thetas - mu) / sd
    d = pdist(X, metric="euclidean")
    # Threshold = 25% of median pairwise distance → merges near-duplicates,
    # leaves geometrically distinct lenses as separate clusters
    thresh = max(float(np.median(d) * 0.25), 0.1)
    Z = linkage(X, method="single", metric="euclidean")
    labels = fcluster(Z, t=thresh, criterion="distance")
    return labels, len(set(labels)), thresh


def main():
    grids = np.load(GRIDS_NPZ)
    origin = grids["origin"]; e1 = grids["e1"]; e2 = grids["e2"]

    seeds, offs = load_lineage_full()

    rng = np.random.default_rng(0)
    seed_idx = rng.choice(len(seeds), min(TARGET_N, len(seeds)), replace=False)
    off_idx = rng.choice(len(offs), min(TARGET_N, len(offs)), replace=False)

    def pack(rows, idxs):
        th = np.array([rows[i]["theta"] for i in idxs])
        rms = np.array([rows[i]["rms_spot"] for i in idxs])
        inv = np.array([rows[i]["invalid_fraction"] for i in idxs])
        return th, rms, inv
    s_th, s_rms, s_inv = pack(seeds, seed_idx)
    o_th, o_rms, o_inv = pack(offs, off_idx)

    def proj_resid(th):
        d = th - origin
        a = d @ e1; b = d @ e2
        return np.linalg.norm(d - a[:, None] * e1 - b[:, None] * e2, axis=1)

    sr = proj_resid(s_th); or_ = proj_resid(o_th)
    s_inplane = sr < np.percentile(sr, RESID_KEEP_FRAC * 100)
    o_inplane = or_ < np.percentile(or_, RESID_KEEP_FRAC * 100)

    # "Legit" filter: non-degenerate RMS AND low-invalid-fraction
    s_legit = s_inplane & (s_rms > MIN_RMS_MM) & (s_inv < MAX_INV_FRAC) & np.isfinite(s_rms)
    o_legit = o_inplane & (o_rms > MIN_RMS_MM) & (o_inv < MAX_INV_FRAC) & np.isfinite(o_rms)

    print(f"Sixlet — in-plane + non-degenerate sample counts")
    print(f"  raw shown: seeds {s_inplane.sum()} · offspring {o_inplane.sum()}")
    print(f"  after dropping degenerate (RMS<{MIN_RMS_MM*1000:.0f}μm) & "
          f"broken (inv_frac>{MAX_INV_FRAC}): seeds {s_legit.sum()} · "
          f"offspring {o_legit.sum()}")
    print()

    print("=== RMS threshold counts (legit samples only, matched-N in-plane) ===")
    print(f"{'threshold':>14} {'seed count':>12} {'off count':>12} "
          f"{'seed %':>10} {'off %':>10}")
    for thr_um in [50, 100, 200, 500, 1000, 2000, 5000]:
        thr_mm = thr_um / 1000.0
        ns = int(((s_rms < thr_mm) & s_legit).sum())
        no = int(((o_rms < thr_mm) & o_legit).sum())
        print(f"  < {thr_um:>5}μm   {ns:>10}   {no:>10}   "
              f"{100*ns/max(s_legit.sum(),1):>8.2f}%   "
              f"{100*no/max(o_legit.sum(),1):>8.2f}%")

    # For each threshold, cluster in θ-space and count distinct geometric
    # families
    print()
    print("=== Geometric diversity of low-RMS samples (distinct basins) ===")
    print(f"{'threshold':>14} {'seed bas/samples':>20} {'off bas/samples':>20}")
    for thr_um in [50, 100, 200, 500, 1000]:
        thr_mm = thr_um / 1000.0
        s_sel = s_legit & (s_rms < thr_mm)
        o_sel = o_legit & (o_rms < thr_mm)
        n_s, n_o = int(s_sel.sum()), int(o_sel.sum())
        if n_s >= 2:
            _, k_s, _ = cluster_basins(s_th[s_sel])
        else:
            k_s = n_s
        if n_o >= 2:
            _, k_o, _ = cluster_basins(o_th[o_sel])
        else:
            k_o = n_o
        print(f"  < {thr_um:>5}μm   {k_s:>5} basins / {n_s:<5} samples  "
              f"{k_o:>5} basins / {n_o:<5} samples")

    # Mean pairwise standardised-θ distance within low-RMS subset — another
    # diversity metric (larger = more spread out)
    print()
    print("=== Mean pairwise θ distance (standardised, larger = more diverse) ===")
    print(f"{'threshold':>14} {'seed mean d':>14} {'off mean d':>14}")
    for thr_um in [50, 100, 200, 500, 1000]:
        thr_mm = thr_um / 1000.0
        s_sel = s_legit & (s_rms < thr_mm)
        o_sel = o_legit & (o_rms < thr_mm)
        n_s, n_o = int(s_sel.sum()), int(o_sel.sum())
        def mean_pairwise(th):
            if len(th) < 2:
                return float("nan")
            mu = th.mean(axis=0); sd = th.std(axis=0) + 1e-9
            X = (th - mu) / sd
            return float(np.mean(pdist(X)))
        ms = mean_pairwise(s_th[s_sel])
        mo = mean_pairwise(o_th[o_sel])
        ms_str = f"{ms:.3f}" if not np.isnan(ms) else "n/a"
        mo_str = f"{mo:.3f}" if not np.isnan(mo) else "n/a"
        print(f"  < {thr_um:>5}μm   {ms_str:>14}   {mo_str:>14}")


if __name__ == "__main__":
    main()
