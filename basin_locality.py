"""
Basin-locality test: how well do different 2-D BC choices predict basin identity?

Comparators (all 2-D, the dimensionality of the BC archive used in the paper):
  - Seidel (S2, S3)        -- the paper's choice
  - Random 2-D projection of theta  -- null baseline
  - (c0, c1) raw curvatures  -- geometric BC
  - (rms_raw, efl_err)       -- behavior-but-not-Seidel BC (if available)

Basins are defined by single-linkage clustering on REFINED theta. Caveat: this
makes d_theta_raw look artificially good (raw and refined theta are tightly
coupled by local L-BFGS), so we exclude it from the headline comparison and
report it only as a sanity-check upper bound.

For each archive:
  - load refined elites (raw theta, BC measures, refined theta, refined RMS)
  - cluster refined-theta -> basin labels
  - compute pairwise distance under each BC choice
  - AUC per BC: how well does small-distance predict same-basin?
  - stratify by RMS tier
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def load_elites(archive_dir):
    paths = sorted(glob.glob(os.path.join(archive_dir, "elite_designs", "*.json")))
    elites = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if "refined" not in d or not d["refined"]:
            continue
        ref = d["refined"]
        if "theta" not in ref or "rms_mm" not in ref:
            continue
        elites.append({
            "name": os.path.basename(p),
            "theta_raw": np.array(d["theta"], dtype=float),
            "seidel": np.array(d["measures"], dtype=float),
            "rms_raw": float(d["rms_mm"]),
            "theta_ref": np.array(ref["theta"], dtype=float),
            "rms_ref": float(ref["rms_mm"]),
            "metrics": d.get("metrics", {}),
        })
    return elites


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def cluster_basins(theta_ref_std, threshold):
    if len(theta_ref_std) < 2:
        return np.zeros(len(theta_ref_std), dtype=int)
    Z = linkage(theta_ref_std, method="single", metric="euclidean")
    labels = fcluster(Z, t=threshold, criterion="distance")
    return labels


def pick_basin_threshold(theta_ref_std, frac=0.25):
    """Threshold = frac * median pairwise distance.
    Smaller frac -> more, tighter basins."""
    d = pdist(theta_ref_std, metric="euclidean")
    if len(d) == 0:
        return 1.0
    return float(np.median(d) * frac)


def auc_safely(y, score):
    if len(set(y.astype(int))) < 2:
        return None
    # smaller distance -> more likely same basin -> negate
    return float(roc_auc_score(y, -score))


def make_random_projection(theta_dim, seed):
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((2, theta_dim))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P


def build_bcs(elites, n_random_seeds=5):
    """Return dict of BC name -> (n, 2) coordinate matrix."""
    theta_raw = np.array([e["theta_raw"] for e in elites])
    seidel = np.array([e["seidel"] for e in elites])
    rms_raw = np.array([e["rms_raw"] for e in elites])

    bcs = {}
    bcs["Seidel (S2,S3)"] = seidel.copy()
    bcs["(c0, c1) curvatures"] = theta_raw[:, :2].copy()

    # Random projections (we'll average AUCs across seeds for a stable null)
    theta_std = standardize(theta_raw)
    for s in range(n_random_seeds):
        P = make_random_projection(theta_raw.shape[1], seed=1000 + s)
        bcs[f"random_proj_{s}"] = (theta_std @ P.T)

    # Behavior-but-not-Seidel: (raw_rms, |efl_err|) if available
    if "efl_error_pct" in elites[0]["metrics"]:
        efl_err = np.array([abs(e["metrics"].get("efl_error_pct", 0.0)) for e in elites])
        bcs["(rms_raw, |EFL_err|)"] = np.column_stack([rms_raw, efl_err])

    return bcs


def pairwise_table(elites, basin_threshold_frac=0.25):
    n = len(elites)
    theta_raw = np.array([e["theta_raw"] for e in elites])
    theta_ref = np.array([e["theta_ref"] for e in elites])
    rms_ref = np.array([e["rms_ref"] for e in elites])

    theta_ref_std = standardize(theta_ref)
    thresh = pick_basin_threshold(theta_ref_std, frac=basin_threshold_frac)
    basin = cluster_basins(theta_ref_std, thresh)
    n_basins = len(set(basin))

    # Pairs (i<j). Store: index pairs, max_rms (for tier stratification).
    pairs_i, pairs_j = np.triu_indices(n, k=1)
    same_basin = (basin[pairs_i] == basin[pairs_j]).astype(int)
    max_rms = np.maximum(rms_ref[pairs_i], rms_ref[pairs_j])

    return {
        "pairs_i": pairs_i,
        "pairs_j": pairs_j,
        "same_basin": same_basin,
        "max_rms": max_rms,
        "basin": basin,
        "n_basins": n_basins,
        "threshold": thresh,
        "rms_ref": rms_ref,
    }


def bc_pairwise_distances(bc_coords, pairs_i, pairs_j):
    diff = bc_coords[pairs_i] - bc_coords[pairs_j]
    return np.linalg.norm(diff, axis=1)


def stratified_aucs(d, same_basin, max_rms, qs):
    out = {"overall": auc_safely(same_basin, d)}
    for t in range(1, len(qs)):
        lo, hi = qs[t - 1], qs[t]
        m = (max_rms >= lo) & (max_rms <= hi)
        sub_y = same_basin[m]
        sub_d = d[m]
        if len(sub_y) < 4 or sub_y.sum() < 1 or sub_y.sum() == len(sub_y):
            out[f"tier_{t}"] = None
        else:
            out[f"tier_{t}"] = auc_safely(sub_y, sub_d)
    return out


def aggregate_random_aucs(per_bc_aucs):
    """Collapse random_proj_0..N into one entry with mean and std per tier."""
    keys = [k for k in per_bc_aucs if k.startswith("random_proj_")]
    if not keys:
        return per_bc_aucs
    new = {k: v for k, v in per_bc_aucs.items() if not k.startswith("random_proj_")}
    tiers = list(per_bc_aucs[keys[0]].keys())
    agg_mean = {}
    agg_std = {}
    for t in tiers:
        vals = [per_bc_aucs[k][t] for k in keys if per_bc_aucs[k][t] is not None]
        if vals:
            agg_mean[t] = float(np.mean(vals))
            agg_std[t] = float(np.std(vals))
        else:
            agg_mean[t] = None
            agg_std[t] = None
    new["random_proj (mean over 5)"] = agg_mean
    new["random_proj (std)"] = agg_std
    return new


def analyze(archive_dir, label, out_dir):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    elites = load_elites(archive_dir)
    print(f"refined elites loaded: {len(elites)}")
    if len(elites) < 5:
        print("not enough elites for analysis"); return

    info = pairwise_table(elites, basin_threshold_frac=0.25)
    print(f"basin clustering: threshold={info['threshold']:.3f} (std-theta units), basins={info['n_basins']}")
    sizes = sorted([int((info['basin'] == b).sum()) for b in set(info['basin'])], reverse=True)
    print(f"basin sizes: {sizes}")
    print(f"pairs: {len(info['same_basin'])}, same-basin: {int(info['same_basin'].sum())}")
    print(f"refined RMS range: {info['rms_ref'].min() * 1000:.2f} - {info['rms_ref'].max() * 1000:.2f} um")

    qs = np.quantile(info['rms_ref'], np.linspace(0, 1, 5))
    print(f"RMS tier boundaries (um): {[f'{q*1000:.1f}' for q in qs]}")

    bcs = build_bcs(elites, n_random_seeds=5)

    per_bc = {}
    for bc_name, coords in bcs.items():
        d = bc_pairwise_distances(coords, info['pairs_i'], info['pairs_j'])
        per_bc[bc_name] = stratified_aucs(d, info['same_basin'], info['max_rms'], qs)

    per_bc = aggregate_random_aucs(per_bc)

    print("\n--- AUC: lower BC distance => same basin (random=0.5, perfect=1.0) ---")
    headline = ["Seidel (S2,S3)", "(c0, c1) curvatures", "random_proj (mean over 5)", "(rms_raw, |EFL_err|)"]
    cols = ["overall", "tier_1", "tier_2", "tier_3", "tier_4"]
    print(f"{'BC':<32} " + " ".join(f"{c:>10}" for c in cols))
    for bc_name in headline:
        if bc_name not in per_bc:
            continue
        row = per_bc[bc_name]
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append(f"{v:>10.3f}" if v is not None else f"{'n/a':>10}")
        std_key = "random_proj (std)" if bc_name.startswith("random_proj") else None
        suffix = ""
        if std_key and std_key in per_bc:
            std_overall = per_bc[std_key].get("overall")
            if std_overall is not None:
                suffix = f"  (std overall={std_overall:.3f})"
        print(f"{bc_name:<32} " + " ".join(cells) + suffix)

    # Plot: AUC by BC choice and tier
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar_x = np.arange(len(cols))
    width = 0.18
    plot_keys = [k for k in headline if k in per_bc]
    colors = ["C0", "C2", "C3", "C4"]
    for i, bc_name in enumerate(plot_keys):
        vals = [per_bc[bc_name].get(c) for c in cols]
        vals_plot = [v if v is not None else 0 for v in vals]
        bars = ax.bar(bar_x + (i - (len(plot_keys) - 1) / 2) * width, vals_plot, width,
                      label=bc_name, color=colors[i % len(colors)])
        for j, v in enumerate(vals):
            if v is None:
                ax.text(bar_x[j] + (i - (len(plot_keys) - 1) / 2) * width, 0.02, "n/a",
                        ha="center", va="bottom", fontsize=7, rotation=90)
    ax.axhline(0.5, ls=":", color="k", alpha=0.4, label="random (0.5)")
    ax.set_xticks(bar_x)
    rms_tier_labels = ["overall"] + [f"tier {t}\n{qs[t-1]*1000:.0f}-{qs[t]*1000:.0f} um" for t in range(1, 5)]
    ax.set_xticklabels(rms_tier_labels, fontsize=9)
    ax.set_ylabel("AUC: P(same basin | small BC distance)")
    ax.set_title(f"{label}: which 2-D BC tracks basin identity?\n"
                 f"(basin = single-linkage cluster on refined-theta, threshold={info['threshold']:.2f})")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(out_dir, f"bc_comparison_{label}.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"saved {out_png}")
    plt.close(fig)


def main():
    out_dir = "/home/hisham26/Desktop/RL/basin_locality_results"
    os.makedirs(out_dir, exist_ok=True)

    archives = [
        ("/home/hisham26/Desktop/RL/results/lensmega/cooke_triplet_materials_20260414_173935",
         "cooke_triplet"),
        ("/home/hisham26/Desktop/RL/results/lensmega/sixlet_discovery_20260421_154452",
         "sixlet"),
    ]
    for path, label in archives:
        analyze(path, label, out_dir)


if __name__ == "__main__":
    main()
