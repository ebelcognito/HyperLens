"""
Design-family gallery for the sixlet (S2, S3) run.

Loads all 39 refined elites, clusters them in standardised refined-theta space
to find distinct geometric families, and shows one lens-diagram representative
per family along with:
  - refined RMS, EFL, sign pattern
  - Seidel (S2, S3) at the refined theta
  - the elite's coordinates on the hero plane (so the user can correlate
    with the landscape figure)

Output: a multi-panel grid plus a per-elite table.
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

RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "sixlet_discovery_20260423_132001")
GRIDS_NPZ = ("/home/hisham26/Desktop/RL/paper_figures/landscape_sixlet/"
             "grids.npz")
OUT = "/home/hisham26/Desktop/RL/paper_figures/sixlet_design_gallery"
os.makedirs(OUT, exist_ok=True)

CLUSTER_FRAC = 0.25   # pick threshold = 25% of median pairwise dist
TOP_N_BASINS = 12     # show this many distinct families


def load_elites():
    paths = sorted(glob.glob(os.path.join(RUN_DIR, "elite_designs", "*.json")))
    out = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        ref = d.get("refined") or {}
        if "theta" not in ref or "rms_mm" not in ref:
            continue
        name = os.path.basename(p)[:-5]
        s2 = float(d.get("measures", [0, 0])[0])
        s3 = float(d.get("measures", [0, 0])[1])
        out.append({
            "name": name,
            "theta_refined": np.array(ref["theta"], dtype=float),
            "rms_um_refined": float(ref["rms_mm"]) * 1000.0,
            "rms_um_raw": float(d.get("rms_mm", float("nan"))) * 1000.0,
            "efl_mm": ref.get("efl_mm", "?"),
            "sign_pattern": d.get("sign_pattern", ""),
            "s2": s2, "s3": s3,
            "lens_png_refined": os.path.join(
                RUN_DIR, "elite_designs", f"{name}_refined.png"
            ),
        })
    return out


def cluster_elites(elites, frac=CLUSTER_FRAC):
    thetas = np.array([e["theta_refined"] for e in elites])
    mu = thetas.mean(axis=0); sd = thetas.std(axis=0) + 1e-9
    X = (thetas - mu) / sd
    if len(X) < 2:
        return np.zeros(len(X), dtype=int)
    from scipy.spatial.distance import pdist
    d = pdist(X)
    thresh = max(float(np.median(d) * frac), 0.5)
    Z = linkage(X, method="single", metric="euclidean")
    labels = fcluster(Z, t=thresh, criterion="distance")
    return labels, thresh


def main():
    elites = load_elites()
    print(f"loaded {len(elites)} refined elites")

    labels, thresh = cluster_elites(elites)
    n_basins = len(set(labels))
    print(f"clustering threshold: {thresh:.3f}  →  {n_basins} basins")

    # Group by cluster, pick best-RMS member of each
    by_cluster = {}
    for e, lab in zip(elites, labels):
        c = int(lab)
        e["_cluster"] = c
        if c not in by_cluster or e["rms_um_refined"] < by_cluster[c]["rms_um_refined"]:
            by_cluster[c] = e
    reps = sorted(by_cluster.values(), key=lambda e: e["rms_um_refined"])
    print(f"\nDesign families found (one rep per basin, sorted by refined RMS):")
    for r in reps:
        print(f"  basin {r['_cluster']:>3}  RMS={r['rms_um_refined']:>6.2f}μm "
              f"sign={r['sign_pattern']:<14}  S2={r['s2']:+.2e}  S3={r['s3']:+.2e}  "
              f"({r['name']})")

    # Project all elites onto the hero plane to correlate with the figure
    grids = np.load(GRIDS_NPZ)
    origin = grids["origin"]; e1 = grids["e1"]; e2 = grids["e2"]

    def proj_one(theta):
        d_ = theta - origin
        a = float(d_ @ e1); b = float(d_ @ e2)
        resid = float(np.linalg.norm(d_ - a * e1 - b * e2))
        return a, b, resid

    for r in reps:
        a, b, residual = proj_one(r["theta_refined"])
        r["plane_a"] = a; r["plane_b"] = b; r["plane_resid"] = residual

    # ===== figure 1: design-family gallery =====
    n_show = min(TOP_N_BASINS, len(reps))
    n_cols = 4
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 2.6 * n_rows))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n_show:
            r = reps[i]
            if os.path.exists(r["lens_png_refined"]):
                ax.imshow(imread(r["lens_png_refined"]))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#333"); sp.set_linewidth(0.5)
            title = (
                f"basin {r['_cluster']}  ·  RMS = {r['rms_um_refined']:.2f} μm\n"
                f"sign {r['sign_pattern']}  ·  EFL {r['efl_mm']} mm\n"
                f"plane (a, b) = ({r['plane_a']:+.2f}, {r['plane_b']:+.2f})  "
                f"resid = {r['plane_resid']:.2f}"
            )
            ax.set_title(title, fontsize=8)
        else:
            ax.axis("off")

    fig.suptitle(
        f"Sixlet design-family gallery — {n_show} distinct geometric basins from one CMA-MEGA run\n"
        "each panel = best-RMS member of a basin (refined θ-space cluster)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(OUT, "sixlet_design_family_gallery.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_png}")

    # ===== figure 2: hero plane with elite-anchor overlays =====
    AA = grids["AA"]; BB = grids["BB"]
    rms_grid = grids["rms_grid"]
    a_lo = AA.min(); a_hi = AA.max()
    b_lo = BB.min(); b_hi = BB.max()

    fig, ax = plt.subplots(figsize=(11, 9))
    rms_vis = np.log10(np.clip(rms_grid, 1e-4, 1e3))
    cf = ax.contourf(AA, BB, rms_vis, levels=30, cmap="magma_r",
                     vmin=np.nanpercentile(rms_vis, 2),
                     vmax=np.nanpercentile(rms_vis, 98))
    ax.contour(AA, BB, rms_vis, levels=10, colors="white",
               linewidths=0.4, alpha=0.35)

    # Plot ALL elites coloured by basin, sized by 1/RMS
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, n_basins))
    for r in reps:
        c = cluster_colors[(r["_cluster"] - 1) % n_basins]
        size = max(80, 600 / r["rms_um_refined"] ** 0.5)
        ax.scatter(r["plane_a"], r["plane_b"], s=size, c=[c],
                   edgecolor="black", linewidths=1.2, zorder=10)
        ax.annotate(
            f"b{r['_cluster']}·{r['rms_um_refined']:.1f}μm",
            (r["plane_a"], r["plane_b"]),
            fontsize=8, color="white", ha="center",
            xytext=(0, 10), textcoords="offset points",
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.7, pad=1.5, boxstyle="round"),
        )

    ax.set_xlabel("plane coord a")
    ax.set_ylabel("plane coord b")
    ax.set_xlim(a_lo, a_hi); ax.set_ylim(b_lo, b_hi)
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, shrink=0.8,
                 label="log₁₀(RMS / mm) · yellow = better")
    fig.suptitle(
        f"All distinct design families projected on the hero plane\n"
        f"({n_show} basins · sizes scale with 1/√RMS)",
        fontsize=12, fontweight="bold",
    )
    out_png2 = os.path.join(OUT, "sixlet_basin_anchors_on_plane.png")
    fig.savefig(out_png2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png2}")

    # ===== summary JSON =====
    summary = {
        "n_elites_total": len(elites),
        "n_basins": n_basins,
        "cluster_threshold_std_units": float(thresh),
        "basins": [
            {
                "id": int(r["_cluster"]),
                "rep_name": r["name"],
                "rms_um_refined": r["rms_um_refined"],
                "rms_um_raw": r["rms_um_raw"],
                "efl_mm": r["efl_mm"],
                "sign_pattern": r["sign_pattern"],
                "S2": r["s2"], "S3": r["s3"],
                "plane_a": r["plane_a"],
                "plane_b": r["plane_b"],
                "plane_residual": r["plane_resid"],
            } for r in reps
        ],
    }
    with open(os.path.join(OUT, "design_family_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {os.path.join(OUT, 'design_family_summary.json')}")


if __name__ == "__main__":
    main()
