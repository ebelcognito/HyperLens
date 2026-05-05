"""
v2: filter to reliable minima, use only curvature axes for PCA,
show top-fitness CMA-MEGA samples.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KNOWN_JSON = "/home/hisham26/Desktop/RL/paper_figures/sample_efficiency_subregion/known_minima.json"
LINEAGE = (
    "/home/hisham26/Desktop/RL/results/lensmega/"
    "cooke_triplet_seidel3d_s2s3s4_20260423_031324/lineage.jsonl"
)
OUT_DIR = "/home/hisham26/Desktop/RL/paper_figures/sample_efficiency_subregion"
os.makedirs(OUT_DIR, exist_ok=True)

# Only keep curvature dims (first 6 of 19 for Cooke full_dof theta)
CURV_IDX = list(range(6))

# Option C + multi-member: curated minima with >=2 members at decent RMS.
# With MIN_MEMBERS=2 and RMS<=20, we get 5 well-spread, well-supported basins.
MIN_MEMBERS = 2
MAX_RMS_UM = 20.0

# Fraction of offspring by fitness to keep
TOP_FITNESS_FRAC = 0.05


def load_reliable_minima():
    with open(KNOWN_JSON) as f:
        d = json.load(f)
    basins = [b for b in d["basins"]
              if b["n_members"] >= MIN_MEMBERS
              and b["best_rms_um"] <= MAX_RMS_UM]
    centroids = np.array([b["centroid_theta"] for b in basins], dtype=float)
    rms = np.array([b["best_rms_um"] for b in basins])
    n_members = np.array([b["n_members"] for b in basins])
    labels = [f"b{b['basin_id']}\n{b['best_rms_um']:.1f}μm ({b['n_members']})"
              for b in basins]
    return centroids, rms, n_members, labels


def load_lineage_thetas(path):
    seeds, offs = [], []
    seed_fits, off_fits = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            th = r.get("theta")
            if th is None:
                continue
            if r["source"] == "seed":
                seeds.append(th); seed_fits.append(float(r.get("fitness", np.nan)))
            else:
                offs.append(th); off_fits.append(float(r.get("fitness", np.nan)))
    return (np.array(seeds), np.array(offs),
            np.array(seed_fits), np.array(off_fits))


def nearest_dist(points, centroids):
    # Euclidean in whatever scaling the caller passed in
    N = len(points); out = np.empty(N)
    chunk = 2000
    for i in range(0, N, chunk):
        sub = points[i:i + chunk]
        diff = sub[np.newaxis, :, :] - centroids[:, np.newaxis, :]
        d = np.linalg.norm(diff, axis=2)
        out[i:i + chunk] = d.min(axis=0)
    return out


def main():
    cents_full, rms, n_members, labels = load_reliable_minima()
    print(f"reliable minima: {len(cents_full)} (>= {MIN_MEMBERS} members, "
          f"RMS <= {MAX_RMS_UM} μm)")
    for lab, r in zip(labels, rms):
        print(f"  {lab}")

    seeds, offs, seed_fits, off_fits = load_lineage_thetas(LINEAGE)
    print(f"loaded {len(seeds):,} seed samples, {len(offs):,} offspring samples")

    # Restrict to curvature dims
    cents = cents_full[:, CURV_IDX]
    seeds_c = seeds[:, CURV_IDX]
    offs_c = offs[:, CURV_IDX]

    # Standardise by std of curvature values ACROSS all samples + centroids
    # (this is a better scale than centroid-only std for the 6-D curvature space)
    all_curv = np.vstack([cents, seeds_c, offs_c])
    mu = all_curv.mean(axis=0); sd = all_curv.std(axis=0) + 1e-9
    cents_s = (cents - mu) / sd
    seeds_s = (seeds_c - mu) / sd
    offs_s = (offs_c - mu) / sd

    # PCA on centroids
    c_mean = cents_s.mean(axis=0)
    X = cents_s - c_mean
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    basis = Vt[:2]
    explained = (S ** 2) / (S ** 2).sum()
    print(f"PCA of centroids in curvature-only space: "
          f"top-2 explains {explained[:2].sum() * 100:.1f}% of variance")

    def proj(P):
        return (P - c_mean) @ basis.T
    p_cents = proj(cents_s)
    p_seeds = proj(seeds_s)
    p_offs = proj(offs_s)

    # Nearest-basin distances in 6-D standardised curvature space
    d_seed = nearest_dist(seeds_s, cents_s)
    d_off = nearest_dist(offs_s, cents_s)
    print(f"mean nearest-basin dist (curvature-only, std units):")
    print(f"  seed   : {d_seed.mean():.3f}  median={np.median(d_seed):.3f}")
    print(f"  offspr : {d_off.mean():.3f}  median={np.median(d_off):.3f}")

    # Top-fitness offspring
    valid = np.isfinite(off_fits)
    q = np.quantile(off_fits[valid], 1.0 - TOP_FITNESS_FRAC)
    top_mask = valid & (off_fits >= q)
    d_off_top = d_off[top_mask]
    p_offs_top = p_offs[top_mask]
    print(f"top-{TOP_FITNESS_FRAC*100:.0f}% offspring (n={top_mask.sum():,}): "
          f"mean dist = {d_off_top.mean():.3f}  median = {np.median(d_off_top):.3f}")

    # --- figure ---
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.30,
                          left=0.06, right=0.98, top=0.92, bottom=0.08)

    lim = max(abs(p_cents).max() * 1.8,
              np.percentile(np.abs(np.concatenate([p_seeds, p_offs])), 95))

    # A: PCA, seed density
    ax = fig.add_subplot(gs[0, 0])
    hb = ax.hexbin(p_seeds[:, 0], p_seeds[:, 1], gridsize=60,
                   cmap="Oranges", mincnt=1, extent=(-lim, lim, -lim, lim))
    ax.scatter(p_cents[:, 0], p_cents[:, 1], s=280, c="red", marker="*",
               edgecolor="black", linewidths=0.8, zorder=5,
               label=f"known minima ({len(p_cents)})")
    for lab, pc in zip(labels, p_cents):
        ax.annotate(lab.split("\n")[1].split(" ")[0], pc,
                    fontsize=7, ha="left", va="bottom", color="darkred")
    ax.set_title(f"A. Random seeds (N={len(seeds):,})\n"
                 "density in curvature-PCA")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.legend(loc="upper right", fontsize=9)
    plt.colorbar(hb, ax=ax, label="# seeds")

    # B: PCA, all offspring density
    ax = fig.add_subplot(gs[0, 1])
    hb = ax.hexbin(p_offs[:, 0], p_offs[:, 1], gridsize=60,
                   cmap="Blues", mincnt=1, extent=(-lim, lim, -lim, lim))
    ax.scatter(p_cents[:, 0], p_cents[:, 1], s=280, c="red", marker="*",
               edgecolor="black", linewidths=0.8, zorder=5,
               label="known minima")
    ax.set_title(f"B. CMA-MEGA offspring (N={len(offs):,})\n"
                 "density in curvature-PCA")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.legend(loc="upper right", fontsize=9)
    plt.colorbar(hb, ax=ax, label="# offspring")

    # C: PCA, top-fitness offspring density
    ax = fig.add_subplot(gs[0, 2])
    hb = ax.hexbin(p_offs_top[:, 0], p_offs_top[:, 1], gridsize=60,
                   cmap="Greens", mincnt=1, extent=(-lim, lim, -lim, lim))
    ax.scatter(p_cents[:, 0], p_cents[:, 1], s=320, c="red", marker="*",
               edgecolor="black", linewidths=0.8, zorder=5,
               label="known minima")
    ax.set_title(f"C. CMA-MEGA top {TOP_FITNESS_FRAC*100:.0f}% fitness "
                 f"(N={top_mask.sum():,})\n"
                 "→ best priors should sit on the minima")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.legend(loc="upper right", fontsize=9)
    plt.colorbar(hb, ax=ax, label="# top-fit offspring")

    # D: Distance histograms (curvature-space)
    ax = fig.add_subplot(gs[1, 0])
    hi = max(d_seed.max(), d_off.max())
    bins = np.linspace(0, hi, 70)
    ax.hist(d_seed, bins=bins, alpha=0.55, color="C1",
            label=f"seed mean {d_seed.mean():.2f}", density=True)
    ax.hist(d_off, bins=bins, alpha=0.55, color="C0",
            label=f"offspring mean {d_off.mean():.2f}", density=True)
    ax.hist(d_off_top, bins=bins, alpha=0.7, color="C2",
            label=f"top-{TOP_FITNESS_FRAC*100:.0f}% off mean {d_off_top.mean():.2f}",
            density=True)
    ax.axvline(d_seed.mean(), color="C1", ls="--", alpha=0.6)
    ax.axvline(d_off.mean(), color="C0", ls="--", alpha=0.6)
    ax.axvline(d_off_top.mean(), color="C2", ls="--", alpha=0.6)
    ax.set_xlabel("distance to nearest reliable minimum (std curv units)")
    ax.set_ylabel("density")
    ax.set_title("D. Distance to nearest minimum")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # E: Survival
    ax = fig.add_subplot(gs[1, 1])
    rr = np.linspace(0, hi, 200)
    cum_s = np.array([(d_seed <= r).mean() for r in rr]) * 100
    cum_o = np.array([(d_off <= r).mean() for r in rr]) * 100
    cum_t = np.array([(d_off_top <= r).mean() for r in rr]) * 100
    ax.plot(rr, cum_s, color="C1", lw=2, label="seed")
    ax.plot(rr, cum_o, color="C0", lw=2, label="offspring (all)")
    ax.plot(rr, cum_t, color="C2", lw=2,
            label=f"offspring top {TOP_FITNESS_FRAC*100:.0f}%")
    ax.set_xlabel("radius r")
    ax.set_ylabel("% of samples within r")
    ax.set_title("E. Cumulative containment\n(how tight samples sit to minima)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # F: Headline + ratios
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ratio_all = d_seed.mean() / max(d_off.mean(), 1e-9)
    ratio_top = d_seed.mean() / max(d_off_top.mean(), 1e-9)

    lines = [
        "HEADLINE (curvature-only, 6-D standardised)",
        f"",
        f"  reliable minima (≥{MIN_MEMBERS} members, RMS≤{MAX_RMS_UM}μm): {len(cents_full)}",
        f"",
        f"  mean dist to nearest minimum:",
        f"    random seeds       : {d_seed.mean():.3f}",
        f"    CMA-MEGA offspring : {d_off.mean():.3f}   (×{ratio_all:.2f})",
        f"    top {TOP_FITNESS_FRAC*100:.0f}% offspring  : {d_off_top.mean():.3f}   (×{ratio_top:.2f})",
        f"",
        f"  % within radius r:",
        f"    {'r':>5}  {'seed':>8}  {'off(all)':>9}  {'off(top)':>9}",
    ]
    for r in [0.5, 1.0, 1.5, 2.0, 3.0]:
        fs = (d_seed <= r).mean() * 100
        fo = (d_off <= r).mean() * 100
        ft = (d_off_top <= r).mean() * 100
        lines.append(f"    {r:>5.1f}  {fs:>6.2f}%  {fo:>7.2f}%  {ft:>7.2f}%")
    ax.text(0.0, 0.95, "\n".join(lines), family="monospace", fontsize=10,
            va="top", transform=ax.transAxes)

    fig.suptitle(
        "Where do CMA-MEGA+Seidel priors land? (Cooke, curvatures only)\n"
        "reliable minima from prior 2-D Cooke archive vs raw emissions",
        fontsize=13, fontweight="bold",
    )
    out_png = os.path.join(OUT_DIR, "cma_mega_samples_v2.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
