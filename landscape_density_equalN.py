"""
Fair N comparison:
  * random seeds   : 3,000 samples (the real count from the run)
  * CMA-MEGA       : random subsample of 3,000 offspring (matched N)
  * refined elites : all of them (individual markers), projected onto plane

Everything else (plane, landscape, anchors) identical to landscape_density.
"""
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KNOWN_JSON = (
    "/home/hisham26/Desktop/RL/paper_figures/"
    "sample_efficiency_subregion/known_minima.json"
)
RUN_DIR = (
    "/home/hisham26/Desktop/RL/results/lensmega/"
    "cooke_triplet_seidel3d_s2s3s4_20260423_031324"
)
GRIDS_NPZ = (
    "/home/hisham26/Desktop/RL/paper_figures/landscape_plane3/"
    "landscape_plane3_grids.npz"
)
OUT = "/home/hisham26/Desktop/RL/paper_figures/landscape_plane3"
os.makedirs(OUT, exist_ok=True)

MIN_MEMBERS = 2
MAX_RMS_UM = 20.0
ANCHOR_BASIN_IDS = [7, 10, 9]
RNG_SEED = 0
TARGET_N = 3000


def load_minima():
    with open(KNOWN_JSON) as f:
        d = json.load(f)
    all_b = [b for b in d["basins"]
             if b["n_members"] >= MIN_MEMBERS
             and b["best_rms_um"] <= MAX_RMS_UM]
    anchors = [b for b in all_b if b["basin_id"] in ANCHOR_BASIN_IDS]
    return anchors


def load_lineage_split():
    seeds, offs = [], []
    with open(os.path.join(RUN_DIR, "lineage.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            th = r.get("theta")
            if th is None:
                continue
            (seeds if r["source"] == "seed" else offs).append(th)
    return np.array(seeds), np.array(offs)


def load_refined_elites():
    paths = sorted(glob.glob(os.path.join(RUN_DIR, "elite_designs", "*.json")))
    ref_thetas, ref_rms = [], []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        ref = d.get("refined") or {}
        if "theta" not in ref or "rms_mm" not in ref:
            continue
        ref_thetas.append(ref["theta"])
        ref_rms.append(float(ref["rms_mm"]) * 1000.0)
    return np.array(ref_thetas), np.array(ref_rms)


def project(theta_arr, origin, e1, e2):
    d = theta_arr - origin
    a = d @ e1
    b = d @ e2
    resid = np.linalg.norm(d - a[:, None] * e1 - b[:, None] * e2, axis=1)
    return a, b, resid


def main():
    anchors = load_minima()

    grids = np.load(GRIDS_NPZ)
    AA = grids["AA"]; BB = grids["BB"]
    rms_grid = grids["rms_grid"]
    origin = grids["origin"]; e1 = grids["e1"]; e2 = grids["e2"]
    anchor_xy = grids["anchor_coords"]
    a_lo = AA.min(); a_hi = AA.max()
    b_lo = BB.min(); b_hi = BB.max()

    seeds, offs = load_lineage_split()
    ref_thetas, ref_rms = load_refined_elites()
    print(f"lineage: {len(seeds):,} seeds · {len(offs):,} offspring")
    print(f"refined elites loaded: {len(ref_thetas)}  RMS range {ref_rms.min():.1f}–{ref_rms.max():.1f} μm")

    # Equal-N sub-sample
    rng = np.random.default_rng(RNG_SEED)
    seeds_N = seeds[rng.choice(len(seeds), TARGET_N, replace=False)]
    offs_N = offs[rng.choice(len(offs), TARGET_N, replace=False)]
    print(f"equal-N comparison at N={TARGET_N} per phase")

    # Project
    sa, sb, sr = project(seeds_N, origin, e1, e2)
    oa, ob, or_ = project(offs_N, origin, e1, e2)
    ra, rb, rr = project(ref_thetas, origin, e1, e2) if len(ref_thetas) else (np.array([]),)*3

    # Binning
    nbin = 50
    a_edges = np.linspace(a_lo, a_hi, nbin + 1)
    b_edges = np.linspace(b_lo, b_hi, nbin + 1)
    H_seed, _, _ = np.histogram2d(sa, sb, bins=[a_edges, b_edges])
    H_off, _, _ = np.histogram2d(oa, ob, bins=[a_edges, b_edges])

    # Normalise (same N, but be robust)
    def norm(H):
        s = H.sum(); return H / s if s > 0 else H
    n_seed = norm(H_seed); n_off = norm(H_off)
    prior = n_seed[n_seed > 0].min() * 0.5 if (n_seed > 0).any() else 1e-6
    log_ratio = np.log10((n_off + prior) / (n_seed + prior))

    a_mid = 0.5 * (a_edges[:-1] + a_edges[1:])
    b_mid = 0.5 * (b_edges[:-1] + b_edges[1:])
    AA_bin, BB_bin = np.meshgrid(a_mid, b_mid, indexing="ij")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(22, 7),
                             gridspec_kw=dict(wspace=0.22))

    rms_vis = np.log10(np.clip(rms_grid, 1e-4, 1e3))
    vmin = np.nanpercentile(rms_vis, 2); vmax = np.nanpercentile(rms_vis, 98)

    # Panel A: random seeds (N=TARGET_N)
    ax = axes[0]
    ax.contourf(AA, BB, rms_vis, levels=30, cmap="magma_r",
                vmin=vmin, vmax=vmax, alpha=0.55)
    ax.pcolormesh(AA_bin, BB_bin, H_seed, cmap="Oranges",
                  shading="nearest", alpha=0.78)
    for (a, b), _ in zip(anchor_xy, anchors):
        ax.scatter(a, b, s=420, c="red", marker="*",
                   edgecolor="black", linewidths=1.6, zorder=20)
    ax.set_title(f"A. Random seeds · N = {TARGET_N:,}\n(matched sample count)",
                 fontsize=11)
    ax.set_xlabel("plane coord a"); ax.set_ylabel("plane coord b")
    ax.set_xlim(a_lo, a_hi); ax.set_ylim(b_lo, b_hi); ax.set_aspect("equal")

    # Panel B: CMA-MEGA offspring subsample (N=TARGET_N)
    ax = axes[1]
    ax.contourf(AA, BB, rms_vis, levels=30, cmap="magma_r",
                vmin=vmin, vmax=vmax, alpha=0.55)
    ax.pcolormesh(AA_bin, BB_bin, H_off, cmap="Greens",
                  shading="nearest", alpha=0.78)
    for (a, b), _ in zip(anchor_xy, anchors):
        ax.scatter(a, b, s=420, c="red", marker="*",
                   edgecolor="black", linewidths=1.6, zorder=20)
    ax.set_title(f"B. CMA-MEGA offspring · N = {TARGET_N:,}\n(random subsample of 50k)",
                 fontsize=11)
    ax.set_xlabel("plane coord a"); ax.set_ylabel("plane coord b")
    ax.set_xlim(a_lo, a_hi); ax.set_ylim(b_lo, b_hi); ax.set_aspect("equal")

    # Panel C: density RATIO (now at equal N — no count bias)
    ax = axes[2]
    ax.contourf(AA, BB, rms_vis, levels=30, cmap="magma_r",
                vmin=vmin, vmax=vmax, alpha=0.35)
    pc = ax.pcolormesh(AA_bin, BB_bin, log_ratio,
                       cmap="PuOr_r", shading="nearest",
                       vmin=-1.0, vmax=1.0, alpha=0.85)
    # Refined elites projected onto plane (annotate residual visually via alpha)
    if len(ref_thetas):
        alpha_ref = np.clip(1.0 - rr / (rr.max() + 1e-9) * 0.6, 0.35, 1.0)
        for i in range(len(ref_thetas)):
            ax.scatter(ra[i], rb[i], s=120, c="#14d140", marker="o",
                       edgecolor="black", linewidths=0.9, zorder=25,
                       alpha=float(alpha_ref[i]))
        ax.scatter([], [], s=120, c="#14d140", marker="o", edgecolor="black",
                   linewidths=0.9, label=f"refined elites (n={len(ref_thetas)})")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
    for (a, b), _ in zip(anchor_xy, anchors):
        ax.scatter(a, b, s=420, c="red", marker="*",
                   edgecolor="black", linewidths=1.6, zorder=20)
    cb = plt.colorbar(pc, ax=ax, shrink=0.9)
    cb.set_label("log₁₀(offspring / seed) at equal N")
    ax.set_title("C. Density RATIO at equal N\n"
                 "orange = CMA-MEGA preferred · refined elites overlaid",
                 fontsize=11)
    ax.set_xlabel("plane coord a"); ax.set_ylabel("plane coord b")
    ax.set_xlim(a_lo, a_hi); ax.set_ylim(b_lo, b_hi); ax.set_aspect("equal")

    fig.suptitle(
        f"Matched-N comparison — {TARGET_N:,} samples per method · "
        f"refined elites projected onto plane",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(OUT, "landscape_density_equalN.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")

    # Also print: residuals of refined elites (how close they sit to the plane)
    if len(ref_thetas):
        print("refined-elite projection residuals to plane (std-θ-ish units):")
        for i in range(len(ref_thetas)):
            print(f"  elite {i}: RMS={ref_rms[i]:.2f} μm · (a,b)=({ra[i]:.2f},{rb[i]:.2f}) "
                  f"· resid={rr[i]:.2f}")


if __name__ == "__main__":
    main()
