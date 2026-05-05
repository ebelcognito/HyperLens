"""
Quantify the quality of the samples shown in the sixlet landscape figure.

Uses the exact same filtering (matched N, closest-50% to plane) as the
landscape figure, then reports RMS / fitness statistics with several
metrics including ones that weight smaller spots more heavily.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "sixlet_discovery_20260423_132001")
GRIDS_NPZ = ("/home/hisham26/Desktop/RL/paper_figures/landscape_sixlet/"
             "grids.npz")
OUT = "/home/hisham26/Desktop/RL/paper_figures/landscape_sixlet"

TARGET_N = 1000
RESID_KEEP_FRAC = 0.50


def load_lineage_full():
    """Read every emission with full theta + fitness + rms_spot."""
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


def main():
    grids = np.load(GRIDS_NPZ)
    origin = grids["origin"]; e1 = grids["e1"]; e2 = grids["e2"]
    anchor_xy = grids["anchor_xy"]

    seeds, offs = load_lineage_full()
    print(f"lineage: {len(seeds):,} seeds · {len(offs):,} offspring\n")

    rng = np.random.default_rng(0)
    seed_idx = rng.choice(len(seeds), min(TARGET_N, len(seeds)), replace=False)
    off_idx = rng.choice(len(offs), min(TARGET_N, len(offs)), replace=False)

    def pack(rows, idxs):
        th = np.array([rows[i]["theta"] for i in idxs])
        fit = np.array([rows[i]["fitness"] for i in idxs])
        rms = np.array([rows[i]["rms_spot"] for i in idxs])
        inv = np.array([rows[i]["invalid_fraction"] for i in idxs])
        return th, fit, rms, inv
    s_th, s_fit, s_rms, s_inv = pack(seeds, seed_idx)
    o_th, o_fit, o_rms, o_inv = pack(offs, off_idx)

    def proj(th):
        d = th - origin
        a = d @ e1; b = d @ e2
        resid = np.linalg.norm(d - a[:, None] * e1 - b[:, None] * e2, axis=1)
        return a, b, resid
    sa, sb, sr = proj(s_th)
    oa, ob, or_ = proj(o_th)

    # Same in-plane filter as figure
    s_mask = sr < np.percentile(sr, RESID_KEEP_FRAC * 100)
    o_mask = or_ < np.percentile(or_, RESID_KEEP_FRAC * 100)
    print(f"after in-plane filter (closest {RESID_KEEP_FRAC*100:.0f}% each):")
    print(f"  seeds:      {s_mask.sum():,}")
    print(f"  offspring:  {o_mask.sum():,}")

    # RMS of filtered displayed samples
    s_rms_shown = s_rms[s_mask]
    o_rms_shown = o_rms[o_mask]
    s_fit_shown = s_fit[s_mask]
    o_fit_shown = o_fit[o_mask]

    # Clean: drop non-finite
    def clean(a):
        return a[np.isfinite(a) & (a > 0)]
    s_rms_clean = clean(s_rms_shown)
    o_rms_clean = clean(o_rms_shown)

    def weighted_mean(rms, power):
        """Small-spot-weighted mean. power<0 emphasises small RMS."""
        w = np.power(rms, power)
        return np.sum(w * rms) / np.sum(w)

    def report(label, rms):
        print(f"\n=== {label} ===  (n = {len(rms)})")
        print(f"  simple metrics:")
        print(f"    mean RMS        : {rms.mean() * 1000:>10.2f} μm")
        print(f"    median RMS      : {np.median(rms) * 1000:>10.2f} μm")
        print(f"    min RMS         : {rms.min() * 1000:>10.2f} μm")
        print(f"    max RMS         : {rms.max() * 1000:>10.2f} μm")
        print(f"  weighted-toward-small metrics:")
        print(f"    geometric mean  : {np.exp(np.log(rms).mean()) * 1000:>10.2f} μm")
        print(f"    harmonic mean   : {len(rms) / np.sum(1.0 / rms) * 1000:>10.2f} μm")
        print(f"    mean of top-25%%: {np.sort(rms)[:len(rms)//4].mean() * 1000:>10.2f} μm")
        print(f"    mean of top-10%%: {np.sort(rms)[:len(rms)//10].mean() * 1000:>10.2f} μm")
        print(f"    mean of top-5%% : {np.sort(rms)[:max(1, len(rms)//20)].mean() * 1000:>10.2f} μm")
        print(f"  fraction below threshold:")
        for thr in [1.0, 0.1, 0.05, 0.02, 0.01]:
            print(f"    RMS < {thr:5.2f} mm : {(rms < thr).mean() * 100:>6.2f} %")

    report("random seeds (shown)", s_rms_clean)
    report("CMA-MEGA priors (shown)", o_rms_clean)

    # Direct ratio table
    print("\n=== side-by-side ===")
    print(f"{'metric':<30} {'seed (μm)':>12} {'CMA-MEGA (μm)':>14} {'ratio s/o':>10}")
    seed_stats = {
        "mean":    s_rms_clean.mean(),
        "median":  np.median(s_rms_clean),
        "geo_mean": np.exp(np.log(s_rms_clean).mean()),
        "harmonic": len(s_rms_clean) / np.sum(1.0 / s_rms_clean),
        "top-10% mean": np.sort(s_rms_clean)[:len(s_rms_clean)//10].mean(),
    }
    off_stats = {
        "mean":    o_rms_clean.mean(),
        "median":  np.median(o_rms_clean),
        "geo_mean": np.exp(np.log(o_rms_clean).mean()),
        "harmonic": len(o_rms_clean) / np.sum(1.0 / o_rms_clean),
        "top-10% mean": np.sort(o_rms_clean)[:len(o_rms_clean)//10].mean(),
    }
    for k in seed_stats:
        s_val = seed_stats[k] * 1000
        o_val = off_stats[k] * 1000
        ratio = s_val / max(o_val, 1e-9)
        print(f"{k:<30} {s_val:>12.2f} {o_val:>14.2f} {ratio:>10.2f}×")

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(wspace=0.3))

    # Panel A: RMS histogram log-scale
    ax = axes[0]
    bins = np.logspace(-3, np.log10(max(s_rms_clean.max(), o_rms_clean.max())), 50)
    ax.hist(s_rms_clean, bins=bins, alpha=0.55, color="C1",
            label=f"random seeds (n={len(s_rms_clean)})  median {np.median(s_rms_clean)*1000:.1f} μm",
            density=True)
    ax.hist(o_rms_clean, bins=bins, alpha=0.55, color="C2",
            label=f"CMA-MEGA priors (n={len(o_rms_clean)})  median {np.median(o_rms_clean)*1000:.1f} μm",
            density=True)
    ax.axvline(np.median(s_rms_clean), color="C1", ls="--", alpha=0.7)
    ax.axvline(np.median(o_rms_clean), color="C2", ls="--", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("spot RMS (mm)")
    ax.set_ylabel("density")
    ax.set_title("A. Spot-RMS distribution of displayed samples\n"
                 f"(closest {int(RESID_KEEP_FRAC*100)}% to plane, matched N)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Panel B: CDF
    ax = axes[1]
    def cdf(a):
        x = np.sort(a); y = np.arange(1, len(x) + 1) / len(x)
        return x, y
    xs, ys = cdf(s_rms_clean); xo, yo = cdf(o_rms_clean)
    ax.plot(xs, ys * 100, color="C1", lw=2, label="random seeds")
    ax.plot(xo, yo * 100, color="C2", lw=2, label="CMA-MEGA priors")
    ax.set_xscale("log")
    ax.set_xlabel("spot RMS (mm)")
    ax.set_ylabel("% of samples ≤ RMS")
    ax.set_title("B. CDF of displayed samples\n"
                 "(left = better; higher CDF at low RMS = method wins)",
                 fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Sample quality of dots shown in the sixlet landscape figure\n"
        f"N = {len(s_rms_clean)} seeds vs {len(o_rms_clean)} CMA-MEGA priors (matched, in-plane)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = os.path.join(OUT, "sample_quality.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
