"""
Audit the residual-filter bias in the sixlet landscape figure.

Questions:
  1. What are the residual distributions of seeds vs CMA-MEGA offspring?
  2. Are CMA-MEGA samples systematically closer to the plane?
  3. If we change the residual filter threshold, does the concentration
     story hold?
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
MIN_RMS_MM = 1e-3
MAX_INV_FRAC = 0.2


def load_lineage():
    seeds, offs = [], []
    with open(os.path.join(RUN_DIR, "lineage.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            th = r.get("theta")
            if th is None:
                continue
            row = {
                "theta": np.asarray(th, dtype=float),
                "rms_spot": float(r.get("rms_spot", np.nan)),
                "invalid_fraction": float(r.get("invalid_fraction", 1.0)),
            }
            (seeds if r["source"] == "seed" else offs).append(row)
    return seeds, offs


def main():
    grids = np.load(GRIDS_NPZ)
    origin = grids["origin"]; e1 = grids["e1"]; e2 = grids["e2"]
    anchor_xy = grids["anchor_xy"]

    seeds, offs = load_lineage()
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

    def resid(th):
        d = th - origin
        a = d @ e1; b = d @ e2
        return np.linalg.norm(d - a[:, None] * e1 - b[:, None] * e2, axis=1)
    s_r = resid(s_th); o_r = resid(o_th)

    print("=== Residual distribution (distance from the plane) ===")
    print(f"{'phase':<12} {'n':>6} {'min':>8} {'p10':>8} {'median':>8} "
          f"{'mean':>8} {'p90':>8} {'max':>8}")
    for label, r in [("seeds", s_r), ("offspring", o_r)]:
        print(f"{label:<12} {len(r):>6} {r.min():>8.2f} "
              f"{np.percentile(r, 10):>8.2f} {np.median(r):>8.2f} "
              f"{r.mean():>8.2f} {np.percentile(r, 90):>8.2f} "
              f"{r.max():>8.2f}")
    ratio = o_r.mean() / s_r.mean()
    print(f"\nmean(offspring residual) / mean(seed residual) = {ratio:.3f}")
    print("(< 1.0 means CMA-MEGA samples live CLOSER to the plane on average)")

    # Fitness-stratified — maybe only LOW-RMS samples cluster near the plane
    legit_s = (s_rms > MIN_RMS_MM) & (s_inv < MAX_INV_FRAC) & np.isfinite(s_rms)
    legit_o = (o_rms > MIN_RMS_MM) & (o_inv < MAX_INV_FRAC) & np.isfinite(o_rms)

    print("\n=== Residuals stratified by RMS quality ===")
    print(f"{'phase':<20} {'RMS range':<20} {'n':>5} {'mean residual':>14}")
    for label, rms, r, mask in [
        ("seeds (all)",     s_rms, s_r, legit_s),
        ("offspring (all)", o_rms, o_r, legit_o),
    ]:
        for lo, hi in [(0, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, 1e9)]:
            sel = mask & (rms >= lo) & (rms < hi)
            if sel.sum() > 0:
                print(f"{label:<20} [{lo:>6.2f}, {hi:>6.1f}] mm  "
                      f"{int(sel.sum()):>5} {r[sel].mean():>14.2f}")

    # ==========
    # How the concentration story changes with residual filter choice
    # ==========
    print("\n=== Good-prior counts vs residual filter choice ===")
    print("How many LEGIT samples below 1 mm RMS are kept, per phase, "
          "at various residual cutoffs?")
    print(f"{'resid ≤':<12} {'seeds (n, <1mm)':>20} {'offspring (n, <1mm)':>22} "
          f"{'ratio':>8}")
    for cutoff_frac in [0.20, 0.50, 0.80, 1.00]:   # 1.00 = no filter
        if cutoff_frac < 1.0:
            s_thr = np.percentile(s_r, cutoff_frac * 100)
            o_thr = np.percentile(o_r, cutoff_frac * 100)
            s_keep = s_r <= s_thr; o_keep = o_r <= o_thr
            label = f"{int(cutoff_frac*100)}%"
        else:
            s_keep = np.ones_like(s_r, dtype=bool)
            o_keep = np.ones_like(o_r, dtype=bool)
            label = "no filter"

        n_s = int((s_keep & legit_s & (s_rms < 1.0)).sum())
        n_o = int((o_keep & legit_o & (o_rms < 1.0)).sum())
        ratio_str = f"{n_o/max(n_s, 1):.1f}×" if n_s > 0 else f"∞ ({n_o})"
        print(f"{label:<12} {n_s:>20} {n_o:>22} {ratio_str:>8}")

    # ==========
    # Histogram of residuals per phase
    # ==========
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(wspace=0.3))

    ax = axes[0]
    bins = np.linspace(0, max(s_r.max(), o_r.max()), 40)
    ax.hist(s_r, bins=bins, alpha=0.55, color="C1",
            label=f"seeds (mean {s_r.mean():.2f}, median {np.median(s_r):.2f})",
            density=True)
    ax.hist(o_r, bins=bins, alpha=0.55, color="C2",
            label=f"offspring (mean {o_r.mean():.2f}, median {np.median(o_r):.2f})",
            density=True)
    ax.axvline(s_r.mean(), color="C1", ls="--")
    ax.axvline(o_r.mean(), color="C2", ls="--")
    ax.set_xlabel("residual θ-distance from the plane")
    ax.set_ylabel("density")
    ax.set_title(f"A. Residual distribution (N={TARGET_N} per phase)\n"
                 f"are CMA-MEGA samples systematically closer to the plane?",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: good-prior count vs residual cutoff (the robustness plot)
    ax = axes[1]
    cutoffs = np.linspace(0.05, 1.00, 40)
    s_counts = []; o_counts = []
    for c in cutoffs:
        if c < 1.0:
            s_thr = np.percentile(s_r, c * 100)
            o_thr = np.percentile(o_r, c * 100)
            s_keep = s_r <= s_thr; o_keep = o_r <= o_thr
        else:
            s_keep = np.ones_like(s_r, dtype=bool)
            o_keep = np.ones_like(o_r, dtype=bool)
        s_counts.append(int((s_keep & legit_s & (s_rms < 1.0)).sum()))
        o_counts.append(int((o_keep & legit_o & (o_rms < 1.0)).sum()))
    ax.plot(cutoffs * 100, s_counts, "o-", color="C1", label="seeds, <1 mm RMS")
    ax.plot(cutoffs * 100, o_counts, "o-", color="C2",
            label="CMA-MEGA offspring, <1 mm RMS")
    ax.axvline(50, color="gray", ls=":", alpha=0.8, label="current filter (50%)")
    ax.set_xlabel("residual-percentile cutoff (% kept)")
    ax.set_ylabel("# legit samples below 1 mm RMS")
    ax.set_title("B. Is the '15× more good priors' claim robust\n"
                 "to the residual-filter choice?",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Residual audit — does the in-plane filter bias the sixlet landscape comparison?",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = os.path.join(OUT, "residual_audit.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
