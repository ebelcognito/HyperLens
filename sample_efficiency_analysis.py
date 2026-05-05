"""
Sample-efficiency analysis of CMA-MEGA+Seidel on a single run.

For a given run directory, read lineage.jsonl (every evaluation logged) and
answer: where do the CMA-MEGA mutations actually land? Do they concentrate
in high-fitness regions, or waste compute on sub-optimal θ?

The seed phase (3,000 random θ draws) provides a built-in "pure random
parameter-space sampling" control — conceptually equivalent to what a
GA with no prior would explore before any selection pressure applies.

Panels:
  A. Fitness histogram: seed vs offspring (log-scale count)
  B. Cumulative fraction of samples above fitness threshold (survival-style)
  C. 2-D sample density in (S2, S3): seed vs offspring
  D. 2-D sample density in (S3, S4): seed vs offspring
  E. Per-iteration mean and max fitness (how fast the search climbs)
  F. Headline numbers printed: mean fitness per phase; fraction below
     various thresholds per phase; mean offspring fitness vs mean seed
     fitness.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_lineage(run_dir):
    path = os.path.join(run_dir, "lineage.jsonl")
    seeds, offs = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            row = {
                "fitness": float(r["fitness"]),
                "rms_spot": float(r.get("rms_spot", float("nan"))),
                "inv_frac": float(r.get("invalid_fraction", 1.0)),
                "seidel": r.get("seidel_all") or r.get("measures") or [float("nan")] * 5,
                "gen": int(r.get("gen", 0)),
                "insert": int(r.get("insert_status", 0)),
                "rejected": bool(r.get("rejected", False)),
            }
            if r["source"] == "seed":
                seeds.append(row)
            else:
                offs.append(row)
    return seeds, offs


def analyse(run_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    seeds, offs = load_lineage(run_dir)
    print(f"run: {run_dir}")
    print(f"  seed phase:  {len(seeds):,} samples")
    print(f"  offspring:   {len(offs):,} samples")

    fit_s = np.array([r["fitness"] for r in seeds])
    fit_o = np.array([r["fitness"] for r in offs])
    sei_s = np.array([r["seidel"] for r in seeds])
    sei_o = np.array([r["seidel"] for r in offs])
    gen_o = np.array([r["gen"] for r in offs])
    inv_s = np.array([r["inv_frac"] for r in seeds])
    inv_o = np.array([r["inv_frac"] for r in offs])

    # Replace -inf / NaN with sentinel very negative value for histogram/sort
    fit_s_clean = np.where(np.isfinite(fit_s), fit_s, -1e6)
    fit_o_clean = np.where(np.isfinite(fit_o), fit_o, -1e6)

    # ---- headline stats ----
    print("\n=== FITNESS ===")
    print(f"  seed   : mean={fit_s_clean.mean():.4f}  median={np.median(fit_s_clean):.4f}  best={fit_s_clean.max():.4f}")
    print(f"  offspr : mean={fit_o_clean.mean():.4f}  median={np.median(fit_o_clean):.4f}  best={fit_o_clean.max():.4f}")
    print(f"  offspring - seed mean: {fit_o_clean.mean() - fit_s_clean.mean():+.4f}")

    # fraction above thresholds
    print("\n=== FRACTION OF SAMPLES ABOVE FITNESS THRESHOLD ===")
    print(f"{'threshold':>10} {'seed':>12} {'offspring':>12} {'ratio':>8}")
    for thr in [-10.0, -1.0, -0.5, -0.2, -0.1, -0.05]:
        fs = float((fit_s_clean > thr).mean())
        fo = float((fit_o_clean > thr).mean())
        ratio = fo / max(fs, 1e-9)
        print(f"{thr:>10.2f} {fs*100:>10.3f}% {fo*100:>10.3f}% {ratio:>7.1f}x")

    # ---- figure ----
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.30,
                          left=0.06, right=0.98, top=0.94, bottom=0.06)

    # A: Fitness histogram
    ax = fig.add_subplot(gs[0, 0])
    # Clip for visualisation
    lo = max(-50, np.percentile(np.concatenate([fit_s_clean, fit_o_clean]), 2))
    hi = 0
    bins = np.linspace(lo, hi, 80)
    ax.hist(np.clip(fit_s_clean, lo, hi), bins=bins, alpha=0.55, color="C1",
            label=f"seed (N={len(fit_s):,})", density=True)
    ax.hist(np.clip(fit_o_clean, lo, hi), bins=bins, alpha=0.55, color="C0",
            label=f"offspring (N={len(fit_o):,})", density=True)
    ax.set_xlabel("fitness (higher = better)")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.set_title("A. Fitness distribution by phase\n(log-density)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # B: Survival / cumulative
    ax = fig.add_subplot(gs[0, 1])
    xx = np.linspace(lo, 0.0, 200)
    surv_s = np.array([(fit_s_clean > x).mean() for x in xx])
    surv_o = np.array([(fit_o_clean > x).mean() for x in xx])
    ax.plot(xx, surv_s * 100, color="C1", lw=2, label="seed")
    ax.plot(xx, surv_o * 100, color="C0", lw=2, label="offspring")
    ax.set_xlabel("fitness threshold")
    ax.set_ylabel("% of phase above threshold")
    ax.set_yscale("log")
    ax.set_title("B. Survival curve\n(how many samples beat a given threshold)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # C: 2-D Seidel density (S2, S3)
    ax = fig.add_subplot(gs[0, 2])
    valid_s = np.isfinite(sei_s[:, 1]) & np.isfinite(sei_s[:, 2])
    valid_o = np.isfinite(sei_o[:, 1]) & np.isfinite(sei_o[:, 2])
    ax.scatter(sei_s[valid_s, 1] * 1e4, sei_s[valid_s, 2] * 1e4,
               s=2, color="C1", alpha=0.25, label=f"seed")
    ax.scatter(sei_o[valid_o, 1] * 1e4, sei_o[valid_o, 2] * 1e4,
               s=2, color="C0", alpha=0.15, label=f"offspring")
    # Clip axes to archive bounds for visibility
    ax.set_xlim(-15, 15); ax.set_ylim(-15, 15)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel(r"$S_2$ (coma) [$\times 10^{-4}$]")
    ax.set_ylabel(r"$S_3$ (astigmatism) [$\times 10^{-4}$]")
    ax.set_title("C. Sample location in $(S_2, S_3)$")
    ax.legend(loc="upper right", fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.3)

    # D: 2-D Seidel density (S3, S4) — most-informative pair
    ax = fig.add_subplot(gs[1, 0])
    valid_s = np.isfinite(sei_s[:, 2]) & np.isfinite(sei_s[:, 3])
    valid_o = np.isfinite(sei_o[:, 2]) & np.isfinite(sei_o[:, 3])
    ax.scatter(sei_s[valid_s, 2] * 1e4, sei_s[valid_s, 3] * 1e4,
               s=2, color="C1", alpha=0.25, label="seed")
    ax.scatter(sei_o[valid_o, 2] * 1e4, sei_o[valid_o, 3] * 1e4,
               s=2, color="C0", alpha=0.15, label="offspring")
    ax.set_xlim(-15, 15); ax.set_ylim(-15, 15)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel(r"$S_3$ (astigmatism) [$\times 10^{-4}$]")
    ax.set_ylabel(r"$S_4$ (Petzval) [$\times 10^{-4}$]")
    ax.set_title("D. Sample location in $(S_3, S_4)$")
    ax.legend(loc="upper right", fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.3)

    # E: Per-generation fitness distribution (heatmap)
    ax = fig.add_subplot(gs[1, 1])
    if len(offs) > 0:
        # Bin generations into 100 buckets, compute [5, 50, 95]%ile of fitness per bucket
        gen_max = gen_o.max()
        n_bins = min(100, max(1, gen_max // 10))
        gen_bins = np.linspace(0, gen_max, n_bins + 1)
        mid_gen = 0.5 * (gen_bins[:-1] + gen_bins[1:])
        p5 = np.full(n_bins, np.nan); p50 = np.full(n_bins, np.nan); p95 = np.full(n_bins, np.nan)
        for i in range(n_bins):
            mask = (gen_o >= gen_bins[i]) & (gen_o < gen_bins[i + 1])
            if mask.sum() > 0:
                vals = np.clip(fit_o_clean[mask], lo, 0)
                p5[i], p50[i], p95[i] = np.percentile(vals, [5, 50, 95])
        ax.fill_between(mid_gen, p5, p95, alpha=0.3, color="C0", label="5-95%ile")
        ax.plot(mid_gen, p50, color="C0", lw=1.5, label="median")
        ax.set_ylim(max(lo, -50), 0.5)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.set_title("E. Offspring fitness distribution per generation")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # F: Cumulative best fitness (seed-phase best line + offspring climb)
    ax = fig.add_subplot(gs[1, 2])
    cum_best_s = np.maximum.accumulate(fit_s_clean)
    cum_best_o = np.maximum.accumulate(np.concatenate([fit_s_clean, fit_o_clean]))
    xs = np.arange(len(cum_best_s))
    xo = np.arange(len(cum_best_o))
    ax.plot(xs, cum_best_s, color="C1", lw=2, label="best so far (seed only)")
    ax.plot(xo, cum_best_o, color="C0", lw=2, label="best so far (full run)")
    ax.axvline(len(seeds), color="k", lw=1, ls="--", alpha=0.5,
               label=f"seed→offspring transition")
    ax.set_xlabel("evaluation number")
    ax.set_ylabel("best fitness so far")
    ax.set_title("F. Best-so-far trajectory")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(max(lo, -50), 0.5)

    # G: Fraction of samples in valid / low-invalid region
    ax = fig.add_subplot(gs[2, 0])
    bins_inv = np.linspace(0, 1, 40)
    ax.hist(inv_s, bins=bins_inv, alpha=0.6, color="C1", density=True,
            label="seed")
    ax.hist(inv_o, bins=bins_inv, alpha=0.6, color="C0", density=True,
            label="offspring")
    ax.set_xlabel("invalid fraction (rays that missed)")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.set_title("G. Ray-validity distribution\n(lower = better lens)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # H: RMS spot distribution
    ax = fig.add_subplot(gs[2, 1])
    rms_s = np.array([r["rms_spot"] for r in seeds])
    rms_o = np.array([r["rms_spot"] for r in offs])
    rms_s = rms_s[np.isfinite(rms_s)]
    rms_o = rms_o[np.isfinite(rms_o)]
    # log scale; clip
    rms_s_c = np.clip(rms_s, 1e-3, 1e3)
    rms_o_c = np.clip(rms_o, 1e-3, 1e3)
    bins_rms = np.logspace(-3, 3, 60)
    ax.hist(rms_s_c, bins=bins_rms, alpha=0.6, color="C1", density=True,
            label=f"seed (median={np.median(rms_s_c):.2e} mm)")
    ax.hist(rms_o_c, bins=bins_rms, alpha=0.6, color="C0", density=True,
            label=f"offspring (median={np.median(rms_o_c):.2e} mm)")
    ax.set_xscale("log")
    ax.set_xlabel("spot RMS (mm)")
    ax.set_ylabel("density")
    ax.set_title("H. Spot-RMS distribution\n(lower = sharper lens)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # I: Headline metrics as text
    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    lines = [
        "HEADLINE METRICS",
        f"  seed samples:      {len(fit_s):,}",
        f"  offspring samples: {len(fit_o):,}",
        "",
        "MEAN FITNESS",
        f"  seed:      {fit_s_clean.mean():.3f}",
        f"  offspring: {fit_o_clean.mean():.3f}",
        f"  gain:      +{fit_o_clean.mean() - fit_s_clean.mean():.3f}",
        "",
        "FRACTION OF SAMPLES WITH FITNESS > −0.5",
        f"  seed:      {(fit_s_clean > -0.5).mean() * 100:.3f}%",
        f"  offspring: {(fit_o_clean > -0.5).mean() * 100:.3f}%",
        f"  ratio:     {(fit_o_clean > -0.5).mean() / max((fit_s_clean > -0.5).mean(), 1e-9):.1f}×",
        "",
        "FRACTION WITH fitness > −0.2 (elite threshold)",
        f"  seed:      {(fit_s_clean > -0.2).mean() * 100:.3f}%",
        f"  offspring: {(fit_o_clean > -0.2).mean() * 100:.3f}%",
        f"  ratio:     {(fit_o_clean > -0.2).mean() / max((fit_s_clean > -0.2).mean(), 1e-9):.1f}×",
    ]
    ax.text(0.0, 0.97, "\n".join(lines), family="monospace", fontsize=10,
            va="top", transform=ax.transAxes)

    fig.suptitle(
        f"Sample efficiency — {os.path.basename(run_dir)}\n"
        "Where do CMA-MEGA+Seidel mutations actually end up?",
        fontsize=14, fontweight="bold",
    )
    out_png = os.path.join(out_dir, "sample_efficiency.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_png}")

    # Save numerical summary
    summary = {
        "run_dir": run_dir,
        "n_seed": len(seeds),
        "n_offspring": len(offs),
        "mean_fitness_seed": float(fit_s_clean.mean()),
        "mean_fitness_offspring": float(fit_o_clean.mean()),
        "best_seed": float(fit_s_clean.max()),
        "best_offspring": float(fit_o_clean.max()),
        "threshold_fractions": {
            str(thr): {
                "seed": float((fit_s_clean > thr).mean()),
                "offspring": float((fit_o_clean > thr).mean()),
            } for thr in [-10.0, -1.0, -0.5, -0.2, -0.1, -0.05]
        },
    }
    with open(os.path.join(out_dir, "sample_efficiency_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {os.path.join(out_dir, 'sample_efficiency_summary.json')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--out", default=None,
                        help="output directory (default: <run_dir>/sample_efficiency)")
    args = parser.parse_args()
    out = args.out or os.path.join(args.run_dir, "sample_efficiency")
    analyse(args.run_dir, out)


if __name__ == "__main__":
    main()
