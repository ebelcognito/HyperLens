"""
Fair prior-quality test.

Take N_SAMPLES random seeds AND N_SAMPLES CMA-MEGA offspring from the Cooke
3-D S2S3S4 run. Refine each with the SAME L-BFGS (same steps, same termination).
Compare the distribution of refined RMS.

If CMA-MEGA priors refine to lower RMS than random-seed priors at matched budget,
that isolates "CMA-MEGA emits better starts for gradient descent" — the core claim.
"""
import glob
import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, "/home/hisham26/Desktop/RL")
from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.run import build_all


CFG = ("/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
       "config/examples/cooke_triplet_seidel3d_s2s3s4.yaml")
RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "cooke_triplet_seidel3d_s2s3s4_20260423_031324")
OUT = "/home/hisham26/Desktop/RL/paper_figures/refined_prior_comparison"
os.makedirs(OUT, exist_ok=True)

N_SAMPLES = 100       # per phase — total 200 refinements, ~7 min
MAX_ITER = 50         # L-BFGS steps per refinement


def load_samples():
    seeds, offs = [], []
    with open(os.path.join(RUN_DIR, "lineage.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            th = r.get("theta")
            if th is None:
                continue
            (seeds if r["source"] == "seed" else offs).append(th)
    return np.array(seeds), np.array(offs)


def refine_one(problem, theta_start, max_iter=MAX_ITER):
    """Minimal L-BFGS-B refinement using problem.evaluate (which gives
    fitness + autograd-derived θ-gradients)."""
    def fg(theta):
        try:
            res = problem.evaluate(theta, with_grads=True)
            # scipy minimizes; we want to maximise fitness → minimise -fitness
            return float(-res.fitness), -res.grads["fitness"].astype(np.float64)
        except Exception:
            return 1e9, np.zeros_like(theta)

    theta_bounds = problem.theta_bounds
    bounds = [(float(lo), float(hi)) for lo, hi in theta_bounds]
    try:
        result = minimize(
            fg, theta_start, jac=True, method="L-BFGS-B",
            bounds=bounds, options=dict(maxiter=max_iter, ftol=1e-9, gtol=1e-6),
        )
        # Evaluate once more to get clean RMS
        res_final = problem.evaluate(result.x, with_grads=False)
        return {
            "theta_final": result.x,
            "fitness_before": float(problem.evaluate(theta_start, with_grads=False).fitness),
            "fitness_after": float(res_final.fitness),
            "rms_before": float(problem.evaluate(theta_start, with_grads=False).metrics.get("rms_spot", np.nan)),
            "rms_after": float(res_final.metrics.get("rms_spot", np.nan)),
            "n_iter": int(result.nit),
            "converged": bool(result.success),
        }
    except Exception as e:
        return None


def main():
    cfg = load_config(CFG)
    problem, _, _, _, _ = build_all(cfg)
    problem._ensure_backend()

    seeds, offs = load_samples()
    rng = np.random.default_rng(0)
    seeds_sub = seeds[rng.choice(len(seeds), N_SAMPLES, replace=False)]
    offs_sub = offs[rng.choice(len(offs), N_SAMPLES, replace=False)]
    print(f"refining {N_SAMPLES} random seeds and {N_SAMPLES} CMA-MEGA priors "
          f"with L-BFGS-B (max_iter={MAX_ITER}) ...")

    def run_all(label, samples):
        results = []
        for i, th in enumerate(samples):
            r = refine_one(problem, th)
            if r is not None:
                results.append(r)
            if (i + 1) % 20 == 0:
                rms_arr = np.array([x["rms_after"] for x in results
                                    if np.isfinite(x["rms_after"])])
                med = np.median(rms_arr) if len(rms_arr) else float("nan")
                print(f"  [{label}] {i+1}/{len(samples)}  median RMS after = {med:.4f} mm")
        return results

    seed_res = run_all("seed", seeds_sub)
    off_res = run_all("offspring", offs_sub)

    # Collect stats
    def stats(res):
        rms_before = np.array([r["rms_before"] for r in res
                                if np.isfinite(r["rms_before"])])
        rms_after = np.array([r["rms_after"] for r in res
                               if np.isfinite(r["rms_after"])])
        fit_before = np.array([r["fitness_before"] for r in res
                                if np.isfinite(r["fitness_before"])])
        fit_after = np.array([r["fitness_after"] for r in res
                               if np.isfinite(r["fitness_after"])])
        return rms_before, rms_after, fit_before, fit_after

    s_rb, s_ra, s_fb, s_fa = stats(seed_res)
    o_rb, o_ra, o_fb, o_fa = stats(off_res)

    def q(a):
        return (np.min(a), np.percentile(a, 25), np.median(a),
                np.percentile(a, 75), np.max(a), np.mean(a))

    print("\n--- Refined RMS (mm) ---")
    print(f"  seed (before): min={s_rb.min():.4f} med={np.median(s_rb):.4f} "
          f"max={s_rb.max():.4f}")
    print(f"  seed (AFTER):  min={s_ra.min():.4f} med={np.median(s_ra):.4f} "
          f"max={s_ra.max():.4f}")
    print(f"  off  (before): min={o_rb.min():.4f} med={np.median(o_rb):.4f} "
          f"max={o_rb.max():.4f}")
    print(f"  off  (AFTER):  min={o_ra.min():.4f} med={np.median(o_ra):.4f} "
          f"max={o_ra.max():.4f}")

    print("\n--- Fraction of refined designs below a spot-RMS threshold ---")
    print(f"{'threshold (mm)':>20} {'seed %':>10} {'offspring %':>12} {'ratio':>8}")
    for t in [0.050, 0.020, 0.010, 0.005]:
        f_s = (s_ra < t).mean() * 100
        f_o = (o_ra < t).mean() * 100
        print(f"{t:>20.3f} {f_s:>9.1f}% {f_o:>11.1f}% "
              f"{f_o/max(f_s, 1e-6):>7.1f}×")

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw=dict(wspace=0.28))

    # A: RMS distribution before/after (log scale)
    ax = axes[0]
    bins = np.logspace(-3, 1, 50)
    ax.hist(np.clip(s_ra, 1e-3, 1e2), bins=bins, alpha=0.55, color="C1",
            label=f"random seed → refined  (N={len(s_ra)})")
    ax.hist(np.clip(o_ra, 1e-3, 1e2), bins=bins, alpha=0.55, color="C2",
            label=f"CMA-MEGA → refined  (N={len(o_ra)})")
    ax.axvline(np.median(s_ra), color="C1", ls="--",
               label=f"seed median {np.median(s_ra)*1000:.2f} μm")
    ax.axvline(np.median(o_ra), color="C2", ls="--",
               label=f"offspring median {np.median(o_ra)*1000:.2f} μm")
    ax.set_xscale("log")
    ax.set_xlabel("refined spot RMS (mm)")
    ax.set_ylabel("# designs")
    ax.set_title("A. Refined RMS distribution\n"
                 "(each sample → L-BFGS to convergence)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # B: CDF comparison
    ax = axes[1]
    def cdf(a):
        x = np.sort(a); y = np.arange(1, len(x) + 1) / len(x)
        return x, y
    xs, ys_s = cdf(s_ra); xo, ys_o = cdf(o_ra)
    ax.plot(xs, ys_s * 100, color="C1", lw=2, label="random seed → refined")
    ax.plot(xo, ys_o * 100, color="C2", lw=2, label="CMA-MEGA → refined")
    ax.set_xscale("log")
    ax.set_xlabel("refined spot RMS (mm)")
    ax.set_ylabel("% of refined designs ≤ RMS")
    ax.set_title("B. CDF of refined RMS\n"
                 "(left = lower-RMS designs; higher CDF at small RMS is better)",
                 fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Fair prior-quality test: same L-BFGS refinement, matched N = {N_SAMPLES}\n"
        "Cooke 3-D (S₂, S₃, S₄) · equal budget, equal refinement",
        fontsize=12, fontweight="bold",
    )
    out_png = os.path.join(OUT, "refined_prior_comparison.png")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_png}")

    # Save numerical summary
    import json as _json
    summary = {
        "n_samples_per_phase": N_SAMPLES,
        "seed": {
            "rms_after_min": float(s_ra.min()),
            "rms_after_median": float(np.median(s_ra)),
            "rms_after_max": float(s_ra.max()),
            "rms_after_p25": float(np.percentile(s_ra, 25)),
            "rms_after_p75": float(np.percentile(s_ra, 75)),
        },
        "offspring": {
            "rms_after_min": float(o_ra.min()),
            "rms_after_median": float(np.median(o_ra)),
            "rms_after_max": float(o_ra.max()),
            "rms_after_p25": float(np.percentile(o_ra, 25)),
            "rms_after_p75": float(np.percentile(o_ra, 75)),
        },
        "fraction_below_threshold": {
            f"{t}": {"seed": float((s_ra < t).mean()),
                     "offspring": float((o_ra < t).mean())}
            for t in [0.050, 0.020, 0.010, 0.005]
        },
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        _json.dump(summary, f, indent=2)
    print(f"saved summary.json")


if __name__ == "__main__":
    main()
