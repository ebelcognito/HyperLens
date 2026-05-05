"""
Multi-start L-BFGS refinement test.

For each chosen elite, take its PRIOR theta (raw θ that CMA-MEGA emitted),
generate N small Gaussian perturbations, run L-BFGS-B from each, and report
whether the best perturbation-refined RMS beats the elite's stored refined RMS.

Catches cases where the unperturbed prior happens to land in a sub-optimal
local minimum nearby, when a small perturbation would have rolled into a
better basin.
"""
import glob
import json
import os
import sys
import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, "/home/hisham26/Desktop/RL")
from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.run import build_all


CFG = ("/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
       "config/examples/sixlet_discovery.yaml")
RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "sixlet_discovery_20260423_132001")
OUT = "/home/hisham26/Desktop/RL/paper_figures/multi_start_refinement"
os.makedirs(OUT, exist_ok=True)

N_PERTURBATIONS = 5
PERTURB_EPS = 0.02     # 2% of each axis's bound range
MAX_ITER = 100
TARGET_BASIN_IDS = None    # None = test top-K best-RMS elites; otherwise list of basin IDs


def load_top_n_elites_by_rms(n=5):
    paths = sorted(glob.glob(os.path.join(RUN_DIR, "elite_designs", "*.json")))
    elites = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        ref = d.get("refined") or {}
        if "theta" not in ref or "rms_mm" not in ref:
            continue
        elites.append({
            "name": os.path.basename(p)[:-5],
            "theta_raw": np.array(d["theta"], dtype=float),
            "theta_refined": np.array(ref["theta"], dtype=float),
            "rms_um_raw": float(d.get("rms_mm", float("nan"))) * 1000.0,
            "rms_um_refined": float(ref["rms_mm"]) * 1000.0,
        })
    elites.sort(key=lambda e: e["rms_um_refined"])
    return elites[:n]


def refine(problem, theta_start, max_iter=MAX_ITER):
    def fg(theta):
        try:
            res = problem.evaluate(theta, with_grads=True)
            return float(-res.fitness), -res.grads["fitness"].astype(np.float64)
        except Exception:
            return 1e9, np.zeros_like(theta)

    bounds = [(float(lo), float(hi)) for lo, hi in problem.theta_bounds]
    try:
        result = minimize(
            fg, theta_start, jac=True, method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=max_iter, ftol=1e-9, gtol=1e-6),
        )
        res_final = problem.evaluate(result.x, with_grads=False)
        return {
            "theta_final": result.x,
            "fitness_after": float(res_final.fitness),
            "rms_um_after": float(res_final.metrics.get("rms_spot", float("nan"))) * 1000.0,
            "n_iter": int(result.nit),
            "converged": bool(result.success),
        }
    except Exception as e:
        return None


def main():
    cfg = load_config(CFG)
    problem, _, _, _, _ = build_all(cfg)
    problem._ensure_backend()
    bounds = problem.theta_bounds

    # Pick top-5 elites by refined RMS
    elites = load_top_n_elites_by_rms(n=5)
    print(f"Testing multi-start on top-{len(elites)} elites by refined RMS\n")
    rng = np.random.default_rng(42)

    # Per-axis perturbation magnitudes
    bound_range = np.array([hi - lo for lo, hi in bounds])

    summary = []
    for k, e in enumerate(elites):
        print(f"=== elite {k+1}: {e['name']}  stored refined RMS = "
              f"{e['rms_um_refined']:.3f} μm ===")

        # First refine from the original raw prior (sanity baseline)
        orig = refine(problem, e["theta_raw"])
        if orig is not None:
            print(f"  baseline refine of raw prior: RMS = {orig['rms_um_after']:.3f} μm "
                  f"(stored: {e['rms_um_refined']:.3f} μm)")

        # Multi-start
        results = []
        for i in range(N_PERTURBATIONS):
            noise = rng.normal(0, PERTURB_EPS, size=len(bound_range)) * bound_range
            theta_pert = np.clip(
                e["theta_raw"] + noise,
                np.array([lo for lo, _ in bounds]),
                np.array([hi for _, hi in bounds]),
            )
            r = refine(problem, theta_pert)
            if r is not None:
                results.append(r)
                print(f"  perturbation {i+1}: RMS = {r['rms_um_after']:.3f} μm "
                      f"(converged={r['converged']}, iters={r['n_iter']})")

        if results:
            best_pert = min(results, key=lambda r: r["rms_um_after"])
            best_rms = best_pert["rms_um_after"]
            improvement_um = e["rms_um_refined"] - best_rms
            improvement_pct = 100 * improvement_um / e["rms_um_refined"]
            print(f"  best of multi-start: RMS = {best_rms:.3f} μm  "
                  f"(Δ = {improvement_um:+.3f} μm, {improvement_pct:+.1f}%)\n")
            summary.append({
                "elite": e["name"],
                "stored_rms_um": e["rms_um_refined"],
                "baseline_rerefine_rms_um": orig["rms_um_after"] if orig else None,
                "multistart_best_rms_um": best_rms,
                "multistart_improvement_um": improvement_um,
                "multistart_improvement_pct": improvement_pct,
                "n_perturbations": len(results),
                "all_perturbation_rms": [r["rms_um_after"] for r in results],
            })

    # Print summary table
    print("\n========== SUMMARY ==========")
    print(f"{'elite':<32} {'stored':>9} {'best m-s':>9} {'Δ (μm)':>9} {'Δ %':>7}")
    for s in summary:
        d = s["multistart_improvement_um"]
        p = s["multistart_improvement_pct"]
        flag = "↓ better" if d > 0.5 else ("≈ tie" if abs(d) < 0.5 else "↑ worse")
        print(f"{s['elite']:<32} {s['stored_rms_um']:>7.2f}μm "
              f"{s['multistart_best_rms_um']:>7.2f}μm  "
              f"{d:>+8.2f}  {p:>+5.1f}%  {flag}")

    out_path = os.path.join(OUT, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
