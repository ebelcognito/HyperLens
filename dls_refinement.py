"""
DLS (Damped Least-Squares) / Levenberg-Marquardt refinement on saved elite priors.

For each elite, take its raw theta (the prior CMA-MEGA emitted) and refine
with scipy.least_squares (method='trf', LM-class trust-region with bounds).
The residual vector is per-ray (dx, dy) from the chief ray on the sensor,
across all field angles and wavelengths in the run config — this is the
multi-residual structure DLS was designed to exploit, unlike L-BFGS which
only sees the scalar sum-of-squares.

Reports refined-RMS comparison vs the stored L-BFGS refined value.
"""
import glob
import json
import os
import sys
import time
import numpy as np
import torch
from scipy.optimize import least_squares

sys.path.insert(0, "/home/hisham26/Desktop/RL")
sys.path.insert(0, "/home/hisham26/Desktop/RL/DeepLens")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from optical_rl.qd.lensmega.run import build_all


CFG = ("/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
       "config/examples/sixlet_discovery.yaml")
RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "sixlet_discovery_20260423_132001")
OUT = "/home/hisham26/Desktop/RL/paper_figures/dls_refinement"
os.makedirs(OUT, exist_ok=True)

N_ELITES = 5         # top-N best-RMS elites to test
N_RAYS = 128         # rays per field per wavelength
EFL_WEIGHT = 5.0     # weight of EFL residual (vs ray residuals)
MAX_NFEV = 500       # max function evals (scipy least_squares)


def load_top_n_elites(n=N_ELITES):
    paths = sorted(glob.glob(os.path.join(RUN_DIR, "elite_designs", "*.json")))
    out = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        ref = d.get("refined") or {}
        if "rms_mm" not in ref:
            continue
        out.append({
            "name": os.path.basename(p)[:-5],
            "theta_raw": np.array(d["theta"], dtype=float),
            "theta_refined": np.array(ref["theta"], dtype=float),
            "rms_um_raw": float(d.get("rms_mm", 0.0)) * 1000.0,
            "rms_um_refined": float(ref["rms_mm"]) * 1000.0,
        })
    out.sort(key=lambda e: e["rms_um_refined"])
    return out[:n]


class DLSContext:
    """Holds problem + lens + template + solver for repeated residual evals."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.problem, _, _, _, _ = build_all(cfg)
        self.problem._ensure_backend()
        # Build a separate GeoLens we can reset per residual call
        from deeplens.optics import GeoLens
        p = cfg.problem
        self.template, self.lens0 = load_lens_template(
            p.lens_path,
            wavelength_um=p.wavelength_um,
            stop_surface_index=p.stop_surface_index,
            free_curvature_indices=tuple(p.free_curvature_indices),
            derived_curvature_index=p.derived_curvature_index,
            device="cpu",
        )
        self.solver = GeneralizedParaxialSolver(self.template)
        self.lens = GeoLens(filename=p.lens_path, device="cpu")
        self.bounds = self.problem.theta_bounds
        self.fields_deg = self._build_fields()
        self.wavelengths = list(p.fitness_wavelengths_um)

    def _build_fields(self):
        # Match the fitness function: uses 3 field heights from 0 to hfov
        hfov_deg = float(np.rad2deg(float(self.lens.rfov)))
        return [0.0, 0.5 * hfov_deg, hfov_deg]

    def apply_theta(self, theta):
        p = self.cfg.problem
        theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=False)
        _apply_theta_generic(
            self.lens, theta_t,
            template=self.template, solver=self.solver,
            focal_length_target_mm=p.focal_length_target_mm,
            device="cpu",
            parameterization=p.parameterization,
            free_curvature_indices=p.free_curvature_indices,
            free_gap_indices=p.free_gap_indices,
            variable_flange=p.variable_flange,
            variable_materials=p.variable_materials,
            free_material_indices=p.free_material_indices,
        )

    def residuals(self, theta):
        """Return vector of per-ray (dx, dy) deviations from chief ray
        across all (field, wavelength) combos + an EFL residual."""
        try:
            self.apply_theta(theta)
        except Exception:
            return np.full(2 * N_RAYS * len(self.fields_deg) * len(self.wavelengths) + 1,
                            1e3, dtype=np.float64)

        all_residuals = []
        depth = float(self.cfg.problem.object_depth_mm)
        try:
            for field_deg in self.fields_deg:
                for wvln in self.wavelengths:
                    ray = self.lens.sample_point_source_2D(
                        fov=float(field_deg), depth=depth,
                        num_rays=int(N_RAYS), wvln=float(wvln),
                    )
                    ray_out, _ = self.lens.trace2sensor(ray=ray, record=False)
                    x = ray_out.o[..., 0].detach().cpu().numpy().reshape(-1)
                    y = ray_out.o[..., 1].detach().cpu().numpy().reshape(-1)
                    valid = np.isfinite(x) & np.isfinite(y)
                    if valid.sum() == 0:
                        all_residuals.extend([1e3] * (2 * N_RAYS))
                        continue
                    x_mean = x[valid].mean(); y_mean = y[valid].mean()
                    dx = x - x_mean; dy = y - y_mean
                    # Replace invalid with large penalty to keep size constant
                    dx[~valid] = 1.0; dy[~valid] = 1.0
                    all_residuals.extend(dx.tolist())
                    all_residuals.extend(dy.tolist())
        except Exception:
            return np.full(2 * N_RAYS * len(self.fields_deg) * len(self.wavelengths) + 1,
                            1e3, dtype=np.float64)

        # EFL residual: encourage focal length to stay near target
        try:
            efl_actual = float(self.lens.foclen)
            target = self.cfg.problem.focal_length_target_mm
            efl_res = EFL_WEIGHT * (efl_actual - target) / target
        except Exception:
            efl_res = 1e3
        all_residuals.append(efl_res)

        return np.array(all_residuals, dtype=np.float64)

    def evaluate_full(self, theta):
        """Compute spot RMS via the standard problem.evaluate (for comparison)."""
        res = self.problem.evaluate(theta, with_grads=False)
        return {
            "rms_um": float(res.metrics.get("rms_spot", float("nan"))) * 1000.0,
            "fitness": float(res.fitness),
        }


def run_dls(ctx, theta_start, max_nfev=MAX_NFEV, verbose=False):
    """Run scipy.least_squares (LM-class) on per-ray residuals."""
    bounds = ctx.bounds
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)

    t0 = time.time()
    try:
        result = least_squares(
            ctx.residuals, theta_start.copy(),
            method='trf', bounds=(lo, hi),
            max_nfev=max_nfev,
            xtol=1e-9, ftol=1e-9, gtol=1e-9,
            verbose=2 if verbose else 0,
        )
    except Exception as e:
        return {"error": str(e)}
    elapsed = time.time() - t0

    final = ctx.evaluate_full(result.x)
    return {
        "theta_final": result.x,
        "rms_um_after": final["rms_um"],
        "fitness_after": final["fitness"],
        "n_eval": int(result.nfev),
        "n_iter": int(getattr(result, "njev", -1)),
        "converged": bool(result.success),
        "elapsed_s": float(elapsed),
        "cost": float(result.cost),
    }


def main():
    cfg = load_config(CFG)
    ctx = DLSContext(cfg)
    elites = load_top_n_elites(N_ELITES)

    print(f"DLS refinement test on top-{N_ELITES} sixlet elites")
    print(f"  fields:      {ctx.fields_deg}")
    print(f"  wavelengths: {ctx.wavelengths}")
    print(f"  rays/field:  {N_RAYS}")
    print(f"  total residuals per eval: ~{2*N_RAYS*len(ctx.fields_deg)*len(ctx.wavelengths) + 1}")
    print()

    summary = []
    for k, e in enumerate(elites):
        print(f"=== elite {k+1}: {e['name']} ===")
        print(f"  raw RMS={e['rms_um_raw']:.2f} μm  →  L-BFGS refined RMS={e['rms_um_refined']:.2f} μm")
        # Sanity check: re-evaluate the stored prior
        baseline = ctx.evaluate_full(e["theta_raw"])
        print(f"  re-eval of raw prior: {baseline['rms_um']:.3f} μm "
              f"(matches stored {e['rms_um_raw']:.2f}μm)")

        # Run DLS from the raw prior
        r = run_dls(ctx, e["theta_raw"])
        if "error" in r:
            print(f"  DLS FAILED: {r['error']}")
            continue
        delta = e["rms_um_refined"] - r["rms_um_after"]
        pct = 100 * delta / e["rms_um_refined"]
        flag = "↓ better" if delta > 0.5 else ("≈ tie" if abs(delta) < 0.5 else "↑ worse")
        print(f"  DLS result: RMS={r['rms_um_after']:.3f} μm  "
              f"(Δ={delta:+.3f} μm, {pct:+.1f}%, {flag})  "
              f"[{r['n_eval']} evals, {r['elapsed_s']:.1f}s]")
        print()
        summary.append({
            "elite": e["name"],
            "rms_lbfgs": e["rms_um_refined"],
            "rms_dls": r["rms_um_after"],
            "delta_um": delta,
            "delta_pct": pct,
            "n_eval": r["n_eval"],
            "elapsed_s": r["elapsed_s"],
        })

    # Summary
    print("\n========== SUMMARY ==========")
    print(f"{'elite':<32} {'L-BFGS':>9} {'DLS':>9} {'Δ μm':>9} {'Δ %':>7}")
    for s in summary:
        flag = "↓" if s["delta_um"] > 0.5 else ("≈" if abs(s["delta_um"]) < 0.5 else "↑")
        print(f"{s['elite']:<32} {s['rms_lbfgs']:>7.2f}μm "
              f"{s['rms_dls']:>7.2f}μm  {s['delta_um']:>+8.2f}  "
              f"{s['delta_pct']:>+5.1f}%  {flag}")

    out_path = os.path.join(OUT, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
