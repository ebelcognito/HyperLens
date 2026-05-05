"""
Debug the DLS implementation. Verbose at every step.
"""
import glob
import json
import os
import sys
import time
import traceback
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

N_RAYS = 64        # halved for speed during debug
EFL_WEIGHT = 5.0


class DLSContext:
    def __init__(self, cfg):
        self.cfg = cfg
        self.problem, _, _, _, _ = build_all(cfg)
        self.problem._ensure_backend()
        from deeplens.optics import GeoLens
        p = cfg.problem
        self.template, _ = load_lens_template(
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
        # Use just one field, one wavelength for debug
        self.fields_deg = [10.0]
        self.wavelengths = [0.587]
        self.call_count = 0

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
        self.call_count += 1
        try:
            self.apply_theta(theta)
        except Exception as e:
            print(f"  [call {self.call_count}] apply_theta EXCEPTION: {e}")
            traceback.print_exc()
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
                    ray_out = self.lens.trace2sensor(ray=ray, record=False)
                    o = ray_out.o
                    print(f"  [call {self.call_count}] field={field_deg}, wvln={wvln}: ray.o.shape={o.shape}, dtype={o.dtype}")
                    x = o[..., 0].detach().cpu().numpy().reshape(-1)
                    y = o[..., 1].detach().cpu().numpy().reshape(-1)
                    print(f"  [call {self.call_count}] x: min={np.nanmin(x):.4e}, max={np.nanmax(x):.4e}, "
                          f"finite count={np.isfinite(x).sum()}/{len(x)}")
                    print(f"  [call {self.call_count}] y: min={np.nanmin(y):.4e}, max={np.nanmax(y):.4e}, "
                          f"finite count={np.isfinite(y).sum()}/{len(y)}")
                    valid = np.isfinite(x) & np.isfinite(y)
                    if valid.sum() == 0:
                        all_residuals.extend([1e3] * (2 * N_RAYS))
                        continue
                    x_mean = x[valid].mean(); y_mean = y[valid].mean()
                    dx = x - x_mean; dy = y - y_mean
                    dx[~valid] = 1.0; dy[~valid] = 1.0
                    print(f"  [call {self.call_count}] dx: |max|={np.max(np.abs(dx)):.4e}, "
                          f"std={np.std(dx):.4e}")
                    all_residuals.extend(dx.tolist())
                    all_residuals.extend(dy.tolist())
        except Exception as e:
            print(f"  [call {self.call_count}] trace EXCEPTION: {e}")
            traceback.print_exc()
            return np.full(2 * N_RAYS * len(self.fields_deg) * len(self.wavelengths) + 1,
                            1e3, dtype=np.float64)

        try:
            efl_actual = float(self.lens.foclen)
            target = self.cfg.problem.focal_length_target_mm
            efl_res = EFL_WEIGHT * (efl_actual - target) / target
        except Exception as e:
            print(f"  [call {self.call_count}] efl EXCEPTION: {e}")
            efl_res = 1e3
        all_residuals.append(efl_res)

        out = np.array(all_residuals, dtype=np.float64)
        print(f"  [call {self.call_count}] residual vector: shape={out.shape}, "
              f"norm={np.linalg.norm(out):.4e}, "
              f"any_nan={np.any(np.isnan(out))}, "
              f"any_inf={np.any(np.isinf(out))}")
        return out


def load_one_elite():
    paths = sorted(glob.glob(os.path.join(RUN_DIR, "elite_designs", "*.json")))
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        ref = d.get("refined") or {}
        if "rms_mm" not in ref:
            continue
        return {
            "name": os.path.basename(p)[:-5],
            "theta_raw": np.array(d["theta"], dtype=float),
            "rms_um_refined": float(ref["rms_mm"]) * 1000.0,
        }


def main():
    cfg = load_config(CFG)
    ctx = DLSContext(cfg)
    elite = load_one_elite()
    print(f"\n>>> Testing on elite: {elite['name']} (refined RMS={elite['rms_um_refined']:.2f}μm)\n")

    print("=== Single residual call (sanity) ===")
    r0 = ctx.residuals(elite["theta_raw"])
    print(f"residual vector size: {r0.size}")
    print(f"residual norm: {np.linalg.norm(r0):.4e}")
    print()

    print("=== Manual finite-diff Jacobian probe (test 3 dims) ===")
    eps = 1e-4
    for k in range(3):
        theta_p = elite["theta_raw"].copy()
        theta_p[k] += eps
        r1 = ctx.residuals(theta_p)
        dr = (r1 - r0) / eps
        print(f"  param {k}: |dr|/{eps:.0e} = {np.linalg.norm(dr):.4e}")
    print()

    print("=== Run scipy.least_squares (verbose) ===")
    bounds = ctx.bounds
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    ctx.call_count = 0
    t0 = time.time()
    result = least_squares(
        ctx.residuals, elite["theta_raw"].copy(),
        method='trf', bounds=(lo, hi),
        max_nfev=20,            # small for debug
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
        verbose=2,
    )
    elapsed = time.time() - t0
    print(f"\nscipy result: success={result.success}, nfev={result.nfev}, njev={result.njev}")
    print(f"  final cost: {result.cost:.4e}")
    print(f"  message: {result.message}")
    print(f"  elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
