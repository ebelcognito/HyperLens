"""Analyze the original EF50mm f/1.8 lens with the same Seidel fingerprint
and rendering pipeline used for the LensMEGA discovered elites.

Outputs:
  - ORIGINAL_ef50_raw.png           (lens drawing with rays, same style as elites)
  - ORIGINAL_ef50_fingerprint.json  (per-element Seidel breakdown + measures)
All saved into the run's elite_designs/ directory.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DEEPLENS_ROOT = PROJECT_ROOT / "DeepLens"
for _p in [str(PROJECT_ROOT), str(DEEPLENS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from deeplens import GeoLens
from optical_rl.qd.lensmega.problem.lens_template import LensTemplate, load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.lensmega.descriptors.seidel import (
    SeidelDescriptor,
    compute_per_element_seidel,
)

# ---- Configuration (matches ef50_discovery config) ----
RUN_DIR = Path(
    "/home/hisham26/Desktop/RL/results/lensmega/ef50_discovery_20260414_232959"
)
LENS_JSON = str(
    PROJECT_ROOT / "DeepLens/datasets/lenses/camera/ef50mm_f1.8.json"
)
WAVELENGTH_UM = 0.587
STOP_INDEX = 6
FREE_CURVATURE_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11)
FREE_GAP_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
VARIABLE_FLANGE = True
DERIVED_CURVATURE_INDEX = None
FOCAL_LENGTH_TARGET = 50.52
DEPTH_MM = -20000.0
SEIDEL_TERMS = ["S2", "S3"]

OUT_DIR = RUN_DIR / "elite_designs"

# ---- Load the original lens (untouched) ----
print("[original] Loading original EF50 lens…")
original_lens = GeoLens(filename=LENS_JSON, device="cpu")
torch.set_default_dtype(torch.float64)

# ---- Build LensTemplate for Seidel computation ----
template, _base_lens = load_lens_template(
    LENS_JSON,
    wavelength_um=WAVELENGTH_UM,
    stop_surface_index=STOP_INDEX,
    free_curvature_indices=FREE_CURVATURE_INDICES,
    derived_curvature_index=DERIVED_CURVATURE_INDEX,
    device="cpu",
)

# ---- Extract theta from the original lens (full_dof parameterization) ----
# theta = [curvatures...] + [gaps...] + [flange]
print("[original] Extracting theta vector from original design…")

# Curvatures at free indices
curvatures = []
for i in FREE_CURVATURE_INDICES:
    c_val = float(original_lens.surfaces[i].c.item())
    curvatures.append(c_val)

# Gaps at free indices
d_axial = [float(original_lens.surfaces[i].d.item()) for i in range(len(original_lens.surfaces))]
all_gaps = []
for i in range(len(original_lens.surfaces)):
    if i < len(original_lens.surfaces) - 1:
        all_gaps.append(d_axial[i + 1] - d_axial[i])
    else:
        all_gaps.append(0.0)

gap_values = [all_gaps[i] for i in FREE_GAP_INDICES]

# Flange: distance from last surface to sensor
last_surface_d = d_axial[-1]
flange = float(original_lens.d_sensor.item()) - last_surface_d

theta = np.array(curvatures + gap_values + [flange], dtype=np.float64)
print(f"[original] theta dim = {len(theta)}")
print(f"[original] curvatures = {curvatures}")
print(f"[original] gaps = {[f'{g:.3f}' for g in gap_values]}")
print(f"[original] flange = {flange:.3f}")

# ---- Compute Seidel measures (S2, S3) ----
print("[original] Computing Seidel descriptors (S2, S3)…")
seidel_desc = SeidelDescriptor(seidel_terms=SEIDEL_TERMS)
seidel_desc.prepare(template, device="cpu")

# We need to apply the theta to a fresh lens and compute
from optical_rl.qd.render_lensmega_archive import _apply_theta_full_dof

eval_lens = GeoLens(filename=LENS_JSON, device="cpu")
torch.set_default_dtype(torch.float64)

_apply_theta_full_dof(
    eval_lens, theta,
    template=template,
    free_curvature_indices=list(FREE_CURVATURE_INDICES),
    free_gap_indices=list(FREE_GAP_INDICES),
    variable_flange=VARIABLE_FLANGE,
    device="cpu",
)

# Compute descriptor field angle
hfov = 14.4
try:
    efl = float(original_lens.foclen)
    half_sensor = template.sensor_size_mm[0] / 2.0
    hfov = math.degrees(math.atan(half_sensor / max(abs(efl), 1.0)))
except Exception:
    pass

# Descriptor uses field_index=1 => 0.5 * hfov
descriptor_field_deg = 0.5 * hfov

measures_t = seidel_desc.compute(
    eval_lens,
    field_y_deg=descriptor_field_deg,
    object_depth_mm=DEPTH_MM,
    wavelength_um=WAVELENGTH_UM,
    device="cpu",
)
measures = [float(m) for m in measures_t.detach().cpu().numpy()]
print(f"[original] S2 (coma) = {measures[0]:.8e}")
print(f"[original] S3 (astigmatism) = {measures[1]:.8e}")

# ---- Compute per-element Seidel fingerprint ----
print("[original] Computing per-element Seidel fingerprint…")
fingerprint = compute_per_element_seidel(
    eval_lens, template,
    field_y_deg=descriptor_field_deg,
    wavelength_um=WAVELENGTH_UM,
    device="cpu",
)

# ---- Compute RMS spot size (polychromatic, multi-field) ----
print("[original] Computing RMS spot sizes…")
fitness_wavelengths = [0.486, 0.587, 0.656]
fov_fracs = [0.0, 0.5, 1.0]
rfov = float(np.rad2deg(float(eval_lens.rfov)))

spot_sizes = {}
total_spot_sq = 0.0
n_fields = 0
for fov_frac in fov_fracs:
    fov_deg = fov_frac * rfov
    spots_wv = []
    for wv in fitness_wavelengths:
        try:
            ray = eval_lens.sample_point_source_2D(
                fov=fov_deg, depth=DEPTH_MM, num_rays=128, wvln=wv,
            )
            ray.prop_to(-1.0)
            ray_out = eval_lens.trace2sensor(ray=ray)
            valid = ray_out.is_valid
            denom = valid.sum().clamp_min(1e-8)
            xy = ray_out.o[..., :2]
            centroid = (xy * valid.unsqueeze(-1)).sum(dim=0) / denom
            err2 = ((xy - centroid) ** 2).sum(dim=-1)
            rms = float(((err2 * valid).sum() / denom + 1e-12).sqrt().detach().cpu().item())
            spots_wv.append(rms)
        except Exception as e:
            print(f"  [warn] spot calc failed fov={fov_deg:.1f} wv={wv}: {e}")
    if spots_wv:
        avg = np.mean(spots_wv)
        spot_sizes[f"fov_{fov_frac:.1f}"] = float(avg)
        total_spot_sq += avg ** 2
        n_fields += 1

rms_overall = float(np.sqrt(total_spot_sq / max(n_fields, 1)))
print(f"[original] RMS spot (polychromatic, multi-field) = {rms_overall*1000:.1f} μm")
print(f"[original] Per-field spots: {spot_sizes}")

efl_mm = float(eval_lens.foclen) if hasattr(eval_lens, "foclen") else float("nan")
print(f"[original] EFL = {efl_mm:.2f} mm")

# ---- Determine sign pattern ----
sign_chars = []
for i in FREE_CURVATURE_INDICES:
    c_val = float(eval_lens.surfaces[i].c.item())
    sign_chars.append("+" if c_val >= 0 else "-")
sign_pattern = "".join(sign_chars)
print(f"[original] Sign pattern: [{sign_pattern}]")

# ---- Render raw PNG (same style as elite renders) ----
print("[original] Rendering lens diagram…")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
eval_lens.draw_lens_2d(ax=ax, fig=fig, color="black")
for fov_frac, color in [(0.0, "#2196F3"), (0.5, "#4CAF50"), (1.0, "#F44336")]:
    try:
        ray = eval_lens.sample_point_source_2D(
            fov=fov_frac * rfov, depth=DEPTH_MM, num_rays=11, wvln=WAVELENGTH_UM,
        )
        ray.prop_to(-1.0)
        _, rec = eval_lens.trace2sensor(ray=ray, record=True)
        eval_lens.draw_ray_2d(rec, ax=ax, fig=fig, color=color)
    except Exception as e:
        print(f"  [warn] ray draw failed fov_frac={fov_frac}: {e}")

title = (
    f"ORIGINAL_ef50 RAW | fitness={-rms_overall:.6f}  "
    f"RMS={rms_overall*1000:.1f}μm  EFL={efl_mm:.1f}mm  [{sign_pattern}]"
)
ax.set_title(title, fontsize=10)
ax.set_xlabel("z (mm)")
ax.set_ylabel("y (mm)")
fig.tight_layout()
raw_png = OUT_DIR / "ORIGINAL_ef50_raw.png"
fig.savefig(raw_png, dpi=150)
plt.close(fig)
print(f"[original] Saved: {raw_png}")

# ---- L-BFGS Refinement (same as runner) ----
print("[original] Running L-BFGS refinement…")
from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem
from optical_rl.qd.lensmega.config.loader import load_config as load_lensmega_config
import copy as _copy

cfg = load_lensmega_config(RUN_DIR / "config.yaml")
refine_cfg = _copy.deepcopy(cfg)
io_cfg = cfg.io
if io_cfg.refine_efl_penalty_weight is not None:
    refine_cfg.problem.efl_penalty_weight = io_cfg.refine_efl_penalty_weight
if io_cfg.refine_invalid_penalty_weight is not None:
    refine_cfg.problem.invalid_penalty_weight = io_cfg.refine_invalid_penalty_weight
if io_cfg.refine_intersect_penalty_weight is not None:
    refine_cfg.problem.intersect_penalty_weight = io_cfg.refine_intersect_penalty_weight
if io_cfg.refine_field_weights is not None:
    refine_cfg.problem.field_weights = tuple(io_cfg.refine_field_weights)
desc = _build_descriptor(refine_cfg)
prob = _build_problem(refine_cfg, desc)
prob._ensure_backend()

# Theta bounds from config
theta_bounds = list(cfg.problem.theta_bounds)
lo = np.array([b[0] for b in theta_bounds], dtype=np.float64)
hi = np.array([b[1] for b in theta_bounds], dtype=np.float64)

theta_t = torch.tensor(
    theta.copy(), dtype=torch.float32, device="cpu", requires_grad=True,
)
optimizer = torch.optim.LBFGS(
    [theta_t], lr=io_cfg.refine_lbfgs_lr, max_iter=10,
    line_search_fn="strong_wolfe",
)
lo_t = torch.tensor(lo, dtype=torch.float32)
hi_t = torch.tensor(hi, dtype=torch.float32)
converge_tol = io_cfg.refine_converge_tol

best_loss = float("inf")
best_theta = theta.copy()
lbfgs_steps = 0
converged = False
final_gnorm = float("inf")
no_improve_count = 0

for step in range(io_cfg.refine_lbfgs_max_steps):
    def closure():
        optimizer.zero_grad()
        with torch.no_grad():
            theta_t.clamp_(lo_t, hi_t)
        fitness, _, _ = prob._fitness_and_metrics(theta_t)
        loss = -fitness
        if loss.requires_grad:
            loss.backward()
        return loss

    loss_val = optimizer.step(closure)
    lbfgs_steps = step + 1

    if loss_val is not None:
        lv = float(loss_val)
        if lv < best_loss - 1e-8:
            best_loss = lv
            no_improve_count = 0
            with torch.no_grad():
                best_theta = theta_t.detach().cpu().numpy().copy()
        else:
            no_improve_count += 1

    if theta_t.grad is not None:
        final_gnorm = float(theta_t.grad.norm().item())
        if final_gnorm < converge_tol:
            converged = True
            break

    if no_improve_count >= 5:
        converged = True
        break

best_theta = np.clip(best_theta, lo, hi)
refined_fitness = -best_loss
raw_fitness = -rms_overall

# Evaluate refined design
refined_eval = prob.evaluate(best_theta, with_grads=False)
ref_rms = refined_eval.metrics.get("rms_spot", abs(refined_fitness))
ref_efl = refined_eval.metrics.get("efl_actual", 0.0)

conv_str = "CONVERGED" if converged else "max_steps"
print(
    f"[original-refine] raw={raw_fitness:.6f} → refined={refined_fitness:.6f} "
    f"RMS={ref_rms*1000:.1f}μm EFL={ref_efl:.1f}mm "
    f"({lbfgs_steps} steps, {conv_str}, gnorm={final_gnorm:.2e})"
)

# ---- Basin stability analysis (same as runner) ----
print("[original] Running stability analysis…")
n_stab = io_cfg.refine_stability_samples
perturb_scale = io_cfg.refine_stability_perturbation
bounds_range = hi - lo
rng = np.random.RandomState(42)
fitness_deltas = []
for _ in range(n_stab):
    noise = rng.randn(len(best_theta)) * perturb_scale * bounds_range
    perturbed = np.clip(best_theta + noise, lo, hi)
    pr = prob.evaluate(perturbed, with_grads=False)
    fitness_deltas.append(pr.fitness - refined_fitness)

fitness_deltas = np.array(fitness_deltas)
stability = {
    "perturbation_scale": perturb_scale,
    "n_samples": n_stab,
    "fitness_delta_mean": round(float(fitness_deltas.mean()), 8),
    "fitness_delta_std": round(float(fitness_deltas.std()), 8),
    "fitness_delta_worst": round(float(fitness_deltas.min()), 8),
    "pct_survived": round(
        float((fitness_deltas > -abs(refined_fitness)).mean()) * 100, 1
    ),
    "sensitivity": round(float(fitness_deltas.std()), 8),
}
print(
    f"[original-stability] sensitivity={stability['sensitivity']:.6f}  "
    f"survived={stability['pct_survived']:.0f}%"
)

# ---- Per-element fingerprint on REFINED design ----
print("[original] Computing refined Seidel fingerprint…")
template2, _base2 = load_lens_template(
    LENS_JSON,
    wavelength_um=WAVELENGTH_UM,
    stop_surface_index=STOP_INDEX,
    free_curvature_indices=FREE_CURVATURE_INDICES,
    derived_curvature_index=DERIVED_CURVATURE_INDEX,
    device="cpu",
)
lens2 = GeoLens(filename=LENS_JSON, device="cpu")
torch.set_default_dtype(torch.float64)

_apply_theta_full_dof(
    lens2, best_theta,
    template=template2,
    free_curvature_indices=list(FREE_CURVATURE_INDICES),
    free_gap_indices=list(FREE_GAP_INDICES),
    variable_flange=VARIABLE_FLANGE,
    device="cpu",
)

fp_refined = compute_per_element_seidel(
    lens2, template2,
    field_y_deg=descriptor_field_deg,
    wavelength_um=WAVELENGTH_UM,
    device="cpu",
)

# ---- Render refined PNG (same format as elite _refined.png) ----
ref_signs = "".join(
    "+" if v > 0 else "-" for v in np.sign(best_theta[:6])
)
fig2, ax2 = plt.subplots(figsize=(10, 4), dpi=150)
lens2.draw_lens_2d(ax=ax2, fig=fig2, color="black")
ref_rfov = float(np.rad2deg(float(lens2.rfov)))
for fov_frac, color in [(0.0, "#2196F3"), (0.5, "#4CAF50"), (1.0, "#F44336")]:
    try:
        ray = lens2.sample_point_source_2D(
            fov=fov_frac * ref_rfov, depth=DEPTH_MM, num_rays=11, wvln=WAVELENGTH_UM,
        )
        ray.prop_to(-1.0)
        _, rec = lens2.trace2sensor(ray=ray, record=True)
        lens2.draw_ray_2d(rec, ax=ax2, fig=fig2, color=color)
    except Exception as e:
        print(f"  [warn] refined ray draw failed fov_frac={fov_frac}: {e}")

sens = stability['sensitivity']
surv = stability['pct_survived']
ref_title = (
    f"ORIGINAL_ef50 REFINED | fitness={refined_fitness:.6f}  "
    f"RMS={ref_rms*1000:.1f}μm  EFL={ref_efl:.1f}mm  [{ref_signs}]  "
    f"stability={sens:.6f}  survived={surv:.0f}%"
)
ax2.set_title(ref_title, fontsize=10)
ax2.set_xlabel("z (mm)")
ax2.set_ylabel("y (mm)")
fig2.tight_layout()
refined_png = OUT_DIR / "ORIGINAL_ef50_refined.png"
fig2.savefig(refined_png, dpi=150)
plt.close(fig2)
print(f"[original] Saved: {refined_png}")

# ---- Save fingerprint JSON ----
info = {
    "eval": "original",
    "elite_index": "ORIGINAL",
    "is_new_best": False,
    "fitness": -rms_overall,
    "rms_mm": rms_overall,
    "rms_um": f"{rms_overall*1000:.1f}",
    "efl_mm": f"{efl_mm:.2f}",
    "sign_pattern": sign_pattern,
    "measures": measures,
    "metrics": {
        "rms_spot": rms_overall,
        "efl_actual": efl_mm,
        "m0_S2_coma": measures[0],
        "m1_S3_astigmatism": measures[1],
    },
    "spot_sizes": spot_sizes,
    "theta": theta.tolist(),
    "fingerprint_raw": fingerprint,
    "refined": {
        "theta": [float(v) for v in best_theta],
        "fitness": float(refined_fitness),
        "rms_mm": float(ref_rms),
        "rms_um": f"{ref_rms * 1000:.1f}",
        "efl_mm": f"{ref_efl:.2f}",
        "lbfgs_steps": lbfgs_steps,
        "converged": converged,
        "final_grad_norm": round(float(final_gnorm), 8),
        "improvement_ratio": round(
            float(raw_fitness / refined_fitness)
            if abs(refined_fitness) > 1e-12 else 0.0,
            4,
        ),
        "raw_fitness": float(raw_fitness),
        "stability": stability,
        "metrics": {
            k: float(v) for k, v in refined_eval.metrics.items()
        },
    },
    "fingerprint_refined": fp_refined,
}
fp_path = OUT_DIR / "ORIGINAL_ef50_fingerprint.json"
fp_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
print(f"[original] Saved: {fp_path}")

# ---- Summary table ----
print("\n" + "=" * 70)
print("ORIGINAL EF50 f/1.8 — Seidel Fingerprint Summary")
print("=" * 70)
print(f"  EFL:            {efl_mm:.2f} mm")
print(f"  RMS spot:       {rms_overall*1000:.1f} μm")
print(f"  Sign pattern:   [{sign_pattern}]")
print(f"  S2 (coma):      {measures[0]:.8e}")
print(f"  S3 (astigmatism): {measures[1]:.8e}")
print()
print("  Per-element Seidel breakdown:")
print(f"  {'Element':<8} {'S1_sph':>12} {'S2_coma':>12} {'S3_astig':>12} {'S4_petz':>12} {'S5_dist':>12}")
print("  " + "-" * 68)
for elem in fingerprint["elements"]:
    print(
        f"  {elem['name']:<8} "
        f"{elem['S1_spherical']:>12.6e} "
        f"{elem['S2_coma']:>12.6e} "
        f"{elem['S3_astigmatism']:>12.6e} "
        f"{elem['S4_petzval']:>12.6e} "
        f"{elem['S5_distortion']:>12.6e}"
    )
print("  " + "-" * 68)
st = fingerprint["system_total"]
print(
    f"  {'TOTAL':<8} "
    f"{st['S1_spherical']:>12.6e} "
    f"{st['S2_coma']:>12.6e} "
    f"{st['S3_astigmatism']:>12.6e} "
    f"{st['S4_petzval']:>12.6e} "
    f"{st['S5_distortion']:>12.6e}"
)
print("=" * 70)
print("[original] Done!")
