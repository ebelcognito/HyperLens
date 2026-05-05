"""Post-hoc LBFGS refinement of all elite archive cells in a finished run.

For every archive cell with raw fitness above the run's elite threshold that
does NOT already have a refined JSON, runs the same LBFGS + stability pipeline
as the in-flight `_EliteRenderWorker._render`, writes a JSON, and (only if the
refined RMS is below the threshold below) writes raw + refined PNG renders.

Usage:
    python refine_posthoc.py <run_dir> [--rms-threshold-um 25.0]
"""
from __future__ import annotations
import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/hisham26/Desktop/RL")
sys.path.insert(0, "/home/hisham26/Desktop/RL/DeepLens")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem
from optical_rl.qd.lensmega.descriptors.seidel import compute_per_element_seidel
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from deeplens.optics import GeoLens


def render_png(lens, filepath, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    lens.draw_lens_2d(ax=ax, fig=fig, color="black")
    rfov = float(np.rad2deg(float(lens.rfov))) if hasattr(lens, "rfov") else 14.0
    for fov_frac, color in [(0.0, "#2196F3"), (0.5, "#4CAF50"), (1.0, "#F44336")]:
        try:
            ray = lens.sample_point_source_2D(
                fov=fov_frac * rfov, depth=-20000.0, num_rays=11, wvln=0.587,
            )
            ray.prop_to(-1.0)
            _, rec = lens.trace2sensor(ray=ray, record=True)
            lens.draw_ray_2d(rec, ax=ax, fig=fig, color=color)
        except Exception:
            pass
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("y (mm)")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def build_lens_from_theta(theta, cfg, device="cpu"):
    p = cfg.problem
    template, _ = load_lens_template(
        p.lens_path,
        wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        device=device,
    )
    solver = GeneralizedParaxialSolver(template)
    lens = GeoLens(filename=p.lens_path, device=device)
    _apply_theta_generic(
        lens, np.asarray(theta, dtype=float),
        template=template, solver=solver,
        focal_length_target_mm=p.focal_length_target_mm,
        device=device,
        parameterization=p.parameterization,
        free_curvature_indices=p.free_curvature_indices,
        free_gap_indices=p.free_gap_indices,
        variable_flange=p.variable_flange,
        variable_materials=p.variable_materials,
        free_material_indices=p.free_material_indices,
    )
    return lens, template


def make_refine_problem(cfg):
    refine_cfg = copy.deepcopy(cfg)
    io = cfg.io
    if io.refine_efl_penalty_weight is not None:
        refine_cfg.problem.efl_penalty_weight = io.refine_efl_penalty_weight
    if io.refine_invalid_penalty_weight is not None:
        refine_cfg.problem.invalid_penalty_weight = io.refine_invalid_penalty_weight
    if io.refine_intersect_penalty_weight is not None:
        refine_cfg.problem.intersect_penalty_weight = io.refine_intersect_penalty_weight
    if io.refine_field_weights is not None:
        refine_cfg.problem.field_weights = tuple(io.refine_field_weights)
    if (refine_cfg.problem.enforce_fnum
            and io.refine_fnum_penalty_weight is not None):
        refine_cfg.problem.fnum_penalty_weight = io.refine_fnum_penalty_weight
    desc = _build_descriptor(refine_cfg)
    prob = _build_problem(refine_cfg, desc)
    prob._ensure_backend()
    return prob


def refine_one(theta, prob, io, lo, hi):
    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)
    theta_t = torch.tensor(
        theta.copy(), dtype=torch.float32, device="cpu", requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        [theta_t], lr=io.refine_lbfgs_lr, max_iter=10,
        line_search_fn="strong_wolfe",
    )
    converge_tol = io.refine_converge_tol
    best_loss = float("inf")
    best_theta = theta.copy()
    lbfgs_steps = 0
    converged = False
    final_gnorm = float("inf")
    no_improve_count = 0
    for step in range(io.refine_lbfgs_max_steps):
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
    return best_theta, refined_fitness, lbfgs_steps, converged, final_gnorm


def stability_analysis(prob, best_theta, refined_fitness, lo, hi, io):
    n_stab = io.refine_stability_samples
    perturb_scale = io.refine_stability_perturbation
    bounds_range = hi - lo
    rng = np.random.RandomState(42)
    deltas = []
    for _ in range(n_stab):
        noise = rng.randn(len(best_theta)) * perturb_scale * bounds_range
        perturbed = np.clip(best_theta + noise, lo, hi)
        pr = prob.evaluate(perturbed, with_grads=False)
        deltas.append(pr.fitness - refined_fitness)
    deltas = np.array(deltas)
    return {
        "perturbation_scale": perturb_scale,
        "n_samples": n_stab,
        "fitness_delta_mean": round(float(deltas.mean()), 8),
        "fitness_delta_std": round(float(deltas.std()), 8),
        "fitness_delta_worst": round(float(deltas.min()), 8),
        "pct_survived": round(
            float((deltas > -abs(refined_fitness)).mean()) * 100, 1
        ),
        "sensitivity": round(float(deltas.std()), 8),
    }


def existing_refined_thetas(elite_dir: Path):
    """Load (theta_array) for every JSON that already has refinement, used
    to skip cells whose raw or refined theta is already on disk."""
    out = []
    for p in sorted(elite_dir.glob("*.json")):
        try:
            d = json.load(p.open())
        except Exception:
            continue
        ref = d.get("refined", {})
        if isinstance(ref, dict) and "theta" in ref:
            out.append(np.array(d.get("theta", []), dtype=np.float64))
    return out


def already_done(theta_raw, prior_thetas, atol=1e-6):
    if not prior_thetas: return False
    for prev in prior_thetas:
        if prev.shape == theta_raw.shape and np.allclose(prev, theta_raw, atol=atol):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--rms-threshold-um", type=float, default=25.0,
                    help="Render PNGs only when refined RMS is below this (μm).")
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    elite_dir = run_dir / "elite_designs"
    elite_dir.mkdir(exist_ok=True)
    cfg = load_config(str(run_dir / "config.yaml"))
    io = cfg.io
    elite_thr = float(io.elite_fitness_threshold) if io.elite_fitness_threshold is not None else -0.3

    print(f"[posthoc] run_dir={run_dir}")
    print(f"[posthoc] elite threshold (raw fit) = {elite_thr}")
    print(f"[posthoc] refined-RMS threshold for rendering = {args.rms_threshold_um} μm")

    archive = pickle.load(open(run_dir / "archive.pkl", "rb"))
    print(f"[posthoc] archive cells: {len(archive.cells)}")

    elite_cells = [
        (k, v) for k, v in archive.cells.items() if v.fitness > elite_thr
    ]
    elite_cells.sort(key=lambda kv: -kv[1].fitness)
    print(f"[posthoc] elite cells (fit > {elite_thr}): {len(elite_cells)}")

    prior = existing_refined_thetas(elite_dir)
    print(f"[posthoc] existing JSONs with refined data: {len(prior)}")

    theta_bounds = list(cfg.problem.theta_bounds)
    lo = np.array([b[0] for b in theta_bounds], dtype=np.float64)
    hi = np.array([b[1] for b in theta_bounds], dtype=np.float64)

    prob = make_refine_problem(cfg)

    n_done = n_skip = n_render = n_norender = n_err = 0
    t0 = time.time()
    for idx, ((row, col), ev) in enumerate(elite_cells):
        if args.max_cells is not None and idx >= args.max_cells:
            break
        theta_raw = np.array(ev.theta, dtype=np.float64)
        tag = f"posthoc_{idx:04d}_r{row}_c{col}"

        if already_done(theta_raw, prior):
            n_skip += 1
            continue

        try:
            best_theta, ref_fit, lb_steps, converged, gnorm = refine_one(
                theta_raw, prob, io, lo, hi
            )
            ref_eval = prob.evaluate(best_theta, with_grads=False)
            ref_rms_mm = ref_eval.metrics.get("rms_spot", abs(ref_fit))
            ref_efl = ref_eval.metrics.get("efl_actual", 0.0)
            stability = stability_analysis(prob, best_theta, ref_fit, lo, hi, io)

            raw_rms_mm = ev.metrics.get("rms_spot", abs(ev.fitness))
            info = {
                "tag": tag,
                "cell": [int(row), int(col)],
                "fitness": float(ev.fitness),
                "rms_mm": float(raw_rms_mm),
                "rms_um": f"{raw_rms_mm * 1000:.1f}",
                "efl_mm": f"{ev.metrics.get('efl_actual', 0):.2f}",
                "theta": [float(v) for v in theta_raw],
                "measures": [float(v) for v in ev.measures],
                "metrics": {k: float(v) for k, v in ev.metrics.items()},
                "refined": {
                    "theta": [float(v) for v in best_theta],
                    "fitness": float(ref_fit),
                    "rms_mm": float(ref_rms_mm),
                    "rms_um": f"{ref_rms_mm * 1000:.1f}",
                    "efl_mm": f"{ref_efl:.2f}",
                    "lbfgs_steps": int(lb_steps),
                    "converged": bool(converged),
                    "final_grad_norm": round(float(gnorm), 8),
                    "improvement_ratio": round(
                        float(ev.fitness / ref_fit) if abs(ref_fit) > 1e-12 else 0.0, 4
                    ),
                    "raw_fitness": float(ev.fitness),
                    "stability": stability,
                    "metrics": {k: float(v) for k, v in ref_eval.metrics.items()},
                },
            }

            ref_rms_um = ref_rms_mm * 1000.0
            if ref_rms_um < args.rms_threshold_um:
                # Build lenses + render PNGs
                try:
                    lens_raw, template_raw = build_lens_from_theta(theta_raw, cfg)
                    lens_ref, template_ref = build_lens_from_theta(best_theta, cfg)
                    raw_signs = "".join("+" if v > 0 else "-" for v in theta_raw[:12])
                    ref_signs = "".join("+" if v > 0 else "-" for v in best_theta[:12])
                    render_png(
                        lens_raw, elite_dir / f"{tag}_raw.png",
                        f"{tag} RAW | fitness={ev.fitness:.6f}  "
                        f"RMS={raw_rms_mm*1000:.1f}μm  EFL={ev.metrics.get('efl_actual',0):.1f}mm  "
                        f"[{raw_signs}]"
                    )
                    render_png(
                        lens_ref, elite_dir / f"{tag}_refined.png",
                        f"{tag} REFINED | fitness={ref_fit:.6f}  "
                        f"RMS={ref_rms_um:.1f}μm  EFL={ref_efl:.1f}mm  "
                        f"[{ref_signs}]  stability={stability['sensitivity']:.6f}  "
                        f"survived={stability['pct_survived']:.0f}%"
                    )
                    # Per-element fingerprint on refined
                    try:
                        info["fingerprint_refined"] = compute_per_element_seidel(
                            lens_ref, template_ref,
                            field_y_deg=7.0,
                            wavelength_um=cfg.problem.wavelength_um,
                            device="cpu",
                        )
                    except Exception as e:
                        info["fingerprint_refined"] = {"error": str(e)}
                    n_render += 1
                except Exception as e:
                    print(f"  [render-fail] {tag}: {e}", flush=True)
            else:
                n_norender += 1

            (elite_dir / f"{tag}.json").write_text(json.dumps(info, indent=2))
            n_done += 1

            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            print(
                f"[{idx+1:4d}/{len(elite_cells)}] cell=({row:>3},{col:>3}) "
                f"raw_fit={ev.fitness:.4f} -> ref_fit={ref_fit:.4f}  "
                f"raw_rms={raw_rms_mm*1000:6.1f}μm -> ref_rms={ref_rms_um:6.1f}μm  "
                f"{'RENDER' if ref_rms_um < args.rms_threshold_um else 'no-render':9s}  "
                f"({lb_steps} steps, {'CONVERGED' if converged else 'maxsteps'})  "
                f"rate={rate:.2f}/s",
                flush=True,
            )
        except Exception as e:
            print(f"  [refine-fail] {tag}: {e}", flush=True)
            n_err += 1

    print()
    print(f"[posthoc] DONE  refined={n_done}  rendered={n_render}  "
          f"no-render(>={args.rms_threshold_um}μm)={n_norender}  "
          f"skipped(prior)={n_skip}  errors={n_err}")
    print(f"[posthoc] wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
