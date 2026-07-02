"""Top-5 diverse elites: L-BFGS refine + clear-aperture prune + render.

Replicates the runner's geometric-diversity selection from a finished
sixlet/cooke/septlet run, then for each picked design:
  1. L-BFGS refine using the same refine_* knobs as the in-flight pipeline.
  2. Apply refined theta to a GeoLens.
  3. Call lens.prune_surf() to shrink clear apertures to ray-traced extent.
  4. Render a 2D ray-trace PNG.
  5. Save lens JSON (post-prune) + a per-design metadata JSON.

Usage:
    python refine_prune_render_top5_diverse.py <run_dir> [--top-n 50] [--pick-k 5]
"""
from __future__ import annotations
import argparse
import copy
import json
import pickle
from pathlib import Path
import numpy as np
import torch

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from deeplens.optics import GeoLens


def select_diverse(cells, problem_cfg, top_n=50, pick_k=5):
    all_designs = sorted(
        cells.items(), key=lambda kr: kr[1].fitness, reverse=True
    )
    candidates = all_designs[: min(top_n, len(all_designs))]
    if len(candidates) < pick_k:
        return candidates
    nc = len(problem_cfg.free_curvature_indices)
    ng = len(problem_cfg.free_gap_indices)
    geo_dim = nc + ng
    bounds = np.array(problem_cfg.theta_bounds[:geo_dim], dtype=np.float64)
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    geo_vecs = np.array([r.theta[:geo_dim] for _, r in candidates])
    geo_norm = (geo_vecs - lo) / span
    selected_idx = [0]
    min_dists = np.full(len(candidates), np.inf)
    for _ in range(pick_k - 1):
        last = geo_norm[selected_idx[-1]]
        d = np.linalg.norm(geo_norm - last, axis=1)
        min_dists = np.minimum(min_dists, d)
        copy_d = min_dists.copy()
        for si in selected_idx:
            copy_d[si] = -1.0
        selected_idx.append(int(np.argmax(copy_d)))
    return [candidates[i] for i in selected_idx]


def make_refine_problem(cfg):
    rc = copy.deepcopy(cfg)
    io = cfg.io
    if io.refine_efl_penalty_weight is not None:
        rc.problem.efl_penalty_weight = io.refine_efl_penalty_weight
    if io.refine_invalid_penalty_weight is not None:
        rc.problem.invalid_penalty_weight = io.refine_invalid_penalty_weight
    if io.refine_intersect_penalty_weight is not None:
        rc.problem.intersect_penalty_weight = io.refine_intersect_penalty_weight
    if io.refine_field_weights is not None:
        rc.problem.field_weights = tuple(io.refine_field_weights)
    if rc.problem.enforce_fnum and io.refine_fnum_penalty_weight is not None:
        rc.problem.fnum_penalty_weight = io.refine_fnum_penalty_weight
    desc = _build_descriptor(rc)
    prob = _build_problem(rc, desc)
    prob._ensure_backend()
    return prob


def refine_lbfgs(theta, prob, io, lo, hi):
    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)
    theta_t = torch.tensor(
        theta.copy(), dtype=torch.float32, requires_grad=True,
    )
    opt = torch.optim.LBFGS(
        [theta_t], lr=io.refine_lbfgs_lr, max_iter=10,
        line_search_fn="strong_wolfe",
    )
    best_loss = float("inf")
    best_theta = theta.copy()
    no_improve = 0
    for step in range(io.refine_lbfgs_max_steps):
        def closure():
            opt.zero_grad()
            with torch.no_grad():
                theta_t.clamp_(lo_t, hi_t)
            fit, _, _ = prob._fitness_and_metrics(theta_t)
            loss = -fit
            if loss.requires_grad:
                loss.backward()
            return loss
        loss_val = opt.step(closure)
        if loss_val is not None:
            lv = float(loss_val)
            if lv < best_loss - 1e-8:
                best_loss = lv
                no_improve = 0
                with torch.no_grad():
                    best_theta = theta_t.detach().cpu().numpy().copy()
            else:
                no_improve += 1
        if theta_t.grad is not None:
            if float(theta_t.grad.norm()) < io.refine_converge_tol:
                break
        if no_improve >= 5:
            break
    return np.clip(best_theta, lo, hi), -best_loss, step + 1


def build_lens(theta, cfg, device="cpu"):
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
    return lens


def render_lens(lens, filepath, title):
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
    ax.set_xlabel("z (mm)"); ax.set_ylabel("y (mm)")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--top-n", type=int, default=50, help="Pool to pick diverse from")
    ap.add_argument("--pick-k", type=int, default=5, help="How many diverse designs")
    ap.add_argument("--prune-expand", type=float, default=None,
                    help="prune_surf expand_factor (default: auto, 5%/10% by r_sensor)")
    ap.add_argument("--prune-margin", type=float, default=None,
                    help="prune_surf mounting_margin in mm (overrides expand_factor)")
    ap.add_argument("--prune-first", action="store_true",
                    help="Prune CAs BEFORE refinement (lets L-BFGS optimize "
                         "within smaller CAs — intersect penalty becomes much "
                         "less aggressive but invalid_fraction may rise)")
    ap.add_argument("--prune-shrink-fraction", type=float, default=None,
                    help="Prune-first only: shrink each CA by this fraction of "
                         "its current value (e.g. 0.3 = keep 70%%). Overrides "
                         "ray-traced prune_surf for the pre-refine step.")
    ap.add_argument("--four-stage", action="store_true",
                    help="4-stage pipeline: shrink-fixed -> refine -> "
                         "prune-real -> refine. Most thorough mode.")
    ap.add_argument("--stage1-shrink", type=float, default=0.30,
                    help="Four-stage: fraction to shrink CAs in stage 1 (default 0.30)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / f"top{args.pick_k}_refined_pruned"
    out_dir.mkdir(exist_ok=True)
    print(f"[top{args.pick_k}] run_dir = {run_dir}")
    print(f"[top{args.pick_k}] out_dir = {out_dir}")

    cfg = load_config(str(run_dir / "config.yaml"))
    archive = pickle.load(open(run_dir / "archive.pkl", "rb"))
    print(f"[top{args.pick_k}] archive cells: {len(archive.cells)}")

    selected = select_diverse(
        archive.cells, cfg.problem,
        top_n=args.top_n, pick_k=args.pick_k,
    )
    print(f"[top{args.pick_k}] picked {len(selected)} diverse designs from top {args.top_n}:")
    for i, (cell, ev) in enumerate(selected):
        print(f"  #{i+1}  cell={cell}  raw_fit={ev.fitness:.4f}  "
              f"S2={ev.measures[0]:.6f}  S3={ev.measures[1]:.6f}")

    theta_bounds = list(cfg.problem.theta_bounds)
    lo = np.array([b[0] for b in theta_bounds], dtype=np.float64)
    hi = np.array([b[1] for b in theta_bounds], dtype=np.float64)
    prob = make_refine_problem(cfg)

    summary = []
    for i, (cell, ev) in enumerate(selected):
        tag = f"top{i+1:02d}_cell_{cell[0]}_{cell[1]}"
        print(f"\n=== [{tag}] raw_fit={ev.fitness:.6f} ===")
        theta_raw = np.array(ev.theta, dtype=np.float64)

        if args.four_stage:
            # ---- 4-stage pipeline ----------------------------------------
            # 1. shrink CAs by fixed fraction (drops intersect penalty floor)
            # 2. L-BFGS refine within the constrained CAs
            # 3. ray-traced prune (true CAs needed by the now-refined design)
            # 4. L-BFGS refine again with the correct CAs
            print(f"  [stage 1/4] fixed-shrink CAs by {args.stage1_shrink:.0%}")
            lens_s1 = build_lens(theta_raw, cfg, device="cpu")
            ca_orig = [float(s.r) for s in lens_s1.surfaces]
            for s in lens_s1.surfaces:
                s.r = float(s.r) * (1.0 - args.stage1_shrink)
            ca_s1 = [float(s.r) for s in lens_s1.surfaces]
            tmp1 = out_dir / f"{tag}_s1_shrunk.json"
            lens_s1.write_lens_json(str(tmp1))
            cfg1 = copy.deepcopy(cfg); cfg1.problem.lens_path = str(tmp1)
            prob1 = make_refine_problem(cfg1)

            print(f"  [stage 2/4] L-BFGS refine within shrunk CAs")
            theta_s2, fit_s2, n_s2 = refine_lbfgs(theta_raw, prob1, cfg.io, lo, hi)
            ev_s2 = prob1.evaluate(theta_s2, with_grads=False)
            rms_s2 = ev_s2.metrics.get("rms_spot", abs(fit_s2)) * 1000.0
            inv_s2 = ev_s2.metrics.get("invalid_fraction", -1)
            print(f"     -> fit={fit_s2:.4f}  RMS={rms_s2:.1f}μm  invalid={inv_s2:.3f}  ({n_s2} steps)")
            lens_s2 = build_lens(theta_s2, cfg1, device="cpu")
            render_lens(lens_s2, out_dir / f"{tag}_s2_refined1.png",
                        title=f"{tag} | s2: shrink+refine  fit={fit_s2:.4f}  RMS={rms_s2:.1f}μm")

            print(f"  [stage 3/4] ray-traced prune_surf at refined design")
            try:
                kw = {}
                if args.prune_expand is not None: kw["expand_factor"] = args.prune_expand
                if args.prune_margin is not None: kw["mounting_margin"] = args.prune_margin
                lens_s2.prune_surf(**kw)
                ca_s3 = [float(s.r) for s in lens_s2.surfaces]
            except Exception as e:
                print(f"     prune FAILED: {e} — falling back to s2 CAs")
                ca_s3 = ca_s1
            print(f"     CA trajectory: orig={[round(x,1) for x in ca_orig]}")
            print(f"                    s1  ={[round(x,1) for x in ca_s1]}")
            print(f"                    s3  ={[round(x,1) for x in ca_s3]}")
            tmp3 = out_dir / f"{tag}_s3_pruned.json"
            lens_s2.write_lens_json(str(tmp3))
            cfg3 = copy.deepcopy(cfg); cfg3.problem.lens_path = str(tmp3)
            prob3 = make_refine_problem(cfg3)

            print(f"  [stage 4/4] L-BFGS refine within ray-traced CAs")
            theta_ref, ref_fit, n_steps = refine_lbfgs(theta_s2, prob3, cfg.io, lo, hi)
            ref_eval = prob3.evaluate(theta_ref, with_grads=False)
            ref_rms_um = ref_eval.metrics.get("rms_spot", abs(ref_fit)) * 1000.0
            inv_final = ref_eval.metrics.get("invalid_fraction", -1)
            print(f"     -> fit={ref_fit:.4f}  RMS={ref_rms_um:.1f}μm  invalid={inv_final:.3f}  ({n_steps} steps)")
            lens = build_lens(theta_ref, cfg3, device="cpu")
            render_lens(lens, out_dir / f"{tag}_s4_final.png",
                        title=f"{tag} | s4: 4-stage final  fit={ref_fit:.4f}  RMS={ref_rms_um:.1f}μm")
            ca_pre, ca_post = ca_orig, ca_s3
            pruned_ok = True

        elif args.prune_first:
            # Build lens at RAW theta, prune CAs, then mutate prob's lens to
            # reflect the new CAs before L-BFGS.  The smaller CAs make the
            # intersection penalty much less aggressive (it samples SAG at
            # 0.5*r..1.0*r, so smaller r => smaller sag samples).
            lens_raw = build_lens(theta_raw, cfg, device="cpu")
            ca_pre = [float(s.r) for s in lens_raw.surfaces]
            if args.prune_shrink_fraction is not None:
                shrink = args.prune_shrink_fraction
                for s in lens_raw.surfaces:
                    s.r = float(s.r) * (1.0 - shrink)
                ca_post = [float(s.r) for s in lens_raw.surfaces]
                print(f"  prune-first (fixed shrink {shrink:.0%}): "
                      f"CAs {ca_pre} -> {ca_post}")
            else:
                kw = {}
                if args.prune_expand is not None: kw["expand_factor"] = args.prune_expand
                if args.prune_margin is not None: kw["mounting_margin"] = args.prune_margin
                lens_raw.prune_surf(**kw)
                ca_post = [float(s.r) for s in lens_raw.surfaces]
                print(f"  prune-first (ray-traced): CAs {ca_pre} -> {ca_post}")
            # Push the new CAs into the problem's underlying lens so L-BFGS
            # sees them.  prob._lens (or similar) gets rebuilt per-eval from
            # the lens_path + theta, so we must overwrite the JSON-derived
            # CAs by patching the template.  Easiest: use these CAs as the
            # base for _apply_theta_generic via problem._template.
            # Note: _apply_theta_generic builds a fresh GeoLens each call from
            # cfg.problem.lens_path. To make pre-pruned CAs persist through
            # refinement, write a temporary lens JSON with the new CAs and
            # override the cfg's lens_path for this design only.
            tmp_lens_json = out_dir / f"{tag}_pre_pruned_lens.json"
            lens_raw.write_lens_json(str(tmp_lens_json))
            cfg_design = copy.deepcopy(cfg)
            cfg_design.problem.lens_path = str(tmp_lens_json)
            prob_design = make_refine_problem(cfg_design)
            theta_ref, ref_fit, n_steps = refine_lbfgs(theta_raw, prob_design, cfg.io, lo, hi)
            ref_eval = prob_design.evaluate(theta_ref, with_grads=False)
            ref_rms_um = ref_eval.metrics.get("rms_spot", abs(ref_fit)) * 1000.0
            print(f"  refine (post-prune): {n_steps} L-BFGS steps -> fit={ref_fit:.6f}  "
                  f"RMS={ref_rms_um:.2f}μm  invalid_frac={ref_eval.metrics.get('invalid_fraction', -1):.3f}")
            lens = build_lens(theta_ref, cfg_design, device="cpu")
            render_lens(lens, out_dir / f"{tag}_pruned_then_refined.png",
                        title=f"{tag} | prune->refine  fit={ref_fit:.4f}  RMS={ref_rms_um:.1f}μm")
            pruned_ok = True
        else:
            # ORIGINAL: refine first, then prune for cosmetics
            theta_ref, ref_fit, n_steps = refine_lbfgs(theta_raw, prob, cfg.io, lo, hi)
            ref_eval = prob.evaluate(theta_ref, with_grads=False)
            ref_rms_um = ref_eval.metrics.get("rms_spot", abs(ref_fit)) * 1000.0
            print(f"  refine: {n_steps} L-BFGS steps -> fit={ref_fit:.6f}  "
                  f"RMS={ref_rms_um:.2f} μm")
            lens = build_lens(theta_ref, cfg, device="cpu")
            ca_pre = [float(s.r) for s in lens.surfaces]
            render_lens(lens, out_dir / f"{tag}_refined.png",
                        title=f"{tag} | refined  fit={ref_fit:.4f}  RMS={ref_rms_um:.1f}μm")
            try:
                kw = {}
                if args.prune_expand is not None: kw["expand_factor"] = args.prune_expand
                if args.prune_margin is not None: kw["mounting_margin"] = args.prune_margin
                lens.prune_surf(**kw)
                ca_post = [float(s.r) for s in lens.surfaces]
                pruned_ok = True
                print(f"  prune: CA reductions (mm): "
                      f"{[round(a-b,2) for a,b in zip(ca_pre, ca_post)]}")
            except Exception as e:
                print(f"  prune FAILED: {e}")
                ca_post = ca_pre
                pruned_ok = False
            render_lens(lens, out_dir / f"{tag}_refined_pruned.png",
                        title=f"{tag} | refined+pruned  fit={ref_fit:.4f}  RMS={ref_rms_um:.1f}μm")

        # 6. Save lens JSON (post-prune) + metadata
        try:
            lens.write_lens_json(str(out_dir / f"{tag}_lens.json"))
        except Exception as e:
            print(f"  write_lens_json failed: {e}")

        meta = {
            "tag": tag,
            "cell": list(cell),
            "raw_fitness": float(ev.fitness),
            "refined_fitness": float(ref_fit),
            "refined_rms_um": float(ref_rms_um),
            "lbfgs_steps": int(n_steps),
            "ca_before_prune_mm": ca_pre,
            "ca_after_prune_mm": ca_post,
            "pruned_ok": pruned_ok,
            "theta_raw": theta_raw.tolist(),
            "theta_refined": theta_ref.tolist(),
            "measures": list(ev.measures),
        }
        json.dump(meta, open(out_dir / f"{tag}_meta.json", "w"), indent=2)
        summary.append(meta)

    json.dump({"top_k": args.pick_k, "designs": summary},
              open(out_dir / "summary.json", "w"), indent=2)
    print(f"\n[top{args.pick_k}] done. outputs in {out_dir}")
    print(f"  *_refined.png         (post-LBFGS, pre-prune)")
    print(f"  *_refined_pruned.png  (post-LBFGS, post-prune)")
    print(f"  *_lens.json           (final pruned lens)")
    print(f"  *_meta.json           (per-design stats)")
    print(f"  summary.json          (all 5 in one file)")


if __name__ == "__main__":
    main()
