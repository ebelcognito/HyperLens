"""Re-refine LensMEGA elites under a fixed-F# spec.

For every refined elite in a discovery run, this script:
  1. Loads the elite's refined theta as a starting point.
  2. Patches the problem's `_build_lens` so that after every lens build, the
     physical aperture-stop radius is rescaled so the paraxial entrance-pupil
     radius equals  EFL_target / (2 * F#_target).  Because the paraxial
     pupil-to-stop relation is linear (single ABCD-matrix product through
     the front group), one rescale per build is exact.
  3. Runs the same L-BFGS recipe as the original refinement.
  4. Saves a per-elite JSON with the F#-targeted refined parameters and
     diagnostics (final entrance-pupil radius, stop radius, F# achieved).

Usage:
    python refine_to_target_fnum.py
        [--run RUN_DIR] [--config CFG] [--fnum 8.0] [--out OUT_DIR]
"""
import argparse
import copy
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/hisham26/Desktop/RL")
sys.path.insert(0, "/home/hisham26/Desktop/RL/DeepLens")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem


DEFAULT_RUN = ("/home/hisham26/Desktop/RL/results/lensmega/"
               "cooke_triplet_materials_20260414_173935")
DEFAULT_CFG = ("/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
               "config/examples/cooke_triplet_materials.yaml")


def make_target_fnum_build_lens(prob, pupil_target_mm, template_r_stop):
    """Wrap prob._build_lens so the stop is rescaled to hit the target pupil.

    The relation is linear in paraxial: pupil_radius = M(theta) * r_stop, so
    setting r_stop *= (pupil_target / pupil_now) produces an exact one-shot fix.
    Resets r_stop to the template default before each rescale so successive
    elites don't drift off the previous one's stop value.
    """
    orig = prob._build_lens

    def patched(theta_t):
        lens, bfl_t = orig(theta_t)
        aper_idx = lens.aper_idx
        if aper_idx is None:
            return lens, bfl_t
        # Reset to template stop, then rescale.
        lens.surfaces[aper_idx].r = float(template_r_stop)
        with torch.no_grad():
            _, pupilr_now = lens.calc_entrance_pupil(paraxial=True)
        pupilr_now = float(pupilr_now)
        if not np.isfinite(pupilr_now) or pupilr_now <= 1e-6:
            return lens, bfl_t
        lens.surfaces[aper_idx].r = float(template_r_stop) * (pupil_target_mm / pupilr_now)
        return lens, bfl_t

    prob._build_lens = patched
    prob._unpatched_build_lens = orig
    return prob


def build_refine_problem(cfg, fnum_target, efl_target):
    """Build a fresh problem with refinement penalties + F# patch."""
    refine_cfg = copy.deepcopy(cfg)
    p = refine_cfg.problem
    io = refine_cfg.io
    if io.refine_efl_penalty_weight is not None:
        p.efl_penalty_weight = io.refine_efl_penalty_weight
    if io.refine_invalid_penalty_weight is not None:
        p.invalid_penalty_weight = io.refine_invalid_penalty_weight
    if io.refine_intersect_penalty_weight is not None:
        p.intersect_penalty_weight = io.refine_intersect_penalty_weight
    if io.refine_field_weights is not None:
        p.field_weights = tuple(io.refine_field_weights)

    desc = _build_descriptor(refine_cfg)
    prob = _build_problem(refine_cfg, desc)
    prob._ensure_backend()

    # Snapshot the template stop radius BEFORE patching so we can reset on
    # every build (otherwise the lens carries state across elites).
    aper_idx = prob._reusable_lens.aper_idx
    template_r_stop = float(prob._reusable_lens.surfaces[aper_idx].r)

    pupil_target_mm = efl_target / (2.0 * fnum_target)
    make_target_fnum_build_lens(prob, pupil_target_mm, template_r_stop)
    return prob, pupil_target_mm, template_r_stop


def lbfgs_refine(prob, theta0, cfg, max_steps=None, lr=None, tol=None):
    """Run the same L-BFGS schedule as the original refiner."""
    io = cfg.io
    p = cfg.problem
    max_steps = max_steps or io.refine_lbfgs_max_steps
    lr = lr or io.refine_lbfgs_lr
    tol = tol or io.refine_converge_tol

    bounds = p.theta_bounds
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)

    theta_t = torch.tensor(
        np.asarray(theta0, dtype=float).copy(),
        dtype=torch.float32, device="cpu", requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        [theta_t], lr=lr, max_iter=10, line_search_fn="strong_wolfe",
    )
    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)

    best_loss = float("inf")
    best_theta = np.asarray(theta0, dtype=float).copy()
    steps = 0
    converged = False
    final_gn = float("inf")
    no_improve = 0

    for step in range(max_steps):
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
        steps = step + 1

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
            final_gn = float(theta_t.grad.norm().item())
            if final_gn < tol:
                converged = True
                break

        if no_improve >= 5:
            converged = True
            break

    best_theta = np.clip(best_theta, lo, hi)
    return best_theta, -best_loss, steps, converged, final_gn


def evaluate_at(prob, theta):
    """Evaluate refined design and pull back rich diagnostics."""
    res = prob.evaluate(theta, with_grads=False)
    # Build the lens once more at this theta to extract stop & pupil.
    theta_t = torch.tensor(theta, dtype=torch.float32)
    with torch.no_grad():
        lens, _ = prob._build_lens(theta_t)
        pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
    aper_idx = lens.aper_idx
    r_stop = float(lens.surfaces[aper_idx].r) if aper_idx is not None else float("nan")
    return res, float(pupilr), r_stop


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--fnum", type=float, default=8.0)
    ap.add_argument("--out", default=None,
                    help="Output dir; default: <run>/elite_designs_F<num>")
    ap.add_argument("--limit", type=int, default=None,
                    help="Refine only first N elites (debug)")
    return ap.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run)
    cfg = load_config(args.config)
    efl_target = cfg.problem.focal_length_target_mm

    fnum_tag = f"{args.fnum:.1f}".replace(".", "p")
    out_dir = Path(args.out) if args.out else run_dir / f"elite_designs_F{fnum_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pupil_target = efl_target / (2.0 * args.fnum)

    print(f"Run dir   : {run_dir}")
    print(f"Config    : {args.config}")
    print(f"F# target : {args.fnum}")
    print(f"EFL target: {efl_target} mm")
    print(f"=> paraxial pupil_radius_target = {pupil_target:.4f} mm")
    print(f"Output    : {out_dir}")

    elite_files = sorted(glob.glob(str(run_dir / "elite_designs" / "*.json")))
    if args.limit:
        elite_files = elite_files[:args.limit]
    print(f"Elites    : {len(elite_files)}\n")

    prob, _, template_r_stop = build_refine_problem(cfg, args.fnum, efl_target)
    print(f"Template stop radius (reset baseline): {template_r_stop:.4f} mm\n")

    summary_rows = []
    t0 = time.time()
    for idx, src_path in enumerate(elite_files, start=1):
        stem = Path(src_path).stem
        with open(src_path) as f:
            d = json.load(f)
        if "refined" not in d or "theta" not in d.get("refined", {}):
            print(f"[{idx:3d}/{len(elite_files)}] {stem}  SKIP (no refined theta)")
            continue
        theta_init = np.asarray(d["refined"]["theta"], dtype=float)

        # Pre-rescale pupil snapshot using the *unpatched* builder so we
        # capture the design's original (template-stop) F# before rescaling.
        try:
            theta_t0 = torch.tensor(theta_init, dtype=torch.float32)
            with torch.no_grad():
                lens0, _ = prob._unpatched_build_lens(theta_t0)
                aper_idx = lens0.aper_idx
                lens0.surfaces[aper_idx].r = float(template_r_stop)
                _, pupilr_pre = lens0.calc_entrance_pupil(paraxial=True)
                r_stop_pre = float(lens0.surfaces[aper_idx].r)
                fnum_pre = efl_target / (2.0 * float(pupilr_pre)) if float(pupilr_pre) > 0 else float("nan")
        except Exception as e:
            print(f"[{idx:3d}/{len(elite_files)}] {stem}  pre-build FAILED: {e}")
            continue

        # L-BFGS refinement at fixed-F# stop (patched _build_lens does rescale every step).
        try:
            theta_ref, fit_ref, steps, conv, gn = lbfgs_refine(prob, theta_init, cfg)
        except Exception as e:
            print(f"[{idx:3d}/{len(elite_files)}] {stem}  LBFGS FAILED: {e}")
            traceback.print_exc()
            continue

        # Final evaluation + spec readback.
        res, pupilr_post, r_stop_post = evaluate_at(prob, theta_ref)
        rms_mm = res.metrics.get("rms_spot", abs(fit_ref))
        efl_actual = res.metrics.get("efl_actual", float("nan"))
        invalid_frac = res.metrics.get("invalid_fraction", 0.0)
        fnum_post = efl_target / (2.0 * pupilr_post) if pupilr_post > 0 else float("nan")
        # also report ACHIEVED F# using actual EFL (not target) — more honest
        fnum_post_actual = (efl_actual / (2.0 * pupilr_post)
                            if (pupilr_post > 0 and np.isfinite(efl_actual))
                            else float("nan"))

        out = {
            "source_run": str(run_dir),
            "source_elite": stem,
            "fnum_target": args.fnum,
            "efl_target_mm": efl_target,
            "pupil_radius_target_mm": pupil_target,
            "starting_state": {
                "theta": [float(v) for v in theta_init],
                "pupilr_mm": float(pupilr_pre),
                "r_stop_mm": float(r_stop_pre),
                "fnum_at_efl_target": float(fnum_pre),
            },
            "refined_F8": {
                "theta": [float(v) for v in theta_ref],
                "fitness": float(fit_ref),
                "rms_mm": float(rms_mm),
                "rms_um": f"{rms_mm * 1000:.2f}",
                "efl_mm_actual": float(efl_actual),
                "pupilr_mm": float(pupilr_post),
                "r_stop_mm": float(r_stop_post),
                "fnum_at_efl_target": float(fnum_post),
                "fnum_at_efl_actual": float(fnum_post_actual),
                "invalid_fraction": float(invalid_frac),
                "lbfgs_steps": int(steps),
                "converged": bool(conv),
                "final_grad_norm": float(gn),
            },
        }
        out_path = out_dir / f"{stem}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        rms_um_pre = float(d["refined"]["rms_mm"]) * 1000.0
        rms_um_post = rms_mm * 1000.0
        delta = rms_um_post - rms_um_pre
        print(f"[{idx:3d}/{len(elite_files)}] {stem}  "
              f"F#: {fnum_pre:5.2f} → {fnum_post_actual:5.2f}  "
              f"r_stop: {r_stop_pre:.3f} → {r_stop_post:.3f}  "
              f"RMS: {rms_um_pre:6.2f} → {rms_um_post:6.2f} µm  "
              f"(Δ {delta:+6.2f})  EFL={efl_actual:.2f}  "
              f"steps={steps} {'CONV' if conv else 'max'}",
              flush=True)
        summary_rows.append({
            "rank_in_dir": idx, "stem": stem,
            "rms_um_pre": rms_um_pre, "rms_um_post": rms_um_post,
            "fnum_pre": fnum_pre, "fnum_post": fnum_post_actual,
            "r_stop_pre": r_stop_pre, "r_stop_post": r_stop_post,
            "efl_actual": float(efl_actual), "invalid_frac": float(invalid_frac),
            "converged": bool(conv),
        })

    # Write a summary CSV.
    csv_path = out_dir / "_summary.csv"
    with open(csv_path, "w") as f:
        f.write("stem,rms_um_pre,rms_um_post,fnum_pre,fnum_post,"
                "r_stop_pre,r_stop_post,efl_actual,invalid_frac,converged\n")
        for r in sorted(summary_rows, key=lambda x: x["rms_um_post"]):
            f.write(f"{r['stem']},{r['rms_um_pre']:.4f},{r['rms_um_post']:.4f},"
                    f"{r['fnum_pre']:.4f},{r['fnum_post']:.4f},"
                    f"{r['r_stop_pre']:.4f},{r['r_stop_post']:.4f},"
                    f"{r['efl_actual']:.4f},{r['invalid_frac']:.4f},"
                    f"{r['converged']}\n")
    print(f"\nDone in {time.time() - t0:.1f}s.  Summary CSV: {csv_path}")
    print(f"Refined {len(summary_rows)} elites at F/{args.fnum:.2f}.")


if __name__ == "__main__":
    main()
