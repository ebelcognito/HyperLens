"""ADAM refinement of a single LensMEGA elite design.

Loads an elite_*.json from a run's elite_designs/ folder, rebuilds the
LensProblem from the run's config.yaml, and runs ADAM on theta starting
from the *raw* (search-time) prior. Mirrors the L-BFGS refinement pipeline
in core/runner.py so the comparison is apples-to-apples.

Usage:
    python adam_refine_elite.py <elite.json> [--steps N] [--lr LR] [--start raw|refined]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem


def _apply_refine_overrides(cfg):
    """Same penalty-weight overrides the L-BFGS refiner uses (runner.py:197-210)."""
    io = cfg.io
    if io.refine_efl_penalty_weight is not None:
        cfg.problem.efl_penalty_weight = io.refine_efl_penalty_weight
    if io.refine_invalid_penalty_weight is not None:
        cfg.problem.invalid_penalty_weight = io.refine_invalid_penalty_weight
    if io.refine_intersect_penalty_weight is not None:
        cfg.problem.intersect_penalty_weight = io.refine_intersect_penalty_weight
    if io.refine_field_weights is not None:
        cfg.problem.field_weights = tuple(io.refine_field_weights)
    if cfg.problem.enforce_fnum and io.refine_fnum_penalty_weight is not None:
        cfg.problem.fnum_penalty_weight = io.refine_fnum_penalty_weight
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elite", type=Path, help="Path to elite_*.json")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--start", choices=["raw", "refined"], default="raw")
    ap.add_argument("--optimizer", choices=["adam", "lion"], default="adam")
    ap.add_argument("--lion-beta1", type=float, default=0.9)
    ap.add_argument("--lion-beta2", type=float, default=0.99)
    ap.add_argument("--cosine", action="store_true",
                    help="Cosine-decay LR from --lr to 0 over --steps")
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="Clip gradient L2 norm to this value")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    elite_path = args.elite.resolve()
    run_dir = elite_path.parent.parent
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        sys.exit(f"config.yaml not found next to elite ({cfg_path})")

    info = json.loads(elite_path.read_text())
    if args.start == "refined" and "refined" in info and "theta" in info["refined"]:
        theta0 = np.array(info["refined"]["theta"], dtype=np.float64)
        start_label = "refined (post-LBFGS)"
        start_fitness = info["refined"]["fitness"]
    else:
        theta0 = np.array(info["theta"], dtype=np.float64)
        start_label = "raw (search-time)"
        start_fitness = info["fitness"]

    cfg = load_config(cfg_path)
    cfg = _apply_refine_overrides(copy.deepcopy(cfg))
    desc = _build_descriptor(cfg)
    prob = _build_problem(cfg, desc)
    prob._ensure_backend()

    bounds = list(cfg.problem.theta_bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)

    # HFOV at the EFL target (same convention as lens_problem._calc_hfov_deg)
    sensor_side_mm = float(prob.template.sensor_size_mm[0])
    half_sensor = sensor_side_mm / 2.0
    efl_tgt = float(cfg.problem.focal_length_target_mm)
    hfov_target_deg = math.degrees(math.atan(half_sensor / efl_tgt))
    half_diag = half_sensor * math.sqrt(2.0)
    diag_half_angle_deg = math.degrees(math.atan(half_diag / efl_tgt))

    print("=" * 72)
    print(f"Elite: {elite_path.name}")
    print(f"Run:   {run_dir.name}")
    print(f"Start: {start_label}  (fitness={start_fitness:.6f})")
    print(f"Sensor: {sensor_side_mm:.3f} mm side  →  half-diag {half_diag:.3f} mm")
    print(f"EFL target: {efl_tgt:.2f} mm")
    print(f"HFOV (fitness convention, side/2): {hfov_target_deg:.2f}°")
    print(f"HFOV (true sensor-corner, diag/2): {diag_half_angle_deg:.2f}°")
    print("=" * 72)

    theta_t = torch.tensor(theta0.copy(), dtype=torch.float32, device="cpu",
                           requires_grad=True)
    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)

    bounds_range_t = (hi_t - lo_t)

    if args.optimizer == "adam":
        optim = torch.optim.Adam([theta_t], lr=args.lr)
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.steps)
                 if args.cosine else None)
        opt_label = f"ADAM lr={args.lr}{' cosine→0' if args.cosine else ''}"
        lion_state = None
    else:
        optim = None
        sched = None
        # Bounds-scaled Lion: sign(momentum) * bounds_range * lr.
        # Every coord moves the same fraction of its bound per step.
        lion_state = {"m": torch.zeros_like(theta_t)}
        opt_label = (f"LION lr={args.lr} (×bounds_range), "
                     f"β1={args.lion_beta1}, β2={args.lion_beta2}")

    print(f"\n{opt_label}, steps={args.steps}")
    print(f"{'step':>5} {'loss':>12} {'rms_um':>10} {'efl_mm':>8} {'fnum':>7} "
          f"{'inv':>6} {'gnorm':>10}")
    print("-" * 72)

    best_loss = float("inf")
    best_theta = theta0.copy()
    t0 = time.perf_counter()

    for step in range(args.steps + 1):
        if optim is not None:
            optim.zero_grad()
        elif theta_t.grad is not None:
            theta_t.grad.zero_()
        with torch.no_grad():
            theta_t.clamp_(lo_t, hi_t)
        fitness, _, metrics = prob._fitness_and_metrics(theta_t)
        loss = -fitness
        if step < args.steps:
            loss.backward()
            gnorm = float(theta_t.grad.norm().item()) if theta_t.grad is not None else 0.0
            if args.grad_clip is not None and theta_t.grad is not None:
                torch.nn.utils.clip_grad_norm_([theta_t], args.grad_clip)
            if optim is not None:
                optim.step()
                if sched is not None:
                    sched.step()
            else:
                with torch.no_grad():
                    g = theta_t.grad
                    m = lion_state["m"]
                    b1, b2 = args.lion_beta1, args.lion_beta2
                    c = b1 * m + (1.0 - b1) * g
                    update_dir = torch.sign(c)
                    theta_t.sub_(args.lr * bounds_range_t * update_dir)
                    lion_state["m"] = b2 * m + (1.0 - b2) * g
        else:
            gnorm = float("nan")

        lv = float(loss.detach().item())
        if lv < best_loss:
            best_loss = lv
            with torch.no_grad():
                best_theta = theta_t.detach().cpu().numpy().copy()

        if step % args.log_every == 0 or step == args.steps:
            print(f"{step:>5d} {lv:>12.6f} "
                  f"{metrics.get('rms_spot', 0)*1000:>10.2f} "
                  f"{metrics.get('efl_actual', 0):>8.2f} "
                  f"{metrics.get('fnum_actual', 0):>7.3f} "
                  f"{metrics.get('invalid_fraction', 0):>6.3f} "
                  f"{gnorm:>10.3e}")

    elapsed = time.perf_counter() - t0
    best_theta = np.clip(best_theta, lo, hi)
    final_eval = prob.evaluate(best_theta, with_grads=False)

    print("-" * 72)
    print(f"Wall: {elapsed:.1f}s   ({elapsed/max(args.steps,1)*1000:.1f} ms/step)")
    print(f"\nFinal best fitness: {-best_loss:.6f}  (start: {start_fitness:.6f})")
    print(f"Improvement ratio:  {start_fitness / -best_loss if best_loss > 0 else float('nan'):.3f}x")
    print(f"Final RMS:  {final_eval.metrics.get('rms_spot', 0)*1000:.2f} μm")
    print(f"Final EFL:  {final_eval.metrics.get('efl_actual', 0):.2f} mm")
    print(f"Final F#:   {final_eval.metrics.get('fnum_actual', 0):.3f}")
    print(f"Invalid:    {final_eval.metrics.get('invalid_fraction', 0):.3%}")

    if "refined" in info:
        ref_fit = info["refined"]["fitness"]
        print(f"\nL-BFGS-refined fitness (for reference): {ref_fit:.6f}")
        print(f"ADAM vs L-BFGS:  {(-best_loss) / ref_fit if abs(ref_fit) > 1e-9 else float('nan'):.3f}x")

    out = args.out or (elite_path.parent / f"{elite_path.stem}_{args.optimizer}.json")
    out.write_text(json.dumps({
        "source_elite": str(elite_path),
        "start": args.start,
        "lr": args.lr,
        "steps": args.steps,
        "wallclock_seconds": elapsed,
        "hfov_fitness_deg": hfov_target_deg,
        "hfov_diagonal_deg": diag_half_angle_deg,
        "start_fitness": start_fitness,
        "final_fitness": -best_loss,
        "theta": [float(v) for v in best_theta],
        "metrics": {k: float(v) for k, v in final_eval.metrics.items()},
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
