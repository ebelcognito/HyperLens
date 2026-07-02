#!/usr/bin/env python3
"""Render a 3x3 grid of the top-9 epsilon-distinct designs from a refined pool.

Filters refined designs by RMS threshold, sorts by refined RMS asc, runs greedy
eps-net in normalized parameter space, picks the first 9, and renders each as
a 2D lens layout with multi-field rays. Tiles into a single 3x3 PNG.

Usage:
  python tools/render_top9_eps_grid.py \
      --refined-json <path/to/refined.json> \
      --problem-config <path/to/system.yaml> \
      --hl-run <path/to/hl_run_dir>           # for config.yaml lookup if needed
      --rms-thresh 20 --eps 0.20 \
      --out <path/to/out.png>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic


def eps_net_indices(thetas_normalized, eps):
    """Greedy: input is in fitness-asc order; return list of accepted indices."""
    accepted = []
    for i, t in enumerate(thetas_normalized):
        if not accepted:
            accepted.append(i)
            continue
        d = min(np.linalg.norm(t - thetas_normalized[a]) / np.sqrt(len(t))
                for a in accepted)
        if d > eps:
            accepted.append(i)
    return accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined-json", required=True)
    ap.add_argument("--problem-config", required=True,
                    help="Path to YAML used for the run, to load problem spec")
    ap.add_argument("--rms-thresh", type=float, default=20.0,
                    help="Refined RMS upper bound (μm)")
    ap.add_argument("--eps", type=float, default=0.20)
    ap.add_argument("--top-n", type=int, default=9)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title-prefix", default="HL")
    args = ap.parse_args()

    refined = json.loads(Path(args.refined_json).read_text())
    designs = refined["designs"] if isinstance(refined, dict) else refined

    # Filter & sort by refined RMS asc
    pool = sorted(
        [d for d in designs if d.get("rms_um", 1e9) < args.rms_thresh],
        key=lambda d: d["rms_um"],
    )
    print(f"[filter] {len(pool)} designs below {args.rms_thresh}μm refined RMS")
    if len(pool) == 0:
        sys.exit("no designs below threshold")

    # Load problem to get bounds + lens setup
    cfg = load_config(args.problem_config)
    p = cfg.problem
    bounds = np.array(p.theta_bounds, dtype=np.float64)
    lo, hi = bounds[:, 0], bounds[:, 1]

    # Greedy eps-net
    thetas_norm = [(np.asarray(d["theta"]) - lo) / (hi - lo) for d in pool]
    keep = eps_net_indices(thetas_norm, args.eps)
    print(f"[eps-net] {len(keep)} eps={args.eps}-distinct, taking top {args.top_n}")
    keep = keep[: args.top_n]
    if len(keep) < args.top_n:
        print(f"[warn] only {len(keep)} eps-distinct designs found")

    # Build lens factory
    def make_fresh_lens():
        template, lens = load_lens_template(
            p.lens_path,
            wavelength_um=p.wavelength_um,
            stop_surface_index=p.stop_surface_index,
            free_curvature_indices=tuple(p.free_curvature_indices),
            derived_curvature_index=p.derived_curvature_index,
            device="cpu",
            cemented_interfaces=tuple(p.cemented_interfaces) if p.cemented_interfaces else (),
        )
        solver = GeneralizedParaxialSolver(template)
        return template, lens, solver

    def apply(lens, theta, template, solver):
        _apply_theta_generic(
            lens, np.asarray(theta, dtype=np.float64),
            template=template, solver=solver,
            focal_length_target_mm=p.focal_length_target_mm,
            device="cpu",
            parameterization=p.parameterization,
            free_curvature_indices=p.free_curvature_indices,
            free_gap_indices=p.free_gap_indices,
            variable_flange=p.variable_flange,
            variable_materials=p.variable_materials,
            free_material_indices=p.free_material_indices,
        )

    # Render 3x3 grid
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = len(keep)
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7), dpi=130)
    axes = np.atleast_2d(axes)

    for panel_idx in range(rows * cols):
        ax = axes[panel_idx // cols, panel_idx % cols]
        if panel_idx >= n_panels:
            ax.axis("off")
            continue
        d = pool[keep[panel_idx]]
        theta = np.asarray(d["theta"], dtype=np.float64)
        template, lens, solver = make_fresh_lens()
        try:
            apply(lens, theta, template, solver)
            lens.draw_lens_2d(ax=ax, fig=fig, color="black")
            rfov = float(np.rad2deg(float(lens.rfov))) if hasattr(lens, "rfov") else 14.0
            for fov_frac, color in [(0.0, "#2196F3"), (0.5, "#4CAF50"), (1.0, "#F44336")]:
                try:
                    ray = lens.sample_point_source_2D(
                        fov=fov_frac * rfov, depth=-20000.0, num_rays=9, wvln=0.587,
                    )
                    ray.prop_to(-1.0)
                    _, rec = lens.trace2sensor(ray=ray, record=True)
                    lens.draw_ray_2d(rec, ax=ax, fig=fig, color=color)
                except Exception:
                    pass
            survived = d.get("stability", {}).get("pct_survived", float("nan"))
            title = (
                f"#{panel_idx+1}  RMS={d['rms_um']:.1f}μm  "
                f"EFL={d.get('efl_mm', 0):.1f}mm  F/{d.get('fnum_actual', 0):.1f}\n"
                f"survived={survived:.0f}%  steps={d.get('lbfgs_steps','?')}"
            )
            ax.set_title(title, fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f"render failed:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red")
        ax.set_xlabel("z (mm)", fontsize=7)
        ax.set_ylabel("y (mm)", fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle(
        f"{args.title_prefix} top-{n_panels} ε={args.eps}-distinct, "
        f"refined RMS<{args.rms_thresh:.0f}μm",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
