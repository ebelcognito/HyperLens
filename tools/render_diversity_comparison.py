#!/usr/bin/env python3
"""Render PNG layouts comparing SA+GA's top-3 families vs HL archive cells.

For each design, draws the 2D lens cross-section with ray fans at three
field positions (axis, mid-FOV, edge-FOV). Layouts are saved as PNGs in
the output directory. Title encodes fitness, RMS, EFL, F#, sign pattern,
and stability if available.

Usage:
  python tools/render_diversity_comparison.py \\
      --hl-run <hl_run_dir> \\
      --sa-ga-run <sa_ga_run_dir> \\
      --out <output_dir>
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic


def render_lens(
    lens, filepath: Path, title: str, n_curv: int = 12,
) -> None:
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
        except Exception as e:
            print(f"  ray trace failed at fov_frac={fov_frac}: {e}",
                  flush=True)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("y (mm)")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  wrote {filepath}", flush=True)


def sign_pattern(theta: np.ndarray, n_curv: int = 12) -> str:
    chars = []
    for i in range(n_curv):
        c = float(theta[i])
        chars.append("+" if c > 1e-9 else ("-" if c < -1e-9 else "0"))
    return "".join(chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hl-run", required=True)
    ap.add_argument("--sa-ga-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-hl-families", type=int, default=6,
                    help="How many HL families to render (one design each)")
    ap.add_argument("--hl-threshold", type=float, default=-0.5,
                    help="Fitness floor for picking HL designs")
    ap.add_argument("--n-curv", type=int, default=12)
    args = ap.parse_args()

    hl_dir = Path(args.hl_run).resolve()
    sg_dir = Path(args.sa_ga_run).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load problem config (used for both methods; same lens spec) ----
    cfg = load_config(str(hl_dir / "config.yaml"))
    p = cfg.problem

    def make_fresh_lens():
        template, lens = load_lens_template(
            p.lens_path,
            wavelength_um=p.wavelength_um,
            stop_surface_index=p.stop_surface_index,
            free_curvature_indices=tuple(p.free_curvature_indices),
            derived_curvature_index=p.derived_curvature_index,
            device="cpu",
            cemented_interfaces=tuple(p.cemented_interfaces),
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

    # ============================================================
    # SA+GA top-3 families
    # ============================================================
    print("[SA+GA] loading refined designs", flush=True)
    sg_designs = json.loads(
        (sg_dir / "refined_designs_sa_ga.json").read_text()
    )
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in sg_designs:
        by_family[d["sign_pattern"]].append(d)
    fam_summary = sorted(
        ((sp, designs) for sp, designs in by_family.items()),
        key=lambda kv: -len(kv[1]),
    )

    for rank, (sp, designs) in enumerate(fam_summary[:3], 1):
        best = min(designs, key=lambda d: d["rms_um"])
        theta = np.asarray(best["theta"], dtype=np.float64)
        template, lens, solver = make_fresh_lens()
        apply(lens, theta, template, solver)
        title = (
            f"SA+GA family #{rank} ({len(designs)} members) [{sp}]\n"
            f"refined RMS={best['rms_um']:.1f}μm  "
            f"EFL={best['efl_mm']:.2f}mm  F/{best.get('fnum_actual', '?'):.2f}  "
            f"survived={best['stability']['pct_survived']:.0f}%"
        )
        out_path = out_dir / f"sa_ga_family{rank}_{sp}.png"
        render_lens(lens, out_path, title, n_curv=args.n_curv)

    # ============================================================
    # HL archive — pick N most-populated families above threshold
    # ============================================================
    print(f"[HL] loading archive cells with fitness >= "
          f"{args.hl_threshold}", flush=True)
    with (hl_dir / "archive.pkl").open("rb") as f:
        a = pickle.load(f)
    cells_above = [
        (float(v.fitness), np.asarray(v.theta), v)
        for v in a.cells.values()
        if v is not None and float(v.fitness) >= args.hl_threshold
    ]
    print(f"[HL] {len(cells_above)} cells above threshold", flush=True)

    # Group by sign pattern
    fams_hl: Dict[str, List[Tuple[float, np.ndarray]]] = defaultdict(list)
    for fit, theta, _ in cells_above:
        sp = sign_pattern(theta, args.n_curv)
        fams_hl[sp].append((fit, theta))
    fam_hl_sorted = sorted(
        fams_hl.items(), key=lambda kv: -len(kv[1])
    )
    print(f"[HL] {len(fam_hl_sorted)} unique sign families above threshold",
          flush=True)

    # Pick top-N most-populated families. From each, pick the highest-
    # fitness representative.
    for rank, (sp, members) in enumerate(fam_hl_sorted[:args.n_hl_families], 1):
        best_fit, best_theta = max(members, key=lambda x: x[0])
        rms_um_approx = -best_fit * 1000  # rough; raw fitness ≈ -RMS_mm
        template, lens, solver = make_fresh_lens()
        apply(lens, best_theta, template, solver)
        title = (
            f"HL family #{rank} ({len(members)} cells) [{sp}]\n"
            f"raw fitness={best_fit:.4f}  RMS≈{rms_um_approx:.0f}μm "
            f"(unrefined)"
        )
        out_path = out_dir / f"hl_family{rank}_{sp}_fit{best_fit:+.3f}.png"
        render_lens(lens, out_path, title, n_curv=args.n_curv)

    # ============================================================
    # Side-by-side summary
    # ============================================================
    print("\n=== rendered ===")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  {f.name}")
    print(f"\nAll PNGs in: {out_dir}")


if __name__ == "__main__":
    main()
