#!/usr/bin/env python3
"""Render the 3 spotlight lens panels for the archive figure.

Each panel: 2D lens cross-section + axis/mid/edge field ray fans.
Output: clean PNG with no axes/title/border, ready to be Inset[]'d in Mathematica.
"""
import sys, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic


def render_panel(lens, out_path, dpi=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
    lens.draw_lens_2d(ax=ax, fig=fig, color="#1a1a1a")
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
            print(f"  ray trace failed at fov_frac={fov_frac}: {e}", flush=True)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.05)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05,
                facecolor="white", transparent=False)
    plt.close(fig)


def main():
    OUT = Path("results/figure_archive_demo")
    spotlights = json.loads((OUT / "spotlights.json").read_text())

    # Sixlet config (the run dir's saved config)
    HL = Path("results/lensmega/sixlet_apsc_discovery_v2_seed42_20260506_034601")
    cfg = load_config(str(HL / "config.yaml"))
    p = cfg.problem

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

    for s in spotlights:
        for which, theta_field, label in [
            ("raw", "raw_theta", "raw"),
            ("refined", "refined_theta", "refined"),
        ]:
            template, lens, solver = make_fresh_lens()
            apply(lens, s[theta_field], template, solver)
            out = OUT / f"spot{s['id']}_{label}.png"
            render_panel(lens, out)
            print(f"[wrote] {out}  rms={s['refined_rms_um']:.2f}μm" if which=="refined" else f"[wrote] {out}  raw rms={s['raw_rms_um']:.0f}μm")


if __name__ == "__main__":
    main()
