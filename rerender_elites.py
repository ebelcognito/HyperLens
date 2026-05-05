#!/usr/bin/env python3
"""Re-render elite designs with thicker lines and fewer rays."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rerender_elite(json_path: Path, out_dir: Path, cfg: dict):
    """Load one elite JSON and re-render raw + refined PNGs."""
    import torch
    from deeplens.optics import GeoLens
    from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
    from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
    from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver

    with open(json_path) as f:
        info = json.load(f)

    p = cfg["problem"]
    tag = json_path.stem.replace(".json", "")  # e.g. BEST_0001_25344k

    def _load():
        template, lens = load_lens_template(
            p["lens_path"],
            wavelength_um=p["wavelength_um"],
            stop_surface_index=p["stop_surface_index"],
            free_curvature_indices=tuple(p["free_curvature_indices"]),
            derived_curvature_index=p["derived_curvature_index"],
            device="cpu",
        )
        solver = GeneralizedParaxialSolver(template)
        return template, lens, solver

    def _apply(lens, theta, template, solver):
        _apply_theta_generic(
            lens, theta,
            template=template, solver=solver,
            focal_length_target_mm=p["focal_length_target_mm"],
            device="cpu",
            parameterization=p["parameterization"],
            free_curvature_indices=p["free_curvature_indices"],
            free_gap_indices=p["free_gap_indices"],
            variable_flange=p["variable_flange"],
            variable_materials=p.get("variable_materials", False),
            free_material_indices=p.get("free_material_indices", []),
        )

    # ------------------------------------------------------------------ #
    #  Rendering parameters                                               #
    # ------------------------------------------------------------------ #
    NUM_RAYS = 3
    LENS_LINEWIDTH = 2.5
    RAY_LINEWIDTH = 2.5
    FIELD_COLORS = [
        ("#2196F3", 0.0),   # on-axis  – blue
        ("#4CAF50", 0.5),   # mid-field – green
        ("#F44336", 1.0),   # full-field – red
    ]

    def _render_png(lens, filepath, title):
        fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
        ax.set_facecolor("white")

        # -- Draw lens outlines --
        n_lines_before = len(ax.lines)
        lens.draw_lens_2d(ax=ax, fig=fig, color="black")
        # Thicken every line that draw_lens_2d just added
        for line in ax.lines[n_lines_before:]:
            line.set_linewidth(LENS_LINEWIDTH)

        rfov = float(np.rad2deg(float(lens.rfov))) if hasattr(lens, "rfov") else 14.0

        # -- Draw rays: 3 per field point --
        for color, fov_frac in FIELD_COLORS:
            try:
                ray = lens.sample_point_source_2D(
                    fov=fov_frac * rfov, depth=-20000.0,
                    num_rays=NUM_RAYS, wvln=0.587,
                )
                ray.prop_to(-1.0)
                _, rec = lens.trace2sensor(ray=ray, record=True)

                n_before = len(ax.lines)
                lens.draw_ray_2d(rec, ax=ax, fig=fig, color=color)
                for line in ax.lines[n_before:]:
                    line.set_linewidth(RAY_LINEWIDTH)
            except Exception:
                pass

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("z (mm)", fontsize=10)
        ax.set_ylabel("y (mm)", fontsize=10)
        fig.tight_layout()
        fig.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {filepath.name}")

    # -- Render RAW --
    theta_raw = np.array(info["theta"], dtype=np.float64)
    template, lens, solver = _load()
    _apply(lens, theta_raw, template, solver)

    rms_um = info.get("rms_um", "?")
    efl_s = info.get("efl_mm", "?")
    signs = info.get("sign_pattern", "?")
    raw_fit = info.get("fitness", "?")
    title_raw = (
        f"{tag} RAW | fitness={raw_fit:.6f}  "
        f"RMS={rms_um}\u03bcm  EFL={efl_s}mm  [{signs}]"
    )
    _render_png(lens, out_dir / f"{tag}_raw.png", title_raw)

    # -- Render REFINED (if available) --
    ref = info.get("refined")
    if ref and "theta" in ref:
        theta_ref = np.array(ref["theta"], dtype=np.float64)
        template2, lens2, solver2 = _load()
        _apply(lens2, theta_ref, template2, solver2)

        ref_fit = ref.get("fitness", "?")
        ref_rms = ref.get("rms_um", "?")
        ref_efl = ref.get("efl_mm", "?")
        ref_signs = "".join(
            "+" if v > 0 else "-" for v in np.sign(theta_ref[:6])
        )
        stab = ref.get("stability", {})
        sens = stab.get("sensitivity", "?")
        surv = stab.get("pct_survived", "?")
        title_ref = (
            f"{tag} REFINED | fitness={ref_fit:.6f}  "
            f"RMS={ref_rms}\u03bcm  EFL={ref_efl}mm  [{ref_signs}]"
        )
        if isinstance(sens, float):
            title_ref += f"  stability={sens:.6f}  survived={surv:.0f}%"

        _render_png(lens2, out_dir / f"{tag}_refined.png", title_ref)


def main():
    run_dir = Path(
        "/home/hisham26/Desktop/RL/results/lensmega/"
        "sixlet_discovery_20260416_160517"
    )
    elite_dir = run_dir / "elite_designs"

    # Load config
    with open(run_dir / "config_resolved.json") as f:
        cfg = json.load(f)

    # Output to a subdirectory so we don't overwrite originals
    out_dir = elite_dir / "rerendered"
    out_dir.mkdir(exist_ok=True)

    # Find all JSON files
    jsons = sorted(elite_dir.glob("*.json"))
    print(f"Found {len(jsons)} elite designs to re-render")
    print(f"Output: {out_dir}\n")

    for jp in jsons:
        print(f"Processing {jp.name} ...")
        try:
            rerender_elite(jp, out_dir, cfg)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone! Re-rendered designs in:\n  {out_dir}")


if __name__ == "__main__":
    main()
