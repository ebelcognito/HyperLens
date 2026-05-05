#!/usr/bin/env python3
"""Render 6 lens families from a QD archive as a 3x2 (2:1 aspect) grid
suitable for ACM TOG 2-column figures.

Ray-disappearance fix:
    - Sample ~3× more rays than displayed per field
    - Drop any ray that is invalid anywhere along the path (NaN anywhere)
    - Apply pupil-fraction clip (e.g., 0.85) to skip the marginal rays
      that most often fail; pick evenly spaced survivors for display
    - Every displayed ray reaches the sensor cleanly

Usage:
    python render_family_grid.py <run_dir> [elite1.json elite2.json ...] \\
        [--out <path>] [--title "Cooke triplet"] [--rays 3] [--pupil-frac 0.85]

If no elite JSONs are provided, auto-picks the best refined design from each
of the top-6 sign families (by refined RMS).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"]
rcParams["font.size"] = 11
rcParams["axes.titlesize"] = 12
rcParams["axes.labelsize"] = 10


# ─── ray rendering helpers ──────────────────────────────────────────────

def _trace_rays_clean(lens, fov_deg: float, wvln: float,
                      object_depth: float, n_display: int,
                      pupil_frac: float):
    """Trace rays for one field; return (n_kept, n_path, 2) array of (z, y)
    for rays that are valid end-to-end."""
    n_sample = max(n_display * 3, 9)
    ray = lens.sample_point_source_2D(
        fov=fov_deg, depth=object_depth,
        num_rays=n_sample, wvln=float(wvln),
    )
    ray.prop_to(-1.0)
    _, rec = lens.trace2sensor(ray=ray, record=True)

    stacked = torch.stack(rec, dim=-2).cpu().numpy()  # (n_rays, n_path, 3)
    zx = np.stack([stacked[..., 2], stacked[..., 0]], axis=-1)  # z, y

    valid = ~np.isnan(zx).any(axis=(1, 2))
    zx = zx[valid]
    if len(zx) == 0:
        return np.zeros((0, stacked.shape[1], 2))

    # Clip to pupil_frac by first-surface y position
    y0 = zx[:, 0, 1]
    if np.abs(y0).max() > 0:
        keep = np.abs(y0) <= pupil_frac * np.abs(y0).max() + 1e-9
        zx = zx[keep]
    if len(zx) == 0:
        return np.zeros((0, stacked.shape[1], 2))

    # Pick n_display rays evenly spaced in initial y
    order = np.argsort(zx[:, 0, 1])
    zx = zx[order]
    if len(zx) <= n_display:
        return zx
    picks = np.linspace(0, len(zx) - 1, n_display).round().astype(int)
    return zx[picks]


def _draw_rays(ax, zx, color, linewidth):
    for r in zx:
        ax.plot(r[:, 0], r[:, 1], color=color, linewidth=linewidth,
                solid_capstyle="round")


# ─── lens rebuild ──────────────────────────────────────────────────────

def _load_lens(run_dir: Path, theta: np.ndarray):
    from optical_rl.qd.lensmega.config.loader import load_config
    from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
    from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
    from optical_rl.qd.render_lensmega_archive import _apply_theta_generic

    cfg = load_config(run_dir / "config.yaml")
    p = cfg.problem
    template, lens = load_lens_template(
        p.lens_path,
        wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        device="cpu",
    )
    solver = GeneralizedParaxialSolver(template)
    _apply_theta_generic(
        lens, theta, template=template, solver=solver,
        focal_length_target_mm=p.focal_length_target_mm,
        device="cpu",
        parameterization=p.parameterization,
        free_curvature_indices=p.free_curvature_indices,
        free_gap_indices=p.free_gap_indices,
        variable_flange=p.variable_flange,
        variable_materials=p.variable_materials,
        free_material_indices=p.free_material_indices,
    )
    return lens, cfg


# ─── grid plot ────────────────────────────────────────────────────────

def plot_family_grid(run_dir: Path, elite_files: list, out_path: Path,
                     title: str, rays_per_field: int, pupil_frac: float,
                     lens_linewidth: float, ray_linewidth: float,
                     use_refined: bool):
    fig = plt.figure(figsize=(12, 6))  # 2:1 aspect
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.08,
                          left=0.04, right=0.99, top=0.90, bottom=0.06)

    field_colors = [("#1565C0", 0.0), ("#2E7D32", 0.5), ("#C62828", 1.0)]

    elite_dir = run_dir / "elite_designs"

    for i, fname in enumerate(elite_files[:6]):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(gs[row, col])

        with open(elite_dir / fname) as f:
            d = json.load(f)

        if use_refined and "refined" in d and "theta" in d["refined"]:
            theta = np.array(d["refined"]["theta"], dtype=np.float64)
            rms_str = d["refined"].get("rms_um", "?")
        else:
            theta = np.array(d["theta"], dtype=np.float64)
            rms_str = d.get("rms_um", "?")

        try:
            lens, cfg = _load_lens(run_dir, theta)
        except Exception as e:
            ax.text(0.5, 0.5, f"load fail: {e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            continue

        n_before = len(ax.lines)
        lens.draw_lens_2d(ax=ax, fig=fig, color="black")
        for line in ax.lines[n_before:]:
            line.set_linewidth(lens_linewidth)

        rfov = float(np.rad2deg(float(lens.rfov))) if hasattr(lens, "rfov") else 14.0
        wvln = float(cfg.problem.wavelength_um)
        obj_depth = float(cfg.problem.object_depth_mm)

        for color, fov_frac in field_colors:
            try:
                zx = _trace_rays_clean(
                    lens, fov_deg=fov_frac * rfov, wvln=wvln,
                    object_depth=obj_depth,
                    n_display=rays_per_field,
                    pupil_frac=pupil_frac,
                )
                _draw_rays(ax, zx, color=color, linewidth=ray_linewidth)
            except Exception as e:
                print(f"  ray trace failed {fname} fov={fov_frac}: {e}")

        sign = d.get("sign_pattern", "?")
        try:
            rms_val = float(rms_str)
            rms_fmt = f"{rms_val:.1f} μm"
        except (ValueError, TypeError):
            rms_fmt = f"{rms_str} μm"

        ax.set_title(f"[{sign}]    RMS = {rms_fmt}",
                     fontsize=11, fontweight="bold", pad=4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(True, alpha=0.12, linestyle=":")
        ax.set_facecolor("#fcfcfc")
        if row == 1:
            ax.set_xlabel("z  (mm)", fontsize=9)
        if col == 0:
            ax.set_ylabel("y  (mm)", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def auto_pick_families(run_dir: Path, n: int = 6) -> list:
    from collections import defaultdict
    elite_dir = run_dir / "elite_designs"
    by_family = defaultdict(list)
    for j in sorted(elite_dir.glob("*.json")):
        with open(j) as f:
            d = json.load(f)
        ref = d.get("refined", {})
        if "rms_um" not in ref:
            continue
        try:
            rms = float(ref["rms_um"])
        except (ValueError, TypeError):
            continue
        by_family[d.get("sign_pattern", "?")].append((rms, j.name))

    best_per_family = []
    for sign, lst in by_family.items():
        lst.sort()
        best_per_family.append((lst[0][0], sign, lst[0][1]))

    best_per_family.sort()
    return [fname for _, _, fname in best_per_family[:n]]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("elites", nargs="*")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--rays", type=int, default=3)
    p.add_argument("--pupil-frac", type=float, default=0.85)
    p.add_argument("--lens-lw", type=float, default=2.2)
    p.add_argument("--ray-lw", type=float, default=2.0)
    p.add_argument("--use-raw", action="store_true")
    args = p.parse_args()

    if not args.elites:
        args.elites = auto_pick_families(args.run_dir, n=6)
        print("Auto-picked 6 families:")
        for f in args.elites:
            print(f"  {f}")

    out = args.out or args.run_dir / "family_showcase_6.png"
    title = args.title or f"{args.run_dir.name} — six representative lens families"

    plot_family_grid(
        run_dir=args.run_dir,
        elite_files=args.elites,
        out_path=out,
        title=title,
        rays_per_field=args.rays,
        pupil_frac=args.pupil_frac,
        lens_linewidth=args.lens_lw,
        ray_linewidth=args.ray_lw,
        use_refined=not args.use_raw,
    )


if __name__ == "__main__":
    main()
