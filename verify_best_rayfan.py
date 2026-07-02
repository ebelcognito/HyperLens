"""Quantitative + visual ray-fan diagnostics for the single best refined design.

For the design with the lowest verified high-N RMS:
  1. Picks the top-1 from paper_diagnostics/summary.csv by rms_match_unpruned_um
     (or specify --tag <tag>).
  2. Rebuilds the UNPRUNED lens from theta_refined.
  3. Computes ray fans for 5 fields × 3 wavelengths × {tangential, sagittal}.
     Reports per-fan peak-to-peak in microns and a shape-classifier
     (linear-fit slope, cubic-fit coefficient, parabolic-fit coefficient).
  4. Renders a clean publication-style ray-fan grid PNG.

Usage:
    python verify_best_rayfan.py <pruned_dir> [--tag <tag>]
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float32)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deeplens.optics import GeoLens
from deeplens.optics.light.ray import Ray
from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic


WAVELENGTHS_UM = [0.486, 0.587, 0.656]
WAVE_COLORS = {0.486: "#2196F3", 0.587: "#4CAF50", 0.656: "#F44336"}
FIELD_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEPTH_MM = -20000.0
N_FAN_RAYS = 51
PUPIL_FRAC_MAX = 0.99   # avoid edge ray clipping at exactly r=±1


def build_unpruned(theta, run_dir, device="cpu"):
    cfg = load_config(str(run_dir / "config.yaml"))
    p = cfg.problem
    template, _ = load_lens_template(
        p.lens_path, wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        device=device,
        cemented_interfaces=tuple(p.cemented_interfaces),
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


def chief_hit(lens, fov_y_deg, wvln):
    fov_y_deg = float(fov_y_deg)
    pupilz, pupilr = lens.calc_entrance_pupil()
    pupilz, pupilr = float(pupilz), float(pupilr)
    fov_rad = math.radians(fov_y_deg)
    obj_y = math.tan(fov_rad) * DEPTH_MM
    o = torch.tensor([[0.0, obj_y, DEPTH_MM]], dtype=torch.float32)
    d = torch.tensor([[0.0, -obj_y, pupilz - DEPTH_MM]], dtype=torch.float32)
    out = lens.trace2sensor(ray=Ray(o, d, wvln=float(wvln), device="cpu"))
    if float(out.is_valid[0]) < 0.5:
        return None, None
    return float(out.o[0, 0]), float(out.o[0, 1])


def ray_fan(lens, fov_y_deg, wvln, axis="y", n=N_FAN_RAYS):
    """1D pupil fan. Returns (norm_pupil, sensor_offset_um) arrays.
    axis='y' = tangential, axis='x' = sagittal."""
    fov_y_deg = float(fov_y_deg)
    pupilz, pupilr = lens.calc_entrance_pupil()
    pupilz, pupilr = float(pupilz), float(pupilr)
    fov_rad = math.radians(fov_y_deg)
    obj_y = math.tan(fov_rad) * DEPTH_MM
    obj = np.array([0.0, obj_y, DEPTH_MM], dtype=np.float32)

    pupil_coord = np.linspace(
        -PUPIL_FRAC_MAX * pupilr, PUPIL_FRAC_MAX * pupilr, n, dtype=np.float32
    )
    pupil_pts = np.zeros((n, 3), dtype=np.float32)
    pupil_pts[:, 2] = pupilz
    if axis == "y":
        pupil_pts[:, 1] = pupil_coord
    else:
        pupil_pts[:, 0] = pupil_coord

    o = torch.tensor(np.tile(obj, (n, 1)), dtype=torch.float32)
    d = torch.tensor(pupil_pts - obj, dtype=torch.float32)
    out = lens.trace2sensor(ray=Ray(o, d, wvln=float(wvln), device="cpu"))
    valid = out.is_valid.cpu().numpy() > 0.5
    pos = out.o.cpu().numpy()
    cx, cy = chief_hit(lens, fov_y_deg, wvln)
    if cy is None:
        return None, None
    if axis == "y":
        offset_mm = pos[:, 1] - cy
    else:
        offset_mm = pos[:, 0] - cx
    norm = pupil_coord / pupilr
    norm = np.where(valid, norm, np.nan)
    offset_um = np.where(valid, offset_mm * 1000.0, np.nan)
    return norm, offset_um


def fan_quantitative(norm, offset_um):
    """Return dict of summary stats for one ray fan curve."""
    valid = np.isfinite(norm) & np.isfinite(offset_um)
    if valid.sum() < 5:
        return {"peak_to_peak_um": float("nan"), "rms_um": float("nan"),
                "linear_slope": float("nan"), "cubic_coef": float("nan"),
                "parab_coef": float("nan"), "shape": "invalid"}
    x = norm[valid]
    y = offset_um[valid]
    p2p = float(y.max() - y.min())
    rms = float(np.sqrt(np.mean(y ** 2)))
    slope = float(np.polyfit(x, y, 1)[0])
    p3 = np.polyfit(x, y, 3)
    cubic_coef = float(p3[0])
    parab_coef = float(p3[1])
    abs_lin   = abs(slope)
    abs_cubic = abs(cubic_coef)
    abs_parab = abs(parab_coef)
    dominant = max(("defocus", abs_lin),
                   ("spherical", abs_cubic),
                   ("coma/asym", abs_parab),
                   key=lambda kv: kv[1])
    return {
        "peak_to_peak_um": round(p2p, 2),
        "rms_um": round(rms, 2),
        "linear_slope": round(slope, 3),
        "cubic_coef": round(cubic_coef, 3),
        "parab_coef": round(parab_coef, 3),
        "shape": dominant[0],
    }


def render_rayfan(lens, fov_max_deg, out_path, title=""):
    fields = [f * fov_max_deg for f in FIELD_FRACTIONS]
    fig, axes = plt.subplots(len(fields), 2, figsize=(8.5, 2.0 * len(fields)),
                             sharex=True, dpi=150)
    # Compute global y-range so all subplots share scale
    all_y = []
    cache = {}
    for i, fov in enumerate(fields):
        for j, axis in enumerate(["y", "x"]):
            key = (i, axis)
            cache[key] = {}
            for wvln in WAVELENGTHS_UM:
                norm, off = ray_fan(lens, fov, wvln, axis=axis)
                cache[key][wvln] = (norm, off)
                if off is not None:
                    all_y.extend(off[np.isfinite(off)].tolist())
    if all_y:
        ymax = max(50.0, np.percentile(np.abs(all_y), 99) * 1.1)
    else:
        ymax = 100.0

    for i, fov in enumerate(fields):
        for j, axis in enumerate(["y", "x"]):
            ax = axes[i, j]
            for wvln in WAVELENGTHS_UM:
                norm, off = cache[(i, axis)][wvln]
                if norm is None: continue
                ax.plot(norm, off, color=WAVE_COLORS[wvln], lw=1.2,
                        label=f"{int(wvln*1000)}nm" if i == 0 else None)
            ax.axhline(0, color="k", lw=0.4, alpha=0.4)
            ax.axvline(0, color="k", lw=0.4, alpha=0.4)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-ymax, ymax)
            ax.grid(alpha=0.25, lw=0.4)
            label_axis = "tangential (Y-fan)" if j == 0 else "sagittal (X-fan)"
            ax.set_ylabel(f"fov={fov:.1f}°\n{label_axis}\nΔ [μm]", fontsize=8)
    axes[0, 0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    axes[-1, 0].set_xlabel("normalized pupil coordinate")
    axes[-1, 1].set_xlabel("normalized pupil coordinate")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pruned_dir", type=str)
    ap.add_argument("--tag", type=str, default=None,
                    help="Specific design tag. Default: best by high-N Fibonacci RMS.")
    args = ap.parse_args()

    pruned_dir = Path(args.pruned_dir)
    summary_csv = pruned_dir / "paper_diagnostics" / "summary.csv"
    rows = list(csv.DictReader(open(summary_csv)))
    def _f(r, k):
        try: return float(r[k])
        except: return float("inf")

    if args.tag is not None:
        picked = [r for r in rows if r["tag"] == args.tag]
        if not picked:
            raise SystemExit(f"tag {args.tag} not found in summary.csv")
        row = picked[0]
    else:
        rows.sort(key=lambda r: _f(r, "rms_match_unpruned_um"))
        row = rows[0]
    tag = row["tag"]
    print(f"[best] picked tag = {tag}")
    print(f"[best] meta_RMS={_f(row,'refined_rms_um_meta'):.2f}μm  "
          f"unp_match={_f(row,'rms_match_unpruned_um'):.2f}μm")
    print()

    meta = json.load(open(pruned_dir / f"{tag}_meta.json"))
    lens = build_unpruned(meta["theta_refined"], pruned_dir.parent)

    f = float(lens.calc_foclen())
    fov_max_deg = math.degrees(math.atan(float(lens.r_sensor) / f))
    print(f"[best] EFL={f:.3f}mm  half-FoV={fov_max_deg:.2f}°")
    print()

    # Quantitative table
    print(f"{'fov°':>6} {'axis':<8} {'wvln_nm':>8} {'P-P_μm':>8} "
          f"{'RMS_μm':>8} {'lin_slope':>10} {'cubic':>9} {'parab':>9} {'shape':<10}")
    print("-" * 88)
    headline = []
    for frac in FIELD_FRACTIONS:
        fov = frac * fov_max_deg
        for axis in ["y", "x"]:
            for wvln in WAVELENGTHS_UM:
                norm, off = ray_fan(lens, fov, wvln, axis=axis)
                if norm is None:
                    continue
                stats = fan_quantitative(norm, off)
                axis_name = "tangential" if axis == "y" else "sagittal"
                print(f"{fov:>6.2f} {axis_name:<8} {int(wvln*1000):>8} "
                      f"{stats['peak_to_peak_um']:>8.2f} {stats['rms_um']:>8.2f} "
                      f"{stats['linear_slope']:>10.3f} {stats['cubic_coef']:>9.3f} "
                      f"{stats['parab_coef']:>9.3f} {stats['shape']:<10}")
                headline.append((fov, axis, wvln, stats))
    print()

    # Render
    out_png = pruned_dir / "paper_diagnostics" / f"{tag}_BEST_rayfan.png"
    render_rayfan(lens, fov_max_deg, out_png,
                  title=f"BEST: {tag}  EFL={f:.2f}mm  FoV±{fov_max_deg:.2f}°  "
                        f"high-N RMS={_f(row,'rms_match_unpruned_um'):.1f}μm")
    print(f"[best] rendered: {out_png}")


if __name__ == "__main__":
    main()
