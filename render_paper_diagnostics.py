"""Tier-1 paper diagnostics: ray fans, spot diagrams, field curves, distortion.

Runs on a directory of pruned lens JSONs (output of
refine_prune_render_top5_diverse.py) and produces, per design:
  <tag>_rayfan.png   3 wavelengths × 5 fields, tangential + sagittal
  <tag>_spots.png    5 fields × 3 wavelengths spot diagrams
  <tag>_field.png    RMS spot vs field height + distortion vs field
  summary.csv        one row per design with key metrics

Usage:
    python render_paper_diagnostics.py <pruned_dir>
e.g. python render_paper_diagnostics.py \\
        results/lensmega/sixlet_apsc_discovery_v2_20260505_140231/top9_refined_pruned
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float32)   # DeepLens meshgrid is dtype-strict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deeplens.optics import GeoLens
from deeplens.optics.light.ray import Ray
from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic


WAVELENGTHS_UM = [0.486, 0.587, 0.656]   # F, d, C lines
FIELD_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEPTH_MM = -20000.0   # match config
N_FAN_RAYS = 51       # rays along the 1D pupil fan
N_SPOT_RAYS = 256     # rays in spot diagram
N_FIELD_SAMPLES = 11  # field-curve resolution
WAVE_COLORS = {0.486: "#2196F3", 0.587: "#4CAF50", 0.656: "#F44336"}


# ---------------------------------------------------------------------------
# Ray-tracing helpers
# ---------------------------------------------------------------------------

def _trace(lens, ray):
    """Trace ray to sensor; return ray with .o at sensor and validity mask."""
    out = lens.trace2sensor(ray=ray, record=False)
    return out


def _chief_ray_hit(lens, fov_y_deg, wvln):
    """Trace a single chief ray (pupil center) at the given field; return (x,y) at sensor."""
    fov_y_deg = float(fov_y_deg)   # numpy.float64 -> python float (DeepLens dtype-strict)
    pupilz, pupilr = lens.calc_entrance_pupil()
    pupilz = float(pupilz); pupilr = float(pupilr)
    fov_rad = math.radians(fov_y_deg)
    obj_y = math.tan(fov_rad) * DEPTH_MM
    o = torch.tensor([[0.0, obj_y, DEPTH_MM]], dtype=torch.float32)
    d = torch.tensor([[0.0, -obj_y, pupilz - DEPTH_MM]], dtype=torch.float32)
    ray = Ray(o, d, wvln=wvln, device="cpu")
    out = _trace(lens, ray)
    if float(out.is_valid[0]) < 0.5:
        return None, None
    return float(out.o[0, 0]), float(out.o[0, 1])


def _ray_fan(lens, fov_y_deg, wvln, axis="y", n=N_FAN_RAYS):
    """1D pupil fan at given field. Returns (norm_pupil, sensor_offset_um) arrays.

    axis='y' = tangential (meridional) fan (pupil sampled along y-axis).
    axis='x' = sagittal fan (pupil sampled along x-axis).
    """
    fov_y_deg = float(fov_y_deg)
    pupilz, pupilr = lens.calc_entrance_pupil()
    pupilz = float(pupilz); pupilr = float(pupilr)
    fov_rad = math.radians(fov_y_deg)
    obj_y = math.tan(fov_rad) * DEPTH_MM
    obj = np.array([0.0, obj_y, DEPTH_MM], dtype=np.float32)

    pupil_coord = np.linspace(-pupilr, pupilr, n, dtype=np.float32)
    pupil_pts = np.zeros((n, 3), dtype=np.float32)
    pupil_pts[:, 2] = pupilz
    if axis == "y":
        pupil_pts[:, 1] = pupil_coord
    else:
        pupil_pts[:, 0] = pupil_coord

    o = torch.tensor(np.tile(obj, (n, 1)), dtype=torch.float32)
    d = torch.tensor(pupil_pts - obj, dtype=torch.float32)
    ray = Ray(o, d, wvln=wvln, device="cpu")
    out = _trace(lens, ray)
    valid = out.is_valid.cpu().numpy() > 0.5
    pos = out.o.cpu().numpy()
    cx, cy = _chief_ray_hit(lens, fov_y_deg, wvln)
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


def _spot_at_field(lens, fov_y_deg, wvln, n_rays=N_SPOT_RAYS):
    """Trace n_rays from a point source at the given field and wavelength.

    Returns (dx_um, dy_um) arrays of valid sensor hits relative to the
    CENTROID (matches lensmega's rms_spot_batch convention; chief-ray
    referencing inflates RMS for off-axis fields with coma).
    """
    fov_y_deg = float(fov_y_deg)
    ray = lens.sample_point_source(
        fov_x=[0.0], fov_y=[fov_y_deg], depth=float(DEPTH_MM),
        num_rays=n_rays, wvln=float(wvln), entrance_pupil=True,
    )
    out = _trace(lens, ray)
    valid = out.is_valid.cpu().numpy().reshape(-1) > 0.5
    pos = out.o.cpu().numpy().reshape(-1, 3)
    if valid.sum() < 1:
        return None, None
    cx = pos[valid, 0].mean()
    cy = pos[valid, 1].mean()
    dx = (pos[valid, 0] - cx) * 1000.0
    dy = (pos[valid, 1] - cy) * 1000.0
    return dx, dy


def _rms_spot_um(lens, fov_y_deg, wvln, n_rays=N_SPOT_RAYS):
    dx, dy = _spot_at_field(lens, fov_y_deg, wvln, n_rays)
    if dx is None or len(dx) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(dx**2 + dy**2)))


def _distortion_pct(lens, fov_y_deg, wvln):
    """Percent distortion: 100 * (y_actual - y_paraxial) / y_paraxial."""
    fov_y_deg = float(fov_y_deg)
    if abs(fov_y_deg) < 1e-6:
        return 0.0
    cx, cy = _chief_ray_hit(lens, fov_y_deg, wvln)
    if cy is None:
        return float("nan")
    f = float(lens.calc_foclen())
    y_para = f * math.tan(math.radians(fov_y_deg))
    return 100.0 * (cy - y_para) / y_para


# ---------------------------------------------------------------------------
# Plotters
# ---------------------------------------------------------------------------

def plot_rayfans(lens, fov_max_deg, out_path, title=""):
    """Grid: rows = fields (5), cols = (tangential, sagittal). 3 wvln overlaid."""
    fields = [f * fov_max_deg for f in FIELD_FRACTIONS]
    fig, axes = plt.subplots(len(fields), 2, figsize=(8, 2.0 * len(fields)),
                              sharex=True, dpi=140)
    for i, fov in enumerate(fields):
        for j, axis in enumerate(["y", "x"]):
            ax = axes[i, j]
            for wvln in WAVELENGTHS_UM:
                norm, off = _ray_fan(lens, fov, wvln, axis=axis)
                if norm is None: continue
                ax.plot(norm, off, color=WAVE_COLORS[wvln], lw=1.0,
                        label=f"{int(wvln*1000)}nm" if i == 0 else None)
            ax.axhline(0, color="k", lw=0.4, alpha=0.4)
            ax.axvline(0, color="k", lw=0.4, alpha=0.4)
            ax.set_xlim(-1.05, 1.05)
            ax.grid(alpha=0.25, lw=0.4)
            label = "tangential (y)" if j == 0 else "sagittal (x)"
            ax.set_ylabel(f"fov={fov:.1f}°\n{label} [μm]", fontsize=8)
    axes[0, 0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    axes[-1, 0].set_xlabel("normalized pupil coordinate")
    axes[-1, 1].set_xlabel("normalized pupil coordinate")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_spots(lens, fov_max_deg, out_path, title=""):
    """Grid of spot diagrams: rows = fields, cols = wavelengths (3 + combined)."""
    fields = [f * fov_max_deg for f in FIELD_FRACTIONS]
    fig, axes = plt.subplots(len(fields), len(WAVELENGTHS_UM) + 1,
                              figsize=(2.0 * (len(WAVELENGTHS_UM) + 1), 2.0 * len(fields)),
                              dpi=140)
    for i, fov in enumerate(fields):
        all_dx, all_dy, all_c = [], [], []
        for j, wvln in enumerate(WAVELENGTHS_UM):
            ax = axes[i, j]
            dx, dy = _spot_at_field(lens, fov, wvln)
            color = WAVE_COLORS[wvln]
            if dx is not None and len(dx):
                ax.scatter(dx, dy, s=2, alpha=0.55, color=color, edgecolors="none")
                all_dx.append(dx); all_dy.append(dy); all_c.append(np.full(len(dx), j))
                rms = float(np.sqrt(np.mean(dx**2 + dy**2)))
                ax.set_title(f"{int(wvln*1000)}nm  RMS={rms:.1f}μm", fontsize=7)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.25, lw=0.4)
            if j == 0:
                ax.set_ylabel(f"fov={fov:.1f}°\nΔy [μm]", fontsize=8)
            if i == len(fields) - 1:
                ax.set_xlabel("Δx [μm]", fontsize=8)
        # combined panel — all wavelengths
        ax = axes[i, -1]
        for c, dx, dy in zip(all_c, all_dx, all_dy):
            ax.scatter(dx, dy, s=2, alpha=0.45,
                       color=[WAVE_COLORS[WAVELENGTHS_UM[int(c[0])]]] * len(dx),
                       edgecolors="none")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25, lw=0.4)
        if i == 0:
            ax.set_title("polychromatic", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_field(lens, fov_max_deg, out_path, title=""):
    """RMS spot vs field + distortion vs field, twin axes."""
    fields = np.linspace(0, fov_max_deg, N_FIELD_SAMPLES)
    fig, (ax_rms, ax_dist) = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    for wvln in WAVELENGTHS_UM:
        rms = np.array([_rms_spot_um(lens, f, wvln) for f in fields])
        ax_rms.plot(fields, rms, color=WAVE_COLORS[wvln], lw=1.5, marker="o", ms=3,
                    label=f"{int(wvln*1000)}nm")
    rms_avg = np.array([np.nanmean([_rms_spot_um(lens, f, w) for w in WAVELENGTHS_UM])
                         for f in fields])
    ax_rms.plot(fields, rms_avg, color="k", lw=1.0, ls="--", label="poly avg")
    ax_rms.set_xlabel("field angle [°]")
    ax_rms.set_ylabel("RMS spot [μm]")
    ax_rms.set_title("RMS spot vs field")
    ax_rms.grid(alpha=0.3)
    ax_rms.legend(fontsize=8)

    dist_d = np.array([_distortion_pct(lens, f, 0.587) for f in fields])
    ax_dist.plot(fields, dist_d, color="k", lw=1.5, marker="o", ms=3)
    ax_dist.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax_dist.set_xlabel("field angle [°]")
    ax_dist.set_ylabel("distortion [%]")
    ax_dist.set_title("distortion (587nm)")
    ax_dist.grid(alpha=0.3)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-design driver
# ---------------------------------------------------------------------------

def _build_unpruned_lens_from_theta(theta, run_dir: Path, device="cpu"):
    """Rebuild the lens at theta_refined using the ORIGINAL lens JSON
    (pre-prune) so the CAs/geometry match what lensmega saw during
    refinement. Returns a GeoLens with refined theta applied."""
    cfg = load_config(str(run_dir / "config.yaml"))
    p = cfg.problem
    template, _ = load_lens_template(
        p.lens_path,
        wavelength_um=p.wavelength_um,
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


def process_design(lens_path: Path, meta_path: Path, out_dir: Path, do_plots: bool = False):
    tag = lens_path.stem.replace("_lens", "")
    print(f"[paper] {tag}", flush=True)
    lens = GeoLens(filename=str(lens_path), device="cpu")

    rfov_rad = float(lens.rfov) if hasattr(lens, "rfov") and lens.rfov is not None else None
    if rfov_rad is None or not np.isfinite(rfov_rad):
        f = float(lens.calc_foclen())
        rfov_rad = math.atan(float(lens.r_sensor) / f)
    fov_max_deg = math.degrees(rfov_rad)

    summary = {
        "tag": tag,
        "lens_path": str(lens_path),
        "efl_mm": round(float(lens.calc_foclen()), 4),
        "fov_deg": round(fov_max_deg, 3),
    }
    if meta_path.exists():
        import json
        m = json.load(open(meta_path))
        summary["raw_fitness"] = m.get("raw_fitness")
        summary["refined_fitness"] = m.get("refined_fitness")
        summary["refined_rms_um_meta"] = m.get("refined_rms_um")
        summary["lbfgs_steps"] = m.get("lbfgs_steps")

    # Per-field RMS at 5 field heights × 3 wavelengths + polychrom average.
    # Use a higher ray count to reduce variance from random pupil sampling.
    field_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    rms_grid = {}   # (frac, wvln_nm) -> rms_um
    for frac in field_fracs:
        for wvln in WAVELENGTHS_UM:
            rms_grid[(frac, int(wvln*1000))] = _rms_spot_um(
                lens, frac * fov_max_deg, wvln, n_rays=1024
            )

    # Per-field polychromatic average
    for frac in field_fracs:
        triple = [rms_grid[(frac, int(w*1000))] for w in WAVELENGTHS_UM]
        summary[f"rms_um_f{int(frac*100):03d}_poly"] = round(float(np.nanmean(triple)), 2)
    # Per-wavelength RMS at axis and max field
    for wvln in WAVELENGTHS_UM:
        nm = int(wvln*1000)
        summary[f"rms_um_axis_{nm}"]    = round(rms_grid[(0.0, nm)], 2)
        summary[f"rms_um_maxfld_{nm}"] = round(rms_grid[(1.0, nm)], 2)
    # Whole-design polychromatic average across all 15 (field × wvln) cells
    summary["rms_polychrom_avg_um"] = round(
        float(np.nanmean(list(rms_grid.values()))), 2)
    summary["rms_max_cell_um"] = round(
        float(np.nanmax([v for v in rms_grid.values() if np.isfinite(v)])), 2)

    # lensmega-aggregation match: 3 fields (0, 0.5, 1.0 hfov) × 3 wavelengths,
    # field weights [1, 2, 4] / 7, then averaged across wavelengths. This is
    # the exact reduction full_dof.py does to produce metrics['rms_spot'],
    # so this column should match the refined_rms_um_meta within numerical
    # tolerance (any large gap is a real disagreement worth investigating).
    fw = np.array([1.0, 2.0, 4.0]) / 7.0
    weighted_rms_per_w = []
    for wvln in WAVELENGTHS_UM:
        nm = int(wvln * 1000)
        per_field = np.array([
            rms_grid[(0.0, nm)],
            rms_grid[(0.5, nm)],
            rms_grid[(1.0, nm)],
        ])
        weighted_rms_per_w.append(float(np.nansum(per_field * fw)))
    summary["rms_match_lensmega_um"] = round(
        float(np.nanmean(weighted_rms_per_w)), 2)

    # Same lensmega aggregation but evaluated on the UNPRUNED lens (built
    # from theta_refined applied to the original lens template). This is
    # the apples-to-apples comparison with refined_rms_um_meta — should
    # match within a few μm. If it does, the gap vs the pruned-lens column
    # is purely a pruning artifact (CAs too tight clipping valid rays).
    if meta_path.exists() and "theta_refined" in (meta := __import__("json").load(open(meta_path))):
        try:
            run_dir = lens_path.parent.parent  # top<N>_refined_pruned/.. = run dir
            lens_un = _build_unpruned_lens_from_theta(meta["theta_refined"], run_dir)
            unp_grid = {}
            for frac in [0.0, 0.5, 1.0]:
                for wvln in WAVELENGTHS_UM:
                    unp_grid[(frac, int(wvln*1000))] = _rms_spot_um(
                        lens_un, frac * fov_max_deg, wvln, n_rays=1024
                    )
            unp_per_w = []
            for wvln in WAVELENGTHS_UM:
                nm = int(wvln*1000)
                vals = np.array([unp_grid[(0.0, nm)], unp_grid[(0.5, nm)], unp_grid[(1.0, nm)]])
                unp_per_w.append(float(np.nansum(vals * fw)))
            summary["rms_match_unpruned_um"] = round(float(np.nanmean(unp_per_w)), 2)
        except Exception as e:
            print(f"  [paper] unpruned eval failed for {tag}: {e}", flush=True)
            summary["rms_match_unpruned_um"] = ""
    # Distortion at multiple field heights (587nm)
    for frac in [0.5, 0.75, 1.0]:
        summary[f"dist_pct_f{int(frac*100):03d}"] = round(
            _distortion_pct(lens, frac * fov_max_deg, 0.587), 3)

    if do_plots:
        title = (f"{tag}  EFL={summary['efl_mm']}mm  FoV±{fov_max_deg:.1f}°  "
                 f"RMS_axis={summary['rms_um_axis_587']:.1f}μm  "
                 f"RMS_max={summary['rms_um_maxfld_587']:.1f}μm")
        plot_rayfans(lens, fov_max_deg, out_dir / f"{tag}_rayfan.png", title=title)
        plot_spots(lens, fov_max_deg, out_dir / f"{tag}_spots.png", title=title)
        plot_field(lens, fov_max_deg, out_dir / f"{tag}_field.png", title=title)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pruned_dir", type=str,
                    help="Path to a top<N>_refined_pruned/ directory")
    ap.add_argument("--plots", action="store_true",
                    help="Also render rayfan/spots/field PNGs (slow). "
                         "Default: numeric-only summary CSV.")
    args = ap.parse_args()

    pruned_dir = Path(args.pruned_dir)
    out_dir = pruned_dir / "paper_diagnostics"
    out_dir.mkdir(exist_ok=True)
    print(f"[paper] in  : {pruned_dir}")
    print(f"[paper] out : {out_dir}")

    lens_paths = sorted(pruned_dir.glob("top*_lens.json"))
    if not lens_paths:
        # fallback for four-stage outputs
        lens_paths = sorted(pruned_dir.glob("top*_s3_pruned.json"))
    if not lens_paths:
        raise SystemExit(f"no top*_lens.json in {pruned_dir}")

    rows = []
    for lp in lens_paths:
        # meta file alongside (tag is e.g. top01_cell_R_C; meta tag matches)
        tag = lp.stem.replace("_lens", "")
        meta = pruned_dir / f"{tag}_meta.json"
        try:
            row = process_design(lp, meta, out_dir, do_plots=args.plots)
            rows.append(row)
        except Exception as e:
            import traceback
            print(f"  [paper] {lp.name} FAILED: {e}")
            traceback.print_exc()

    # Write summary CSV
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        ordered = [
            "tag", "efl_mm", "fov_deg",
            "raw_fitness", "refined_fitness", "refined_rms_um_meta",
            "rms_match_unpruned_um", "rms_match_lensmega_um",
            "rms_polychrom_avg_um", "rms_max_cell_um",
            "rms_um_f000_poly", "rms_um_f025_poly", "rms_um_f050_poly",
            "rms_um_f075_poly", "rms_um_f100_poly",
            "rms_um_axis_486",  "rms_um_axis_587",  "rms_um_axis_656",
            "rms_um_maxfld_486","rms_um_maxfld_587","rms_um_maxfld_656",
            "dist_pct_f050", "dist_pct_f075", "dist_pct_f100",
            "lbfgs_steps", "lens_path",
        ]
        keys = [k for k in ordered if k in keys] + [k for k in keys if k not in ordered]
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"[paper] wrote summary.csv with {len(rows)} rows")

    print(f"[paper] done -> {out_dir}")


if __name__ == "__main__":
    main()
