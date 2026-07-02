#!/usr/bin/env python
"""Render the three persistence-demo stages (A: gen824 ancestor, B: elite_0020 raw,
C: elite_0020 refined) as:

  1) one pretty layout PNG per stage  — same style as
     figures/method_library_octuplet_unpruned_with_rays.png (single-panel,
     filled glass, dark outlines, three-field 587 nm ray fans). All three
     stages share the same elegant blue glass colour.
  2) per-stage RMS-vs-field ribbon (a "robust" PNG, ribbon = min/max over
     the three fitness wavelengths) and a per-stage |S₂|/|S₃| bar chart
     (a "seidel" PNG) — same compact bold-axis style as
     results/figure_archive_demo_octuplet/spotlight_panels_stable3/.

Outputs in persistence_argument_export/figures/:
    layout_<tag>.png
    robust_<tag>.png
    seidel_<tag>.png
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

EXPORT = ROOT / "persistence_argument_export"
OUT_DIR = EXPORT / "figures"
OUT_DIR.mkdir(exist_ok=True)

CFG_PATH = ROOT / "optical_rl/qd/lensmega/config/examples/septlet_F3_discovery.yaml"
LENS_TEMPLATE_OVERRIDE = ROOT / "DeepLens/datasets/lenses/septlet_50mm_f3.json"

# ---- styling (matches method_library_octuplet_unpruned_with_rays.png) ----
BLUE_FILL = "#7BAFD4"
RED_FILL  = "#E07B7B"
GREEN_FILL = "#7BC47B"
OUTLINE   = "#1F1F1F"
SENSOR    = "#000000"
LINEWIDTH_LENS = 1.4
LINEWIDTH_SENSOR = 1.8
FILL_NUM_SAMPLES = 192

# Layout-figure rays (3 fields, single wavelength, line rays).
FIELD_COLORS = ["#1565C0", "#2E7D32", "#C62828"]
RAY_LW = 0.55
RAY_ALPHA = 0.85
RAY_NUM_PER_FIELD = 13
RAY_WVLN_UM = 0.587
RAY_DEPTH_MM = -20000.0

# Robust-ribbon settings (mirrors render_spotlight_panels_split.py).
ROBUST_N_FIELDS = 7
ROBUST_NUM_RAYS = 256


from scripts_throwaway.fig5_extract import (
    build_lens_template_and_solver, make_geolens,
)


# --------------------------------------------------------------------
# Lens drawing helpers (copied / adapted from
# scripts_throwaway/method_library_render_unpruned_octuplet.py)
# --------------------------------------------------------------------

def lens_fill_polygons(lens, num_samples=FILL_NUM_SAMPLES):
    polys = []
    for i in range(len(lens.surfaces) - 1):
        s0 = lens.surfaces[i]
        s1 = lens.surfaces[i + 1]
        if not (hasattr(s0, "mat2") and float(s0.mat2.n) > 1.1):
            continue
        r0 = float(s0.draw_r()) * 0.999
        r1 = float(s1.draw_r()) * 0.999
        if r0 <= 1e-8 or r1 <= 1e-8:
            continue
        rs0 = np.linspace(-r0, r0, num_samples, dtype=np.float64)
        rs1 = np.linspace(-r1, r1, num_samples, dtype=np.float64)
        rt0 = torch.tensor(rs0, dtype=torch.float32, device=lens.device)
        rt1 = torch.tensor(rs1, dtype=torch.float32, device=lens.device)
        z0 = s0.surface_with_offset(rt0, torch.zeros_like(rt0), valid_check=False).detach().cpu().numpy()
        z1 = s1.surface_with_offset(rt1, torch.zeros_like(rt1), valid_check=False).detach().cpu().numpy()
        z_poly = np.concatenate([z0, z1[::-1]], axis=0)
        x_poly = np.concatenate([rs0, rs1[::-1]], axis=0)
        polys.append((z_poly, x_poly))
    return polys


def draw_lens_clean(lens, ax, color=OUTLINE, lw=LINEWIDTH_LENS):
    for s in lens.surfaces:
        cls = s.__class__.__name__
        if cls == "Aperture":
            d = float(s.d.item()) if torch.is_tensor(s.d) else float(s.d)
            r = float(s.r)
            wedge_l = 0.05 * r
            wedge_h = 0.18 * r
            for sign in (-1, +1):
                ax.plot([d - wedge_l, d + wedge_l], [sign * r, sign * r],
                        color="#666", lw=lw, solid_capstyle="round")
                ax.plot([d, d], [sign * r, sign * (r + wedge_h)],
                        color="#666", lw=lw, solid_capstyle="round")
        else:
            r_eff = float(s.draw_r())
            r = torch.linspace(-r_eff, r_eff, 256, device=s.device)
            z = s.surface_with_offset(
                r, torch.zeros(len(r), device=s.device), valid_check=False
            ).cpu().detach().numpy()
            r_np = r.cpu().detach().numpy()
            ax.plot(z, r_np, color=color, lw=lw, solid_capstyle="round")

    surfs = lens.surfaces
    for i in range(len(surfs) - 1):
        s_prev = surfs[i]
        if not (hasattr(s_prev, "mat2") and float(s_prev.mat2.n) > 1.1):
            continue
        s_next = surfs[i + 1]
        r_prev = float(s_prev.draw_r())
        r_next = float(s_next.draw_r())
        sag_prev = s_prev.surface_with_offset(
            torch.tensor(r_prev, device=s_prev.device),
            torch.tensor(0.0, device=s_prev.device),
            valid_check=False,
        ).item()
        sag_next = s_next.surface_with_offset(
            torch.tensor(r_next, device=s_next.device),
            torch.tensor(0.0, device=s_next.device),
            valid_check=False,
        ).item()
        for sign in (-1, +1):
            ax.plot([sag_prev, sag_next],
                    [sign * r_prev, sign * r_next],
                    color=color, lw=lw, solid_capstyle="round")

    d_sensor = float(lens.d_sensor.item()) if torch.is_tensor(lens.d_sensor) else float(lens.d_sensor)
    r_sensor = float(lens.r_sensor)
    ax.plot([d_sensor, d_sensor], [-r_sensor, r_sensor],
            color=SENSOR, lw=LINEWIDTH_SENSOR, solid_capstyle="round")


def hfov_deg_for(lens):
    if hasattr(lens, "rfov") and lens.rfov is not None:
        try:
            return float(np.rad2deg(float(lens.rfov)))
        except Exception:
            pass
    half_sensor = float(lens.sensor_size[0]) / 2.0 if hasattr(lens, "sensor_size") else float(lens.r_sensor)
    efl = float(getattr(lens, "foclen", 50.0))
    return math.degrees(math.atan(half_sensor / abs(efl)))


def draw_line_rays(lens, ax):
    hfov = hfov_deg_for(lens)
    for fov_frac, color in zip([0.0, 0.5, 1.0], FIELD_COLORS):
        try:
            ray = lens.sample_point_source_2D(
                fov=float(fov_frac * hfov), depth=RAY_DEPTH_MM,
                num_rays=RAY_NUM_PER_FIELD, wvln=RAY_WVLN_UM,
            )
            ray.prop_to(-1.0)
            _, rec = lens.trace2sensor(ray=ray, record=True)
            n_before = len(ax.lines)
            lens.draw_ray_2d(rec, ax=ax, fig=ax.figure, color=color)
            for line in ax.lines[n_before:]:
                line.set_linewidth(RAY_LW)
                line.set_alpha(RAY_ALPHA)
        except Exception as e:
            print(f"    line-ray draw failed for fov={fov_frac}: {e}")


# --------------------------------------------------------------------
# Stage 1: pretty layout panels
# --------------------------------------------------------------------

def render_layout(lens, fill_color, title, subtitle, out_path, draw_rays=True):
    fig, ax = plt.subplots(figsize=(11.0, 4.4), facecolor="white")
    ax.set_facecolor("white")

    if draw_rays:
        draw_line_rays(lens, ax)
    for z_poly, x_poly in lens_fill_polygons(lens):
        ax.fill(z_poly, x_poly, color=fill_color, alpha=0.85,
                linewidth=0.0, zorder=3)
    draw_lens_clean(lens, ax)

    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.suptitle(title, fontsize=14, y=0.98)
    if subtitle:
        ax.set_title(subtitle, fontsize=10.5, color="#444", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------
# Stage 2: robust ribbon + Seidel bars  (matches the visual language of
# results/figure_archive_demo_octuplet/spotlight_panels_stable3/)
# --------------------------------------------------------------------

def per_field_per_wvln_rms_um(lens, p_problem,
                              n_fields=ROBUST_N_FIELDS,
                              num_rays=ROBUST_NUM_RAYS):
    """RMS [µm] for an n_fields x len(fitness_wavelengths) grid."""
    from optical_rl.qd.lensmega.problem.tracing import (
        disk_samples, trace_fields_batch, rms_spot_batch,
    )
    device = "cpu"
    trace_dtype = lens.surfaces[
        p_problem.free_curvature_indices[0]
    ].c.dtype
    aperture = torch.from_numpy(disk_samples(num_rays)).to(
        dtype=trace_dtype, device=device,
    )
    with torch.no_grad():
        pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
    aperture_xy = aperture * float(pupilr)
    pupil_z = float(pupilz)

    template_sensor_half = (lens.sensor_size[0] / 2.0
                            if hasattr(lens, "sensor_size")
                            else float(lens.r_sensor))
    efl = float(getattr(lens, "foclen", float(p_problem.focal_length_target_mm)))
    hfov_deg = math.degrees(math.atan(template_sensor_half / abs(efl)))
    field_fracs = np.linspace(0.0, 1.0, n_fields)
    fields = [(0.0, float(f * hfov_deg)) for f in field_fracs]

    wvlns = list(p_problem.fitness_wavelengths_um)
    rms_arr = np.zeros((n_fields, len(wvlns)), dtype=np.float64)
    for wi, w in enumerate(wvlns):
        ray = trace_fields_batch(
            lens, fields, aperture_xy=aperture_xy, pupil_z=pupil_z,
            object_depth_mm=p_problem.object_depth_mm,
            wavelength_um=float(w), device=device, dtype=trace_dtype,
        )
        rms, _ = rms_spot_batch(ray)
        rms_arr[:, wi] = rms.detach().cpu().numpy() * 1000.0  # mm -> µm
    return field_fracs, wvlns, rms_arr


def _bold_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#444"); ax.spines["bottom"].set_linewidth(1.3)
    ax.spines["left"].set_color("#444");   ax.spines["left"].set_linewidth(1.3)


def render_robust(field_fracs, rms_arr, fill_color, out_path,
                  forced_ymax=None):
    """Ribbon = min/max-across-wavelengths RMS vs field, median centre line.
    Three numbers on each axis; bold tick labels."""
    rms_arr = np.asarray(rms_arr)
    field_fracs = np.asarray(field_fracs)
    rms_min = rms_arr.min(axis=1)
    rms_max = rms_arr.max(axis=1)
    rms_mid = np.median(rms_arr, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.2), facecolor="white")
    ax.fill_between(field_fracs, rms_min, rms_max,
                    color=fill_color, alpha=0.45, linewidth=0.0)
    ax.plot(field_fracs, rms_mid, color=fill_color, lw=2.6,
            solid_capstyle="round")
    ax.plot(field_fracs, rms_min, color=fill_color, lw=0.8, alpha=0.7)
    ax.plot(field_fracs, rms_max, color=fill_color, lw=0.8, alpha=0.7)

    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["axis", "½", "edge"], fontsize=15, fontweight="bold")
    ax.set_xlabel("field", fontsize=15, fontweight="bold")

    ymax = float(forced_ymax) if forced_ymax is not None else float(rms_max.max()) * 1.15
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, ymax / 2.0, ymax])
    if ymax >= 100:
        fmt = lambda v, _: f"{v:.0f}"
    elif ymax >= 10:
        fmt = lambda v, _: f"{v:.0f}"
    else:
        fmt = lambda v, _: f"{v:.1f}"
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt))
    ax.set_ylabel("RMS  [µm]", fontsize=15, fontweight="bold")
    ax.tick_params(axis="y", labelsize=14)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    _bold_axes(ax)
    ax.grid(axis="y", lw=0.4, alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}  (RMS range {rms_min.min():.2f}..{rms_max.max():.2f} µm)")


def render_seidel(s2_val, s3_val, fill_color, out_path, forced_ymax=None):
    """Two bars: |S₂| and |S₃|. Three y-ticks, bold/large.

    forced_ymax shares the y-scale across stages so the dramatic
    parent → refined collapse is visible at a glance.
    """
    vals = [abs(float(s2_val)), abs(float(s3_val))]
    labels = ["S₂", "S₃"]
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.2), facecolor="white")
    ax.bar([0, 1], vals, color=fill_color, edgecolor="#333",
           linewidth=0.9, width=0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=22, fontweight="bold")
    if forced_ymax is not None:
        vmax = float(forced_ymax)
    else:
        vmax = max(vals) * 1.18 if max(vals) > 0 else 1.0
    ax.set_ylim(0, vmax)
    ax.set_yticks([0, vmax / 2.0, vmax])
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v*1e3:.2f}")
    )
    ax.set_ylabel("|S|  (×10⁻³)", fontsize=15, fontweight="bold")
    ax.tick_params(axis="y", labelsize=14)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    _bold_axes(ax)
    ax.grid(axis="y", lw=0.4, alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}  (|S2|={vals[0]:.2e}, |S3|={vals[1]:.2e})")


# --------------------------------------------------------------------
# Stages: which JSON, which theta, which colour, which spot-window scale
# --------------------------------------------------------------------

def load_theta(path: Path, key_chain):
    obj = json.loads(path.read_text())
    cur = obj
    for k in key_chain:
        cur = cur[k]
    return np.asarray(cur, dtype=np.float64)


STAGES = [
    dict(
        tag="A_gen824_parent",
        json_path=EXPORT / "state_A_gen824_parent.json",
        theta_keys=("theta",),
        title="Stage A — gen 824 ancestor (bad-but-stored)",
        subtitle="Field-averaged RMS = 924.5 µm    Admitted to (S₂, S₃) cell (135, 141) on optical novelty, not fitness",
        S2=5.8546087680302626e-05,
        S3=1.0185972540114616e-04,
        robust_ymax=1500.0,
    ),
    dict(
        tag="B_elite0020_raw",
        json_path=EXPORT / "state_B_elite_0020_raw.json",
        theta_keys=("theta",),
        title="Stage B — elite_0020 RAW (single CMA-MEGA emission from Stage A)",
        subtitle="Field-averaged RMS = 443.1 µm    Direct gradient-mixed child of Stage A — no L-BFGS yet",
        S2=3.9275498164881876e-05,
        S3=7.397474007159073e-05,
        robust_ymax=700.0,
    ),
    dict(
        tag="C_elite0020_refined",
        json_path=EXPORT / "state_C_elite_0020_refined.json",
        theta_keys=("refined", "theta"),
        title="Stage C — elite_0020 after L-BFGS polish",
        subtitle="Field-averaged RMS = 2.7 µm    Standard post-hoc refinement of Stage B",
        S2=1.0452573345878853e-06,
        S3=1.1670927162270439e-06,
        robust_ymax=8.0,
    ),
]


def _make_patched_cfg():
    """The shipped septlet config lists an absolute lens_path that is valid on
    a different machine; rewrite it to the local DeepLens copy."""
    import yaml, tempfile
    cfg = yaml.safe_load(CFG_PATH.read_text())
    cfg["problem"]["lens_path"] = str(LENS_TEMPLATE_OVERRIDE)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def main():
    cfg_path = _make_patched_cfg()
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(cfg_path)
    print(f"[loaded septlet config]  lens_path = {lens_path}")

    for st in STAGES:
        print(f"\n=== Stage {st['tag']} ===")
        theta = load_theta(st["json_path"], st["theta_keys"])
        lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)

        # --- pretty layout (uniform blue across all stages) ---
        layout_out = OUT_DIR / f"layout_{st['tag']}.png"
        render_layout(lens, BLUE_FILL, st["title"], st["subtitle"], layout_out)
        layout_norays_out = OUT_DIR / f"layout_norays_{st['tag']}.png"
        render_layout(lens, BLUE_FILL, st["title"], st["subtitle"],
                      layout_norays_out, draw_rays=False)

        # --- robust ribbon: RMS vs field, ribbon = min/max across the 3 fitness λ.
        # Shared y-scale across all stages (worst-stage's ymax) so the dramatic
        # parent → refined collapse is read off the curve height directly. ---
        field_fracs, wvlns, rms_arr = per_field_per_wvln_rms_um(lens, p_problem)
        robust_out = OUT_DIR / f"robust_{st['tag']}.png"
        shared_robust_ymax = max(s["robust_ymax"] for s in STAGES)
        render_robust(field_fracs, rms_arr, BLUE_FILL, robust_out,
                      forced_ymax=shared_robust_ymax)

        # --- |S2|, |S3| bars (shared y-scale across stages) ---
        seidel_out = OUT_DIR / f"seidel_{st['tag']}.png"
        seidel_ymax = max(abs(s["S2"]) for s in STAGES) * 1.18
        seidel_ymax = max(seidel_ymax, max(abs(s["S3"]) for s in STAGES) * 1.18)
        render_seidel(st["S2"], st["S3"], BLUE_FILL, seidel_out,
                      forced_ymax=seidel_ymax)

    # --- on-axis spot triptych: random sampling, single λ, shared scale,
    #     white dots on black, all three stages side-by-side ---
    render_onaxis_triptych(prob_cfg, p_problem, lens_path, template, solver)


# --------------------------------------------------------------------
# On-axis monochromatic spot triptych (white-on-black, shared scale)
# --------------------------------------------------------------------

ONAXIS_NUM_RAYS = 6000
ONAXIS_WVLN_UM = 0.587


def _trace_onaxis(lens, p_problem, num_rays, wavelength_um, rng):
    """Random uniform-disk samples through the entrance pupil for an on-axis
    object point at p_problem.object_depth_mm. Returns sensor-plane (Δx, Δy)
    in microns relative to the centroid."""
    from optical_rl.qd.lensmega.problem.tracing import trace_fields_batch
    # uniform disk via rejection
    pts = []
    while len(pts) < num_rays:
        n_need = num_rays - len(pts)
        cand = rng.uniform(-1.0, 1.0, size=(int(n_need * 1.4), 2))
        keep = (cand[:, 0] ** 2 + cand[:, 1] ** 2) <= 1.0
        pts.extend(cand[keep].tolist())
    pts = np.asarray(pts[:num_rays], dtype=np.float64)

    device = "cpu"
    trace_dtype = lens.surfaces[
        p_problem.free_curvature_indices[0]
    ].c.dtype
    aperture = torch.from_numpy(pts).to(dtype=trace_dtype, device=device)
    with torch.no_grad():
        pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
    aperture_xy = aperture * float(pupilr)
    pupil_z = float(pupilz)

    ray = trace_fields_batch(
        lens, [(0.0, 0.0)], aperture_xy=aperture_xy, pupil_z=pupil_z,
        object_depth_mm=p_problem.object_depth_mm,
        wavelength_um=float(wavelength_um),
        device=device, dtype=trace_dtype,
    )
    valid = ray.is_valid.detach().cpu().numpy().astype(bool)
    xy = ray.o[..., :2].detach().cpu().numpy()
    if valid.ndim == 3:
        valid = valid.squeeze(-1)
    mask = valid[0]
    pts_mm = xy[0][mask]
    if pts_mm.shape[0] == 0:
        return np.zeros((0, 2)), 0.0
    cxy = pts_mm.mean(axis=0)
    d_um = (pts_mm - cxy) * 1e3
    rms_um = float(np.sqrt((d_um ** 2).sum(axis=1).mean()))
    return d_um, rms_um


def render_onaxis_triptych(prob_cfg, p_problem, lens_path, template, solver):
    rng = np.random.default_rng(42)
    panels = []
    for st in STAGES:
        theta = load_theta(st["json_path"], st["theta_keys"])
        lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
        d_um, rms_um = _trace_onaxis(lens, p_problem, ONAXIS_NUM_RAYS,
                                     ONAXIS_WVLN_UM, rng)
        panels.append((st["tag"], d_um, rms_um))

    # shared scale = 99th-percentile radius of the WORST stage (Stage A),
    # so the refined stage collapses to a tiny dot — that's the point.
    radii = []
    for _, d_um, _ in panels:
        if d_um.shape[0] > 0:
            radii.append(np.percentile(np.linalg.norm(d_um, axis=1), 99.0))
    R = float(max(radii)) * 1.05 if radii else 1.0

    for tag, d_um, rms_um in panels:
        fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0), facecolor="white")
        ax.set_facecolor("white")
        if d_um.shape[0] > 0:
            ax.scatter(d_um[:, 0], d_um[:, 1], s=0.7, c=BLUE_FILL,
                       alpha=0.9, edgecolors="none")
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_aspect("equal")
        ax.set_axis_off()
        out_path = OUT_DIR / f"onaxis_spot_{tag}.png"
        fig.savefig(out_path, dpi=220, facecolor="white", bbox_inches="tight",
                    pad_inches=0.0)
        plt.close(fig)
        print(f"  wrote {out_path}  (RMS = {rms_um:.1f} µm, window ±{R:.0f} µm)")


if __name__ == "__main__":
    main()
