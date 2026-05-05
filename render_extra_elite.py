"""Render one extra elite using the same clean template as
render_method_overview.py, but with a single HFOV ray bundle (8 rays, dark
red, slightly thinner than the multi-FOV diagrams).

Hard-coded for elite_0012_11988k from the cooke_triplet_materials run, F/8
refined; outputs the clean lens diagram only (no USAF render) into
paper_figures/method_overview_renders/ as rank 5.
"""
import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/hisham26/Desktop/RL")
sys.path.insert(0, "/home/hisham26/Desktop/RL/DeepLens")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from render_method_overview import (
    CFG, OUT, FNUM_TARGET, EFL_TARGET,
    build_lens_from_theta, draw_lens_clean, lens_specs,
    LINEWIDTH_LENS, WAVELENGTH_UM,
)

ELITE_JSON = (
    "/home/hisham26/Desktop/RL/results/lensmega/"
    "cooke_triplet_materials_20260414_173935/"
    "elite_designs_F8p0/elite_0012_11988k.json"
)
RANK = 5  # extends the existing 1..4 set

# Single bundle from the HFOV (max field), 8 rays, dark red, slightly thinner
NUM_RAYS = 8
RAY_COLOR = "#8B0000"   # darkred
RAY_LW = 1.1            # was 1.4 in the multi-FOV variant
RAY_ALPHA = 0.9

# RGB three-FOV variant
RGB_NUM_RAYS = 5
RGB_FIELD_FRACS = (0.0, 0.5, 1.0)
RGB_COLORS = ("#D62728", "#2CA02C", "#1F77B4")  # R, G, B (mpl-style, clean on white)
RGB_LW = 1.1
RGB_ALPHA = 0.9

# Spot-diagram knobs
SPOT_NUM_RAYS = 25
SPOT_DEPTH = -20000.0
SPOT_DOT_COLOR = "#8B0000"
SPOT_DOT_SIZE = 90        # larger for visual punch
SPOT_DOT_ALPHA = 0.9
SPOT_DOT_EDGE = "#400000"
SPOT_DOT_LW = 0.4

# Paraxial-Seidel ray-trace knobs
PARAX_FIELD_FRAC = 0.5      # descriptor uses 0.5 × HFOV (descriptor_field_index=1)
PARAX_MARGINAL_COLOR = "#8B0000"   # darkred (matches HFOV bundle / spot)
PARAX_CHIEF_COLOR = "#1F4E8B"      # dark blue
PARAX_LW = 1.6
PARAX_ALPHA = 0.95
PARAX_BACK_EXTEND_MM = 18.0  # how far to extend rays into object space


def render_lens_diagram_single_bundle(lens, out_path):
    fig = plt.figure(figsize=(6.0, 3.0), dpi=300, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    draw_lens_clean(lens, ax, color="black", lw_lens=LINEWIDTH_LENS)

    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    fov_hfov = rfov_deg * 0.999

    rays = lens.sample_parallel_2D(
        fov=fov_hfov,
        num_rays=NUM_RAYS,
        wvln=WAVELENGTH_UM,
        depth=-1.0,
        plane="sagittal",
    )
    _, ray_o_record = lens.trace2sensor(ray=rays, record=True)
    pts = torch.stack(ray_o_record, dim=-2).cpu().numpy()
    if pts.ndim == 3:
        pts = pts[None, ...]
    for ray_idx in range(pts.shape[1]):
        ax.plot(
            pts[0, ray_idx, :, 2],
            pts[0, ray_idx, :, 0],
            color=RAY_COLOR, lw=RAY_LW, alpha=RAY_ALPHA,
        )

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("")
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                pad_inches=0.02, facecolor="white", transparent=False)
    plt.close(fig)


def render_lens_diagram_rgb_3fov(lens, out_path):
    fig = plt.figure(figsize=(6.0, 3.0), dpi=300, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    draw_lens_clean(lens, ax, color="black", lw_lens=LINEWIDTH_LENS)

    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    for fov_frac, color in zip(RGB_FIELD_FRACS, RGB_COLORS):
        fov = max(rfov_deg * fov_frac, 1e-6) * (0.999 if fov_frac > 0 else 1.0)
        rays = lens.sample_parallel_2D(
            fov=fov,
            num_rays=RGB_NUM_RAYS,
            wvln=WAVELENGTH_UM,
            depth=-1.0,
            plane="sagittal",
        )
        _, rec = lens.trace2sensor(ray=rays, record=True)
        pts = torch.stack(rec, dim=-2).cpu().numpy()
        if pts.ndim == 3:
            pts = pts[None, ...]
        for ray_idx in range(pts.shape[1]):
            ax.plot(
                pts[0, ray_idx, :, 2],
                pts[0, ray_idx, :, 0],
                color=color, lw=RGB_LW, alpha=RGB_ALPHA,
                solid_capstyle="round",
            )

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("")
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                pad_inches=0.02, facecolor="white", transparent=False)
    plt.close(fig)


def _live_lens_paraxial_data(lens, template):
    """Pull (z, c, n_before, n_after, is_stop) per surface from the live lens
    + template, in the same order/convention the Seidel descriptor uses."""
    N = len(template.surfaces)
    z = []
    c = []
    n_bef = []
    n_aft = []
    is_stop = []
    def _get_n(mat):
        # mat may have refractive_index(wvln) method, or just .n attribute.
        try:
            return float(mat.refractive_index(WAVELENGTH_UM))
        except Exception:
            return float(getattr(mat, "n", 1.0))

    # mat_before[i] = mat_after[i-1]; air (n=1) before surface 0.
    for i in range(N):
        z.append(float(lens.surfaces[i].d.detach().cpu().item()))
        if template.surfaces[i].is_stop:
            c.append(0.0)
        else:
            c.append(float(lens.surfaces[i].c.detach().cpu().item()))
        is_stop.append(bool(template.surfaces[i].is_stop))
        s = lens.surfaces[i]
        n_aft.append(_get_n(s.mat2))
        if i == 0:
            n_bef.append(1.0)
        else:
            n_bef.append(n_aft[i - 1])
    return z, c, n_bef, n_aft, is_stop


def _trace_paraxial(y0, nu0, c, n_bef, n_aft, gaps):
    """Step a paraxial ray through surfaces 0..N-1 with the same recurrence
    the Seidel descriptor uses: refract, then propagate to next surface."""
    N = len(c)
    ys = []
    nus_before = []
    y = y0
    nu = nu0
    for i in range(N):
        ys.append(y)
        nus_before.append(nu)
        phi = (n_aft[i] - n_bef[i]) * c[i]
        nu = nu - phi * y
        if i < N - 1:
            y = y + (gaps[i] / n_aft[i]) * nu
    nu_after_last = nu
    return ys, nus_before, nu_after_last


def render_paraxial_rays(lens, template, out_path):
    z, c, n_bef, n_aft, is_stop = _live_lens_paraxial_data(lens, template)
    N = len(z)
    gaps = [z[i + 1] - z[i] for i in range(N - 1)]
    stop_idx = template.stop_index

    # ---- Marginal ray: y₀=1, ν₀=0 (Seidel normalization) ----
    ys_m, _, nu_after_m = _trace_paraxial(1.0, 0.0, c, n_bef, n_aft, gaps)
    # Scale to the entrance-pupil radius so the visual ray rides the pupil edge.
    try:
        with torch.no_grad():
            _, pupilr = lens.calc_entrance_pupil(paraxial=True)
        marginal_scale = float(pupilr)
    except Exception:
        marginal_scale = 1.0
    ys_m = [y * marginal_scale for y in ys_m]
    nu_after_m_scaled = nu_after_m * marginal_scale

    # ---- Chief ray: solve h̄₀ so the ray hits stop center at the field ----
    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    field_deg = PARAX_FIELD_FRAC * rfov_deg
    tan_field = np.tan(np.deg2rad(field_deg))

    # Trace two basis rays to surface[stop_idx] to find ABCD elements
    yA, nuA = 1.0, 0.0
    yB, nuB = 0.0, 1.0
    for i in range(stop_idx):
        phi = (n_aft[i] - n_bef[i]) * c[i]
        nuA = nuA - phi * yA
        nuB = nuB - phi * yB
        yA = yA + (gaps[i] / n_aft[i]) * nuA
        yB = yB + (gaps[i] / n_aft[i]) * nuB
    safe_yA = yA if abs(yA) > 1e-10 else 1e-10
    hbar_0 = -tan_field * yB / safe_yA

    ys_c, _, nu_after_c = _trace_paraxial(
        hbar_0, tan_field, c, n_bef, n_aft, gaps
    )

    # ---- Build the (z, y) polylines for plotting ----
    # Forward to sensor for both rays:
    d_sensor = float(lens.d_sensor.item())
    d_last_to_sensor = d_sensor - z[-1]
    y_m_sensor = ys_m[-1] + (d_last_to_sensor / n_aft[-1]) * nu_after_m_scaled
    y_c_sensor = ys_c[-1] + (d_last_to_sensor / n_aft[-1]) * nu_after_c

    # Backward extension into object space:
    # Marginal: enters parallel to axis (nu before surface 0 = 0) at y = ys_m[0]
    # Chief:    enters tilted; nu before surface 0 = tan_field, so y(z) = ys_c[0] - tan_field*(z[0]-z)
    z_back = z[0] - PARAX_BACK_EXTEND_MM
    y_m_back = ys_m[0]                          # parallel
    y_c_back = ys_c[0] - tan_field * (z[0] - z_back)

    z_marg = [z_back] + list(z) + [d_sensor]
    y_marg = [y_m_back] + list(ys_m) + [y_m_sensor]
    z_chief = [z_back] + list(z) + [d_sensor]
    y_chief = [y_c_back] + list(ys_c) + [y_c_sensor]

    # ---- Plot ----
    fig = plt.figure(figsize=(6.0, 3.0), dpi=300, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    draw_lens_clean(lens, ax, color="black", lw_lens=LINEWIDTH_LENS)

    ax.plot(
        z_marg, y_marg,
        color=PARAX_MARGINAL_COLOR, lw=PARAX_LW, alpha=PARAX_ALPHA,
        solid_capstyle="round",
    )
    ax.plot(
        z_chief, y_chief,
        color=PARAX_CHIEF_COLOR, lw=PARAX_LW, alpha=PARAX_ALPHA,
        solid_capstyle="round",
    )

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("")
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                pad_inches=0.02, facecolor="white", transparent=False)
    plt.close(fig)
    return field_deg, marginal_scale, ys_c[stop_idx]


def render_spot_diagram(lens, out_path):
    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    fov_hfov = rfov_deg * 0.999

    rays = lens.sample_point_source(
        fov_x=[0.0],
        fov_y=[fov_hfov],
        depth=SPOT_DEPTH,
        num_rays=SPOT_NUM_RAYS,
        wvln=WAVELENGTH_UM,
    )
    out_ray = lens.trace2sensor(ray=rays)
    pts = out_ray.o.detach().cpu().numpy().reshape(-1, 3)
    valid = out_ray.is_valid.detach().cpu().numpy().reshape(-1) > 0
    xy = pts[valid][:, :2]

    cx, cy = float(xy[:, 0].mean()), float(xy[:, 1].mean())
    dx_um = (xy[:, 0] - cx) * 1000.0
    dy_um = (xy[:, 1] - cy) * 1000.0
    rms_um = float(np.sqrt(np.mean(dx_um ** 2 + dy_um ** 2)))

    extent = max(np.abs(dx_um).max(), np.abs(dy_um).max()) * 1.2 + 1.0

    fig = plt.figure(figsize=(4.0, 4.0), dpi=300, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    ax.scatter(
        dx_um, dy_um,
        s=SPOT_DOT_SIZE, c=SPOT_DOT_COLOR, alpha=SPOT_DOT_ALPHA,
        edgecolors=SPOT_DOT_EDGE, linewidths=SPOT_DOT_LW,
    )
    ax.axhline(0.0, color="#cccccc", lw=0.6, zorder=0)
    ax.axvline(0.0, color="#cccccc", lw=0.6, zorder=0)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                pad_inches=0.04, facecolor="white", transparent=False)
    plt.close(fig)
    return rms_um, extent, len(xy)


def main():
    cfg = load_config(CFG)

    with open(ELITE_JSON) as f:
        d = json.load(f)
    ref = d.get("refined_F8") or d.get("refined")
    theta = ref["theta"]
    rms_um = float(ref["rms_mm"]) * 1000.0
    rms_tag = f"{rms_um:.1f}"

    print(f"Source: {os.path.basename(ELITE_JSON)}")
    print(f"  refined RMS: {rms_um:.2f} um")

    lens_path = os.path.join(OUT, f"lens_design_{RANK}_rms_{rms_tag}um.png")

    lens = build_lens_from_theta(
        theta, cfg, device="cpu",
        target_fnum=FNUM_TARGET, target_efl=EFL_TARGET,
    )
    efl, fnum, dfov, diag = lens_specs(lens)
    print(f"  built lens: EFL={efl:.3f}mm  F/{fnum:.2f}  dFOV={dfov:.2f}deg "
          f"sensor_diag={diag:.2f}mm")

    print(f"  drawing lens diagram -> {os.path.basename(lens_path)}")
    render_lens_diagram_single_bundle(lens, lens_path)

    rgb_path = os.path.join(OUT, f"lens_design_{RANK}_rgb_rms_{rms_tag}um.png")
    print(f"  drawing RGB 3-FOV diagram -> {os.path.basename(rgb_path)}")
    rgb_lens = build_lens_from_theta(
        theta, cfg, device="cpu",
        target_fnum=FNUM_TARGET, target_efl=EFL_TARGET,
    )
    render_lens_diagram_rgb_3fov(rgb_lens, rgb_path)

    spot_path = os.path.join(OUT, f"spot_{RANK}_rms_{rms_tag}um.png")
    print(f"  drawing spot diagram -> {os.path.basename(spot_path)}")
    spot_rms, extent, n_valid = render_spot_diagram(lens, spot_path)
    print(f"    spot: {n_valid}/{SPOT_NUM_RAYS} valid, "
          f"geom_rms={spot_rms:.2f}um, plot_extent=±{extent:.1f}um")

    # Load the template for the paraxial trace (Seidel descriptor uses it
    # to identify the stop and is_stop flags). Use a fresh lens build so
    # the paraxial render doesn't share state with the chart-render lens.
    p = cfg.problem
    template, _ = load_lens_template(
        p.lens_path,
        wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        device="cpu",
    )
    parax_path = os.path.join(OUT, f"paraxial_{RANK}_rms_{rms_tag}um.png")
    print(f"  drawing paraxial seidel rays -> {os.path.basename(parax_path)}")
    parax_lens = build_lens_from_theta(
        theta, cfg, device="cpu",
        target_fnum=FNUM_TARGET, target_efl=EFL_TARGET,
    )
    field_deg, marg_scale, chief_at_stop = render_paraxial_rays(
        parax_lens, template, parax_path,
    )
    print(f"    paraxial: chief field={field_deg:.2f}°  "
          f"marginal_scale={marg_scale:.3f}mm  "
          f"chief_y_at_stop={float(chief_at_stop):.3e} (should be ~0)")

    print()
    print(f"OK -> {lens_path}")
    print(f"OK -> {rgb_path}")
    print(f"OK -> {spot_path}")
    print(f"OK -> {parax_path}")


if __name__ == "__main__":
    main()
