"""Filled-style lens diagram renderer for paper figures.

Differences from DeepLens's draw_layout(style='filled'):
  1. Tight figsize matching lens aspect ratio (no large white margins).
  2. Robust ray-bundle: samples 9 rays per field, filters out NaN/inf, then
     uses the surviving topmost+bottommost as the bundle envelope. This
     prevents 'disappearing' bundles when extreme marginals fail to trace
     through poorly-corrected lenses.

After rendering all individual diagrams, builds two composite figures:
  - triplet_collection.png  : 4 triplet picks (lens + USAF row each)
  - sixlet_septlet_collection.png : 3 sixlet scaling + 3 septlet picks

USAF chart renders are NOT regenerated — composites compose existing PNGs.
"""
import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

sys.path.insert(0, "/home/hisham26/Desktop/RL")
sys.path.insert(0, "/home/hisham26/Desktop/RL/DeepLens")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from deeplens.optics import GeoLens

OUT = "/home/hisham26/Desktop/RL/result_figs_all"

# Triplet picks (F/8 stop-rescaled, same as method_overview)
TRIPLET_CFG = ("/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
               "config/examples/cooke_triplet_materials.yaml")
TRIPLET_F8_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
                  "cooke_triplet_materials_20260414_173935/elite_designs_F8p0")
TRIPLET_PICKS = ["elite_0024_23328k", "elite_0035_29070k",
                 "BEST_0005_5544k", "elite_0026_23724k"]
TRIPLET_FNUM = 8.0
TRIPLET_EFL = 50.0

# Sixlet/septlet scaling sets (scaling section)
SETS = {
    "sixlet_scaling": dict(
        cfg=("/home/hisham26/Desktop/RL/results/lensmega/"
             "sixlet_discovery_materials_tightBC_20260426_051752/config.yaml"),
        run_dir=("/home/hisham26/Desktop/RL/results/lensmega/"
                 "sixlet_discovery_materials_tightBC_20260426_051752"),
        prefix="sixlet_scaling",
        picks=["posthoc_0089_r133_c156", "elite_0033_97248k",
               "posthoc_0005_r162_c132"],
    ),
    "sixlet_deeper": dict(
        cfg=("/home/hisham26/Desktop/RL/results/lensmega/"
             "sixlet_discovery_materials_tightBC_20260426_051752/config.yaml"),
        run_dir=("/home/hisham26/Desktop/RL/results/lensmega/"
                 "sixlet_discovery_materials_tightBC_20260426_051752"),
        prefix="sixlet_deeper",
        picks=["elite_0003_32384k", "elite_0026_83520k", "elite_0033_97248k"],
    ),
    "septlet_scaling": dict(
        cfg=("/home/hisham26/Desktop/RL/results/lensmega/"
             "septlet_F3_20260425_224405/config.yaml"),
        run_dir=("/home/hisham26/Desktop/RL/results/lensmega/"
                 "septlet_F3_20260425_224405"),
        prefix="septlet",
        picks=["BEST_0001_4944k", "BEST_0002_8304k", "elite_0020_34872k"],
    ),
}

# Aesthetic params
GLASS_FACECOLOR = "#BFE3F2"
GLASS_EDGECOLOR = "black"
GLASS_LW = 1.4
BUNDLE_ALPHA = 0.22
BUNDLE_COLORS = ["#CC0000", "#006600", "#0066CC"]   # 3 fields (R/G/B)
NUM_FIELDS = 3
NUM_RAYS_PER_FIELD = 9   # over-sample so a few NaN failures don't kill the bundle
WAVE_RGB = [0.486, 0.587, 0.656]


def build_lens(theta, cfg, target_fnum=None, target_efl=None, device="cpu"):
    p = cfg.problem
    template, _ = load_lens_template(
        p.lens_path, wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index, device=device,
    )
    solver = GeneralizedParaxialSolver(template)
    lens = GeoLens(filename=p.lens_path, device=device)
    _apply_theta_generic(
        lens, np.asarray(theta, dtype=float),
        template=template, solver=solver,
        focal_length_target_mm=p.focal_length_target_mm, device=device,
        parameterization=p.parameterization,
        free_curvature_indices=p.free_curvature_indices,
        free_gap_indices=p.free_gap_indices,
        variable_flange=p.variable_flange,
        variable_materials=p.variable_materials,
        free_material_indices=p.free_material_indices,
    )
    if target_fnum is not None and target_efl is not None and lens.aper_idx is not None:
        pupil_target = target_efl / (2.0 * target_fnum)
        with torch.no_grad():
            _, pupilr_now = lens.calc_entrance_pupil(paraxial=True)
        pupilr_now = float(pupilr_now)
        if pupilr_now > 1e-6:
            r_old = float(lens.surfaces[lens.aper_idx].r)
            lens.surfaces[lens.aper_idx].r = r_old * (pupil_target / pupilr_now)
            try:
                lens.post_computation()
            except Exception:
                pass
    return lens


def draw_glass_filled(lens, ax):
    """Filled glass + aperture wedges + sensor — adapted from
    GeoLens.draw_lens_2d_filled but kept inline for full control."""
    surfs = lens.surfaces
    rendered = [False] * len(surfs)
    for i in range(len(surfs)):
        cls = surfs[i].__class__.__name__
        if cls == "Aperture":
            d = float(surfs[i].d.item())
            r = float(surfs[i].r)
            wedge_l = 0.05 * r; wedge_h = 0.18 * r
            for sign in (-1, +1):
                ax.plot([d - wedge_l, d + wedge_l], [sign * r, sign * r],
                        color="#666", lw=GLASS_LW)
                ax.plot([d, d], [sign * r, sign * (r + wedge_h)],
                        color="#666", lw=GLASS_LW)
            rendered[i] = True
            continue
        if rendered[i]:
            continue
        if (i + 1 < len(surfs) and getattr(surfs[i], "mat2", None) is not None
                and surfs[i].mat2.n > 1.1):
            s_front, s_back = surfs[i], surfs[i + 1]
            r_front = float(s_front.draw_r()); r_back = float(s_back.draw_r())
            rr_f = torch.linspace(-r_front, r_front, 256, device=s_front.device)
            z_f = s_front.surface_with_offset(
                rr_f, torch.zeros_like(rr_f), valid_check=False
            ).cpu().detach().numpy()
            y_f = rr_f.cpu().detach().numpy()
            rr_b = torch.linspace(-r_back, r_back, 256, device=s_back.device)
            z_b = s_back.surface_with_offset(
                rr_b, torch.zeros_like(rr_b), valid_check=False
            ).cpu().detach().numpy()
            y_b = rr_b.cpu().detach().numpy()
            zs = np.concatenate([z_f, z_b[::-1]])
            ys = np.concatenate([y_f, y_b[::-1]])
            poly = Polygon(np.column_stack([zs, ys]), closed=True,
                           facecolor=GLASS_FACECOLOR, edgecolor=GLASS_EDGECOLOR,
                           linewidth=GLASS_LW, joinstyle="round")
            ax.add_patch(poly)
            rendered[i] = True; rendered[i + 1] = True
        else:
            surfs[i].draw_widget(ax)
            rendered[i] = True
    d_sensor = float(lens.d_sensor.item())
    r_sensor = float(lens.r_sensor)
    ax.plot([d_sensor, d_sensor], [-r_sensor, r_sensor],
            color=GLASS_EDGECOLOR, lw=GLASS_LW)


def trace_field_robust(lens, fov_deg, wvln_um, num_rays):
    """Sample num_rays parallel rays at fov, trace to sensor with record=True.
    Returns (z, y) arrays of shape (num_rays, num_path) — entries that failed
    along the path (NaN/inf) are kept; we'll filter them per-ray downstream.
    """
    rays = lens.sample_parallel_2D(
        fov=fov_deg, num_rays=num_rays, wvln=wvln_um,
        depth=-1.0, plane="sagittal",
    )
    _, ray_o_record = lens.trace2sensor(ray=rays, record=True)
    pts = torch.stack(ray_o_record, dim=-2).cpu().numpy()
    if pts.ndim == 3:
        pts = pts[None, ...]
    # pts shape: (1, num_rays, num_path, 3)  → axis 2 is path, axis-1 is ray
    z = pts[0, :, :, 2]
    y = pts[0, :, :, 0]
    return z, y


def draw_bundle_robust(ax, z, y, color, alpha):
    """Build a bundle envelope from the surviving topmost & bottommost rays.

    Filtering rule: a ray is 'survivor' if all path points are finite.  Among
    survivors, pick the ray with greatest entry-y (top) and least entry-y
    (bottom). Polygon goes top-fwd + bot-rev (closed).
    """
    finite_along_path = np.all(np.isfinite(z) & np.isfinite(y), axis=1)
    if finite_along_path.sum() < 2:
        return  # nothing to draw, skip silently
    surv_idx = np.where(finite_along_path)[0]
    entry_y = y[surv_idx, 0]
    i_top = surv_idx[np.argmax(entry_y)]
    i_bot = surv_idx[np.argmin(entry_y)]
    z_top, y_top = z[i_top], y[i_top]
    z_bot, y_bot = z[i_bot], y[i_bot]
    zs = np.concatenate([z_top, z_bot[::-1]])
    ys = np.concatenate([y_top, y_bot[::-1]])
    poly = Polygon(np.column_stack([zs, ys]), closed=True,
                   facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(poly)


def render_filled(lens, out_path):
    """Render lens in filled style with tight figsize matching content extent."""
    # Pre-compute lens-extent so we can choose a tight figsize.
    surfs = lens.surfaces
    all_z = [float(s.d.item()) for s in surfs]
    all_z.append(float(lens.d_sensor.item()))
    z_min, z_max = min(all_z), max(all_z)
    width_mm = z_max - z_min + 4.0
    height_mm = 2.0 * float(lens.r_sensor) + 4.0  # leave a touch of padding

    # Figure size in inches: aim for ~9-inch wide, height proportional.
    target_width_in = 9.0
    aspect = width_mm / max(height_mm, 1.0)
    fig_w = target_width_in
    fig_h = max(1.5, target_width_in / max(aspect, 0.1))

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])  # axes fill figure → no margins
    ax.set_facecolor("white")

    draw_glass_filled(lens, ax)

    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    fovs = np.linspace(0.0, rfov_deg * 0.999, NUM_FIELDS)
    for fov, color, wvln in zip(fovs, BUNDLE_COLORS, WAVE_RGB):
        z, y = trace_field_robust(lens, fov, wvln, NUM_RAYS_PER_FIELD)
        draw_bundle_robust(ax, z, y, color, BUNDLE_ALPHA)

    ax.set_aspect("equal")
    # Tight data limits
    ax.set_xlim(z_min - 2.0, z_max + 2.0)
    ax.set_ylim(-height_mm / 2, height_mm / 2)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close(fig)


def render_all_filled():
    print("\n========== triplet ==========")
    cfg = load_config(TRIPLET_CFG)
    items = []
    for stem in TRIPLET_PICKS:
        with open(os.path.join(TRIPLET_F8_DIR, f"{stem}.json")) as f:
            d = json.load(f)
        r = d["refined_F8"]
        rms_um = float(r["rms_mm"]) * 1000.0
        items.append((stem, rms_um, r["theta"]))
    items.sort(key=lambda t: t[1])
    triplet_outpaths = []
    for rank, (stem, rms_um, theta) in enumerate(items, start=1):
        out_path = os.path.join(
            OUT, f"triplet_lens_design_{rank}_rms_{rms_um:.1f}um_filled.png"
        )
        print(f"  rank {rank}: {stem}  rms={rms_um:.2f} µm")
        lens = build_lens(theta, cfg, target_fnum=TRIPLET_FNUM, target_efl=TRIPLET_EFL)
        render_filled(lens, out_path)
        triplet_outpaths.append((rank, rms_um, out_path))
        print(f"    -> {os.path.basename(out_path)}")

    set_outpaths = {}
    for set_name, spec in SETS.items():
        print(f"\n========== {set_name} ==========")
        cfg = load_config(spec["cfg"])
        elite_dir = os.path.join(spec["run_dir"], "elite_designs")
        items = []
        for stem in spec["picks"]:
            with open(os.path.join(elite_dir, f"{stem}.json")) as f:
                d = json.load(f)
            r = d.get("refined") or d.get("refined_F8")
            rms_um = float(r["rms_mm"]) * 1000.0
            items.append((stem, rms_um, r["theta"]))
        items.sort(key=lambda t: t[1])
        outs = []
        for rank, (stem, rms_um, theta) in enumerate(items, start=1):
            out_path = os.path.join(
                OUT, f"{spec['prefix']}_lens_{rank}_rms_{rms_um:.1f}um_filled.png"
            )
            print(f"  rank {rank}: {stem}  rms={rms_um:.2f} µm")
            lens = build_lens(theta, cfg)
            render_filled(lens, out_path)
            outs.append((rank, rms_um, out_path))
            print(f"    -> {os.path.basename(out_path)}")
        set_outpaths[set_name] = outs

    return triplet_outpaths, set_outpaths


# =============================================================================
# Composite figures
# =============================================================================

def find_render(prefix, rank, rms_um):
    """Locate the existing _render_*.png whose RMS tag matches."""
    candidates = [
        os.path.join(OUT, f"{prefix}_render_{rank}_rms_{rms_um:.1f}um.png"),
        os.path.join(OUT, f"{prefix}_render_{rank}_rms_{rms_um:.0f}um.png"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # try fuzzy match
    for f in os.listdir(OUT):
        if f.startswith(f"{prefix}_render_{rank}_rms_") and f.endswith(".png"):
            return os.path.join(OUT, f)
    return None


def composite_grid(rows_data, out_path, family_label_fn, suptitle=None):
    """rows_data: list of (label, lens_png_path, render_png_path | None).
    Layout: each row = [lens diagram | render]; if render_path is None, just lens.
    """
    n_rows = len(rows_data)
    has_renders = any(r[2] is not None for r in rows_data)
    n_cols = 2 if has_renders else 1

    fig_w = 12.0 if has_renders else 9.0
    row_h = 2.6
    fig_h = row_h * n_rows + (0.4 if suptitle else 0)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h),
        gridspec_kw=dict(wspace=0.04, hspace=0.10,
                         width_ratios=[1.4, 1.0] if has_renders else [1]),
    )
    if n_rows == 1:
        axes = np.array([axes]) if n_cols == 1 else np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for r_idx, (label, lens_p, render_p) in enumerate(rows_data):
        ax_l = axes[r_idx, 0]
        if lens_p and os.path.exists(lens_p):
            ax_l.imshow(Image.open(lens_p))
        ax_l.set_xticks([]); ax_l.set_yticks([])
        for sp in ax_l.spines.values(): sp.set_visible(False)
        ax_l.text(0.012, 0.92, label, transform=ax_l.transAxes,
                  fontsize=10, fontweight="bold", color="black",
                  bbox=dict(facecolor="white", edgecolor="none",
                            alpha=0.85, pad=2.0))
        if has_renders:
            ax_r = axes[r_idx, 1]
            if render_p and os.path.exists(render_p):
                ax_r.imshow(Image.open(render_p))
            ax_r.set_xticks([]); ax_r.set_yticks([])
            for sp in ax_r.spines.values(): sp.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.997)

    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


def build_composites(triplet_outpaths, set_outpaths):
    # ----- triplet collection: 4 rows × (lens, render) -----
    triplet_labels = [
        ("T1 — Cooke form", "++-++-"),
        ("T2 — compound-positive", "+++++-"),
        ("T3 — all-meniscus", "------"),
        ("T4 — front-negative split", "--++--"),
    ]
    rows = []
    for (rank, rms_um, lens_p), (label_short, sign) in zip(
        triplet_outpaths, triplet_labels
    ):
        render_p = find_render("triplet", rank, rms_um)
        # Triplet renders use legacy filename pattern
        if render_p is None:
            for f in os.listdir(OUT):
                if (f.startswith(f"triplet_render_{rank}_") and f.endswith(".png")):
                    render_p = os.path.join(OUT, f); break
        label = f"{label_short}  ·  RMS {rms_um:.1f} μm  ·  signs {sign}"
        rows.append((label, lens_p, render_p))
    composite_grid(
        rows,
        os.path.join(OUT, "triplet_collection.png"),
        family_label_fn=None,
        suptitle="Cooke triplet — top-4 geometrically distinct refined families (F/8)",
    )

    # ----- sixlet+septlet collection: sixlet scaling (3) + septlet (3) = 6 rows -----
    sixlet_labels = ["S1 — leading sixlet", "S2 — alternative sixlet",
                     "S3 — alternative sixlet"]
    septlet_labels = ["V1 — leading seventuplet", "V2 — alternative seventuplet",
                      "V3 — alternative seventuplet"]
    rows = []
    for (rank, rms_um, lens_p), label_short in zip(
        set_outpaths["sixlet_scaling"], sixlet_labels
    ):
        render_p = find_render("sixlet_scaling", rank, rms_um)
        rows.append((f"{label_short}  ·  RMS {rms_um:.2f} μm", lens_p, render_p))
    for (rank, rms_um, lens_p), label_short in zip(
        set_outpaths["septlet_scaling"], septlet_labels
    ):
        render_p = find_render("septlet", rank, rms_um)
        rows.append((f"{label_short}  ·  RMS {rms_um:.2f} μm", lens_p, render_p))
    composite_grid(
        rows,
        os.path.join(OUT, "sixlet_septlet_collection.png"),
        family_label_fn=None,
        suptitle="Six- and seven-element archives — representative refined designs",
    )


def main():
    triplet_outpaths, set_outpaths = render_all_filled()
    print("\n========== composites ==========")
    build_composites(triplet_outpaths, set_outpaths)
    print("\nAll done.")


if __name__ == "__main__":
    main()
