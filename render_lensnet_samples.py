"""Render LensNet ZMX samples in the same style as LensMEGA elite renderings.

Loads each .zmx file from lensnet_samples/ using DeepLens, sets sensor
parameters to match the LensMEGA triplet configuration, renders the 2D lens
layout with ray traces at multiple field angles, and annotates with spot size,
focal length, FOV, F/#, and number of elements.

Usage:
    python render_lensnet_samples.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
_DEEPLENS_ROOT = PROJECT_ROOT / "DeepLens"
for _p in [str(PROJECT_ROOT), str(_DEEPLENS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from deeplens.optics import GeoLens

from optical_rl.qd.render_singlet_archive_designs import (
    PlotTheme,
    _compute_rms_spot,
    _lens_fill_polygons,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ZMX_DIR = PROJECT_ROOT / "lensnet_samples"
OUT_PATH = PROJECT_ROOT / "lensnet_samples" / "lensnet_samples_rendered.png"

# Match LensMEGA triplet discovery sensor config
SENSOR_SIZE = (18.94, 18.94)
R_SENSOR = 13.395

DEPTH_MM = 20000.0       # object at ~20 m (effectively infinity for camera lenses)
WAVELENGTH_UM = 0.5875   # d-line
NUM_RAYS = 11
FOV_FRACS = [0.0, 0.5, 0.9]
DPI = 220
DEVICE = "cpu"

THEME = PlotTheme(
    background="#F0EDE4",
    lens_fill="#7BAFD4",
    lens_outline="#1A4E8C",
    ray_colors=["#E07B39", "#2E8B57", "#8B2252"],
)


# ---------------------------------------------------------------------------
# Spot size computation (multi-field weighted RMS)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_rms_spot_multifield(
    lens,
    depth_mm: float,
    wavelength_um: float,
    num_rays: int = 200,
    num_fields: int = 5,
) -> tuple[float, list[float]]:
    """Compute per-field and weighted-average RMS spot size.

    Returns (avg_rms_mm, list_of_per_field_rms_mm).
    """
    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    field_angles = np.linspace(0.0, rfov_deg, num_fields)
    spots = []
    for fov_deg in field_angles:
        ray = lens.sample_point_source_2D(
            fov=float(fov_deg), depth=depth_mm,
            num_rays=num_rays, wvln=wavelength_um,
        )
        ray.prop_to(-1.0)
        ray_out = lens.trace2sensor(ray=ray)
        valid = ray_out.is_valid
        denom = valid.sum().clamp_min(1)
        xy = ray_out.o[..., :2]
        centroid = (xy * valid.unsqueeze(-1)).sum(dim=0) / denom
        err2 = ((xy - centroid) ** 2).sum(dim=-1)
        rms = ((err2 * valid).sum() / denom + 1e-12).sqrt()
        spots.append(float(rms.item()))

    avg = float(np.mean(spots))
    return avg, spots


# ---------------------------------------------------------------------------
# Load and configure a ZMX lens
# ---------------------------------------------------------------------------

def load_zmx_lens(zmx_path: Path, device: str = DEVICE) -> GeoLens:
    """Load a ZMX file and configure sensor to match LensMEGA triplets."""
    lens = GeoLens(filename=str(zmx_path), device=device)

    # Set sensor to match LensMEGA triplet config
    lens.r_sensor = R_SENSOR
    lens.sensor_size = SENSOR_SIZE
    lens.sensor_res = (2000, 2000)

    # Recompute derived properties with corrected sensor
    try:
        lens.post_computation()
    except Exception:
        pass

    return lens


# ---------------------------------------------------------------------------
# Single-lens renderer
# ---------------------------------------------------------------------------

def render_zmx_lens(
    zmx_path: Path,
    ax,
    theme: PlotTheme,
    label: str,
    depth_mm: float = DEPTH_MM,
    wavelength_um: float = WAVELENGTH_UM,
    num_rays: int = NUM_RAYS,
    fov_fracs: list[float] = FOV_FRACS,
    device: str = DEVICE,
) -> dict:
    lens = load_zmx_lens(zmx_path, device=device)

    # Background and lens fill
    ax.set_facecolor(theme.background)
    if theme.lens_fill is not None:
        for z_poly, x_poly in _lens_fill_polygons(lens):
            ax.fill(z_poly, x_poly, color=theme.lens_fill, alpha=1.0,
                    linewidth=0.0, zorder=1)

    # Draw lens outline
    lens.draw_lens_2d(ax=ax, fig=ax.figure, color=theme.lens_outline)

    # Draw rays at each field fraction
    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    for i, frac in enumerate(fov_fracs):
        fov = frac * rfov_deg
        ray = lens.sample_point_source_2D(
            fov=fov, depth=depth_mm,
            num_rays=num_rays, wvln=wavelength_um,
        )
        ray.prop_to(-1.0)
        _ray_out, ray_o_record = lens.trace2sensor(ray=ray, record=True)
        ray_color = theme.ray_colors[i % len(theme.ray_colors)]
        n_lines_before = len(ax.lines)
        lens.draw_ray_2d(ray_o_record, ax=ax, fig=ax.figure, color=ray_color)
        for line in ax.lines[n_lines_before:]:
            line.set_linewidth(1.8)

    # Compute metrics
    on_axis_spot = _compute_rms_spot(
        lens=lens, depth_mm=depth_mm,
        wavelength_um=wavelength_um, num_rays=200,
    )
    avg_spot, field_spots = compute_rms_spot_multifield(
        lens, depth_mm, wavelength_um, num_rays=200, num_fields=5,
    )

    foclen = float(lens.foclen)
    fov_full_deg = float(np.rad2deg(float(lens.rfov))) * 2
    fnum = float(lens.fnum) if hasattr(lens, "fnum") else foclen / (float(lens.enpd) if lens.enpd else 1.0)

    # Count distinct glass elements
    n_glass_groups = 0
    in_glass = False
    for s in lens.surfaces:
        if float(s.mat2.n) > 1.1:
            if not in_glass:
                n_glass_groups += 1
                in_glass = True
        else:
            in_glass = False

    # Title
    title_lines = [
        f"{label}",
        f"RMS spot: {on_axis_spot*1000:.1f} \u00b5m (on-axis)  |  {avg_spot*1000:.1f} \u00b5m (avg)",
        f"EFL: {foclen:.1f} mm  |  F/{fnum:.1f}  |  FOV: {fov_full_deg:.1f}\u00b0  |  {n_glass_groups} elements",
    ]
    ax.set_title("\n".join(title_lines), fontsize=8, fontweight="bold", pad=8)
    ax.set_axis_off()

    return {
        "label": label,
        "on_axis_spot_mm": on_axis_spot,
        "avg_spot_mm": avg_spot,
        "field_spots_mm": field_spots,
        "focal_length_mm": foclen,
        "fnum": fnum,
        "fov_deg": fov_full_deg,
        "n_elements": n_glass_groups,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import matplotlib.pyplot as plt

    zmx_files = sorted(ZMX_DIR.glob("*.zmx"))
    if not zmx_files:
        print("No .zmx files found in", ZMX_DIR)
        return

    n = len(zmx_files)
    cols = min(n, 3)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(6.0 * cols, 4.5 * rows), dpi=DPI,
    )
    fig.patch.set_facecolor(THEME.background)

    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes]) if n > 1 else np.array([[axes]])
    elif cols == 1:
        axes = np.array([[a] for a in axes])

    all_metrics = []
    for idx, zmx_path in enumerate(zmx_files):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        label = f"LensNet #{idx+1} ({zmx_path.name})"
        print(f"Rendering {label} ...", flush=True)
        metrics = render_zmx_lens(zmx_path, ax, THEME, label)
        all_metrics.append(metrics)
        print(
            f"  EFL={metrics['focal_length_mm']:.1f}mm  "
            f"F/{metrics['fnum']:.1f}  "
            f"FOV={metrics['fov_deg']:.1f}deg  "
            f"spot(on-axis)={metrics['on_axis_spot_mm']*1000:.1f}um  "
            f"spot(avg)={metrics['avg_spot_mm']*1000:.1f}um",
            flush=True,
        )

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        "LensNet Samples \u2014 Comparison Renders",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_PATH, dpi=DPI, facecolor=THEME.background)
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}", flush=True)

    # Also save the converted JSON files for reference
    print("\nConverting ZMX -> JSON for reference...")
    for zmx_path in zmx_files:
        json_path = zmx_path.with_suffix(".json")
        lens = load_zmx_lens(zmx_path)
        lens.write_lens_json(str(json_path))
        print(f"  {zmx_path.name} -> {json_path.name}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Lens':<30} {'EFL':>6} {'F/#':>5} {'FOV':>6} {'Spot(ax)':>10} {'Spot(avg)':>10}")
    print("-" * 70)
    for m in all_metrics:
        print(
            f"{m['label']:<30} "
            f"{m['focal_length_mm']:>5.1f}mm "
            f"f/{m['fnum']:>4.1f} "
            f"{m['fov_deg']:>5.1f}\u00b0 "
            f"{m['on_axis_spot_mm']*1000:>8.1f}\u00b5m "
            f"{m['avg_spot_mm']*1000:>8.1f}\u00b5m"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
