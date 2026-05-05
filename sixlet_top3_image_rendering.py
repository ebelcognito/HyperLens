"""
Render a test chart through each of the top-3 sixlet design families.

For each family:
  1. Reconstruct GeoLens from refined theta
  2. Resize the USAF chart to match the lens's sensor resolution
  3. Run lens.render() (PSF-map method, fast) on the chart
  4. Save the rendered image side-by-side with the original

Output: a comparison figure showing the original chart and the same chart as
imaged through each design family. Differences in sharpness and distortion
are visible.
"""
import glob
import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, "/home/hisham26/Desktop/RL")
sys.path.insert(0, "/home/hisham26/Desktop/RL/DeepLens")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic


CFG = ("/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
       "config/examples/sixlet_discovery.yaml")
RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "sixlet_discovery_20260423_132001")
GALLERY_JSON = ("/home/hisham26/Desktop/RL/paper_figures/sixlet_design_gallery/"
                "design_family_summary.json")
CHART_PATH = "/home/hisham26/Desktop/RL/DeepLens/datasets/charts/USAF_1951_1k.png"
OUT = "/home/hisham26/Desktop/RL/paper_figures/sixlet_top3_imaging"
os.makedirs(OUT, exist_ok=True)

SENSOR_RES = (512, 512)     # render-time resolution; balance speed vs sharpness
DEPTH = -20000.0
PSF_GRID = (12, 12)
PSF_KS = 51                  # PSF kernel size


def load_top3_distinct():
    with open(GALLERY_JSON) as f:
        d = json.load(f)
    by_pattern = {}
    for b in d["basins"]:
        sp = b["sign_pattern"]
        if sp not in by_pattern or b["rms_um_refined"] < by_pattern[sp]["rms_um_refined"]:
            by_pattern[sp] = b
    return sorted(by_pattern.values(), key=lambda b: b["rms_um_refined"])[:3]


def load_chart(target_size):
    img = Image.open(CHART_PATH).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # to (1, 3, H, W)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def build_geolens_from_theta(theta_arr, cfg, device="cpu"):
    """Build a GeoLens from a refined theta vector."""
    from deeplens.optics import GeoLens
    p = cfg.problem
    template, _ = load_lens_template(
        p.lens_path,
        wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        device=device,
    )
    solver = GeneralizedParaxialSolver(template)
    lens = GeoLens(filename=p.lens_path, device=device)
    _apply_theta_generic(
        lens, theta_arr,
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


def main():
    cfg = load_config(CFG)
    top3 = load_top3_distinct()

    print("Top-3 structurally distinct families:")
    for r in top3:
        print(f"  basin {r['id']} · {r['rep_name']} · RMS {r['rms_um_refined']:.2f}μm "
              f"· sign {r['sign_pattern']}")

    # Load chart
    chart = load_chart(SENSOR_RES)   # (1, 3, H, W)
    print(f"\nchart size: {chart.shape}")

    # Render through each lens
    results = []
    for r in top3:
        path = os.path.join(RUN_DIR, "elite_designs", f"{r['rep_name']}.json")
        with open(path) as f:
            d = json.load(f)
        theta = np.array(d["refined"]["theta"], dtype=float)

        lens = build_geolens_from_theta(theta, cfg, device="cpu")
        # Force sensor resolution to match chart
        lens.sensor_res = SENSOR_RES
        lens.set_sensor(sensor_res=SENSOR_RES, sensor_size=lens.sensor_size)

        print(f"\nrendering basin {r['id']} (RMS {r['rms_um_refined']:.2f}μm)...")
        try:
            with torch.no_grad():
                rendered = lens.render(
                    chart, depth=DEPTH, method="psf_map",
                    psf_grid=PSF_GRID, psf_ks=PSF_KS,
                )
            arr = rendered.squeeze(0).permute(1, 2, 0).cpu().numpy()
            arr = np.clip(arr, 0, 1)
        except Exception as e:
            print(f"  PSF-map render failed: {e}; falling back to single PSF patch")
            arr = None
        results.append((r, arr))

    # Build figure: original + 3 rendered, side-by-side
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6),
                             gridspec_kw=dict(wspace=0.04))

    # Original
    chart_np = chart.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ax = axes[0]
    ax.imshow(chart_np); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Original USAF 1951 chart\n(input to all lenses)",
                 fontsize=11, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_edgecolor("black"); sp.set_linewidth(2)

    family_colors = ["#E45756", "#4C78A8", "#54A24B"]
    for i, ((r, arr), color) in enumerate(zip(results, family_colors)):
        ax = axes[i + 1]
        if arr is not None:
            ax.imshow(arr)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3)
        ax.set_title(
            f"Family {i+1} — basin {r['id']}\n"
            f"sign {r['sign_pattern']} · RMS {r['rms_um_refined']:.2f} μm",
            fontsize=11, color=color, fontweight="bold")

    fig.suptitle(
        "USAF chart rendered through each of the top-3 sixlet design families\n"
        "(PSF-map convolution, depth = ∞)",
        fontsize=13, fontweight="bold",
    )
    out_png = os.path.join(OUT, "sixlet_top3_imaging.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_png}")

    # Also save individual renders for inspection
    for (r, arr) in results:
        if arr is None:
            continue
        outp = os.path.join(OUT, f"basin_{r['id']}_USAF_render.png")
        Image.fromarray((arr * 255).astype(np.uint8)).save(outp)
        print(f"saved {outp}")


if __name__ == "__main__":
    main()
