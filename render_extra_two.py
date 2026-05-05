"""Render two extra cooke-triplet elites in the same thick-line style as
paper_figures/method_overview_renders/lens_design_1..4.png, but with an RGB
palette instead of the Okabe-Ito one.

Reuses build_lens_from_theta + draw_lens_clean from render_method_overview.py
(same F/8 stop rescale, same EFL lock, same lens-body line weights), and
overrides only the ray-bundle palette + count.
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
from render_method_overview import (
    CFG, OUT, FNUM_TARGET, EFL_TARGET, WAVELENGTH_UM,
    LINEWIDTH_LENS, LINEWIDTH_RAY, RAY_ALPHA, NUM_FIELDS,
    build_lens_from_theta, draw_lens_clean, lens_specs,
)

F8_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
          "cooke_triplet_materials_20260414_173935/elite_designs_F8p0")

PICKS = [
    ("elite_0009_9972k",  6),   # rank 6
    ("elite_0018_16596k", 7),   # rank 7
]

# RGB palette (R, G, B) for the three FOV bundles (0H, 0.5H, 1.0H)
RGB_FIELD_COLORS = ("#D62728", "#2CA02C", "#1F77B4")
NUM_RAYS_PER_FIELD = 3        # match originals


def render_lens_diagram_rgb(lens, out_path):
    fig = plt.figure(figsize=(6.0, 3.0), dpi=300, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    draw_lens_clean(lens, ax, color="black", lw_lens=LINEWIDTH_LENS)

    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    fovs = np.linspace(0.0, rfov_deg * 0.999, NUM_FIELDS)

    for fov, color in zip(fovs, RGB_FIELD_COLORS):
        rays = lens.sample_parallel_2D(
            fov=fov,
            num_rays=NUM_RAYS_PER_FIELD,
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
                color=color, lw=LINEWIDTH_RAY, alpha=RAY_ALPHA,
            )

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("")
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                pad_inches=0.02, facecolor="white", transparent=False)
    plt.close(fig)


def main():
    cfg = load_config(CFG)
    for stem, rank in PICKS:
        path = os.path.join(F8_DIR, f"{stem}.json")
        d = json.load(open(path))
        ref = d.get("refined_F8") or d.get("refined")
        theta = ref["theta"]
        rms_um = float(ref["rms_mm"]) * 1000.0
        rms_tag = f"{rms_um:.1f}"

        out_path = os.path.join(OUT, f"lens_design_{rank}_rgb_rms_{rms_tag}um.png")
        print(f"[rank {rank}] {stem}  refined_rms={rms_um:.2f}um")

        lens = build_lens_from_theta(
            theta, cfg, device="cpu",
            target_fnum=FNUM_TARGET, target_efl=EFL_TARGET,
        )
        efl, fnum, dfov, _ = lens_specs(lens)
        print(f"  built: EFL={efl:.3f}mm  F/{fnum:.2f}  dFOV={dfov:.2f}deg")
        print(f"  drawing -> {os.path.basename(out_path)}")
        render_lens_diagram_rgb(lens, out_path)
        print(f"  OK -> {out_path}")


if __name__ == "__main__":
    main()
