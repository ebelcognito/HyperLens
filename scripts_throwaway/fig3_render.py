#!/usr/bin/env python
"""Render figures/fig3_sanity.png from figures/fig3_data/."""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys, json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic

CFG_PATH = ROOT / "optical_rl/qd/lensmega/config/examples/cooke_triplet_materials_modernstack.yaml"
DATA = ROOT / "figures/fig3_data"
OUT_PNG = ROOT / "figures/fig3_sanity.png"

start = json.load(open(DATA/"start.json"))
brA = json.load(open(DATA/"branch_A_refined.json"))
brB = json.load(open(DATA/"branch_B_refined.json"))
chosen = json.load(open(DATA/"chosen_seed.json"))
spots_npz = np.load(DATA/"spot_diagrams.npz", allow_pickle=True)
tags = list(spots_npz['tags'])

cfg = load_config(CFG_PATH)
p = cfg.problem


def build_lens(theta):
    template, lens = load_lens_template(
        p.lens_path, wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        device="cpu", cemented_interfaces=tuple(p.cemented_interfaces),
    )
    solver = GeneralizedParaxialSolver(template)
    _apply_theta_generic(
        lens, np.asarray(theta), template=template, solver=solver,
        focal_length_target_mm=p.focal_length_target_mm, device="cpu",
        parameterization=p.parameterization,
        free_curvature_indices=p.free_curvature_indices,
        free_gap_indices=p.free_gap_indices,
        variable_flange=p.variable_flange,
        variable_materials=p.variable_materials,
        free_material_indices=p.free_material_indices,
    )
    return lens, template


def lens_panel(ax, theta, title):
    lens, template = build_lens(theta)
    lens.draw_lens_2d(ax=ax, fig=ax.figure, color="black")
    half_sensor = template.sensor_size_mm[0] / 2.0
    efl = float(lens.foclen)
    hfov_deg = float(np.degrees(np.arctan(half_sensor / max(abs(efl), 1.0))))
    fovs = [(0.0, "tab:blue"), (0.5, "tab:green"), (1.0, "tab:red")]
    for frac, color in fovs:
        try:
            ray = lens.sample_point_source_2D(
                fov=frac * hfov_deg, depth=-20000.0, num_rays=11, wvln=0.587,
            )
            ray.prop_to(-1.0)
            _, rec = lens.trace2sensor(ray=ray, record=True)
            lens.draw_ray_2d(rec, ax=ax, fig=ax.figure, color=color)
        except Exception as e:
            print(f"[draw rays {title}] {e}", flush=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("y (mm)")


def spot_inset(parent_ax, tag, loc=(0.62, 0.05, 0.36, 0.36)):
    """Add a small spot diagram inset to parent_ax."""
    ax = parent_ax.inset_axes(loc)
    x = spots_npz[f"{tag}__x_sensor"]
    y = spots_npz[f"{tag}__y_sensor"]
    fi = spots_npz[f"{tag}__field_index"]
    wi = spots_npz[f"{tag}__wavelength_index"]
    valid = spots_npz[f"{tag}__valid"]
    # color by wavelength
    wcolors = ["tab:blue", "tab:green", "tab:red"]
    # offset spots by field index horizontally for clarity
    field_centers = []
    for f in sorted(set(fi.tolist())):
        mask_f = (fi == f) & valid
        if not mask_f.any(): continue
        cy = float(np.median(y[mask_f]))
        field_centers.append((f, cy))
    # draw rays grouped by (field, wavelength); keep raw x relative to chief, y centered on field
    for f, cy in field_centers:
        for w in sorted(set(wi.tolist())):
            mask = (fi == f) & (wi == w) & valid
            if not mask.any(): continue
            cx = float(np.median(x[mask]))
            ax.scatter(x[mask] - cx, y[mask] - cy, s=1.0,
                       color=wcolors[int(w) % 3], alpha=0.5)
        ax.text(0, cy - cy + 0.0, "", fontsize=6)
    ax.set_aspect('equal')
    ax.set_title('spots (centered)', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.2)


fig = plt.figure(figsize=(12, 12))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])

# Row 1: single full-width panel for start
ax_start = fig.add_subplot(gs[0, :])
S2 = start['S_2']; S3 = start['S_3']
start_label = chosen.get('label', f"seed={start['random_seed']}")
lens_panel(ax_start, start['theta'], f"start: {start_label}  L={start['L']:.3f}  S_2={S2:+.2e}  S_3={S3:+.2e}")
spot_inset(ax_start, "start", loc=(0.66, 0.04, 0.32, 0.42))

# Row 2: branches side-by-side
ax_A = fig.add_subplot(gs[1, 0])
lens_panel(ax_A, brA['theta_post_lbfgs'],
           f"S_2-biased descent + L-BFGS  RMS={brA['L_post']*1000:.1f} um")
spot_inset(ax_A, "branch_A_refined", loc=(0.66, 0.04, 0.32, 0.42))
ax_B = fig.add_subplot(gs[1, 1])
lens_panel(ax_B, brB['theta_post_lbfgs'],
           f"S_3-biased descent + L-BFGS  RMS={brB['L_post']*1000:.1f} um")
spot_inset(ax_B, "branch_B_refined", loc=(0.66, 0.04, 0.32, 0.42))

# Row 3: (S_2, S_3) destinations
ax_d = fig.add_subplot(gs[2, :])
ax_d.scatter([S2], [S3], s=120, facecolors='none', edgecolors='black', label='start')
ax_d.scatter([brA['S_2_post']], [brA['S_3_post']], s=120, color='tab:orange', label='branch A (S_2 bias) refined')
ax_d.scatter([brB['S_2_post']], [brB['S_3_post']], s=120, color='tab:purple', label='branch B (S_3 bias) refined')
for x, y, txt in [(S2, S3, "start"),
                  (brA['S_2_post'], brA['S_3_post'], "A"),
                  (brB['S_2_post'], brB['S_3_post'], "B")]:
    ax_d.annotate(txt, (x, y), xytext=(6,6), textcoords='offset points')
ax_d.annotate("", xy=(brA['S_2_post'], brA['S_3_post']), xytext=(S2, S3),
              arrowprops=dict(arrowstyle='->', alpha=0.4, color='tab:orange'))
ax_d.annotate("", xy=(brB['S_2_post'], brB['S_3_post']), xytext=(S2, S3),
              arrowprops=dict(arrowstyle='->', alpha=0.4, color='tab:purple'))
ax_d.axhline(0, color='gray', lw=0.5, alpha=0.5)
ax_d.axvline(0, color='gray', lw=0.5, alpha=0.5)
ax_d.set_xlabel("S_2 (coma)")
ax_d.set_ylabel("S_3 (astigmatism)")
ax_d.set_title("destinations in descriptor space")
ax_d.legend(fontsize=8)
ax_d.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=110)
print(f"wrote {OUT_PNG}")
