"""Re-render all paper lens diagrams in the FILLED style (no USAF re-render).

For all 4 sets (triplet, sixlet_scaling, sixlet_deeper, septlet) build each
lens from its theta and call GeoLens.draw_layout(style='filled', ...) — the
patched DeepLens vis.py handles the filled-glass + ray-bundle rendering.

USAF chart renders are NOT regenerated — keeps the existing
result_figs_all/*_render_*.png files intact.

Output filenames mirror the existing line-style ones with `_filled` suffix
(e.g. triplet_lens_design_1_rms_1.6um.png  →
       triplet_lens_design_1_rms_1.6um_filled.png).
"""
import json
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/hisham26/Desktop/RL")
sys.path.insert(0, "/home/hisham26/Desktop/RL/DeepLens")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from deeplens.optics import GeoLens

OUT = "/home/hisham26/Desktop/RL/result_figs_all"
os.makedirs(OUT, exist_ok=True)

# ============================================================================
# Triplet — re-renders the same 4 picks as method_overview, at F/8
# (existing line-style images are paper_figures/method_overview_renders/)
TRIPLET_CFG = ("/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
               "config/examples/cooke_triplet_materials.yaml")
TRIPLET_F8_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
                  "cooke_triplet_materials_20260414_173935/elite_designs_F8p0")
TRIPLET_PICKS = [
    "elite_0024_23328k",   # rms ~1.6 µm
    "elite_0035_29070k",   # rms ~9.7 µm
    "BEST_0005_5544k",     # rms ~12.6 µm
    "elite_0026_23724k",   # rms ~12.9 µm
]
TRIPLET_FNUM = 8.0
TRIPLET_EFL = 50.0

# Sixlet + septlet sets — same picks as render_paper_results.py
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

# Filled-style aesthetic params. Single bundle alpha; 3 RGB fields by default.
GLASS_FACECOLOR = "#BFE3F2"
GLASS_EDGECOLOR = "black"
GLASS_LW = 1.2
BUNDLE_ALPHA = 0.22   # slightly lower than prototype's 0.35 to reduce
                      # axial-overlap muddiness (per past tuning notes).


def build_lens(theta, cfg, target_fnum=None, target_efl=None,
               aperture_shrink=1.0, device="cpu"):
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
    # F/# rescale (used for triplet at F/8)
    if target_fnum is not None and target_efl is not None:
        pupil_target = target_efl / (2.0 * target_fnum)
        if lens.aper_idx is not None:
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
    # Optional aperture shrink for visualization — keeps marginal rays from
    # vignetting at the pupil edge, which would otherwise leave gaps in the
    # filled bundle envelope.
    if aperture_shrink is not None and aperture_shrink != 1.0 and lens.aper_idx is not None:
        r_old = float(lens.surfaces[lens.aper_idx].r)
        lens.surfaces[lens.aper_idx].r = r_old * float(aperture_shrink)
        try:
            lens.post_computation()
        except Exception:
            pass
    return lens


def render_filled(lens, out_path):
    """Call DeepLens's filled-style draw_layout, then strip the title."""
    lens.draw_layout(
        filename=out_path,
        depth=float("inf"),
        style="filled",
        lens_title="",
        glass_facecolor=GLASS_FACECOLOR,
        glass_edgecolor=GLASS_EDGECOLOR,
        glass_lw=GLASS_LW,
        bundle_alpha=BUNDLE_ALPHA,
    )


def render_triplet():
    print("\n========== TRIPLET (F/8) ==========")
    cfg = load_config(TRIPLET_CFG)
    items = []
    for stem in TRIPLET_PICKS:
        path = os.path.join(TRIPLET_F8_DIR, f"{stem}.json")
        with open(path) as f:
            d = json.load(f)
        r = d["refined_F8"]
        rms_um = float(r["rms_mm"]) * 1000.0
        items.append((stem, rms_um, r["theta"]))
    items.sort(key=lambda t: t[1])

    for rank, (stem, rms_um, theta) in enumerate(items, start=1):
        rms_tag = f"{rms_um:.1f}"
        out_path = os.path.join(
            OUT, f"triplet_lens_design_{rank}_rms_{rms_tag}um_filled.png"
        )
        print(f"  rank {rank}: {stem}  rms={rms_um:.2f} µm")
        lens = build_lens(theta, cfg,
                          target_fnum=TRIPLET_FNUM, target_efl=TRIPLET_EFL)
        render_filled(lens, out_path)
        print(f"    -> {os.path.basename(out_path)}")


def render_set(set_name):
    spec = SETS[set_name]
    print(f"\n========== {set_name} ==========")
    cfg = load_config(spec["cfg"])
    elite_dir = os.path.join(spec["run_dir"], "elite_designs")

    items = []
    for stem in spec["picks"]:
        path = os.path.join(elite_dir, f"{stem}.json")
        with open(path) as f:
            d = json.load(f)
        r = d.get("refined") or d.get("refined_F8")
        rms_um = float(r["rms_mm"]) * 1000.0
        items.append((stem, rms_um, r["theta"]))
    items.sort(key=lambda t: t[1])

    for rank, (stem, rms_um, theta) in enumerate(items, start=1):
        rms_tag = f"{rms_um:.1f}"
        out_path = os.path.join(
            OUT, f"{spec['prefix']}_lens_{rank}_rms_{rms_tag}um_filled.png"
        )
        print(f"  rank {rank}: {stem}  rms={rms_um:.2f} µm")
        # 0.92 — 8% pupil shrink so marginal rays don't vignette at edge
        lens = build_lens(theta, cfg, aperture_shrink=0.92)
        render_filled(lens, out_path)
        print(f"    -> {os.path.basename(out_path)}")


def main():
    render_triplet()
    for s in SETS:
        render_set(s)
    print("\nAll filled-style lens diagrams rendered.")


if __name__ == "__main__":
    main()
