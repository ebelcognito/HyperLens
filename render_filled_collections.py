"""Final pass: re-render filled lens diagrams (DEFAULT MARGINS — no crop) for
the designs going into the 3 paper collections, and build lens-only composites.

Inputs:
  Triplet  — cooke_triplet_materials_20260414_173935 / elite_designs_F8p0
              (4 picks: T1 Cooke, T2 compound-positive, T3 all-meniscus,
              T4 front-negative split)
  Sixlet   — sixlet_discovery_materials_tightBC_20260426_051752
              (3 picks: posthoc_0089, elite_0033, posthoc_0005)
  Septlet  — septlet_F3_20260425_224405
              (3 picks: BEST_0001, BEST_0002, elite_0020)
  Decuplet — decuplet_15plus15_F5_20260426_002407
              (2 picks: elite_0003_4770k, elite_0006_6510k)

Style:
  - GeoLens.draw_layout(style="filled", ...) for individual diagrams (default
    matplotlib margins; NO aggressive cropping).
  - Aperture shrunk by 0.92 to keep marginal rays from vignetting at the edge.
  - Composites are pure lens diagrams (NO USAF chart renders).

Outputs to result_figs_all/:
  - triplet_lens_design_{1..4}_rms_*um_filled.png
  - sixlet_scaling_lens_{1..3}_rms_*um_filled.png  (already exists)
  - septlet_lens_{1..3}_rms_*um_filled.png         (already exists)
  - decuplet_lens_{1..2}_rms_*um_filled.png        (NEW)
  - triplet_collection.png        (rebuilt, lens-only, 4 rows)
  - sixlet_septlet_collection.png (NEW, lens-only, 6 rows)
  - decuplet_collection.png       (NEW, lens-only, 2 rows)
"""
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
from deeplens.optics import GeoLens

OUT = "/home/hisham26/Desktop/RL/result_figs_all"
os.makedirs(OUT, exist_ok=True)

GLASS_FACECOLOR = "#BFE3F2"
GLASS_EDGECOLOR = "black"
GLASS_LW = 1.2
BUNDLE_ALPHA = 0.22


# ----- definitions of the 4 sets ---------------------------------------------
TRIPLET = dict(
    cfg=("/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
         "config/examples/cooke_triplet_materials.yaml"),
    elite_dir=("/home/hisham26/Desktop/RL/results/lensmega/"
               "cooke_triplet_materials_20260414_173935/elite_designs_F8p0"),
    refined_key="refined_F8",
    aperture_shrink=1.0,   # F/8 already stopped down — no extra shrink needed
    target_fnum=8.0,
    target_efl=50.0,
    prefix="triplet_lens_design",
    picks=[
        ("elite_0024_23328k", "T1 — Cooke form",            "+ + - + + -"),
        ("elite_0035_29070k", "T2 — all-meniscus",          "- - - - - -"),
        ("BEST_0005_5544k",   "T3 — compound-positive",     "+ + + + + -"),
        ("elite_0026_23724k", "T4 — front-negative split",  "- - + + - -"),
    ],
)

SIXLET = dict(
    cfg=("/home/hisham26/Desktop/RL/results/lensmega/"
         "sixlet_discovery_materials_tightBC_20260426_051752/config.yaml"),
    elite_dir=("/home/hisham26/Desktop/RL/results/lensmega/"
               "sixlet_discovery_materials_tightBC_20260426_051752/elite_designs"),
    refined_key="refined",
    aperture_shrink=1.0,
    target_fnum=None, target_efl=None,
    prefix="sixlet_scaling_lens",
    picks=[
        ("posthoc_0089_r133_c156", "S1 — leading sixlet",     None),
        ("elite_0033_97248k",       "S2 — alternative sixlet", None),
        ("posthoc_0005_r162_c132", "S3 — alternative sixlet", None),
    ],
)

SEPTLET = dict(
    cfg=("/home/hisham26/Desktop/RL/results/lensmega/"
         "septlet_F3_20260425_224405/config.yaml"),
    elite_dir=("/home/hisham26/Desktop/RL/results/lensmega/"
               "septlet_F3_20260425_224405/elite_designs"),
    refined_key="refined",
    aperture_shrink=1.0,
    target_fnum=None, target_efl=None,
    prefix="septlet_lens",
    picks=[
        ("elite_0020_34872k", "V1 — leading seventuplet",     None),
        ("BEST_0001_4944k",   "V2 — alternative seventuplet", None),
        ("BEST_0002_8304k",   "V3 — alternative seventuplet", None),
    ],
)

DECUPLET = dict(
    cfg=("/home/hisham26/Desktop/RL/results/lensmega/"
         "decuplet_15plus15_F5_20260426_002407/config.yaml"),
    elite_dir=("/home/hisham26/Desktop/RL/results/lensmega/"
               "decuplet_15plus15_F5_20260426_002407/elite_designs"),
    refined_key="refined",
    aperture_shrink=1.0,
    target_fnum=None, target_efl=None,
    prefix="decuplet_lens",
    picks=[
        ("elite_0003_4770k", "D1 — decuplet pick 1", None),
        ("elite_0006_6510k", "D2 — decuplet pick 2", None),
    ],
)


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
    if target_fnum is not None and target_efl is not None and lens.aper_idx is not None:
        pupil_target = target_efl / (2.0 * target_fnum)
        with torch.no_grad():
            _, pupilr_now = lens.calc_entrance_pupil(paraxial=True)
        pupilr_now = float(pupilr_now)
        if pupilr_now > 1e-6:
            r_old = float(lens.surfaces[lens.aper_idx].r)
            lens.surfaces[lens.aper_idx].r = r_old * (pupil_target / pupilr_now)
            try: lens.post_computation()
            except Exception: pass
    if aperture_shrink and aperture_shrink != 1.0 and lens.aper_idx is not None:
        r_old = float(lens.surfaces[lens.aper_idx].r)
        lens.surfaces[lens.aper_idx].r = r_old * float(aperture_shrink)
        try: lens.post_computation()
        except Exception: pass
    return lens


def render_set(spec):
    print(f"\n========== {spec['prefix']} ==========")
    cfg = load_config(spec["cfg"])
    items = []
    for stem, label, sign in spec["picks"]:
        with open(os.path.join(spec["elite_dir"], f"{stem}.json")) as f:
            d = json.load(f)
        ref = d.get(spec["refined_key"]) or d.get("refined")
        rms_um = float(ref["rms_mm"]) * 1000.0
        items.append({"stem": stem, "label": label, "sign": sign,
                      "rms_um": rms_um, "theta": ref["theta"]})
    items.sort(key=lambda x: x["rms_um"])

    out_paths = []
    for rank, it in enumerate(items, start=1):
        rms_tag = f"{it['rms_um']:.1f}"
        out_path = os.path.join(
            OUT, f"{spec['prefix']}_{rank}_rms_{rms_tag}um_filled.png"
        )
        print(f"  rank {rank}: {it['stem']}  rms={it['rms_um']:.2f} µm")
        lens = build_lens(it["theta"], cfg,
                          target_fnum=spec.get("target_fnum"),
                          target_efl=spec.get("target_efl"),
                          aperture_shrink=spec["aperture_shrink"])
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
        it["png"] = out_path
        out_paths.append(it)
        print(f"    -> {os.path.basename(out_path)}")
    return out_paths


def composite_lens_only(items_with_pngs, out_path, suptitle):
    n = len(items_with_pngs)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 2.4 * n),
                             gridspec_kw=dict(hspace=0.10))
    if n == 1:
        axes = [axes]
    for ax, it in zip(axes, items_with_pngs):
        ax.imshow(Image.open(it["png"]))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        label = it["label"]
        if it.get("sign"):
            label = f"{label}  ·  RMS {it['rms_um']:.1f} μm  ·  signs {it['sign']}"
        else:
            label = f"{label}  ·  RMS {it['rms_um']:.2f} μm"
        ax.text(0.012, 0.94, label, transform=ax.transAxes,
                fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=2.0))
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.997)
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    render_set(TRIPLET)
    render_set(SIXLET)
    render_set(SEPTLET)
    render_set(DECUPLET)
    print("\nAll individual filled lens diagrams rendered. No composites built.")


if __name__ == "__main__":
    main()
