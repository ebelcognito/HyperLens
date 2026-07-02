"""Pretty (filled-style) renders for the 6 f/4 designs in fnum4_designs/.

Uses the CANONICAL render recipe documented in
figures/method_library_data/RENDERING_RECIPE.md (the same path that produced
figures/octuplet_top17_eps030_lenses/). The key reason we do NOT use
GeoLens.draw_layout(style="filled"): its default fill helper clips each glass
element to the SMALLER of its two surface radii, which mangles meniscus and
cemented (fused) elements — visible white wedges / broken cement interfaces.

Correct pipeline (see RENDERING_RECIPE.md §3):
  build lens from theta  ->  render_one(ax, lens):
    draw_filled_bundles  (3-field R/G/B envelopes)
    lens_fill_polygons   (per-surface semi-aperture spans, cement-safe)
    draw_lens_clean      (thin outlines + edge connectors + dumbbell stop)

Each design JSON carries {system, design:{theta}}. We map the system to the
matching REFSWEEP problem config (same template + parameterization +
cemented_interfaces the design was optimized under) so the theta reconstructs
the exact lens, then hand the GeoLens to the recipe's render_one.

Outputs -> fnum4_designs_renders/<stem>_filled.png
"""
import os
import sys
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nas/longleaf/home/hisham26/Desktop/hyperlens"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "DeepLens"))
os.chdir(ROOT)

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from deeplens.optics import GeoLens

# Canonical recipe renderer (cement-safe). See RENDERING_RECIPE.md.
from scripts_throwaway.render_spotlight_sixlet_designs import render_one, BLUE_FILL

IN_DIR = os.path.join(ROOT, "fnum4_designs")
OUT_DIR = os.path.join(ROOT, "fnum4_designs_renders")

# system -> problem config providing the template + parameterization the
# designs were optimized under (chosen to match each design's source_run).
SYSTEM_CFG = {
    "sixlet": os.path.join(
        ROOT, "configs/ablations/sixlet_REFSWEEP2_G_grid2048.yaml"),
    "octuplet": os.path.join(
        ROOT, "configs/ablations/octuplet_REFSWEEP_D_long_lbfgs.yaml"),
}


def build_lens(theta, cfg, device="cpu"):
    p = cfg.problem
    template, _ = load_lens_template(
        p.lens_path, wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        cemented_interfaces=tuple(getattr(p, "cemented_interfaces", ()) or ()),
        device=device,
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
    return lens


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg_cache = {}

    paths = sorted(glob.glob(os.path.join(IN_DIR, "*.json")))
    print(f"Rendering {len(paths)} designs -> {OUT_DIR}")
    for jp in paths:
        with open(jp) as f:
            d = json.load(f)
        system = d["system"]
        theta = d["design"]["theta"]
        cfg_path = SYSTEM_CFG.get(system)
        if cfg_path is None:
            print(f"  SKIP {os.path.basename(jp)}: no config for '{system}'")
            continue
        if cfg_path not in cfg_cache:
            cfg_cache[cfg_path] = load_config(cfg_path)
        cfg = cfg_cache[cfg_path]

        stem = os.path.splitext(os.path.basename(jp))[0]
        out_path = os.path.join(OUT_DIR, f"{stem}_filled.png")
        fn = d.get("fnum_actual")
        rms = d.get("rms_um_poly")
        print(f"  {stem}  (f/{fn:.3f}, RMS {rms:.2f} um)")

        lens = build_lens(theta, cfg)
        fig, ax = plt.subplots(figsize=(7.0, 3.4), facecolor="white")
        ax.set_facecolor("white")
        render_one(ax, lens, fill_color=BLUE_FILL)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
