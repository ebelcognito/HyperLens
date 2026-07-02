"""Render spotlight lens panels using the canonical 'pretty / filled' style
documented in render_instructions.md.

Adapts the build_lens + lens.draw_layout pipeline from
render_filled_collections.py to render the 3 sixlet APS-C spotlights
identified in results/figure_archive_demo/spotlights.json.

Sixlet uses aperture_shrink=0.92 per the canonical setting.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "DeepLens"))

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from deeplens.optics import GeoLens

GLASS_FACECOLOR = "#BFE3F2"
GLASS_EDGECOLOR = "black"
GLASS_LW = 1.2
BUNDLE_ALPHA = 0.22


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


def main():
    OUT = REPO / "results/figure_archive_demo"
    spotlights = json.loads((OUT / "spotlights.json").read_text())

    HL = REPO / "results/lensmega/sixlet_apsc_discovery_v2_seed42_20260506_034601"
    cfg = load_config(str(HL / "config.yaml"))

    for s in spotlights:
        for which, theta_field in [("raw", "raw_theta"), ("refined", "refined_theta")]:
            out_path = OUT / f"spot{s['id']}_{which}_pretty.png"
            print(f"[render] {out_path.name}  rms={s['refined_rms_um']:.2f}μm" if which=="refined"
                  else f"[render] {out_path.name}  raw rms={s['raw_rms_um']:.0f}μm")
            lens = build_lens(s[theta_field], cfg, aperture_shrink=0.92)
            lens.draw_layout(
                filename=str(out_path),
                depth=float("inf"),
                style="filled",
                lens_title="",
                glass_facecolor=GLASS_FACECOLOR,
                glass_edgecolor=GLASS_EDGECOLOR,
                glass_lw=GLASS_LW,
                bundle_alpha=BUNDLE_ALPHA,
            )
            print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main()
