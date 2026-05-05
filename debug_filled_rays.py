"""Debug ray-tracing failures in filled-style renders for the 4 problem lenses.

For each of the 4 specified renders, builds the lens (with the SAME pipeline as
render_filled_collections.py — including the 0.92 aperture shrink for sixlet/
septlet, F/8 stop rescale for triplet) and traces num_rays at each of the 3
fields. Reports per-field per-ray:
  - whether all path points are finite (survivor or failure)
  - entry-y and final-z (to see how far along it got before failure)

This tells us whether the issue is (a) too few rays sampled (need oversampling
+ survivor selection in draw_ray_2d_bundle), (b) some other issue.
"""
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/hisham26/Desktop/RL")
sys.path.insert(0, "/home/hisham26/Desktop/RL/DeepLens")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from deeplens.optics import GeoLens

PROBLEMS = [
    dict(
        name="triplet T2 (9.7µm) — elite_0035_29070k",
        cfg=("/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
             "config/examples/cooke_triplet_materials.yaml"),
        elite=("/home/hisham26/Desktop/RL/results/lensmega/"
               "cooke_triplet_materials_20260414_173935/elite_designs_F8p0/"
               "elite_0035_29070k.json"),
        refined_key="refined_F8",
        target_fnum=8.0, target_efl=50.0, aperture_shrink=1.0,
    ),
    dict(
        name="sixlet S1 (3.4µm) — posthoc_0089_r133_c156",
        cfg=("/home/hisham26/Desktop/RL/results/lensmega/"
             "sixlet_discovery_materials_tightBC_20260426_051752/config.yaml"),
        elite=("/home/hisham26/Desktop/RL/results/lensmega/"
               "sixlet_discovery_materials_tightBC_20260426_051752/elite_designs/"
               "posthoc_0089_r133_c156.json"),
        refined_key="refined",
        target_fnum=None, target_efl=None, aperture_shrink=0.92,
    ),
    dict(
        name="sixlet S3 (22.4µm) — posthoc_0005_r162_c132",
        cfg=("/home/hisham26/Desktop/RL/results/lensmega/"
             "sixlet_discovery_materials_tightBC_20260426_051752/config.yaml"),
        elite=("/home/hisham26/Desktop/RL/results/lensmega/"
               "sixlet_discovery_materials_tightBC_20260426_051752/elite_designs/"
               "posthoc_0005_r162_c132.json"),
        refined_key="refined",
        target_fnum=None, target_efl=None, aperture_shrink=0.92,
    ),
    dict(
        name="septlet V3 (36.4µm) — BEST_0002_8304k",
        cfg=("/home/hisham26/Desktop/RL/results/lensmega/"
             "septlet_F3_20260425_224405/config.yaml"),
        elite=("/home/hisham26/Desktop/RL/results/lensmega/"
               "septlet_F3_20260425_224405/elite_designs/"
               "BEST_0002_8304k.json"),
        refined_key="refined",
        target_fnum=None, target_efl=None, aperture_shrink=0.92,
    ),
]


def build_lens(theta, cfg, target_fnum=None, target_efl=None, aperture_shrink=1.0):
    p = cfg.problem
    template, _ = load_lens_template(
        p.lens_path, wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index, device="cpu",
    )
    solver = GeneralizedParaxialSolver(template)
    lens = GeoLens(filename=p.lens_path, device="cpu")
    _apply_theta_generic(
        lens, np.asarray(theta, dtype=float),
        template=template, solver=solver,
        focal_length_target_mm=p.focal_length_target_mm, device="cpu",
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


def trace_n(lens, fov_deg, wvln_um, num_rays):
    rays = lens.sample_parallel_2D(
        fov=fov_deg, num_rays=num_rays, wvln=wvln_um, depth=-1.0, plane="sagittal",
    )
    _, ray_o_record = lens.trace2sensor(ray=rays, record=True)
    pts = torch.stack(ray_o_record, dim=-2).cpu().numpy()
    if pts.ndim == 3: pts = pts[None, ...]
    return pts[0]   # (num_rays, num_path, 3)


def diagnose(prob):
    print(f"\n========== {prob['name']} ==========")
    cfg = load_config(prob["cfg"])
    with open(prob["elite"]) as f:
        d = json.load(f)
    ref = d.get(prob["refined_key"]) or d.get("refined")
    theta = ref["theta"]
    lens = build_lens(theta, cfg,
                      target_fnum=prob["target_fnum"],
                      target_efl=prob["target_efl"],
                      aperture_shrink=prob["aperture_shrink"])

    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    fovs = np.linspace(0.0, rfov_deg * 0.999, 3)
    waves = [0.486, 0.587, 0.656]
    print(f"  rfov = {rfov_deg:.2f}°, fields = {[f'{f:.2f}' for f in fovs]}°")

    for fov, wvln, name in zip(fovs, waves, ["0H (B)", "0.5H (G)", "1.0H (R)"]):
        for nrays in [2, 9, 15]:
            pts = trace_n(lens, fov, wvln, nrays)
            ok_each = np.all(np.isfinite(pts), axis=(1, 2))   # per-ray
            entry_y = pts[:, 0, 0]
            n_ok = ok_each.sum()
            i_top = int(np.argmax(entry_y)); i_bot = int(np.argmin(entry_y))
            top_ok = bool(ok_each[i_top]); bot_ok = bool(ok_each[i_bot])
            print(f"  field {name}  num_rays={nrays:>2}: surviving={n_ok}/{nrays}  "
                  f"entry-extreme rays — top_idx={i_top} ok={top_ok}  "
                  f"bot_idx={i_bot} ok={bot_ok}")
            if n_ok > 0 and n_ok < nrays:
                # Show which entry-y values failed
                fails = np.where(~ok_each)[0]
                fy = entry_y[fails]
                print(f"      failed rays at entry-y = {[f'{y:.3f}' for y in fy]}")


def main():
    for p in PROBLEMS:
        diagnose(p)


if __name__ == "__main__":
    main()
