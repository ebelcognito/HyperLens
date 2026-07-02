#!/usr/bin/env python
"""Sanity-check the PSFs: re-trace the 4 designs with the CANONICAL tracing
path, compare per-design spot RMS to the stored metric, and quantify the
lateral colour that per-wavelength PSF recentering hides."""
from __future__ import annotations
import os, sys, json, math
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
os.environ.setdefault("MPMATH_NOGMPY", "1")
from pathlib import Path
ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "DeepLens"))
import numpy as np, torch
from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens)
from optical_rl.qd.lensmega.problem.tracing import disk_samples, trace_fields_batch, rms_spot_batch

sel = json.loads((ROOT/"figures/sixlet_top4_designs_json/selected_top4.json").read_text())
template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(SYSTEMS["sixlet"]["cfg"])
wvlns = list(p_problem.fitness_wavelengths_um)           # [0.486, 0.587, 0.656]
fw = [1.0, 2.0, 4.0]                                     # refine_field_weights
device, dtype = "cpu", torch.float64
N = 512
aperture = torch.from_numpy(disk_samples(N)).to(dtype=dtype)

for di, entry in enumerate(sel):
    theta = np.asarray(entry["design"]["theta"], dtype=np.float64)
    lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
    with torch.no_grad():
        pz, pr = lens.calc_entrance_pupil(paraxial=True)
    ap_xy = aperture * float(pr)
    efl = float(lens.foclen)
    half = lens.sensor_size[0] / 2.0
    hfov = math.degrees(math.atan(half / abs(efl)))
    fields = ((0.0, 0.0), (0.0, 0.5*hfov), (0.0, hfov))

    per_fw_rms = np.zeros((3,3)); inv = np.zeros((3,3))
    cen = np.zeros((3,3,2))
    for wi,w in enumerate(wvlns):
        ray = trace_fields_batch(lens, fields, aperture_xy=ap_xy, pupil_z=float(pz),
            object_depth_mm=p_problem.object_depth_mm, wavelength_um=float(w),
            device=device, dtype=dtype)
        rms, invf = rms_spot_batch(ray)
        per_fw_rms[:,wi] = rms.cpu().numpy()*1e3   # mm->um
        inv[:,wi] = invf.cpu().numpy()
        v = ray.is_valid.unsqueeze(-1)
        cen[:,wi,:] = ((ray.o[...,:2]*v).sum(-2)/v.sum(-2).clamp_min(1e-8)).cpu().numpy()*1e3

    # polychromatic per-field RMS (mean over wvln), then field-weighted aggregate
    perfield = per_fw_rms.mean(axis=1)
    agg = float(np.average(perfield, weights=fw))
    stored = entry["rms_um"]
    print(f"\n=== design {di+1}  stored rms_um={stored:.3f}  | f/{entry['design']['fnum_actual']:.2f}  efl={efl:.2f}mm  hfov={hfov:.1f}deg")
    print(f"   per-field poly RMS (um):  on-axis={perfield[0]:.2f}  mid={perfield[1]:.2f}  edge={perfield[2]:.2f}")
    print(f"   field-weighted(1:2:4) aggregate = {agg:.3f} um   (stored {stored:.3f})  ratio={agg/stored:.2f}")
    # lateral colour = spread of per-wvln centroids at each field
    for fi,nm in enumerate(["on-axis","mid","edge"]):
        c = cen[fi]                      # (3wvln,2)
        latcol = np.linalg.norm(c - c.mean(0), axis=1).max()*2  # full spread um
        print(f"   {nm:7s}: lateral-colour(chief spread)={latcol:6.2f}um   invalid-frac/wvln={inv[fi].round(3).tolist()}")
