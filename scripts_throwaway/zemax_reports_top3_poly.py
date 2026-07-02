#!/usr/bin/env python
"""Zemax-style reports for the top-3 polychromatic designs per topology
(triplet, sixlet, octuplet) using the existing DeepLens plotters only.

Per design, calls:
  lens.analysis(save_name=)   -> layout, spot diagram, MTF (3 PNGs)
  lens.draw_psf_map(...)       -> RGB PSF field map (1 PNG)
  lens.draw_psf_radial(...)    -> radial PSF cuts (1 PNG)

Output dir: figures/zemax_reports_top3_poly/{system}/rank{N}_polyRMS{R}/
"""
from __future__ import annotations
import json, os, sys, glob, time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch
torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DeepLens"))

from scripts_throwaway.fig5_extract import (
    build_lens_template_and_solver, make_geolens,
)

OUT_ROOT = ROOT / "figures" / "zemax_reports_top3_poly"
DEPTH_MM = -20000.0  # match runs' object_depth_mm
PSF_GRID = (7, 7)
PSF_KS = 64

# Source pools: (system, glob prefix). Used to gather all refined_top*.json.
POOLS = {
    "triplet":  "cooke_triplet",
    "sixlet":   "sixlet",
    "octuplet": "octuplet",
}


def top3_for(system: str):
    """Return list of (run_dir, json_file, design_idx, design_record) for the
    top-3 designs by polychromatic refined_fitness."""
    prefix = POOLS[system]
    all_recs = []
    for fp in sorted(glob.glob(str(ROOT / f"results/lensmega/{prefix}*/refined_top*.json"))):
        try:
            data = json.load(open(fp))
        except Exception:
            continue
        for i, des in enumerate(data.get("designs", [])):
            f = des.get("refined_fitness")
            r = des.get("rms_um")
            if f is None or r is None:
                continue
            all_recs.append((float(f), float(r), fp, i, des))
    all_recs.sort(key=lambda x: -x[0])
    return all_recs[:3]


def render_one(system: str, rank: int, fp: str, idx: int, des: dict):
    run_dir = Path(fp).parent
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        print(f"  ! no config.yaml in {run_dir}", flush=True)
        return None

    theta = np.asarray(des["theta"], dtype=np.float64)
    rms = float(des.get("rms_um", float("nan")))

    # Output dir.
    out_dir = OUT_ROOT / system / f"rank{rank}_polyRMS{rms:.2f}um"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build template + solver from this design's own run config.
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(cfg_path)
    lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver,
                        device="cpu")

    base = str(out_dir / "design")
    t0 = time.time()

    # 1-3. layout + spot + MTF (DeepLens lens.analysis writes 3 PNGs).
    print(f"   - analysis() (layout + spot + MTF)", flush=True)
    lens.analysis(save_name=base, depth=DEPTH_MM)
    # 4. RGB PSF field map (Angular Spectrum Method).
    print(f"   - draw_psf_map(grid={PSF_GRID})", flush=True)
    lens.draw_psf_map(grid=PSF_GRID, ks=PSF_KS, depth=DEPTH_MM,
                      save_name=f"{base}_psf_map.png")
    # 5. radial PSF cuts.
    print(f"   - draw_psf_radial(M=5)", flush=True)
    lens.draw_psf_radial(M=5, depth=DEPTH_MM, ks=PSF_KS,
                          save_name=f"{base}_psf_radial.png")

    # Manifest.
    manifest = {
        "system": system, "rank": rank,
        "poly_rms_um": rms,
        "poly_refined_fitness": float(des.get("refined_fitness", 0.0)),
        "source_run": run_dir.name,
        "source_json": Path(fp).name, "source_idx": idx,
        "theta": [float(x) for x in theta],
        "wavelengths_um": [0.486, 0.587, 0.656],
        "depth_mm": DEPTH_MM,
        "psf_grid": list(PSF_GRID), "psf_ks": PSF_KS,
        "files": {
            "layout": "design.png",
            "spot": "design_spot.png",
            "mtf": "design_mtf.png",
            "psf_map": "design_psf_map.png",
            "psf_radial": "design_psf_radial.png",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"   wrote {out_dir}  ({time.time()-t0:.1f}s)", flush=True)
    return out_dir


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for system in POOLS:
        print(f"\n=== {system}: top-3 polychromatic ===", flush=True)
        picks = top3_for(system)
        for rank, (f, r, fp, idx, des) in enumerate(picks, 1):
            src = Path(fp).parent.name
            print(f"\n #{rank}  poly_RMS={r:.2f}µm  fit={f:+.5f}  src={src}#{idx}",
                  flush=True)
            try:
                render_one(system, rank, fp, idx, des)
            except Exception as e:
                print(f"  ! render failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
