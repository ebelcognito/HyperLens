#!/usr/bin/env python
"""Re-pick top-4 ε-distinct octuplets restricted to an S₁ band, so the
on-axis cores are matched and the off-axis tail-divergence story can be
read cleanly. Same RMS / f-num filters as the original ε-distinct picks
(f/# < 5, RMS < 30 µm), additional constraint |S₁ - median| < 0.002e-3.
Re-runs the PSF radial + encircled-energy overlay for the new picks.
"""
from __future__ import annotations
import json, os, sys, glob, time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("MPMATH_NOGMPY", "1")

import numpy as np
import torch
torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DeepLens"))

import yaml
from scripts_throwaway.psf_radial_and_ee_overlay import render

FNUM_MAX = 5.0
RMS_MAX_UM = 30.0
S1_BAND = 0.002e-3  # raw Seidel units
EPS = 0.30
TOP_N = 4
CFG = ROOT / "experiments/gradient_alignment/octuplet_apsc_champion_config.yaml"


def load_bounds(cfg_path):
    return np.asarray(yaml.safe_load(cfg_path.read_text())["problem"]["theta_bounds"],
                       dtype=np.float64)


def gather_pool():
    out = []
    for fp in sorted(glob.glob(str(ROOT / "results/lensmega/octuplet*/refined_top*.json"))):
        try: data = json.load(open(fp))
        except: continue
        run = Path(fp).parent.name
        for i, des in enumerate(data.get("designs", [])):
            m = des.get("metrics") or {}
            s1 = m.get("seidel_S1")
            s2 = m.get("seidel_S2")
            s3 = m.get("seidel_S3")
            r  = des.get("rms_um")
            fn = des.get("fnum_actual") or m.get("fnum_actual")
            f  = des.get("refined_fitness")
            th = des.get("theta")
            if any(v is None for v in (s1, s2, s3, r, fn, f, th)):
                continue
            out.append(dict(s1=float(s1), s2=float(s2), s3=float(s3),
                             rms=float(r), fnum=float(fn), fit=float(f),
                             theta=np.asarray(th, dtype=np.float64),
                             run=run, idx=i, json_file=Path(fp).name,
                             des=des))
    return out


def eps_net(pool, bounds, eps, top_n):
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    sqrtD = float(np.sqrt(bounds.shape[0]))
    pool_sorted = sorted(pool, key=lambda x: -x["fit"])
    sel = []; sel_u = []
    for rec in pool_sorted:
        if rec["theta"].shape[0] != bounds.shape[0]:
            continue
        u = (rec["theta"] - lo) / span
        ok = True
        for su in sel_u:
            if np.linalg.norm(u - su) / sqrtD <= eps:
                ok = False; break
        if ok:
            sel.append(rec); sel_u.append(u)
            if len(sel) >= top_n: break
    return sel


def main():
    bounds = load_bounds(CFG)
    pool = gather_pool()
    print(f"raw pool with S1,S2,S3: {len(pool)}", flush=True)
    pool_f = [p for p in pool if p["fnum"] < FNUM_MAX and p["rms"] < RMS_MAX_UM]
    print(f"after f/# < {FNUM_MAX} & RMS < {RMS_MAX_UM}: {len(pool_f)}", flush=True)
    s1_med = float(np.median([p["s1"] for p in pool_f]))
    pool_s1 = [p for p in pool_f if abs(p["s1"] - s1_med) < S1_BAND]
    print(f"median S1 (raw) = {s1_med:.6e}  (×1e3 = {s1_med*1e3:+.4f})", flush=True)
    print(f"after |S1 - median| < {S1_BAND:.6e}: {len(pool_s1)}", flush=True)
    picks = eps_net(pool_s1, bounds, EPS, TOP_N)
    print(f"\nε-distinct picks (ε={EPS}, top-{TOP_N}):", flush=True)
    for k, p in enumerate(picks, 1):
        print(f"  #{k}  RMS={p['rms']:6.2f}µm  f/{p['fnum']:.2f}  "
              f"S1={p['s1']*1e3:+.4f}e-3  S2={p['s2']*1e3:+.4f}e-3  "
              f"S3={p['s3']*1e3:+.4f}e-3  src={p['run']}#{p['idx']}", flush=True)

    # Build manifests in the shape render() expects.
    manifests = []
    for k, p in enumerate(picks, 1):
        m = {
            "system": "octuplet",
            "rank": k,
            "source_run": p["run"],
            "source_json": p["json_file"],
            "source_idx": p["idx"],
            "theta": [float(x) for x in p["theta"]],
            "fnum_actual": p["fnum"],
            "fnum_cap": FNUM_MAX, "rms_cap_um": RMS_MAX_UM,
            "s1_band_raw": S1_BAND, "s1_median_raw": s1_med,
            "poly_rms_um": p["rms"],
            "seidel": {
                "seidel_S1": p["s1"], "seidel_S2": p["s2"], "seidel_S3": p["s3"],
            },
        }
        # also grab S4, S5 from metrics if present
        m_src = p["des"].get("metrics") or {}
        for k2 in ("seidel_S4", "seidel_S5"):
            if k2 in m_src: m["seidel"][k2] = float(m_src[k2])
        manifests.append(m)

    # Persist manifests for reproducibility.
    out_dir = ROOT / "figures/psf_radial_ee_overlay/octuplet_S1matched_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, m in enumerate(manifests, 1):
        (out_dir / f"rank{k}_manifest.json").write_text(json.dumps(m, indent=2))

    # Re-run render with these manifests, but force the output filename.
    # render() in the radial-ee script writes to OUT/psf_radial_ee_{sys}.png;
    # we want a distinct name so we don't clobber the original octuplet PNG.
    import scripts_throwaway.psf_radial_and_ee_overlay as mod
    orig_out = mod.OUT
    new_out = ROOT / "figures" / "psf_radial_ee_overlay_S1matched"
    new_out.mkdir(parents=True, exist_ok=True)
    mod.OUT = new_out
    try:
        result = render("octuplet", manifests)
        # Save numerical results.
        (new_out / "psf_radial_ee_results_octuplet_S1matched.json").write_text(
            json.dumps({"octuplet_S1matched": result}, indent=2)
        )
    finally:
        mod.OUT = orig_out


if __name__ == "__main__":
    main()
