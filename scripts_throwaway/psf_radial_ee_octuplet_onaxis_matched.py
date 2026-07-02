#!/usr/bin/env python
"""Re-pick top-4 ε-distinct octuplets restricted to designs whose on-axis
RMS (re-evaluated at λ=587 nm) falls in a tight band — i.e., enforce
matched on-axis cores by the most direct measure rather than via S₁.

Steps:
  1. Filter octuplet pool to (f/# < 5, RMS < 30 µm).
  2. For each surviving design, re-compute monochromatic on-axis RMS at
     587 nm (single field, single wavelength, fast eval).
  3. Keep designs whose on-axis RMS is within a tight band (±10% of the
     in-band median, narrowed if too many survive).
  4. Apply ε-net (ε=0.30, normalized θ-space) and pick top-4 by fitness.
  5. Re-render the PSF radial + EE overlay for these picks.
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
from scripts_throwaway.fig5_extract import build_lens_template_and_solver, make_geolens
from scripts_throwaway.psf_radial_and_ee_overlay import render

CFG = ROOT / "experiments/gradient_alignment/octuplet_apsc_champion_config.yaml"
FNUM_MAX = 5.0
RMS_MAX_UM = 30.0
EPS = 0.30
TOP_N = 4
WVLN_UM = 0.587
N_RAYS = 128
CACHE = ROOT / "scripts_throwaway/octuplet_onaxis_rms_cache.json"


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
            s1 = m.get("seidel_S1"); s2 = m.get("seidel_S2"); s3 = m.get("seidel_S3")
            r = des.get("rms_um")
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


def onaxis_rms_um(lens, depth_mm=-20000.0) -> float:
    from optical_rl.qd.lensmega.problem.tracing import disk_samples, trace_fields_batch
    aperture = torch.from_numpy(disk_samples(N_RAYS)).to(dtype=torch.float64, device="cpu")
    with torch.no_grad():
        pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
    pupil_z = float(pupilz); pupil_r = float(pupilr)
    aperture_xy = aperture * pupil_r
    ray = trace_fields_batch(lens, ((0.0, 0.0),), aperture_xy=aperture_xy,
                              pupil_z=pupil_z, object_depth_mm=depth_mm,
                              wavelength_um=float(WVLN_UM),
                              device="cpu", dtype=torch.float64)
    xy = ray.o[..., :2].detach().cpu().numpy().reshape(-1, 2) * 1000.0
    valid = (ray.is_valid > 0).detach().cpu().numpy().reshape(-1)
    if valid.sum() == 0:
        return float("nan")
    pts = xy[valid]
    cen = pts.mean(axis=0)
    r2 = ((pts - cen) ** 2).sum(axis=1)
    return float(np.sqrt(r2.mean()))


def build_cache(pool):
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(CFG)
    out = {}
    t0 = time.time()
    for k, p in enumerate(pool):
        key = f"{p['run']}#{p['idx']}"
        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path, p["theta"],
                                 template, solver, device="cpu")
            r = onaxis_rms_um(lens)
        except Exception:
            r = float("nan")
        out[key] = r
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(pool)} on-axis RMS computed ({time.time()-t0:.1f}s)",
                  flush=True)
    return out


def eps_net(pool, bounds, eps, top_n):
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    sqrtD = float(np.sqrt(bounds.shape[0]))
    pool_sorted = sorted(pool, key=lambda x: -x["fit"])
    sel = []; sel_u = []
    for rec in pool_sorted:
        if rec["theta"].shape[0] != bounds.shape[0]: continue
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
    pool_f = [p for p in pool if p["fnum"] < FNUM_MAX and p["rms"] < RMS_MAX_UM]
    print(f"pool after f/#<{FNUM_MAX} & RMS<{RMS_MAX_UM}: {len(pool_f)}", flush=True)

    if CACHE.exists():
        print(f"using cache {CACHE}", flush=True)
        cache = json.loads(CACHE.read_text())
        # ensure every design in pool_f has cache entry; recompute if not
        missing = [p for p in pool_f if f"{p['run']}#{p['idx']}" not in cache]
        if missing:
            print(f"  computing on-axis RMS for {len(missing)} missing designs",
                  flush=True)
            extra = build_cache(missing)
            cache.update(extra)
            CACHE.write_text(json.dumps(cache, indent=2))
    else:
        print(f"computing on-axis RMS for all {len(pool_f)} candidates", flush=True)
        cache = build_cache(pool_f)
        CACHE.write_text(json.dumps(cache, indent=2))

    # Attach on-axis RMS to each design.
    for p in pool_f:
        p["onaxis_rms_um"] = cache.get(f"{p['run']}#{p['idx']}", float("nan"))
    pool_finite = [p for p in pool_f if np.isfinite(p["onaxis_rms_um"])]
    onaxis = np.array([p["onaxis_rms_um"] for p in pool_finite])
    print(f"\non-axis RMS distribution (µm) over n={len(onaxis)}:", flush=True)
    print(f"  min={onaxis.min():.2f}  p10={np.percentile(onaxis,10):.2f}  "
          f"p25={np.percentile(onaxis,25):.2f}  median={np.median(onaxis):.2f}  "
          f"p75={np.percentile(onaxis,75):.2f}  max={onaxis.max():.2f}", flush=True)

    # Target: pick a tight band around the BEST on-axis values (so we get
    # designs that are individually good on-axis AND matched to each other).
    # Try a few band widths and pick one yielding >= 8 candidates (gives the
    # ε-net some slack), starting at the lowest available bucket.
    best_floor = np.percentile(onaxis, 5)  # near-best on-axis (5th percentile)
    for band_frac in (0.20, 0.30, 0.40, 0.50, 0.75, 1.00):
        band_hi = best_floor * (1 + band_frac)
        in_band = [p for p in pool_finite
                   if best_floor * (1 - band_frac) <= p["onaxis_rms_um"] <= band_hi]
        if len(in_band) >= 8:
            print(f"  band: on-axis RMS in [{best_floor*(1-band_frac):.2f}, "
                  f"{band_hi:.2f}] µm  (±{int(band_frac*100)}% of best-quintile floor "
                  f"{best_floor:.2f}): n={len(in_band)}", flush=True)
            break
    else:
        in_band = sorted(pool_finite, key=lambda x: x["onaxis_rms_um"])[:20]
        print(f"  fallback: take 20 lowest-on-axis-RMS designs", flush=True)

    picks = eps_net(in_band, bounds, EPS, TOP_N)
    print(f"\nε-distinct picks (ε={EPS}, top-{TOP_N}):", flush=True)
    for k, p in enumerate(picks, 1):
        print(f"  #{k}  on-axis RMS={p['onaxis_rms_um']:5.2f}µm  field-avg RMS={p['rms']:5.2f}µm  "
              f"f/{p['fnum']:.2f}  S1={p['s1']*1e3:+.4f}e-3  "
              f"S2={p['s2']*1e3:+.4f}e-3  S3={p['s3']*1e3:+.4f}e-3  "
              f"src={p['run']}#{p['idx']}", flush=True)

    # Build manifests + render.
    manifests = []
    for k, p in enumerate(picks, 1):
        m = {
            "system": "octuplet", "rank": k,
            "source_run": p["run"], "source_json": p["json_file"],
            "source_idx": p["idx"],
            "theta": [float(x) for x in p["theta"]],
            "fnum_actual": p["fnum"], "fnum_cap": FNUM_MAX,
            "rms_cap_um": RMS_MAX_UM,
            "onaxis_band_um": [float(best_floor * (1 - band_frac)),
                                float(best_floor * (1 + band_frac))],
            "onaxis_rms_um": float(p["onaxis_rms_um"]),
            "poly_rms_um": p["rms"],
            "seidel": {
                "seidel_S1": p["s1"], "seidel_S2": p["s2"], "seidel_S3": p["s3"],
            },
        }
        m_src = p["des"].get("metrics") or {}
        for k2 in ("seidel_S4", "seidel_S5"):
            if k2 in m_src: m["seidel"][k2] = float(m_src[k2])
        manifests.append(m)

    out_dir = ROOT / "figures/psf_radial_ee_overlay_onaxis_matched"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, m in enumerate(manifests, 1):
        (out_dir / f"octuplet_rank{k}_manifest.json").write_text(json.dumps(m, indent=2))

    import scripts_throwaway.psf_radial_and_ee_overlay as mod
    orig_out = mod.OUT
    mod.OUT = out_dir
    try:
        result = render("octuplet", manifests)
        (out_dir / "psf_radial_ee_results_octuplet_onaxis_matched.json").write_text(
            json.dumps({"octuplet_onaxis_matched": result}, indent=2)
        )
    finally:
        mod.OUT = orig_out


if __name__ == "__main__":
    main()
