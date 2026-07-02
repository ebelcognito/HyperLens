#!/usr/bin/env python
"""Top-4 sixlet + top-4 octuplet designs at f/# < 5 and poly RMS < 30 µm,
selected ε-distinct in normalized-θ space (ε = 0.30) — same selector used
for the mono-λ top-32 JSONs. Fixes the 'they all look the same' problem
of taking pure top-N by fitness.

Output: figures/zemax_reports_top4_diverse_fnum5/{sixlet,octuplet}/rank*/
"""
from __future__ import annotations
import json, os, sys, glob
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

from scripts_throwaway.zemax_reports_top3_poly import render_one
from scripts_throwaway.zemax_reports_top3_poly import POOLS as POOL_PREFIXES
from scripts_throwaway.elegant_reports import run_one as run_elegant
import yaml

OUT_ROOT = ROOT / "figures" / "zemax_reports_top4_diverse_fnum5"
FNUM_MAX = 5.0
RMS_MAX_UM = 30.0
EPS = 0.30
TOP_N = 4
TARGETS = ("sixlet", "octuplet")
CANONICAL_CFG = {
    "sixlet":   ROOT / "experiments/gradient_alignment/sixlet_apsc_champion_config.yaml",
    "octuplet": ROOT / "experiments/gradient_alignment/octuplet_apsc_champion_config.yaml",
}


def load_bounds(cfg_path: Path) -> np.ndarray:
    cfg = yaml.safe_load(cfg_path.read_text())
    return np.asarray(cfg["problem"]["theta_bounds"], dtype=np.float64)


def gather_pool(system: str):
    """Return list of (refined_fitness, rms_um, fnum, theta, fp, idx, des)."""
    prefix = POOL_PREFIXES[system]
    out = []
    for fp in sorted(glob.glob(str(ROOT / f"results/lensmega/{prefix}*/refined_top*.json"))):
        try:
            data = json.load(open(fp))
        except Exception:
            continue
        for i, des in enumerate(data.get("designs", [])):
            m = des.get("metrics", {}) or {}
            fn = des.get("fnum_actual") or m.get("fnum_actual")
            f  = des.get("refined_fitness")
            r  = des.get("rms_um")
            th = des.get("theta")
            if fn is None or f is None or r is None or th is None:
                continue
            out.append((float(f), float(r), float(fn),
                        np.asarray(th, dtype=np.float64), fp, i, des))
    return out


def eps_net_pick(pool, bounds: np.ndarray, eps: float, top_n: int):
    """Greedy ε-net in normalized θ-space. Sort by fitness desc, accept if
    dist/sqrt(D) > eps from all previously accepted."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    D = bounds.shape[0]
    sqrtD = float(np.sqrt(D))
    pool_sorted = sorted(pool, key=lambda x: -x[0])
    selected = []
    selected_u = []
    for rec in pool_sorted:
        theta = rec[3]
        if theta.shape[0] != D:
            continue
        u = (theta - lo) / span
        ok = True
        for su in selected_u:
            if np.linalg.norm(u - su) / sqrtD <= eps:
                ok = False; break
        if ok:
            selected.append(rec)
            selected_u.append(u)
            if len(selected) >= top_n:
                break
    return selected


def main():
    import scripts_throwaway.zemax_reports_top3_poly as zr
    zr.OUT_ROOT = OUT_ROOT
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for system in TARGETS:
        print(f"\n=== {system}: ε-distinct (ε={EPS}) top-{TOP_N}, "
              f"f/# < {FNUM_MAX}, RMS < {RMS_MAX_UM} µm ===", flush=True)
        bounds = load_bounds(CANONICAL_CFG[system])
        pool = gather_pool(system)
        pool_filt = [p for p in pool if p[2] < FNUM_MAX and p[1] < RMS_MAX_UM]
        print(f"  pool: {len(pool)}  ->  after f/# & RMS filter: {len(pool_filt)}",
              flush=True)
        picks = eps_net_pick(pool_filt, bounds, EPS, TOP_N)
        print(f"  ε-distinct picks: {len(picks)}", flush=True)
        for rank, (f, r, fn, _, fp, idx, _) in enumerate(picks, 1):
            print(f"   #{rank}  RMS={r:6.2f}µm  f/{fn:.2f}  fit={f:+.5f}  "
                  f"src={Path(fp).parent.name}#{idx}", flush=True)

        for rank, (f, r, fn, theta, fp, idx, des) in enumerate(picks, 1):
            print(f"\n--- rendering {system} #{rank} ---", flush=True)
            try:
                out_dir = render_one(system, rank, fp, idx, des)
                if out_dir is None:
                    continue
                manifest = out_dir / "manifest.json"
                m = json.loads(manifest.read_text())
                m["fnum_actual"] = float(fn)
                m["fnum_cap"] = FNUM_MAX
                m["rms_cap_um"] = RMS_MAX_UM
                m["epsilon"] = EPS
                manifest.write_text(json.dumps(m, indent=2))
                run_elegant(manifest)
            except Exception as e:
                print(f"  ! {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
