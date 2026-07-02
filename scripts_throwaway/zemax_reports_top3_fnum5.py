#!/usr/bin/env python
"""Same as zemax_reports_top3_poly.py but with an f/# < 5 prefilter on each
pool. Produces top-3 sixlet + top-3 octuplet under that constraint, then
renders the elegant PSF + spot + MTF panels via elegant_reports.run_one.

Output: figures/zemax_reports_top3_poly_fnum5/{sixlet,octuplet}/rank*/...
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

from scripts_throwaway.zemax_reports_top3_poly import render_one
from scripts_throwaway.elegant_reports import run_one as run_elegant
from scripts_throwaway.zemax_reports_top3_poly import POOLS as POOL_PREFIXES

OUT_ROOT = ROOT / "figures" / "zemax_reports_top3_poly_fnum5"
FNUM_MAX = 5.0
TARGETS = ("sixlet", "octuplet")


def top3_fnum_filtered(system: str):
    prefix = POOL_PREFIXES[system]
    candidates = []
    for fp in sorted(glob.glob(str(ROOT / f"results/lensmega/{prefix}*/refined_top*.json"))):
        try:
            data = json.load(open(fp))
        except Exception:
            continue
        for i, des in enumerate(data.get("designs", [])):
            m = des.get("metrics", {}) or {}
            fn = des.get("fnum_actual") or m.get("fnum_actual")
            f = des.get("refined_fitness")
            r = des.get("rms_um")
            if fn is None or f is None or r is None:
                continue
            if float(fn) >= FNUM_MAX:
                continue
            candidates.append((float(f), float(r), float(fn), fp, i, des))
    candidates.sort(key=lambda x: -x[0])
    return candidates[:3]


def main():
    # Need to monkey-patch the elegant_reports / zemax_reports_top3_poly
    # so they write to OUT_ROOT rather than the default figures/zemax_reports_top3_poly.
    import scripts_throwaway.zemax_reports_top3_poly as zr
    import scripts_throwaway.elegant_reports as er
    zr.OUT_ROOT = OUT_ROOT
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for system in TARGETS:
        print(f"\n=== {system}: top-3 polychromatic, f/# < {FNUM_MAX} ===",
              flush=True)
        picks = top3_fnum_filtered(system)
        for rank, (f, r, fn, fp, idx, des) in enumerate(picks, 1):
            src = Path(fp).parent.name
            print(f"\n #{rank}  RMS={r:.2f}µm  f/{fn:.2f}  fit={f:+.5f}  "
                  f"src={src}#{idx}", flush=True)
            try:
                out_dir = render_one(system, rank, fp, idx, des)
                if out_dir is None:
                    continue
                manifest = out_dir / "manifest.json"
                # Augment manifest with fnum info.
                m = json.loads(manifest.read_text())
                m["fnum_actual"] = float(fn)
                m["fnum_cap"] = FNUM_MAX
                manifest.write_text(json.dumps(m, indent=2))
                # Elegant panels.
                run_elegant(manifest)
            except Exception as e:
                print(f"  ! {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
