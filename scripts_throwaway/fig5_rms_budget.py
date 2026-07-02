#!/usr/bin/env python
"""Extension: eps-distinct count as a function of RMS budget at fixed eps=0.20.

Tells the post-refinement story directly: at tight RMS budgets (e.g. <20 um)
HL still has many distinct designs while SA+GA is stuck at its saturation
count. Companion to fig5_extract.py.

Outputs:
  figures/fig5_data/count_vs_rms_budget_<system>_<method>.csv  (6 files)
  figures/fig5_data/rms_pool_summary.json
  figures/fig5_rms_budget_sanity.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
DATA = ROOT / "figures" / "fig5_data"
OUT_PNG = ROOT / "figures" / "fig5_rms_budget_sanity.png"

from scripts_throwaway.fig5_extract import (
    SYSTEMS, load_bounds, load_pool, eps_net_select, EPS_TARGET,
)

# Reasonable thresholds spanning "very tight" to "loose"
RMS_BUDGETS_UM = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

# A finer log-spaced sweep for the curve
RMS_CURVE_UM = np.unique(np.round(np.geomspace(2.0, 1000.0, 60), 2))


def eps_distinct_at_budget(pool, budget_um, eps, dim):
    sub = [r for r in pool if r["rms_um"] <= budget_um]
    if not sub:
        return 0
    return len(eps_net_select(sub, eps, dim))


def main():
    pools = {}
    bounds_by_sys = {}
    for sys_name, info in SYSTEMS.items():
        bounds = load_bounds(info["cfg"])
        bounds_by_sys[sys_name] = bounds
        pools[(sys_name, "hl")] = load_pool(info["hl_json"], bounds)
        pools[(sys_name, "saga")] = load_pool(info["saga_json"], bounds)

    # Per-pool RMS summary (post-refinement)
    summary = {}
    for sys_name in ("triplet", "sixlet", "octuplet"):
        for method in ("hl", "saga"):
            pool = pools[(sys_name, method)]
            rms = np.array([r["rms_um"] for r in pool])
            counts_at_budget = {}
            for b in RMS_BUDGETS_UM:
                counts_at_budget[f"<= {b:.0f} um"] = int((rms <= b).sum())
            summary[f"{sys_name}_{method}"] = {
                "n_total": len(pool),
                "rms_min_um": float(rms.min()) if len(rms) else None,
                "rms_p25_um": float(np.percentile(rms, 25)) if len(rms) else None,
                "rms_p50_um": float(np.percentile(rms, 50)) if len(rms) else None,
                "rms_p75_um": float(np.percentile(rms, 75)) if len(rms) else None,
                "rms_p95_um": float(np.percentile(rms, 95)) if len(rms) else None,
                "rms_max_um": float(rms.max()) if len(rms) else None,
                "n_below": counts_at_budget,
            }
    (DATA / "rms_pool_summary.json").write_text(json.dumps(summary, indent=2))
    print("RMS pool summary (post-refinement):")
    for sys_name in ("triplet", "sixlet", "octuplet"):
        for method in ("hl", "saga"):
            s = summary[f"{sys_name}_{method}"]
            print(f"  {sys_name:9s} {method:5s}  n={s['n_total']:4d}  "
                  f"med={s['rms_p50_um']:7.2f} um   "
                  f"p25={s['rms_p25_um']:7.2f}  p75={s['rms_p75_um']:7.2f}  "
                  f"min={s['rms_min_um']:6.2f}")

    # Eps-distinct count vs RMS budget curves
    print("\neps-distinct(eps=0.20) at RMS budgets (um):")
    print(f"  {'system':10s} {'method':6s}  " +
          "  ".join(f"<={int(b):4d}" for b in RMS_BUDGETS_UM))
    curve_rows = {}
    for sys_name in ("triplet", "sixlet", "octuplet"):
        D = bounds_by_sys[sys_name].shape[0]
        for method in ("hl", "saga"):
            pool = pools[(sys_name, method)]
            counts_table = [
                eps_distinct_at_budget(pool, b, EPS_TARGET, D)
                for b in RMS_BUDGETS_UM
            ]
            counts_curve = [
                eps_distinct_at_budget(pool, float(b), EPS_TARGET, D)
                for b in RMS_CURVE_UM
            ]
            print(f"  {sys_name:9s}  {method:5s}  " +
                  "  ".join(f"{c:6d}" for c in counts_table))
            csv = DATA / f"count_vs_rms_budget_{sys_name}_{method}.csv"
            with csv.open("w") as f:
                f.write("rms_budget_um,eps_distinct_count\n")
                for b, c in zip(RMS_CURVE_UM, counts_curve):
                    f.write(f"{b:.4f},{c}\n")
            curve_rows[(sys_name, method)] = (RMS_CURVE_UM, counts_curve)

    # Specific call-out at 20 um
    print("\nCall-out at refined RMS <= 20 um:")
    for sys_name in ("triplet", "sixlet", "octuplet"):
        D = bounds_by_sys[sys_name].shape[0]
        hl_n = (np.array([r["rms_um"] for r in pools[(sys_name, "hl")]]) <= 20.0).sum()
        sa_n = (np.array([r["rms_um"] for r in pools[(sys_name, "saga")]]) <= 20.0).sum()
        hl_d = eps_distinct_at_budget(pools[(sys_name, "hl")], 20.0, EPS_TARGET, D)
        sa_d = eps_distinct_at_budget(pools[(sys_name, "saga")], 20.0, EPS_TARGET, D)
        print(f"  {sys_name:9s}: HL  n_pop={hl_n:3d}  eps20-distinct={hl_d:3d}     "
              f"SAGA  n_pop={sa_n:3d}  eps20-distinct={sa_d:3d}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    sys_names = ("triplet", "sixlet", "octuplet")
    for ax, sys_name in zip(axes, sys_names):
        for method, color, ls in (("hl", "C0", "-"), ("saga", "C3", "--")):
            x, y = curve_rows[(sys_name, method)]
            ax.plot(x, y, ls=ls, color=color, lw=1.6, label=method.upper())
        ax.set_xscale("log")
        ax.set_xlabel("RMS budget [um] (refined RMS <= budget)")
        ax.set_ylabel("eps-distinct count (eps=0.20)")
        ax.set_title(f"{sys_name}")
        ax.axvline(20, color="gray", ls=":", alpha=0.7)
        ax.text(20, ax.get_ylim()[1] * 0.95, "20um", color="gray",
                rotation=90, ha="right", va="top", fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("eps-distinct count vs refined-RMS budget (eps=0.20)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
