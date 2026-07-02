#!/usr/bin/env python
"""Compare the 5 refinement-sweep outputs (sixlet seed=42).

For each variant:
  - pool_size at multiple RMS thresholds {30, 50, 100, 300, 1000} um
  - eps-distinct count at eps in {0.20, 0.30}
  - median + min refined RMS
  - HL/SAGA-steelman ratio at eps=0.20

Outputs:
  figures/refsweep_data/headline.csv
  figures/refsweep_data/eps_curves.csv
  figures/refsweep_sanity.png
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
OUT = ROOT / "figures/refsweep_data"
OUT.mkdir(parents=True, exist_ok=True)

REFSWEEP_GLOB = "results/refsweep_outputs/refined_top500_sixlet_seed42_*.json"
SAGA_PATH = ROOT / "results/steelman_div020_20260506_123801/sixlet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json"
HL_BASELINE_PATH = ROOT / "results/lensmega/sixlet_apsc_discovery_v2_seed42_20260506_034601/refined_top500_seed42_hl.json"
CFG_PATH = ROOT / "optical_rl/qd/lensmega/config/examples/sixlet_apsc_discovery_v2.yaml"

EPS_GRID = np.arange(0.05, 0.50 + 1e-9, 0.025)
RMS_THRESHOLDS_UM = [30, 50, 100, 300, 1000]


def load(path):
    with open(path) as f: d = json.load(f)
    return [(x['theta'], x['rms_um'], x['refined_fitness']) for x in d['designs']]


def eps_distinct(pool, lo, hi, eps, D):
    if not pool: return 0
    s = sorted(pool, key=lambda r: -r[2])  # best fitness first
    rng = hi - lo
    acc = []
    for theta, _, _ in s:
        u = (np.asarray(theta) - lo) / rng
        ok = True
        for a in acc:
            if np.linalg.norm(u - a) / np.sqrt(D) <= eps:
                ok = False; break
        if ok: acc.append(u)
    return len(acc)


def main():
    cfg = yaml.safe_load(open(CFG_PATH))
    bounds = np.array(cfg['problem']['theta_bounds'], float)
    lo, hi = bounds[:,0], bounds[:,1]
    D = len(lo)

    sources = {}
    sources['HL_fig4_baseline'] = HL_BASELINE_PATH
    sources['SAGA_steelman'] = SAGA_PATH
    for p in sorted(ROOT.glob(REFSWEEP_GLOB)):
        tag = p.stem.replace('refined_top500_sixlet_seed42_', 'REFSWEEP_')
        sources[tag] = p

    headline_rows = []
    long_rows = []
    method_curves = {}
    for name, path in sources.items():
        if not Path(path).exists():
            print(f"  [skip] {name}: missing"); continue
        pool = load(path)
        rms = np.array([r[1] for r in pool])
        med = float(np.median(rms))
        mn = float(np.min(rms))
        print(f"\n{name}: n={len(pool)}, median_rms={med:.2f}um, min_rms={mn:.2f}um")
        method_curves[name] = {}
        for thresh in RMS_THRESHOLDS_UM:
            sub = [r for r in pool if r[1] < thresh]
            counts = [eps_distinct(sub, lo, hi, e, D) for e in EPS_GRID]
            method_curves[name][thresh] = counts
            for e, c in zip(EPS_GRID, counts):
                long_rows.append({
                    'method': name, 'rms_threshold_um': thresh,
                    'epsilon': float(e), 'eps_distinct_count': c,
                    'pool_size': len(sub),
                })
            c20 = counts[np.argmin(np.abs(EPS_GRID - 0.20))]
            c30 = counts[np.argmin(np.abs(EPS_GRID - 0.30))]
            headline_rows.append({
                'method': name, 'rms_threshold_um': thresh,
                'pool_size': len(sub),
                'eps_distinct_at_0.20': c20,
                'eps_distinct_at_0.30': c30,
                'median_rms_um': med,
                'min_rms_um': mn,
            })

    pd.DataFrame(long_rows).to_csv(OUT/"eps_curves.csv", index=False)
    head_df = pd.DataFrame(headline_rows)
    head_df.to_csv(OUT/"headline.csv", index=False)

    # Pivot for headline view
    print("\n=== HEADLINE: eps-distinct @ 0.20 ===")
    pv = head_df.pivot(index='method', columns='rms_threshold_um',
                       values='eps_distinct_at_0.20')
    print(pv.to_string())
    print("\n=== HEADLINE: median refined RMS (um) ===")
    print(head_df.groupby('method')['median_rms_um'].first().to_string())
    print("\n=== HEADLINE: HL/SAGA ratio at eps=0.20 ===")
    saga = head_df[head_df['method']=='SAGA_steelman'].set_index('rms_threshold_um')['eps_distinct_at_0.20']
    for m in pv.index:
        if m == 'SAGA_steelman': continue
        ratios = (pv.loc[m] / saga).round(2)
        print(f"  {m:<28} {ratios.to_dict()}")

    # ---------- figure ----------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = RMS_THRESHOLDS_UM
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4.5), sharey=False)
    if n == 1: axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_curves)))
    for ax, thresh in zip(axes, panels):
        for (name, curves), col in zip(method_curves.items(), colors):
            if thresh not in curves: continue
            ls = '-' if 'REFSWEEP' in name else '--'
            lw = 2 if name in ('HL_fig4_baseline','SAGA_steelman') else 1.4
            ax.plot(EPS_GRID, curves[thresh], 'o-', label=name,
                    color=col, ms=3, lw=lw, ls=ls)
        ax.set_title(f"RMS < {thresh} um")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("eps-distinct count")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Sixlet refinement-config sweep, seed=42")
    fig.tight_layout()
    out_png = ROOT / "figures/refsweep_sanity.png"
    fig.savefig(out_png, dpi=110)
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
