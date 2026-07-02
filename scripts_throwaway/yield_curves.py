#!/usr/bin/env python
"""Yield-curve analysis: number of refined designs below an RMS threshold,
across methods. Tests the "thick catalog of refinable designs" reframe.

Inputs:
  - HL_fig4 baseline (existing)
  - NoGrad ablation (existing)
  - RandomBC ablation (existing)
  - SAGA-steelman (existing)
  - Refsweep variants A-E (new this session)

Outputs:
  figures/yield_data/yield_curves.csv   — long-form (method, rms_cap, n_designs_below)
  figures/yield_sanity.png              — log-log yield plot
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
OUT = ROOT / "figures/yield_data"
OUT.mkdir(parents=True, exist_ok=True)

# Triplet sources (adversarial ablation)
TRIPLET_SOURCES = {
    'HL_triplet':       ROOT / "results/lensmega/cooke_triplet_materials_modernstack_seed42_20260506_033503/refined_top500_seed42_hl.json",
    'NoGrad_triplet':   None,  # will be resolved
    'RandomBC_triplet': None,
    'SAGA_triplet':     ROOT / "results/steelman_div020_20260506_123801/triplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json",
}
nograd = sorted(ROOT.glob("results/lensmega/cooke_triplet_modernstack_NOGRAD_seed42_*/refined_top500_seed42_NOGRAD.json"))
randbc = sorted(ROOT.glob("results/lensmega/cooke_triplet_modernstack_RANDBC_seed42_*/refined_top500_seed42_RANDBC.json"))
if nograd: TRIPLET_SOURCES['NoGrad_triplet'] = nograd[-1]
if randbc: TRIPLET_SOURCES['RandomBC_triplet'] = randbc[-1]

# Sixlet sources (refsweep variants)
SIXLET_SOURCES = {
    'HL_sixlet_baseline_fig4': ROOT / "results/lensmega/sixlet_apsc_discovery_v2_seed42_20260506_034601/refined_top500_seed42_hl.json",
    'SAGA_sixlet':             ROOT / "results/steelman_div020_20260506_123801/sixlet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json",
}
for v in ['A_baseline', 'B_efl_tight', 'C_uniform_fw', 'D_long_lbfgs', 'E_periphery']:
    p = ROOT / f"results/refsweep_outputs/refined_top500_sixlet_seed42_{v}.json"
    if p.exists():
        SIXLET_SOURCES[f'sixlet_REFSWEEP_{v}'] = p

# RMS thresholds — log-spaced from 1µm to 1mm
RMS_CAPS = np.unique(np.round(np.logspace(0, 3, 41))).astype(int)


def load_rms(path):
    if path is None or not Path(path).exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return np.array([x['rms_um'] for x in d['designs']])


def yield_curve(rms_array, caps):
    return [int((rms_array < c).sum()) for c in caps]


rows = []
for system_label, sources in [('triplet', TRIPLET_SOURCES), ('sixlet', SIXLET_SOURCES)]:
    for name, path in sources.items():
        rms = load_rms(path)
        if rms is None:
            print(f"[skip] {name}: missing"); continue
        ys = yield_curve(rms, RMS_CAPS)
        for cap, y in zip(RMS_CAPS, ys):
            rows.append({
                'system': system_label, 'method': name,
                'rms_cap_um': int(cap),
                'n_designs_below': y,
                'pool_size': len(rms),
                'median_rms_um': float(np.median(rms)),
            })

df = pd.DataFrame(rows)
df.to_csv(OUT / "yield_curves.csv", index=False)
print(f"wrote {OUT/'yield_curves.csv'}: {len(rows)} rows")

# Headline summary
print("\n=== headline (n_designs_below at key RMS caps) ===")
for cap in [5, 10, 30, 100, 1000]:
    print(f"\n--- rms_cap = {cap} um ---")
    sub = df[df['rms_cap_um'] == cap].sort_values(['system','n_designs_below'], ascending=[True, False])
    for _, r in sub.iterrows():
        print(f"  {r['method']:<32} n={r['n_designs_below']:>4} (pool={r['pool_size']}, median={r['median_rms_um']:.1f}um)")

# ---------- figure ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
for ax, system in zip(axes, ['triplet', 'sixlet']):
    sub = df[df['system'] == system]
    methods = sub['method'].unique()
    cmap = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for m, c in zip(methods, cmap):
        d = sub[sub['method'] == m].sort_values('rms_cap_um')
        ls = ':' if 'REFSWEEP' in m else '-'
        lw = 2.5 if 'HL_' in m and 'REFSWEEP' not in m else 1.6
        ax.plot(d['rms_cap_um'], d['n_designs_below'], 'o-', label=m,
                color=c, ms=3, lw=lw, ls=ls)
    ax.set_xscale('log')
    ax.set_xlabel('refined RMS cap (um)')
    ax.set_ylabel('# designs in pool below cap')
    ax.set_title(f"{system}: yield curve (more = thicker fab catalog)")
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7, loc='upper left')

fig.tight_layout()
fig.savefig(ROOT / "figures/yield_sanity.png", dpi=110)
print(f"\nwrote {ROOT/'figures/yield_sanity.png'}")
