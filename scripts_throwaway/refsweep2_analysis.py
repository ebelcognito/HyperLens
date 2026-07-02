#!/usr/bin/env python
"""Final analysis of all sixlet HL hyperparameter sweeps.
Produces:
  figures/refsweep2_data/headline.csv
  figures/refsweep2_data/eps_curves.csv
  figures/refsweep2_sanity.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
OUT = ROOT / "figures/refsweep2_data"
OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open(ROOT / "optical_rl/qd/lensmega/config/examples/sixlet_apsc_discovery_v2.yaml"))
b = np.array(cfg['problem']['theta_bounds'], float)
lo, hi = b[:,0], b[:,1]
D = len(lo); rng_ = hi - lo

EPS_GRID = np.arange(0.05, 0.50 + 1e-9, 0.025)
RMS_THRESHOLDS = [10, 30, 100, 300, 1000]

SOURCES = {
    'A_baseline (512×512, std bounds)':  ROOT / "results/refsweep_outputs/refined_top500_sixlet_seed42_A_baseline.json",
    'SAGA_steelman':                      ROOT / "results/steelman_div020_20260506_123801/sixlet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json",
    'G_grid128':                          ROOT / "results/lensmega/sixlet_REFSWEEP2_G_grid128_seed42_20260509_004316/refined_top500_seed42_G_grid128.json",
    'G_grid1024':                         ROOT / "results/lensmega/sixlet_REFSWEEP2_G_grid1024_seed42_20260509_012705/refined_top500_seed42_G_grid1024.json",
    'G_grid2048':                         ROOT / "results/lensmega/sixlet_REFSWEEP2_G_grid2048_seed42_20260509_043849/refined_top500_seed42_G_grid2048.json",
    'W_wider_bd (5× narrower BD)':       ROOT / "results/lensmega/sixlet_REFSWEEP2_W_wider_bd_seed42_20260509_015423/refined_top500_seed42_W_wider_bd.json",
    'T_tighter_bd (16× narrower BD)':    ROOT / "results/lensmega/sixlet_REFSWEEP2_T_tighter_bd_seed42_20260509_015423/refined_top500_seed42_T_tighter_bd.json",
    'P_pop36':                            ROOT / "results/lensmega/sixlet_REFSWEEP2_P_pop36_seed42_20260509_045502/refined_top500_seed42_P_pop36.json",
    'S_sigma10':                          ROOT / "results/lensmega/sixlet_REFSWEEP2_S_sigma10_seed42_20260509_032358/refined_top500_seed42_S_sigma10.json",
    'F_patience30':                       ROOT / "results/lensmega/sixlet_REFSWEEP2_F_patience30_seed42_20260509_032358/refined_top500_seed42_F_patience30.json",
    'F_patience300':                      ROOT / "results/lensmega/sixlet_REFSWEEP2_F_patience300_seed42_20260509_032358/refined_top500_seed42_F_patience300.json",
}


def load(p):
    if not Path(p).exists(): return None
    return [(x['theta'], x['rms_um'], x['refined_fitness']) for x in json.load(open(p))['designs']]


def eps_dist(pool, eps):
    if not pool: return 0
    s = sorted(pool, key=lambda r: -r[2])
    acc = []
    for theta, _, _ in s:
        u = (np.asarray(theta) - lo) / rng_
        if all(np.linalg.norm(u - a) / np.sqrt(D) > eps for a in acc):
            acc.append(u)
    return len(acc)


headline_rows = []
long_rows = []
method_curves = {}
for name, path in SOURCES.items():
    pool = load(path)
    if pool is None:
        print(f"[skip] {name}: missing"); continue
    rms = np.array([r[1] for r in pool])
    method_curves[name] = {}
    for thresh in RMS_THRESHOLDS:
        sub = [r for r in pool if r[1] < thresh]
        counts = [eps_dist(sub, e) for e in EPS_GRID]
        method_curves[name][thresh] = counts
        for e, c in zip(EPS_GRID, counts):
            long_rows.append({'method': name, 'rms_threshold_um': thresh,
                              'epsilon': float(e), 'eps_distinct_count': c, 'pool_size': len(sub)})
        c20 = counts[np.argmin(np.abs(EPS_GRID - 0.20))]
        c30 = counts[np.argmin(np.abs(EPS_GRID - 0.30))]
        headline_rows.append({
            'method': name, 'rms_threshold_um': thresh,
            'pool_size': len(sub),
            'eps_distinct_at_0.20': c20,
            'eps_distinct_at_0.30': c30,
            'median_rms_um': float(np.median(rms)),
            'min_rms_um': float(rms.min()),
        })

pd.DataFrame(long_rows).to_csv(OUT/"eps_curves.csv", index=False)
head_df = pd.DataFrame(headline_rows)
head_df.to_csv(OUT/"headline.csv", index=False)

print("\n=== HEADLINE SUMMARY ===")
print(f"\n{'method':<40} {'med':>7} {'min':>5} {'<10':>5} {'<30':>5} {'<100':>5} {'<1mm':>5} {'@0.20':>6} {'@0.30':>6}")
for name, path in SOURCES.items():
    if name not in method_curves: continue
    m = head_df[(head_df['method']==name) & (head_df['rms_threshold_um']==1000)].iloc[0]
    pool = load(path); rms = np.array([r[1] for r in pool])
    n5 = int((rms<5).sum()); n10 = int((rms<10).sum())
    n30 = int((rms<30).sum()); n100 = int((rms<100).sum()); n1k = int((rms<1000).sum())
    print(f"{name:<40} {m['median_rms_um']:>7.2f} {m['min_rms_um']:>5.2f} {n10:>5} {n30:>5} {n100:>5} {n1k:>5} {int(m['eps_distinct_at_0.20']):>6} {int(m['eps_distinct_at_0.30']):>6}")

# ---------- figure ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

panels = [10, 30, 100, 1000]
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
n_methods = len(method_curves)
colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
for ax, thresh in zip(axes, panels):
    for (name, curves), col in zip(method_curves.items(), colors):
        if thresh not in curves: continue
        is_baseline = 'baseline' in name or 'SAGA' in name
        lw = 2.5 if is_baseline else 1.5
        ax.plot(EPS_GRID, curves[thresh], 'o-', label=name, color=col, ms=3, lw=lw)
    ax.set_title(f"RMS < {thresh} um")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("eps-distinct count")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc='upper right')

fig.suptitle("Sixlet HL hyperparameter sweep, seed=42, 100k iter")
fig.tight_layout()
out_png = ROOT/"figures/refsweep2_sanity.png"
fig.savefig(out_png, dpi=110)
print(f"\nwrote {out_png}")
