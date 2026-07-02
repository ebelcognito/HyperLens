#!/usr/bin/env python
"""Ablation analysis: HL vs NoGrad vs RandomBC vs SAGA-steelman on the
canonical (smaller-aperture) cooke triplet, seed=42.

For each of {HL, NoGrad, RandomBC, SAGA}:
  - load refined-top-500 (or refined_above_-1.0 for SAGA)
  - filter by refined RMS < threshold (sweep over {30, 100, 300, 1000} um)
  - compute eps-distinct curve over eps in [0.05, 0.50] step 0.025

Produces:
  figures/ablation_data/eps_curves.csv     — long-form (method, threshold, eps, count)
  figures/ablation_data/headline_table.csv — pivot at eps=0.20, RMS={30,100,1000}
  figures/ablation_sanity.png              — 3 panels (one per RMS threshold)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

# ---------- configurable inputs ----------
HL_REFINED = ROOT / "results/lensmega/cooke_triplet_materials_modernstack_seed42_20260506_033503/refined_top500_seed42_hl.json"
SAGA_REFINED = ROOT / "results/steelman_div020_20260506_123801/triplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json"
# Ablation outputs (will be filled in by run_ablations.sh)
NOGRAD_GLOB = "results/lensmega/cooke_triplet_modernstack_NOGRAD_seed42_*/refined_top500_seed42_NOGRAD.json"
RANDBC_GLOB = "results/lensmega/cooke_triplet_modernstack_RANDBC_seed42_*/refined_top500_seed42_RANDBC.json"
# theta bounds (same for all three ablation variants — same problem)
HL_CFG = ROOT / "optical_rl/qd/lensmega/config/examples/cooke_triplet_materials_modernstack.yaml"

OUT = ROOT / "figures/ablation_data"
OUT.mkdir(parents=True, exist_ok=True)

EPS_GRID = np.arange(0.05, 0.50 + 1e-9, 0.025)
RMS_THRESHOLDS_UM = [30, 100, 300, 1000]


def load_refined(path):
    """Returns DataFrame with columns: theta, refined_fitness, refined_rms_um."""
    with open(path) as f:
        d = json.load(f)
    designs = d['designs']
    rows = []
    for x in designs:
        rows.append({
            'theta': x['theta'],
            'refined_fitness': x['refined_fitness'],
            'refined_rms_um': x['rms_um'],
        })
    return pd.DataFrame(rows)


def eps_curve(df, lo, hi, eps_grid, rms_max_um):
    """Greedy eps-net, sort by refined_fitness DESC (best-first).
    Returns list of counts aligned with eps_grid."""
    sub = df[df['refined_rms_um'] < rms_max_um].copy()
    sub = sub.sort_values('refined_fitness', ascending=False).reset_index(drop=True)
    range_ = hi - lo
    D = len(lo)
    if len(sub) == 0:
        return [0]*len(eps_grid)
    thetas_unit = np.array([(np.asarray(t) - lo) / range_ for t in sub['theta']])
    counts = []
    for eps in eps_grid:
        accepted = []
        for v in thetas_unit:
            ok = True
            for a in accepted:
                if np.linalg.norm(v - a) / np.sqrt(D) <= eps:
                    ok = False; break
            if ok:
                accepted.append(v)
        counts.append(len(accepted))
    return counts


def main():
    cfg = yaml.safe_load(open(HL_CFG))
    bounds = np.array(cfg['problem']['theta_bounds'], dtype=float)
    lo, hi = bounds[:,0], bounds[:,1]

    sources = {}
    sources['HL'] = HL_REFINED
    sources['SAGA-steelman'] = SAGA_REFINED
    # Resolve ablation globs
    nograd_paths = sorted(ROOT.glob(NOGRAD_GLOB))
    randbc_paths = sorted(ROOT.glob(RANDBC_GLOB))
    if nograd_paths:
        sources['NoGrad'] = nograd_paths[-1]
    else:
        print(f"[warn] NOGRAD not yet present: {NOGRAD_GLOB}")
    if randbc_paths:
        sources['RandomBC'] = randbc_paths[-1]
    else:
        print(f"[warn] RANDBC not yet present: {RANDBC_GLOB}")

    long_rows = []
    headline_rows = []
    method_curves = {}
    for method, path in sources.items():
        print(f"loading {method}: {path}")
        if not Path(path).exists():
            print(f"  [skip] missing")
            continue
        df = load_refined(path)
        print(f"  pool size: {len(df)}, refined RMS: median={df['refined_rms_um'].median():.2f}um  min={df['refined_rms_um'].min():.2f}um")
        method_curves[method] = {}
        for thresh in RMS_THRESHOLDS_UM:
            counts = eps_curve(df, lo, hi, EPS_GRID, thresh)
            method_curves[method][thresh] = counts
            n_pool = int((df['refined_rms_um'] < thresh).sum())
            for eps, c in zip(EPS_GRID, counts):
                long_rows.append({
                    'method': method, 'rms_threshold_um': thresh,
                    'epsilon': float(eps), 'eps_distinct_count': int(c),
                    'pool_size': n_pool,
                })
            # headline at eps=0.20
            c20 = counts[np.argmin(np.abs(EPS_GRID - 0.20))]
            c30 = counts[np.argmin(np.abs(EPS_GRID - 0.30))]
            headline_rows.append({
                'method': method, 'rms_threshold_um': thresh,
                'pool_size': n_pool,
                'eps_distinct_at_0.20': int(c20),
                'eps_distinct_at_0.30': int(c30),
            })

    pd.DataFrame(long_rows).to_csv(OUT/"eps_curves.csv", index=False)
    head_df = pd.DataFrame(headline_rows)
    head_df.to_csv(OUT/"headline_table.csv", index=False)
    print("\n=== headline table ===")
    print(head_df.to_string(index=False))

    # ---------- figure ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_color = {'HL': 'C0', 'SAGA-steelman': 'C3', 'NoGrad': 'C1', 'RandomBC': 'C2'}
    panels = [t for t in RMS_THRESHOLDS_UM if t <= 1000]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5), sharey=False)
    if n == 1: axes = [axes]
    for ax, thresh in zip(axes, panels):
        for method, curves in method_curves.items():
            counts = curves.get(thresh, [])
            if not counts: continue
            ax.plot(EPS_GRID, counts, 'o-', label=method,
                    color=method_color.get(method, None), markersize=3)
        ax.set_title(f"RMS < {thresh} um")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("eps-distinct count")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Ablations on canonical cooke triplet, seed=42")
    fig.tight_layout()
    out_png = ROOT / "figures/ablation_sanity.png"
    fig.savefig(out_png, dpi=110)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
