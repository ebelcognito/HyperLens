#!/usr/bin/env python3
"""Analyze the sixlet method ablation: Seidel CHAMPION vs Random-projection BC.

Aggregates over 3 seeds per arm.  Only includes runs whose refined JSON exists,
so the script is re-runnable on partial data as the alloc progresses.

Outputs:
  figures/sixlet_methodabl_data/headline.csv      (per-arm mean ± std at fixed RMS+ε)
  figures/sixlet_methodabl_data/per_seed.csv      (per-(arm,seed) raw counts)
  figures/sixlet_methodabl_data/eps_curves.csv    (per arm/seed, ε vs count at each RMS threshold)
  figures/sixlet_methodabl_data/trajectories.csv  (per-run search trajectory: evals, coverage, best_fit)
  figures/sixlet_methodabl_sanity.png             (4-panel quicklook)
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
OUT = ROOT / "figures/sixlet_methodabl_data"
OUT.mkdir(parents=True, exist_ok=True)

# θ-bounds for ε-net normalization (37-D for the sixlet APS-C v2)
cfg = yaml.safe_load(open(ROOT / "configs/ablations/sixlet_REFSWEEP2_CHAMPION.yaml"))
b = np.array(cfg['problem']['theta_bounds'], float)
lo, hi = b[:, 0], b[:, 1]
D = len(lo)
rng_ = hi - lo

EPS_GRID = np.arange(0.05, 0.40 + 1e-9, 0.025)
RMS_THRESHOLDS = [10, 30, 100, 300, 1000]
SEEDS = [42, 43, 44]
COLORS = {'seidel': '#2196F3', 'randbc': '#FF9800'}
LABELS = {'seidel': 'Seidel S₂+S₃ (CHAMPION)', 'randbc': 'Random projection BC'}

# ── locate runs ─────────────────────────────────────────────────────────

def find_run_dir(arm: str, seed: int) -> Path | None:
    """Find newest matching run dir for (arm, seed)."""
    if arm == 'seidel' and seed == 42:
        # Reuse the existing CHAMPION run (different naming convention)
        cand = ROOT / "results/lensmega/sixlet_REFSWEEP2_CHAMPION_seed42_20260509_080537"
        return cand if cand.exists() else None
    pat = str(ROOT / f"results/lensmega/sixlet_methodabl_{arm}_seed{seed}_*")
    matches = sorted(glob.glob(pat))
    return Path(matches[-1]) if matches else None


def refined_json_path(run_dir: Path, arm: str, seed: int) -> Path | None:
    """Return refined JSON path if present."""
    # The existing CHAMPION uses a different naming pattern.
    candidates = [
        run_dir / f"refined_top500_{arm}_seed{seed}.json",
        run_dir / f"refined_top500_seed{seed}_CHAMPION.json",
        run_dir / f"refined_top500_seed{seed}_{arm}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: glob
    matches = list(run_dir.glob("refined_top500*.json"))
    return matches[0] if matches else None


def load_refined(path: Path):
    """Return list of (theta, rms_um, refined_fitness) entries."""
    d = json.loads(path.read_text())
    return [(x['theta'], x['rms_um'], x['refined_fitness']) for x in d['designs']]


def eps_distinct(pool, eps: float) -> int:
    """Greedy ε-net count on normalized θ in [0,1]^D."""
    if not pool:
        return 0
    s = sorted(pool, key=lambda r: -r[2])  # by refined_fitness desc
    acc = []
    for theta, _, _ in s:
        u = (np.asarray(theta) - lo) / rng_
        if all(np.linalg.norm(u - a) / np.sqrt(D) > eps for a in acc):
            acc.append(u)
    return len(acc)


# ── gather data ─────────────────────────────────────────────────────────

per_seed_rows = []
eps_curve_rows = []
traj_rows = []
present = {arm: [] for arm in ('seidel', 'randbc')}

for arm in ('seidel', 'randbc'):
    for seed in SEEDS:
        run_dir = find_run_dir(arm, seed)
        if run_dir is None:
            print(f"[miss] {arm} seed={seed}: no run dir yet")
            continue

        # Trajectory (optional — only present if run completed at all)
        tr = run_dir / 'trajectory.csv'
        if tr.exists():
            try:
                df = pd.read_csv(tr)
                last = df.iloc[-1]
                traj_rows.append({
                    'arm': arm, 'seed': seed,
                    'final_evals': int(last['evals']),
                    'final_best_fitness': float(last['best_fitness']),
                    'final_coverage': float(last['coverage']),
                    'final_restarts': int(last['restarts']) if 'restarts' in df else -1,
                    'wallclock_s': float(last['wallclock_s']),
                })
            except Exception as e:
                print(f"[warn] traj read failed for {arm} seed={seed}: {e}")

        # Refined pool
        rp = refined_json_path(run_dir, arm, seed)
        if rp is None:
            print(f"[partial] {arm} seed={seed}: search done but no refined JSON yet")
            continue

        pool = load_refined(rp)
        present[arm].append(seed)
        rms = np.array([r[1] for r in pool])
        row = {'arm': arm, 'seed': seed,
               'n_pool': len(pool),
               'min_rms_um': float(rms.min()),
               'median_rms_um': float(np.median(rms))}
        for t in RMS_THRESHOLDS:
            sub = [r for r in pool if r[1] < t]
            row[f'n_rms_lt_{t}um'] = len(sub)
            c20 = eps_distinct(sub, 0.20)
            c30 = eps_distinct(sub, 0.30)
            row[f'eps0p20_rms_lt_{t}um'] = c20
            row[f'eps0p30_rms_lt_{t}um'] = c30
            for e in EPS_GRID:
                eps_curve_rows.append({
                    'arm': arm, 'seed': seed, 'rms_threshold_um': t,
                    'epsilon': float(e),
                    'eps_distinct_count': eps_distinct(sub, float(e)),
                    'pool_size': len(sub),
                })
        per_seed_rows.append(row)
        print(f"[ok] {arm} seed={seed}: n_pool={len(pool)} min_rms={rms.min():.2f}µm")

per_seed = pd.DataFrame(per_seed_rows)
traj_df = pd.DataFrame(traj_rows)
eps_df = pd.DataFrame(eps_curve_rows)

per_seed.to_csv(OUT / 'per_seed.csv', index=False)
traj_df.to_csv(OUT / 'trajectories.csv', index=False)
eps_df.to_csv(OUT / 'eps_curves.csv', index=False)

# ── headline aggregate (mean ± std over seeds) ──────────────────────────

if not per_seed.empty:
    numeric_cols = [c for c in per_seed.columns if c not in ('arm', 'seed')]
    agg = per_seed.groupby('arm')[numeric_cols].agg(['mean', 'std', 'count'])
    agg.to_csv(OUT / 'headline.csv')
    print('\nHeadline:')
    print(agg)

# ── sanity panel ────────────────────────────────────────────────────────

if not per_seed.empty:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Refined-RMS CDF per arm (each seed = light line, mean = bold)
    ax = axes[0, 0]
    for arm in ('seidel', 'randbc'):
        if not present[arm]:
            continue
        all_rms = []
        for seed in present[arm]:
            run_dir = find_run_dir(arm, seed)
            rp = refined_json_path(run_dir, arm, seed)
            pool = load_refined(rp)
            rms = np.sort(np.array([r[1] for r in pool]))
            cdf = np.arange(1, len(rms) + 1)
            ax.plot(rms, cdf, color=COLORS[arm], alpha=0.35, lw=1)
            all_rms.append(rms)
        # Combined CDF
        if all_rms:
            combined = np.sort(np.concatenate(all_rms))
            cdf = np.arange(1, len(combined) + 1) / len(all_rms)  # avg-per-seed
            ax.plot(combined, cdf, color=COLORS[arm], lw=2.5, label=LABELS[arm])
    ax.set_xscale('log')
    ax.set_xlabel('Refined RMS (µm)')
    ax.set_ylabel('Cumulative count (designs ≤ RMS)')
    ax.set_title('Refined-RMS CDF (lighter lines = per-seed, bold = combined)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 5000)

    # 2. ε-distinct curves @ rms<100µm
    ax = axes[0, 1]
    target_rms = 100
    for arm in ('seidel', 'randbc'):
        sub = eps_df[(eps_df.arm == arm) & (eps_df.rms_threshold_um == target_rms)]
        if sub.empty:
            continue
        # Per-seed light lines
        for seed in present[arm]:
            ss = sub[sub.seed == seed]
            ax.plot(ss.epsilon, ss.eps_distinct_count, color=COLORS[arm], alpha=0.35, lw=1)
        # Mean line
        mean = sub.groupby('epsilon').eps_distinct_count.mean()
        ax.plot(mean.index, mean.values, color=COLORS[arm], lw=2.5, label=LABELS[arm])
    ax.set_xlabel('ε (normalized θ-distance)')
    ax.set_ylabel(f'ε-distinct count (RMS < {target_rms} µm)')
    ax.set_title(f'Structural diversity: ε-net curves at RMS<{target_rms}µm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Counts at fixed thresholds, mean ± std
    ax = axes[1, 0]
    cols = [f'n_rms_lt_{t}um' for t in RMS_THRESHOLDS]
    x = np.arange(len(RMS_THRESHOLDS))
    w = 0.35
    for i, arm in enumerate(('seidel', 'randbc')):
        sub = per_seed[per_seed.arm == arm]
        if sub.empty:
            continue
        means = [sub[c].mean() for c in cols]
        stds = [sub[c].std() for c in cols]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=4,
               color=COLORS[arm], label=f'{LABELS[arm]} (n={len(sub)})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'<{t}µm' for t in RMS_THRESHOLDS])
    ax.set_ylabel('Designs in pool')
    ax.set_title('Refined-RMS threshold counts (mean ± std over seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Trajectory: coverage vs evals
    ax = axes[1, 1]
    for arm in ('seidel', 'randbc'):
        for seed in SEEDS:
            run_dir = find_run_dir(arm, seed)
            if run_dir is None:
                continue
            tr = run_dir / 'trajectory.csv'
            if not tr.exists():
                continue
            df = pd.read_csv(tr)
            ax.plot(df.evals / 1000, df.coverage * 100, color=COLORS[arm], alpha=0.6, lw=1.5,
                    label=LABELS[arm] if seed == 42 else None)
    ax.set_xlabel('Evaluations (×1000)')
    ax.set_ylabel('Archive coverage (%)')
    ax.set_title('Coverage trajectory (per seed)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Sixlet method ablation: Seidel S₂+S₃ vs Random-projection BC',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(ROOT / 'figures/sixlet_methodabl_sanity.png', dpi=180, bbox_inches='tight')
    print(f"\nSaved sanity figure -> figures/sixlet_methodabl_sanity.png")

print(f"\nDone. Outputs in {OUT}")
print(f"Present runs: seidel={present['seidel']}, randbc={present['randbc']}")
