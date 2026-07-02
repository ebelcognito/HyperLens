#!/usr/bin/env python
"""Two follow-up checks on the ablation results:

  Check 1 — aberration-space distinctness:
    Take each method's refined pool (filtered to RMS < 30um), project to
    (S2, S3), and count how many ε-distinct points exist in *Seidel
    space* (not parameter space). HL should win here even if it loses
    in parameter space, because the Seidel BC was the whole point.

  Check 2 — quality-conditional ε-distinct:
    For each refined-RMS quantile q in {0.1, 0.25, 0.5}, take only the
    top-q fraction of each method's pool, then compute ε-distinct
    counts at ε ∈ {0.20, 0.30}. Tests whether HL's smaller-but-better
    pool wins when we condition on quality.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
OUT = ROOT / "figures/ablation_data"

REFINED_PATHS = {
    'HL':            ROOT / "results/lensmega/cooke_triplet_materials_modernstack_seed42_20260506_033503/refined_top500_seed42_hl.json",
    'SAGA-steelman': ROOT / "results/steelman_div020_20260506_123801/triplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json",
    'NoGrad':        next(ROOT.glob("results/lensmega/cooke_triplet_modernstack_NOGRAD_seed42_*/refined_top500_seed42_NOGRAD.json")),
    'RandomBC':      next(ROOT.glob("results/lensmega/cooke_triplet_modernstack_RANDBC_seed42_*/refined_top500_seed42_RANDBC.json")),
}

CFG_PATH = ROOT / "optical_rl/qd/lensmega/config/examples/cooke_triplet_materials_modernstack.yaml"
cfg = yaml.safe_load(open(CFG_PATH))
bounds = np.array(cfg['problem']['theta_bounds'], float)
lo, hi = bounds[:,0], bounds[:,1]
range_ = hi - lo
D = len(lo)


def load(path):
    with open(path) as f: d = json.load(f)
    rows = []
    for x in d['designs']:
        m = x['metrics']
        rows.append({
            'theta': x['theta'],
            'refined_fitness': x['refined_fitness'],
            'refined_rms_um': x['rms_um'],
            'S2': m.get('seidel_S2'),
            'S3': m.get('seidel_S3'),
        })
    return pd.DataFrame(rows)


def curvature_sign_pattern(theta):
    """6 free curvatures in cooke triplet are theta[0:6]. Returns string like '++--+-'."""
    c = theta[:6]
    return ''.join('+' if x >= 0 else '-' for x in c)


def per_element_signed_powers(theta):
    """Cheap proxy for per-element optical power: c_front - c_back per element, with
    refractive-index sign. For Cooke triplet, theta has free curvatures
    (c0..c3, c5, c6) at indices 0..5 corresponding to E1f, E1b, E2f, E2b, E3f, E3b.
    Element power ~ (n-1) * (c_front - c_back). We use n_after - n_before per element,
    sign of which is set by the glass index. Use index thicknesses & materials in
    theta to make a coarse 'lens topology' fingerprint.
    Returns a tuple suitable for hashing."""
    # E1: c0 - c1, E2: c2 - c3, E3: c5 - c6
    p1 = theta[0] - theta[1]
    p2 = theta[2] - theta[3]
    p3 = theta[4] - theta[5]
    return (np.sign(p1), np.sign(p2), np.sign(p3))


def eps_distinct_param(df, eps):
    """Greedy in unit-cube parameter space."""
    if len(df) == 0: return 0
    sub = df.sort_values('refined_fitness', ascending=False).reset_index(drop=True)
    thetas_unit = np.array([(np.asarray(t)-lo)/range_ for t in sub['theta']])
    accepted = []
    for v in thetas_unit:
        ok = True
        for a in accepted:
            if np.linalg.norm(v - a)/np.sqrt(D) <= eps:
                ok = False; break
        if ok: accepted.append(v)
    return len(accepted)


def eps_distinct_seidel(df, eps_seidel_norm):
    """Greedy in normalized (S2, S3) space.
    Use the archive bounds as the normalization range: [-0.001, 0.001] on each axis,
    so theta range is 0.002. Compute ||(S2, S3)/0.002||_2 / sqrt(2) > eps."""
    if len(df) == 0: return 0
    sub = df.sort_values('refined_fitness', ascending=False).reset_index(drop=True)
    pts = np.column_stack([sub['S2'].values/0.002, sub['S3'].values/0.002])
    accepted = []
    for v in pts:
        ok = True
        for a in accepted:
            if np.linalg.norm(v - a)/np.sqrt(2) <= eps_seidel_norm:
                ok = False; break
        if ok: accepted.append(v)
    return len(accepted)


# ------------------------------------------------------------------
# Check 1: how many distinct *physical regimes* per method (RMS < 30um)
# Two metrics: curvature-sign pattern (6-bit topology) and signed
# per-element powers (3-bit element-power topology). Refined designs
# all have S2,S3 ≈ 0 (LBFGS zeros them), so post-refine Seidel space
# is meaningless; topology fingerprint is the right "physical
# regime" metric.
print("=" * 70)
print("CHECK 1 — physical-regime distinctness (RMS < 30um)")
print("=" * 70)
print(f"{'method':<16} {'pool':>5} {'param@.20':>10} {'param@.30':>10} "
      f"{'curv-sign':>10} {'el-power':>9}")
rows = []
for name, p in REFINED_PATHS.items():
    df = load(p)
    sub = df[df['refined_rms_um'] < 30.0]
    n = len(sub)
    p20 = eps_distinct_param(sub, 0.20)
    p30 = eps_distinct_param(sub, 0.30)
    curv_patterns = set()
    elpow_patterns = set()
    for t in sub['theta']:
        t = np.asarray(t)
        curv_patterns.add(curvature_sign_pattern(t))
        elpow_patterns.add(per_element_signed_powers(t))
    print(f"{name:<16} {n:>5} {p20:>10} {p30:>10} {len(curv_patterns):>10} {len(elpow_patterns):>9}")
    rows.append({'method': name, 'pool_lt_30um': n,
                 'param_eps_distinct_0.20': p20, 'param_eps_distinct_0.30': p30,
                 'distinct_curvature_sign_patterns': len(curv_patterns),
                 'distinct_element_power_signs': len(elpow_patterns)})
pd.DataFrame(rows).to_csv(OUT/"check1_physical_regimes.csv", index=False)
print(f"\nwrote {OUT/'check1_physical_regimes.csv'}")

# ------------------------------------------------------------------
# Check 2: quality-conditional eps-distinct
print("\n" + "=" * 70)
print("CHECK 2 — eps-distinct conditional on top-q fraction by refined RMS")
print("=" * 70)
print(f"{'method':<16} {'q':>6} {'pool':>5} {'rms_max':>9} {'@0.20':>6} {'@0.30':>6}")
rows2 = []
for name, p in REFINED_PATHS.items():
    df = load(p)
    df = df.sort_values('refined_rms_um').reset_index(drop=True)  # ascending RMS = best first
    for q in [0.1, 0.25, 0.5, 1.0]:
        n_take = max(1, int(round(q * len(df))))
        sub = df.head(n_take)
        rms_max = sub['refined_rms_um'].max()
        e20 = eps_distinct_param(sub, 0.20)
        e30 = eps_distinct_param(sub, 0.30)
        print(f"{name:<16} {q:>6.2f} {n_take:>5} {rms_max:>9.2f} {e20:>6} {e30:>6}")
        rows2.append({'method': name, 'q': q, 'pool': n_take,
                      'rms_max_um': rms_max,
                      'eps_distinct_at_0.20': e20,
                      'eps_distinct_at_0.30': e30})
pd.DataFrame(rows2).to_csv(OUT/"check2_quality_conditional.csv", index=False)
print(f"\nwrote {OUT/'check2_quality_conditional.csv'}")
