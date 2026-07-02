#!/usr/bin/env python
"""Figure 4 data extraction + sanity plot. Throwaway script.

Note: HL data files are not at the path quoted in the task brief.
The actual locations (per the diversity_run logs) are under
results/lensmega/cooke_triplet_materials_modernstack_seed42_20260506_033503/.
SA+GA paths match the brief.

raw_fitness in HL CSV = the cell's *current* archive fitness at the time
the refined-pool was extracted (archive does not store occupation history).
This is the same value the refined JSON records as "raw_fitness".
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
HL_DIR = ROOT / "results/lensmega/cooke_triplet_materials_modernstack_seed42_20260506_033503"
SAGA_DIR = ROOT / "results/steelman_div020_20260506_123801/triplet/seed42/sa_ga"
CFG = ROOT / "results/diversity_run_20260506_033447/configs/hl_triplet_seed42.yaml"
OUT = ROOT / "figures/fig4_data"
OUT.mkdir(parents=True, exist_ok=True)

EPS_GRID = np.arange(0.05, 0.5 + 1e-9, 0.025)
RMS_LIMIT_UM = 1000.0  # 1 mm

# --- file existence -----------------------------------------------------
for p in [HL_DIR/"refined_top500_seed42_hl.json",
          HL_DIR/"archive.pkl",
          SAGA_DIR/"refined_above_-1.0_seed42_sa_ga.json",
          SAGA_DIR/"unrefined_elites_sa_ga.json",
          CFG]:
    if not p.exists():
        sys.exit(f"MISSING: {p}")

# --- bounds for theta normalization ------------------------------------
with open(CFG) as f:
    cfg = yaml.safe_load(f)
bounds = np.array(cfg['problem']['theta_bounds'], dtype=float)  # (19, 2)
D = bounds.shape[0]
assert D == 19, f"expected D=19, got {D}"
lo, hi = bounds[:,0], bounds[:,1]
span = hi - lo

def norm_theta(theta):
    return (np.asarray(theta, dtype=float) - lo) / span

# --- load HL ------------------------------------------------------------
with open(HL_DIR/"refined_top500_seed42_hl.json") as f:
    hl_json = json.load(f)
with open(HL_DIR/"archive.pkl", "rb") as f:
    archive = pickle.load(f)

# Map cell_id by raw_fitness (verified unique earlier)
fit_to_key = {}
for k, c in archive.cells.items():
    fit_to_key.setdefault(c.fitness, []).append(k)

hl_rows = []
for d in hl_json['designs']:
    rf = d['raw_fitness']
    keys = fit_to_key.get(rf, [])
    cell_id = keys[0] if len(keys)==1 else None
    cid_str = f"{cell_id[0]}_{cell_id[1]}" if cell_id else None
    m = d['metrics']
    hl_rows.append({
        'cell_id': cid_str,
        'theta': json.dumps(d['theta']),
        'raw_fitness': d['raw_fitness'],
        'raw_rms_um': d['raw_rms_um'],
        'refined_fitness': d['refined_fitness'],
        'refined_rms_um': d['rms_um'],
        'S2': m['seidel_S2'],
        'S3': m['seidel_S3'],
    })
hl_df = pd.DataFrame(hl_rows)

# --- load SA+GA ---------------------------------------------------------
with open(SAGA_DIR/"refined_above_-1.0_seed42_sa_ga.json") as f:
    saga_json = json.load(f)
with open(SAGA_DIR/"unrefined_elites_sa_ga.json") as f:
    saga_un = json.load(f)
un_by_rf = {round(u['raw_fitness'], 12): u for u in saga_un}

saga_rows = []
for d in saga_json['designs']:
    m = d['metrics']
    # raw_fitness is already in refined JSON; verify match with unrefined
    rf = round(d['raw_fitness'], 12)
    assert rf in un_by_rf, f"unrefined raw_fitness mismatch for {rf}"
    saga_rows.append({
        'cell_id': None,
        'theta': json.dumps(d['theta']),
        'raw_fitness': d['raw_fitness'],
        'raw_rms_um': d.get('raw_rms_um'),
        'refined_fitness': d['refined_fitness'],
        'refined_rms_um': d['rms_um'],
        'S2': m['seidel_S2'],
        'S3': m['seidel_S3'],
    })
saga_df = pd.DataFrame(saga_rows)

# --- epsilon-distinct curve --------------------------------------------
def eps_distinct_curve(df, eps_grid, D):
    sub = df[df['refined_rms_um'] < RMS_LIMIT_UM].copy()
    # Sort best-first: refined_fitness DESCENDING (least-negative = lowest RMS first).
    # This matches tools/analyze_diversity.py:epsilon_net_count.
    # The brief's "ascending" appears to refer to RMS, not signed fitness.
    sub = sub.sort_values('refined_fitness', ascending=False).reset_index(drop=True)
    thetas = np.array([norm_theta(json.loads(t)) for t in sub['theta']])
    counts = []
    for eps in eps_grid:
        accepted = []
        for v in thetas:
            ok = True
            for a in accepted:
                d = np.linalg.norm(v - a) / np.sqrt(D)
                if d <= eps:
                    ok = False; break
            if ok:
                accepted.append(v)
        counts.append(len(accepted))
    return counts

hl_counts = eps_distinct_curve(hl_df, EPS_GRID, D)
saga_counts = eps_distinct_curve(saga_df, EPS_GRID, D)

hl_eps_df = pd.DataFrame({'epsilon': EPS_GRID, 'eps_distinct_count': hl_counts})
saga_eps_df = pd.DataFrame({'epsilon': EPS_GRID, 'eps_distinct_count': saga_counts})

# --- write CSVs ---------------------------------------------------------
hl_df.to_csv(OUT/"hl_icicles.csv", index=False)
saga_df.to_csv(OUT/"saga_icicles.csv", index=False)
hl_eps_df.to_csv(OUT/"hl_eps_curve.csv", index=False)
saga_eps_df.to_csv(OUT/"saga_eps_curve.csv", index=False)

# --- summary JSON -------------------------------------------------------
hl_filt = hl_df[hl_df['refined_rms_um'] < RMS_LIMIT_UM]
hl_med_raw = float(hl_df['raw_rms_um'].median())
hl_med_ref = float(hl_df['refined_rms_um'].median())
# icicle = raw_rms_um / refined_rms_um ratio
ratios = hl_df['raw_rms_um'] / hl_df['refined_rms_um']
order = ratios.sort_values(ascending=False).index
top5 = []
for idx in order[:5]:
    r = hl_df.loc[idx]
    top5.append({
        'raw_rms_um': float(r['raw_rms_um']),
        'refined_rms_um': float(r['refined_rms_um']),
        'ratio': float(r['raw_rms_um']/r['refined_rms_um']),
    })
longest_idx = order[0]
longest = hl_df.loc[longest_idx]
longest_d = {
    'raw_rms_um': float(longest['raw_rms_um']),
    'refined_rms_um': float(longest['refined_rms_um']),
    'ratio': float(longest['raw_rms_um']/longest['refined_rms_um']),
    'theta': json.loads(longest['theta']),
}

def at(df, e):
    row = df[np.isclose(df['epsilon'], e)]
    return int(row['eps_distinct_count'].iloc[0])

hl20 = at(hl_eps_df, 0.20); hl30 = at(hl_eps_df, 0.30)
sa20 = at(saga_eps_df, 0.20); sa30 = at(saga_eps_df, 0.30)
dominates = bool(np.all(np.array(hl_counts) >= np.array(saga_counts)))
strict_dom = bool(np.all(np.array(hl_counts) > np.array(saga_counts)))

summary = {
    'data_source_note': "HL data found at results/lensmega/cooke_triplet_materials_modernstack_seed42_20260506_033503/ (the path in the brief, results/diversity_run_20260506_033447/triplet/seed42/hl/, does not exist; the diversity_run dir holds only logs and configs for the HL run, while the actual run_dir is under results/lensmega/). raw_fitness here is the archive cell's current fitness, NOT first-occupation; the GridArchive in this codebase does not store history. Across all 500 refined designs, raw_fitness uniquely matched a single archive cell, so cell_id was recovered by raw_fitness lookup.",
    'hl_refined_population_size': int(len(hl_df)),
    'hl_refined_filtered_lt_1mm': int(len(hl_filt)),
    'hl_median_raw_rms_um': hl_med_raw,
    'hl_median_refined_rms_um': hl_med_ref,
    'hl_longest_icicle': longest_d,
    'hl_top5_longest_icicles': top5,
    'saga_refined_population_size': int(len(saga_df)),
    'saga_refined_filtered_lt_1mm': int((saga_df['refined_rms_um']<RMS_LIMIT_UM).sum()),
    'eps_distinct_at_0.20': {'hl': hl20, 'saga': sa20},
    'eps_distinct_at_0.30': {'hl': hl30, 'saga': sa30},
    'hl_dominates_saga_at_every_eps': dominates,
    'hl_strictly_dominates_saga_at_every_eps': strict_dom,
}
with open(OUT/"summary.json","w") as f:
    json.dump(summary, f, indent=2)

# --- sanity checks against expected values -----------------------------
expected = {'hl@0.20': 39, 'hl@0.30': 19, 'saga@0.20': 24, 'saga@0.30': 8}
got      = {'hl@0.20': hl20, 'hl@0.30': hl30, 'saga@0.20': sa20, 'saga@0.30': sa30}

print("=== Fig 4 sanity checks ===")
print(f"HL refined population:        {len(hl_df)}")
print(f"HL filtered (refined<1mm):    {len(hl_filt)}")
print(f"HL longest icicle: raw={longest_d['raw_rms_um']:.3f} um, refined={longest_d['refined_rms_um']:.4f} um, ratio={longest_d['ratio']:.2f}")
print(f"HL median raw RMS:            {hl_med_raw:.3f} um")
print(f"HL median refined RMS:        {hl_med_ref:.4f} um")
print(f"HL eps-distinct @0.20:        got={hl20}  expected=39  {'OK' if hl20==39 else 'MISMATCH'}")
print(f"HL eps-distinct @0.30:        got={hl30}  expected=19  {'OK' if hl30==19 else 'MISMATCH'}")
print(f"SA+GA refined population:     {len(saga_df)}")
print(f"SA+GA eps-distinct @0.20:     got={sa20}  expected=24  {'OK' if sa20==24 else 'MISMATCH'}")
print(f"SA+GA eps-distinct @0.30:     got={sa30}  expected=8   {'OK' if sa30==8 else 'MISMATCH'}")
print(f"HL dominates SA+GA at every eps:  {dominates}")
print(f"HL strictly > SA+GA at every eps: {strict_dom}")
mismatch = [k for k in expected if expected[k] != got[k]]
if mismatch:
    print("\n!!! MISMATCH detected. Stopping before plot. !!!")
    print("expected:", expected)
    print("got     :", got)
    print("Full HL eps curve: ", list(zip([round(e,3) for e in EPS_GRID], hl_counts)))
    print("Full SAGA eps curve:", list(zip([round(e,3) for e in EPS_GRID], saga_counts)))
    sys.exit(2)

# --- sanity plot -------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

# top: icicles
order_idx = hl_df.sort_values('refined_rms_um', ascending=True).index.tolist()
hl_sorted = hl_df.loc[order_idx].reset_index(drop=True)
top5_orig_idx = list(order[:5])
top5_pos = [order_idx.index(i) for i in top5_orig_idx]

for x, (_, r) in enumerate(hl_sorted.iterrows()):
    color = 'C3' if x in top5_pos else 'C0'
    lw = 2.0 if x in top5_pos else 0.6
    ax1.plot([x, x], [r['refined_rms_um'], r['raw_rms_um']],
             color=color, lw=lw, alpha=0.8 if x in top5_pos else 0.6)

ax1.axhline(hl_med_raw, ls='--', color='k', alpha=0.5,
            label=f'median raw RMS = {hl_med_raw:.1f} um')
ax1.set_yscale('log')
ax1.set_xlabel('elite index (sorted by refined RMS asc.)')
ax1.set_ylabel('RMS (um, log scale)')
ax1.set_title('HL refined population, triplet seed=42')
ax1.legend()

# bottom: eps-distinct curves
ax2.plot(EPS_GRID, hl_counts, 'o-', label='HyperLens', color='C0')
ax2.plot(EPS_GRID, saga_counts, 's-', label='SA+GA steelman', color='C3')
for e, hv, sv in [(0.20, hl20, sa20), (0.30, hl30, sa30)]:
    ax2.scatter([e],[hv], color='C0', zorder=5)
    ax2.scatter([e],[sv], color='C3', zorder=5)
    ax2.annotate(f'{hv}', (e, hv), xytext=(4,4), textcoords='offset points', color='C0')
    ax2.annotate(f'{sv}', (e, sv), xytext=(4,-12), textcoords='offset points', color='C3')
ax2.set_xlabel('epsilon (normalized parameter-space distance)')
ax2.set_ylabel('eps-distinct count')
ax2.set_title('Pool diversity, matched compute')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(ROOT/"figures/fig4_sanity.png", dpi=120)
print("\nWrote:")
for p in [OUT/"hl_icicles.csv", OUT/"saga_icicles.csv", OUT/"hl_eps_curve.csv",
          OUT/"saga_eps_curve.csv", OUT/"summary.json", ROOT/"figures/fig4_sanity.png"]:
    print(" ", p)
