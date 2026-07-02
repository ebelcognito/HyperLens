# Runbook — HL Hyperparameter Ablation, Octuplet Topology

**Audience:** another agent (or operator) executing the same study on the octuplet that was just done on the sixlet.

**Goal:** answer "what's the tightest single-seed alpha we can extract on the octuplet from search-side hyperparameters?" and produce a directly-comparable report.

**Time estimate:** ~3-5 hours of compute on a 32-core SLURM allocation (or faster with multiple allocs in parallel). Plus 30 min of analysis at the end.

---

## What we already know from the sixlet (read first)

See `figures/REPORT_hl_hyperparam_ablation.md` for full results. Key findings to test for replication on octuplet:

1. **Phase A (refinement sweep) was saturated** — re-refining with longer LBFGS / tighter tolerance changed nothing because the plateau detector (`no_improve_count ≥ 5`) trips before the cap. Expected to be the same on octuplet, but worth confirming with one variant.
2. **Three positive search-side alphas on sixlet:**
   * `bins: 512² → 2048²`: +84% designs <30 µm, +144% <10 µm, min RMS 3.35 → 2.09 µm
   * `fitness_patience: 100 → 30`: +211% designs <10 µm, **min RMS 0.89 µm** (sub-micron)
   * `initial_sigma: 0.5 → 1.0`: +12% ε-distinct, +27% designs <30 µm
3. **Negative results (don't bother):** coarser grid (128²), tighter bounds, `refine_efl_penalty 20→40`, longer LBFGS (max_steps 100→300).
4. **CHAMPION combo** (`bins=2048² + σ=1.0 + patience=30`) is in flight on sixlet — check `figures/refsweep2_data/headline.csv` for its result before launching the octuplet study so you know if alphas compose.

**Octuplet may differ in interesting ways:**
* Higher D (33 → ~45) = harder fitness landscape
* Two cemented interfaces — different basin structure
* Baseline already has `population_size: 32` and `fitness_patience: 50` (sixlet was 18 / 100), so the operating point is shifted

---

## Files / paths to know

```
# Octuplet baseline config (the v2 paper uses this)
optical_rl/qd/lensmega/config/examples/octuplet_apsc_2cement_v2.yaml

# Existing octuplet HL run (fig4 baseline; ALREADY EXISTS, do NOT re-run)
results/lensmega/octuplet_apsc_2cement_v2_seed42_20260506_040025/
  ├── archive.pkl                                # for reading existing baseline
  ├── checkpoint.pkl                             # backup if archive.pkl corrupt
  └── refined_top500_seed42_hl.json              # paper's reference numbers

# SAGA-steelman octuplet (for comparison)
results/steelman_div020_20260506_123801/octuplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json

# Existing sixlet ablation results (for cross-comparison)
figures/refsweep2_data/headline.csv                                        # sixlet table
figures/REPORT_hl_hyperparam_ablation.md                                   # sixlet report

# Tools you'll need
optical_rl/qd/lensmega/run.py                    # the search driver (entry point: -m optical_rl.qd.lensmega.run)
tools/refine_all_parallel.py                     # the refinement driver
```

**Octuplet baseline knobs (from the YAML):**
* `archive.bins: [512, 512]`
* `archive.bounds: [[-0.005, 0.005], [-0.008, 0.008]]` ← different from sixlet, do **not** copy sixlet bounds
* `algorithm.population_size: 32`
* `algorithm.initial_sigma: 0.5`
* `algorithm.fitness_patience: 50` ← already lower than sixlet's 100
* `algorithm.cma_eta: 0.15`
* `algorithm.new_cell_bonus: 1`
* `algorithm.scale_bonus_by_fitness: true`
* `descriptor.kind: seidel`, `params.terms: [S2, S3]`

---

## Step 1 — verify the existing octuplet baseline reproduces fig4

Don't run a new baseline search — re-refine the **existing archive** at `results/lensmega/octuplet_apsc_2cement_v2_seed42_20260506_040025/` and confirm the refined population matches what fig4 used. (This catches any pipeline drift since fig4 was run.)

```bash
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
RUN_DIR=results/lensmega/octuplet_apsc_2cement_v2_seed42_20260506_040025
$HPY tools/refine_all_parallel.py \
  --pool-from hl --hl-run "$RUN_DIR" \
  --threshold -1.0 --top-n 500 \
  --workers 30 --out /tmp/octuplet_baseline_recheck.json
```

Expected: median refined RMS ≈ same as fig4's `refined_top500_seed42_hl.json` (compare the two JSONs). If it doesn't match, **STOP** — investigate before running variants.

---

## Step 2 — write 9 variant configs

Mirror the sixlet sweep. Each config is a clone of `octuplet_apsc_2cement_v2.yaml` with exactly one knob changed and a unique `io.run_name`.

```python
# scripts_throwaway/make_octuplet_ablation_configs.py
import yaml
base = 'optical_rl/qd/lensmega/config/examples/octuplet_apsc_2cement_v2.yaml'
sweeps = {
    # Phase A — refinement (just one to confirm saturation)
    'D_long_lbfgs':     {'io.refine_lbfgs_max_steps': 300, 'io.refine_converge_tol': 1e-5},

    # Phase B — search-side
    'G_grid128':        {'archive.bins': [128, 128]},
    'G_grid1024':       {'archive.bins': [1024, 1024]},
    'G_grid2048':       {'archive.bins': [2048, 2048]},
    'W_wider_bd':       {'archive.bounds': [[-0.001, 0.001], [-0.001, 0.001]]},   # ~5x narrower than baseline
    'T_tighter_bd':     {'archive.bounds': [[-0.0003, 0.0003], [-0.0005, 0.0005]]},  # ~16x narrower
    'P_pop64':          {'algorithm.population_size': 64},   # baseline is 32, so test 2x
    'S_sigma10':        {'algorithm.initial_sigma': 1.0},
    'F_patience20':     {'algorithm.fitness_patience': 20},  # baseline is 50, so aggressive
    'F_patience150':    {'algorithm.fitness_patience': 150}, # baseline is 50, so lazy
}
def set_nested(d, dotted, val):
    keys = dotted.split('.')
    for k in keys[:-1]: d = d[k]
    d[keys[-1]] = val
for tag, mut in sweeps.items():
    with open(base) as f: c = yaml.safe_load(f)
    for k, v in mut.items(): set_nested(c, k, v)
    c['algorithm']['cma_seed'] = 42
    c['io']['run_name'] = f'octuplet_REFSWEEP_{tag}_seed42'
    out = f'configs/ablations/octuplet_REFSWEEP_{tag}.yaml'
    with open(out, 'w') as f: yaml.safe_dump(c, f, sort_keys=False)
    print(f'wrote {out}')
```

**Notes on choices vs sixlet:**
* `W_wider_bd` is ~5× narrower than the octuplet's actual baseline (which uses `±0.005, ±0.008`). Same name kept for cross-system comparison; the meaning is *narrower-than-baseline*, not *wider*. Re-name in the report if confusing.
* `P_pop64` (not `pop36`) because octuplet baseline is already 32. Test 2× upward.
* `F_patience20` (not `30`) because baseline is 50. Same proportional aggressiveness as sixlet's 30 vs 100.

---

## Step 3 — run the searches

**One search at a time per allocation** (in parallel across allocations is fine). **Do NOT run multiple searches per allocation in parallel** — that path leads to:
* OOM kills if total memory exceeds alloc
* CPU contention slowing each by 3-5×
* Race conditions if two scripts touch the same archive paths

**Compute estimate per variant on octuplet at 100k evals:**
* Smaller grids (128²): ~5 min wall (search short due to fast cell-fill)
* Standard grids (512², 1024²): ~20 min wall
* Largest grid (2048²): ~25-30 min wall (more restarts due to cell-novelty pathology, but still finishes)
* Refinement: ~12 min wall with 30 workers (octuplet is 33-D theta, similar to sixlet)

**Total compute:** 9 variants × ~30 min (search + refine) = ~4.5 hours sequential, or ~1.5 hours if 3 allocs run in parallel (one variant each).

**Allocation params (use these — don't deviate):**
```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=48G --time=4:00:00
```
* Anything less than `--mem=24G` per srun risks OOM during 1024²/2048² archive saves
* `--cpus-per-task=32` per srun matches the runner's worker pool
* `--time=4:00:00` is enough for 4-5 sequential variants per alloc

**The launch pattern that worked (use this exact form):**
```bash
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
ALLOC=<your_jobid>
LOGDIR=results/octuplet_refsweep_logs_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"

for TAG in G_grid128 G_grid1024 G_grid2048 W_wider_bd T_tighter_bd P_pop64 S_sigma10 F_patience20 F_patience150; do
    CFG=configs/ablations/octuplet_REFSWEEP_${TAG}.yaml
    echo "[$(date +%H:%M:%S)] [search] $TAG"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=40G \
        --chdir=/nas/longleaf/home/hisham26/Desktop/hyperlens \
        $HPY -m optical_rl.qd.lensmega.run \
        --config $CFG --iterations 100000 \
        > "$LOGDIR/${TAG}_search.log" 2>&1
    echo "[$(date +%H:%M:%S)] [search] $TAG done"

    RUN_DIR=$(ls -td results/lensmega/octuplet_REFSWEEP_${TAG}_seed42_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then continue; fi
    OUT=$RUN_DIR/refined_top500_seed42_${TAG}.json
    echo "[$(date +%H:%M:%S)] [refine] $TAG"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=40G \
        --chdir=/nas/longleaf/home/hisham26/Desktop/hyperlens \
        $HPY tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUN_DIR" \
        --threshold -1.0 --top-n 500 --workers 30 \
        --out "$OUT" \
        > "$LOGDIR/${TAG}_refine.log" 2>&1
    echo "[$(date +%H:%M:%S)] [refine] $TAG done"
done
```

**Run this with `nohup` and `disown`:**
```bash
nohup bash scripts_throwaway/run_octuplet_ablation.sh $ALLOC > /tmp/octuplet_run.log 2>&1 &
disown -a
```
Otherwise the script dies when the launching shell exits (this happened repeatedly during the sixlet study).

---

## Step 4 — pitfalls we hit on the sixlet (don't repeat)

1. **Disk quota.** Each octuplet HL search produces ~250-300 MB of archive snapshots. With 9 variants, that's ~3 GB. **Check `quota -s` before starting.** Hard limit is 76.8 GB on Longleaf; budget for ~5 GB headroom. If you hit the limit mid-run, the search silently fails to write its archive.
2. **Alloc expiry mid-write.** A 4-hour alloc that expires while a search is in its final-save phase produces a `checkpoint.pkl` but no `archive.pkl`. Recovery: load `checkpoint.pkl`, extract `ckpt.archive`, save as `archive.pkl`, retry refine.
   ```python
   import pickle
   ckpt = pickle.load(open('checkpoint.pkl', 'rb'))
   pickle.dump(ckpt.archive, open('archive.pkl', 'wb'))
   ```
3. **pkill regex too broad.** `pkill -f "lensmega.run.*S_sigma10"` matches *any* process whose command line contains "S_sigma10" — including refine srun for that variant. Use PID-based `kill -9 <PID>` once you've identified the exact process.
4. **Parent script death.** `bash scripts_throwaway/run_X.sh & disown` is *not* enough — when the launching shell exits, the bash job tree can also exit. Use `nohup bash ... &; disown -a` instead. Verify with `pgrep -af "run_octuplet_ablation"`.
5. **Login-node `pgrep` only shows srun, not the python on the compute node.** This is normal. The actual workers run on the compute node and aren't visible via local `pgrep`. To verify a job is alive, check the log file's mtime.
6. **Corrupted archives.** A search killed *during* `archive.pkl` write produces a half-written file with `_pickle.UnpicklingError: invalid load key`. The `checkpoint.pkl` from the same dir often has the valid archive embedded — extract it as in (2).
7. **Bounds vs autoscale.** Sixlet baseline has hand-set bounds. Setting `archive.bounds: null` triggers auto-calibration during the seed phase (uses `percentile_bounds`). For *bound-sweep variants*, override with explicit values; do NOT use null for those.

---

## Step 5 — analysis

After all 9 refined JSONs land, run an analysis script analogous to the sixlet's `scripts_throwaway/refsweep2_analysis.py`. Adapt it by:
* Pointing to octuplet refined JSON paths
* Loading the octuplet config's `theta_bounds` for ε-distinct normalization (octuplet has 33-D theta vs sixlet's 33-D — actually similar, just confirm `D = len(theta_bounds)`)
* Adding the existing octuplet baseline (`results/lensmega/octuplet_apsc_2cement_v2_seed42_20260506_040025/refined_top500_seed42_hl.json`) and SAGA-steelman as reference rows

Outputs:
```
figures/octuplet_refsweep_data/
  ├── headline.csv             # method × RMS_threshold × (ε@.20, ε@.30, median, min, pool_size)
  └── eps_curves.csv           # full ε-distinct curves
figures/octuplet_refsweep_sanity.png   # 4-panel ε-vs-eps plot at RMS thresholds {10, 30, 100, 1000}
figures/REPORT_octuplet_hl_ablation.md # final report, mirroring the sixlet one
```

**Predictions to test (from the sixlet study):**

| sixlet result | octuplet expected | what tells us if true / false |
|---|---|---|
| 2048² wins big | should also win, possibly bigger | HL is grid-resolution-bottlenecked across topologies. Confirms paper recommendation. |
| patience=30 finds sub-µm RMS | may or may not — octuplet has more parameters, deep basins may be harder to find by random restart | If yes → universal recommendation. If no → sixlet was a special case. |
| σ=1.0 wins on diversity | should generalize | Same as σ_init being a width-of-exploration knob, mostly topology-independent. |
| Refinement saturated (D=A) | should also saturate | Confirms LBFGS plateau detector is the bottleneck, not max_steps. |
| Coarser grid + tighter bounds = bad | should also be bad | Robustness footnote. |

**If 2048² does NOT win on octuplet** — that's a paper-relevant negative result (suggests sixlet was a sweet spot for grid resolution, alpha doesn't generalize). Don't suppress it.

---

## Step 6 — combo champion

After all individual variants land, run the combo:
```yaml
# configs/ablations/octuplet_REFSWEEP_CHAMPION.yaml
algorithm:
  population_size: 32           # unchanged
  initial_sigma: 1.0            # if S_sigma10 won
  fitness_patience: 20          # if F_patience20 won
archive:
  bins: [2048, 2048]            # if G_grid2048 won
  bounds: [[-0.005, 0.005], [-0.008, 0.008]]   # baseline, unchanged
```
**Use only the knobs that won individually on octuplet** — don't blindly copy sixlet's combo. If patience=20 *lost* on octuplet, drop it from the combo.

Compare CHAMPION to the best individual variant. Two outcomes:
* **Combo > best individual** → alphas compose, headline result for the paper
* **Combo ≈ best individual** → only one knob is load-bearing; pick that and save compute
* **Combo < best individual** → alphas conflict (e.g., aggressive restart on a fine grid amplifies the cell-novelty pathology). Diagnostic finding.

---

## Step 7 — produce the final report

Mirror `figures/REPORT_hl_hyperparam_ablation.md` (sixlet) for the octuplet. Use the same section structure: TL;DR → setup → Phase A → Phase B → table → three orthogonal alphas → negative results → recommended config → limitations → files.

Cross-reference the sixlet report ("on sixlet, X gave +84%; here it gave Y%") so the two reports can be read side-by-side.

---

## Sanity-check log lines to expect during the runs

A healthy octuplet 100k-iter HL search should produce log entries like:
```
[lensmega] starting CPU worker pool: 32 workers
[lensmega] run_dir=results/lensmega/octuplet_REFSWEEP_..._seed42_<ts>
[seed] insert code=2 fitness=-X.XX measures=[..., ...]
... ~50 sec of seeding ...
[lensmega] gen=50 evals=1600 coverage=0.XX best=-X.XX new=N/32 ... wall=Xm
... search proceeds ...
[lensmega] final stats: {'filled': N, 'total_cells': M, 'coverage': X.X, 'best_fitness': -X.XX, 'iterations': 100000.0, ...}
[lensmega] archive plot saved: ...
[lensmega] diverse elites grid saved: ...
```

If you see `restart_count` > 50 by gen 1000, the search is in the cell-novelty pathology — typical for 2048² grids. Should resolve itself by mid-run as the modern anti-runaway stack damps the bonus.

If you see `archive is empty` at gen 0 — the BD bounds don't contain the seeded designs' measures. Either set `bounds: null` (auto-calibrate) or widen them.

---

## Quick reference — exact commands for each step

```bash
# 0. Verify quota (need ~5 GB free)
quota -s

# 1. Generate configs
$HPY scripts_throwaway/make_octuplet_ablation_configs.py

# 2. Sanity-check baseline reproduces (skip if you trust fig4)
$HPY tools/refine_all_parallel.py --pool-from hl \
  --hl-run results/lensmega/octuplet_apsc_2cement_v2_seed42_20260506_040025 \
  --threshold -1.0 --top-n 500 --workers 30 --out /tmp/octuplet_baseline_recheck.json

# 3. Allocate compute
salloc --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=48G --time=4:00:00
# (note the JOBID it prints)

# 4. Launch (use nohup + disown!)
nohup bash scripts_throwaway/run_octuplet_ablation.sh $JOBID > /tmp/octuplet_run.log 2>&1 &
disown -a

# 5. Monitor
tail -F /tmp/octuplet_run.log

# 6. After all 9 land, run analysis
$HPY scripts_throwaway/octuplet_refsweep_analysis.py

# 7. Run combo champion if individual results show alphas
$HPY -m optical_rl.qd.lensmega.run \
  --config configs/ablations/octuplet_REFSWEEP_CHAMPION.yaml --iterations 100000

# 8. Write the report (mirror figures/REPORT_hl_hyperparam_ablation.md)
```

---

*Generated 2026-05-09, by the agent that just finished the sixlet study. All sixlet artifacts live under `figures/refsweep2_data/` and `figures/REPORT_hl_hyperparam_ablation.md`.*
