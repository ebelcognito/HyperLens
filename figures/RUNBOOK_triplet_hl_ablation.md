# Runbook — HL Hyperparameter Ablation, APS-C Cooke Triplet

**Audience:** another agent (or operator) executing the HL hyperparameter ablation on the APS-C cooke triplet (r_sensor=17mm, EFL=85mm, F/4). Sister runbooks: sixlet (`figures/REPORT_hl_hyperparam_ablation.md` — completed) and octuplet (`figures/RUNBOOK_octuplet_hl_ablation.md` — queued).

**Goal:** find hyperparameter settings that **break through the apparent ~135 µm RMS basin floor** observed on this template's existing (untuned) HL runs. The whole point of this ablation is to test whether HL *can* be pushed below that floor — not to accept it. If a combination of knobs (esp. grid resolution + patience + σ_init) reaches sub-50 µm or sub-30 µm refined designs, that's a topology-class result for the paper. If nothing breaks the floor across all 12 variants + a combo champion, then we have evidence the floor is a *real structural limit* of the cooke topology at APS-C, not a tuning failure. Both outcomes are paper-relevant; **interpret aggressively**.

**Time estimate:** ~2-3 hours of compute on a 32-core SLURM allocation. APS-C triplet runs are fast (~5-10 min per 100k-iter search) like the smaller-aperture triplet.

---

## ⚠️ READ THIS FIRST — what we're trying to find out

**The big question:** the APS-C cooke triplet's *existing* HL runs (5 archives from May 4) all hit a refined-RMS floor of ~135 µm. Those runs used the **default** hyperparameters. The hypothesis driving this ablation: **the 135 µm floor is a tuning artifact, not a structural limit.** Sixlet showed that going from 512² → 2048² grid resolution moved baseline RMS from 95 µm → 39 µm without changing anything else. Maybe APS-C cooke triplet has similar headroom hiding behind the wrong default knobs.

**Set the framing aggressively:**
* **Test whether sub-50 µm or sub-30 µm designs are reachable** with the right knobs. If yes, HL is the lever; the paper claims this. If no, we have hard evidence of a structural limit.
* The yaml's claim that "Best refined RMS at this template will likely land in the 15-40 µm range" is testable — this ablation is the test.
* **Don't pre-commit to APS-C-specific RMS thresholds.** Use both sixlet-style {10, 30, 100, 300, 1000, 3000} µm AND APS-C-specific {150, 200, 300, 500, 1000, 3000} µm in the analysis. If any variant gets sub-30 µm designs, the sixlet-style table will catch it. If nothing does, the APS-C-specific table tells the relative story.
* **Min refined RMS is the headline metric.** If patience=30 (or any combo) finds a design below 100 µm, that's a genuine break-through. The sixlet got 0.89 µm minimum on a topology where baseline was 3.35 µm; could happen here too.

If the answer turns out to be "yes, HL can reach sub-30 µm at APS-C with right knobs," that **inverts the structural-limit interpretation** I gave you earlier — write up that interpretation honestly. If "no", report the floor as the genuine topology limit.

If the user wants the **smaller-aperture cooke triplet** (which matches fig4 and gets sub-30 µm easily), see the prior version of this runbook in git history or use this command to switch templates:
```bash
# Smaller-aperture (fig4-matching, NOT APS-C):
BASE_CFG=optical_rl/qd/lensmega/config/examples/cooke_triplet_materials_modernstack.yaml
# APS-C (this runbook):
BASE_CFG=optical_rl/qd/lensmega/config/examples/cooke_apsc_85mm_f4_discovery.yaml
```

---

## What we already know from the sixlet study (read first)

See `figures/REPORT_hl_hyperparam_ablation.md`. **Three positive search-side alphas to test for replication on APS-C triplet:**

1. `bins: 512² → 2048²`: +84% sub-30 µm designs on sixlet (here: replace metric with "sub-200 µm")
2. `fitness_patience: 100 → 30`: +211% sub-10 µm on sixlet (here: replace with "sub-150 µm" and check minimum)
3. `initial_sigma: 0.5 → 1.0`: +27% sub-30 µm on sixlet (here: same axis, different threshold)
4. **Refinement saturated** (Phase A): re-refining with longer LBFGS was byte-identical. **Confirm on APS-C triplet** — expected same.

**APS-C triplet may differ because:**
* Lowest D in the study (19-D θ, 6 free curvatures)
* Only 5 free Seidels = simpler basin structure but also a lower aberration-correction ceiling
* The cell-novelty pathology (which 2048² introduces) might bite harder here because the BD region has limited deep basins
* `fitness_patience: 30` may *not* find sub-µm designs (sixlet's 0.89 µm minimum was likely a 33-D effect — more parameters to wiggle into deep wells)

---

## Files / paths to know

```
# APS-C cooke triplet baseline config
optical_rl/qd/lensmega/config/examples/cooke_apsc_85mm_f4_discovery.yaml

# Existing APS-C cooke triplet runs (use the most-recent / best one as baseline reference)
results/lensmega/cooke_apsc_85mm_f4_20260504_225808/   # 17746 cells, best_fitness=-0.20  ← BEST
results/lensmega/cooke_apsc_85mm_f4_20260504_230312/   # 26333 cells, best_fitness=-0.53
results/lensmega/cooke_apsc_85mm_f4_20260504_230831/   # 26186 cells, best_fitness=-0.29
results/lensmega/cooke_apsc_85mm_f4_20260504_231615/   # 14583 cells, best_fitness=-0.32
results/lensmega/cooke_apsc_85mm_f4_20260504_232212/   # 19114 cells, best_fitness=-0.38

# NOTE: none of these have a saved refined_top500_seed42_hl.json in their dir.
# You'll need to refine the *_225808 archive yourself as the "baseline" reference.

# SAGA-steelman APS-C triplet — does NOT exist as a separate run.
# The steelman/triplet/seed42/sa_ga in the diversity_run uses the smaller-aperture cooke,
# so there's no direct SAGA reference for APS-C cooke triplet at the same compute budget.
# Either skip SAGA from the comparison or run a fresh sa_ga_refine_baseline.py against
# the APS-C config (out of scope for this runbook — flag in the report).

# Existing sixlet ablation results (for cross-comparison)
figures/refsweep2_data/headline.csv
figures/REPORT_hl_hyperparam_ablation.md

# Tools
python -m optical_rl.qd.lensmega.run                # search driver
tools/refine_all_parallel.py                        # refinement driver
```

**APS-C triplet baseline knobs (from the YAML):**
```yaml
problem:
  free_curvature_indices: [0, 1, 2, 3, 5, 6]    # 6 free curvatures
  # 19-D theta total (same as smaller-aperture cooke)
  focal_length_target_mm: 85.0
  enforce_fnum: true
  f_number_target: 4.0
algorithm:
  population_size: 20                            # NOTE: 20, not 18 like sixlet/smaller-aperture
  initial_sigma: 0.5
  cma_eta: 0.2                                   # NOTE: 0.2, not 0.15 like other templates
  fitness_patience: 100
archive:
  bins: [256, 256]
  bounds: [[-0.0008, 0.0008], [-0.0008, 0.0008]]   # symmetric, even tighter than smaller-aperture (±0.001)
descriptor:
  kind: seidel
  params:
    terms: [S2, S3]
io:
  refine_efl_penalty_weight: 0.0                 # NOTE: 0.0 here vs sixlet's 20
  refine_invalid_penalty_weight: 0.0             # NOTE: 0.0 here vs sixlet's 10
  refine_intersect_penalty_weight: 2.0
  refine_fnum_penalty_weight: 0.0                # F/# unconstrained during refine
  refine_lbfgs_max_steps: 200                    # NOTE: 200 here vs sixlet's 100
  refine_lbfgs_lr: 0.05                          # NOTE: 0.05 here vs sixlet's 0.5
  refine_converge_tol: 0.00008
  refine_field_weights: [1.0, 2.0, 16.0]         # NOTE: [1, 2, 16] here vs sixlet's [1, 2, 4]
```

The refine config differs significantly from the sixlet's. **Don't use sixlet's refine settings as baseline** — keep the APS-C triplet's existing refine config for both the baseline check and all variants.

---

## Step 1 — establish the APS-C triplet baseline

Pick the best existing archive (`*_20260504_225808`, best_fitness=-0.20) and refine its top-500 with the APS-C config's standard refine settings:

```bash
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
RUN_DIR=results/lensmega/cooke_apsc_85mm_f4_20260504_225808
$HPY tools/refine_all_parallel.py \
  --pool-from hl --hl-run "$RUN_DIR" \
  --threshold -1.0 --top-n 500 \
  --workers 30 --out "$RUN_DIR/refined_top500_seed42_apsc_baseline.json"
```

**Expected baseline numbers** (from earlier exploratory check on the same archive):
* n_pool: ~500
* min refined RMS: ~135 µm
* median refined RMS: ~140-150 µm (estimate — verify)
* sub-200 µm count: ~few hundred
* sub-150 µm count: ~10-50

**Sanity check:** if min refined RMS is below 30 µm, something is wrong (this template can't reach there). Re-verify the config and re-run the baseline refine.

---

## Step 2 — sweep design

The APS-C cooke triplet's existing 100k-iter HL run took ~5 min per the diversity-run logs. Variants will be fast. **11 variants** mirror the sixlet/smaller-aperture sweep with APS-C-tuned values.

### Wisely-chosen sweep axes

| variant tag | knob change vs APS-C baseline | rationale |
|---|---|---|
| **D_long_lbfgs** | `io.refine_lbfgs_max_steps: 200→500, io.refine_converge_tol→1e-6` | Phase A check — does extra LBFGS help on this hard template? Note baseline is already 200 not 100 |
| **G_grid128** | `archive.bins: 256²→128²` | Coarser; expect negative |
| **G_grid512** | `archive.bins: 256²→512²` | One step finer |
| **G_grid1024** | `archive.bins: 256²→1024²` | Two steps finer |
| **W_wider_bd** | `archive.bounds: ±0.0008 → ±0.0016` | 2× wider — APS-C bounds are even tighter than smaller-aperture. Test widening. |
| **T_tighter_bd** | `archive.bounds: ±0.0008 → ±0.0004` | 2× tighter |
| **B_autoscale_bd** | `archive.bounds: null` | Let calibration auto-discover. Important for APS-C because measure distribution may not match the hand-set ±0.0008 |
| **P_pop40** | `algorithm.population_size: 20→40` | 2× pop |
| **S_sigma10** | `algorithm.initial_sigma: 0.5→1.0` | Sixlet winner |
| **F_patience30** | `algorithm.fitness_patience: 100→30` | Sixlet extreme-quality winner — **most likely to break the 135µm floor if it can be broken** |
| **F_patience300** | `algorithm.fitness_patience: 100→300` | Sixlet diversity winner |
| **E_cma_eta01** | `algorithm.cma_eta: 0.2→0.1` | Test damping CMA step (the YAML comment says 0.2 was already gentler than 0.15 for APS-C; try going further) |
| **G_grid2048** | `archive.bins: 256²→2048²` | **Aggressive — tests if sixlet's biggest alpha generalizes.** May trigger cell-novelty pathology hard, but worth the test given the goal of breaking through 135 µm |

**Total: 13 variants.** (Two more than sixlet/smaller-aperture: cma_eta knob is APS-C-specific, and 2048² is included aggressively because the goal is to *break through* the floor, not just measure relative gains.)

**If you have to cut:** drop `B_autoscale_bd` first (the existing bounds are calibrated). Then `P_pop40` (sixlet showed pop alpha is small). Then `E_cma_eta01` (third-tier).

### Why these specific values (different from smaller-aperture/sixlet)

* **No 2048².** APS-C triplet baseline is 256²; jumping to 2048² (8× cells) would heavily trigger the cell-novelty pathology because the BD region has *fewer* deep basins than sixlet (only 5 free Seidels). 1024² is the highest justifiable.
* **Bounds 2× both directions.** Baseline ±0.0008 is calibrated tight; testing both directions plus autoscale brackets the sweet spot.
* **Pop=40 not 36.** Baseline is 20, so 2× rather than the sixlet's 36 (which was 2× of 18).
* **cma_eta variant added.** APS-C YAML notes say 0.2 was a tuning choice; worth testing 0.1 to see if even gentler helps with the harder feasibility manifold.
* **`refine_fnum_penalty_weight: 0.0`** in baseline means F/# is allowed to drift during refine. Don't change this in variants — keeping it consistent so the sweep tests *only* the named axis.

### Config-generation script

```python
# scripts_throwaway/make_apsc_triplet_ablation_configs.py
import yaml
base = 'optical_rl/qd/lensmega/config/examples/cooke_apsc_85mm_f4_discovery.yaml'
sweeps = {
    'D_long_lbfgs':    {'io.refine_lbfgs_max_steps': 500, 'io.refine_converge_tol': 1e-6},
    'G_grid128':       {'archive.bins': [128, 128]},
    'G_grid512':       {'archive.bins': [512, 512]},
    'G_grid1024':      {'archive.bins': [1024, 1024]},
    'W_wider_bd':      {'archive.bounds': [[-0.0016, 0.0016], [-0.0016, 0.0016]]},
    'T_tighter_bd':    {'archive.bounds': [[-0.0004, 0.0004], [-0.0004, 0.0004]]},
    'B_autoscale_bd':  {'archive.bounds': None},
    'P_pop40':         {'algorithm.population_size': 40},
    'S_sigma10':       {'algorithm.initial_sigma': 1.0},
    'F_patience30':    {'algorithm.fitness_patience': 30},
    'F_patience300':   {'algorithm.fitness_patience': 300},
    'E_cma_eta01':     {'algorithm.cma_eta': 0.1},
    'G_grid2048':      {'archive.bins': [2048, 2048]},
}
def set_nested(d, dotted, val):
    keys = dotted.split('.')
    for k in keys[:-1]: d = d[k]
    d[keys[-1]] = val
for tag, mut in sweeps.items():
    with open(base) as f: c = yaml.safe_load(f)
    for k, v in mut.items(): set_nested(c, k, v)
    c['algorithm']['cma_seed'] = 42
    c['io']['run_name'] = f'apsc_triplet_REFSWEEP_{tag}_seed42'
    out = f'configs/ablations/apsc_triplet_REFSWEEP_{tag}.yaml'
    with open(out, 'w') as f: yaml.safe_dump(c, f, sort_keys=False)
    print(f'wrote {out}')
```

---

## Step 3 — run the searches

**Compute estimate per variant:**
* Smaller grids (128²): ~3 min wall
* Standard grids (256², 512²): ~5-7 min wall
* Larger grids (1024²): ~10-12 min wall (cell-novelty pathology adds restarts)
* Refinement: ~10 min wall with 30 workers (200 LBFGS steps default, longer than smaller-aperture)

**Total compute:** 12 variants × ~17 min (search + refine) = **~3.5 hours sequential**, comfortably in a 4hr alloc.

**Allocation params:**
```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=48G --time=4:00:00
```

**Critical: launch pattern that worked on sixlet** (use this exact form):
```bash
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
ALLOC=<your_jobid>
LOGDIR=results/apsc_triplet_refsweep_logs_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"

for TAG in D_long_lbfgs G_grid128 G_grid512 G_grid1024 G_grid2048 W_wider_bd T_tighter_bd B_autoscale_bd P_pop40 S_sigma10 F_patience30 F_patience300 E_cma_eta01; do
    CFG=configs/ablations/apsc_triplet_REFSWEEP_${TAG}.yaml
    echo "[$(date +%H:%M:%S)] [search] $TAG"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=40G \
        --chdir=/nas/longleaf/home/hisham26/Desktop/hyperlens \
        $HPY -m optical_rl.qd.lensmega.run \
        --config $CFG --iterations 100000 \
        > "$LOGDIR/${TAG}_search.log" 2>&1
    echo "[$(date +%H:%M:%S)] [search] $TAG done"

    # D_long_lbfgs: re-refine the EXISTING APS-C archive with new refine config; don't re-search.
    if [[ "$TAG" == "D_long_lbfgs" ]]; then
        RUN_DIR=results/lensmega/cooke_apsc_85mm_f4_20260504_225808
    else
        RUN_DIR=$(ls -td results/lensmega/apsc_triplet_REFSWEEP_${TAG}_seed42_* 2>/dev/null | head -1)
    fi
    if [[ -z "$RUN_DIR" ]]; then continue; fi
    OUT=$RUN_DIR/refined_top500_seed42_${TAG}.json
    echo "[$(date +%H:%M:%S)] [refine] $TAG"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=40G \
        --chdir=/nas/longleaf/home/hisham26/Desktop/hyperlens \
        $HPY tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUN_DIR" \
        --problem-config $CFG \
        --threshold -1.0 --top-n 500 --workers 30 \
        --out "$OUT" \
        > "$LOGDIR/${TAG}_refine.log" 2>&1
    echo "[$(date +%H:%M:%S)] [refine] $TAG done"
done
```

**Run with `nohup` and `disown`:**
```bash
nohup bash scripts_throwaway/run_apsc_triplet_ablation.sh $ALLOC > /tmp/apsc_triplet_run.log 2>&1 &
disown -a
```
**Bare `&` is not enough** — when the launching shell exits, the bash job tree can also exit. The sixlet study lost ~3 hours to this.

**Important: pass `--problem-config $CFG` to the refine driver** so it picks up the variant's refinement overrides (especially for `D_long_lbfgs`).

---

## Step 4 — pitfalls (don't repeat sixlet's chaos)

1. **Disk quota.** Each APS-C triplet HL search produces ~80-100 MB. With 12 variants, budget ~1.5 GB. Check `quota -s` before starting; hard limit on Longleaf is 76.8 GB.
2. **Alloc expiry mid-write.** Recovery snippet:
   ```python
   import pickle
   ckpt = pickle.load(open('checkpoint.pkl','rb'))
   pickle.dump(ckpt.archive, open('archive.pkl','wb'))
   ```
3. **`pkill -f` regex too broad.** `pkill -f "T_tighter"` matches both search and refine sruns plus anything else. Use `pgrep -af "..." | grep` to identify exact PIDs, then `kill -9 <PID>`.
4. **Parent script death.** Always use `nohup ... & ; disown -a`. Verify with `pgrep -af "run_apsc_triplet_ablation"`.
5. **Login-node `pgrep` only shows srun.** Actual python is on the compute node. Verify health by checking log mtime (`ls -la <log>` updates every few seconds during a healthy run).
6. **Corrupted archive.** Killed-mid-write produces `_pickle.UnpicklingError`. Fall back to `checkpoint.pkl` extraction (#2).
7. **Don't run multiple variants in parallel on same alloc.** OOMs or 5-10× slowdown. **One variant per alloc at a time.** Triplet is fast enough that sequential is fine.
8. **`bounds: null` only for `B_autoscale_bd`.** Other bound-sweep variants need explicit values.
9. **F/# is unconstrained during APS-C refine** (`refine_fnum_penalty_weight: 0.0`). Some refined designs may drift to F/# != 4.0 — don't filter them out, but flag in the report (the diversity_run accepted this drift).

---

## Step 5 — analysis

Mirror `scripts_throwaway/refsweep2_analysis.py`. Adapt by:
* Pointing to APS-C triplet refined JSON paths
* Loading APS-C triplet config's `theta_bounds` for ε-distinct normalization (D = 19, same as smaller-aperture cooke)
* Adding the existing APS-C triplet baseline (refined from `*_225808`) and *omitting* SAGA-steelman (no APS-C cooke triplet SAGA exists)
* Using **APS-C-appropriate RMS thresholds**: `[150, 200, 300, 500, 1000, 3000]` µm (NOT `[10, 30, 100, 1000]` from sixlet)
* Computing min RMS, since the *direction* of the patience-30 alpha (does it find rare deep basins?) is critical to test

Outputs:
```
figures/apsc_triplet_refsweep_data/
  ├── headline.csv
  └── eps_curves.csv
figures/apsc_triplet_refsweep_sanity.png    # 4-panel ε-vs-eps plot at RMS thresholds {200, 300, 500, 1000}
figures/REPORT_apsc_triplet_hl_ablation.md  # final report
```

### Predictions to test

| sixlet result | APS-C triplet expectation | what tells us if true / false |
|---|---|---|
| 2048² wins big | **1024² may help modestly** (no 2048² variant) | Confirms grid resolution is universally better, even at lower D |
| patience=30 finds sub-µm RMS | **Probably no sub-µm**, but may find ~100-130 µm minimum (better than baseline ~135 µm) | Tells us patience alpha is about *escaping local basins*, not *parameter dimension* |
| σ=1.0 wins on both quality and diversity | Should generalize (σ controls exploration breadth, topology-independent) | Cleanest test of generalization |
| Refinement saturated (D=A) | **May NOT saturate here** — APS-C baseline already uses max_steps=200 (sixlet was 100). Going to 500 may or may not bite. | If D variant differs from baseline, we have a result |
| Coarser grid (G_grid128) hurts | Should also hurt | Robustness footnote |
| Tighter bounds (T_tighter_bd) hurts | **Maybe not on APS-C** — baseline ±0.0008 is already very tight for the actual measure distribution | If T helps, calibration was sub-optimal |

### Cross-system table (fill in once all 3 topology runs are done)

| axis | smaller-aperture triplet | APS-C cooke triplet | sixlet | octuplet | universal recommendation? |
|---|---|---|---|---|---|
| Grid resolution | (TBD — see smaller-aperture runbook) | (TBD here) | 2048² (+84%) | (TBD) | (TBD) |
| Patience | (TBD) | (TBD) | 30 (sub-µm RMS) | (TBD) | (TBD) |
| Sigma | (TBD) | (TBD) | 1.0 (+12%) | (TBD) | (TBD) |
| Refinement | saturated | (test here) | saturated | (TBD) | likely yes |

**This 4-row table is the paper-headline cross-system summary.** Fill it in honestly — negative results (alpha doesn't generalize) are paper-relevant.

---

## Step 6 — combo champion

Run the combo only with knobs that **win individually** on APS-C triplet. Don't blindly copy sixlet's combo.

```yaml
# configs/ablations/apsc_triplet_REFSWEEP_CHAMPION.yaml — fill in based on Step 5 results
algorithm:
  population_size: 20                            # unchanged unless P_pop40 won
  initial_sigma: <best_S_value>
  fitness_patience: <best_F_value>
  cma_eta: <best_eta_or_baseline_0.2>
archive:
  bins: <best_grid>
  bounds: <best_bounds>
```

Compare CHAMPION to best individual variant. Three outcomes:
* **Combo > best individual** → alphas compose, headline cross-system result
* **Combo ≈ best individual** → only one knob is load-bearing on APS-C triplet
* **Combo < best individual** → alphas conflict on this topology

---

## Step 7 — produce the final report

Mirror `figures/REPORT_hl_hyperparam_ablation.md` (sixlet) at `figures/REPORT_apsc_triplet_hl_ablation.md`.

**Report structure:**
1. **TL;DR** — 1-paragraph summary of which alphas replicated, which didn't
2. **Big disclaimer up front:** APS-C cooke triplet is structurally limited to ~135 µm minimum RMS. The metrics here are sub-150/200/300 µm, not sub-30 µm. This is a *replication study* not a *headline performance study*.
3. **Setup** — APS-C config knobs, why baseline differs from smaller-aperture/sixlet
4. **Phase A** — refinement saturation check (note baseline is already max_steps=200)
5. **Phase B** — search-side variant table
6. **Per-axis findings** — replicate sixlet structure
7. **Negative / robustness results** — same structure as sixlet
8. **Cross-system table** — most important deliverable. Fill in alongside the smaller-aperture and octuplet results.
9. **No SAGA comparison** — flag this as a gap; if needed, run a fresh `sa_ga_refine_baseline.py` against `cooke_apsc_85mm_f4_discovery.yaml` (out of scope for this runbook)
10. **Limitations** — single seed, no SAGA reference, F/# unconstrained during refine

---

## Sanity-check log lines

Healthy APS-C triplet 100k-iter HL search:
```
[lensmega] starting CPU worker pool: 32 workers
[lensmega] run_dir=results/lensmega/apsc_triplet_REFSWEEP_..._seed42_<ts>
[seed] insert code=2 fitness=-X.XX measures=[..., ...]
... ~30 sec of seeding ...
[lensmega] gen=50 evals=1000 coverage=0.XX best=-X.XX new=N/20 ... wall=Xm
... search proceeds ...
[lensmega] final stats: {'filled': N, 'total_cells': M, 'coverage': X.X, 'best_fitness': -X.XX, 'iterations': 100000.0, ...}
```

* Best raw fitness should land in the range **-0.20 to -1.0** (i.e. raw RMS 200-1000 µm)
* If best > -0.10 (raw RMS < 100 µm) → suspicious for this template, double-check the config didn't accidentally use the smaller-aperture lens path
* If best < -2.0 (raw RMS > 2000 µm) → search failed, check for invalid-ray / bfl rejection in the seed phase

---

## Quick reference — exact commands

```bash
# 0. Verify quota
quota -s

# 1. Generate configs
$HPY scripts_throwaway/make_apsc_triplet_ablation_configs.py

# 2. Establish baseline by refining the existing best APS-C archive
RUN_DIR=results/lensmega/cooke_apsc_85mm_f4_20260504_225808
$HPY tools/refine_all_parallel.py --pool-from hl --hl-run "$RUN_DIR" \
  --threshold -1.0 --top-n 500 --workers 30 --out "$RUN_DIR/refined_top500_seed42_apsc_baseline.json"

# 3. Allocate compute
salloc --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=48G --time=4:00:00
# (note JOBID)

# 4. Launch (nohup + disown!)
nohup bash scripts_throwaway/run_apsc_triplet_ablation.sh $JOBID > /tmp/apsc_triplet_run.log 2>&1 &
disown -a

# 5. Monitor
tail -F /tmp/apsc_triplet_run.log

# 6. Analysis after all variants land
$HPY scripts_throwaway/apsc_triplet_refsweep_analysis.py

# 7. Combo champion if individual results show alphas
$HPY -m optical_rl.qd.lensmega.run \
  --config configs/ablations/apsc_triplet_REFSWEEP_CHAMPION.yaml --iterations 100000

# 8. Write the report
```

---

## Anticipated outcomes — bracketing the goal

The goal is to **break the 135 µm floor.** Here's how the outcomes break down:

**Best case (alpha generalizes, floor was a tuning artifact):**
* G_grid2048 reaches sub-50 µm minimum RMS, like sixlet's 39 µm median
* F_patience30 reaches sub-100 µm minimum RMS (sixlet got 3.8× improvement)
* CHAMPION combo reaches sub-30 µm minimum RMS — would be a paper headline
* The yaml's "15-40 µm" comment was prophetic; it just needed the right knobs

**Mid case (some alpha, but floor is real):**
* Best variant reaches ~80-120 µm minimum RMS — improvement but not the dramatic break sixlet showed
* Confirms HL has *some* tunability headroom but the topology genuinely limits absolute performance
* Cross-system table shows sixlet alphas partially generalize

**Worst case (floor is structural):**
* All 13 variants land near 130-150 µm minimum RMS, like the existing untuned runs
* CHAMPION combo doesn't help either
* **This is still paper-relevant**: confirms the v2 paper's diversity claim is the only way HL helps on APS-C cooke triplet — quality-side is topology-bound. Suggests Tessar / cemented-pair extensions are needed for this aperture-class.

**A negative result is paper-relevant.** Don't suppress. The cross-system table (smaller-aperture / APS-C / sixlet / octuplet) tells the universal-vs-topology-specific story for the paper.

---

## Why no SAGA comparison

The v2 paper's `sa_ga_refine_baseline.py` was run on the smaller-aperture cooke triplet config, not the APS-C variant. There's no existing SAGA archive against which to compare HL on this template. Two options:

1. **Skip SAGA from the report**, flag as a gap. (Recommended — keeps this runbook scope tight.)
2. Run `sa_ga_refine_baseline.py --problem-config cooke_apsc_85mm_f4_discovery.yaml --baseline-config <saga_baseline>.yaml --budget 100000 --out <path>` separately. ~15 min compute. Then re-run analysis with SAGA included.

Note in the report whichever choice you make.

---

*Generated 2026-05-09. Sister runbooks: sixlet at `figures/REPORT_hl_hyperparam_ablation.md` (executed); octuplet at `figures/RUNBOOK_octuplet_hl_ablation.md`. The previous version of this runbook (smaller-aperture cooke triplet) is in git history if you need both.*
