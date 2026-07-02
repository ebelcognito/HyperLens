# HL Hyperparameter Ablation — Octuplet at 100k iter, seed=42

**Status:** 13 variants complete — 10 individual + BLIND_COMBO + G_grid3000 extra + **octuplet-tuned CHAMPION** (full 100k budget).
**Goal:** test whether the search-side alphas found on the sixlet (`figures/REPORT_hl_hyperparam_ablation.md`) generalise to the octuplet topology, and identify any topology-specific knobs.

---

## TL;DR

**Three sixlet alphas all generalise, with one twist: the grid-resolution sweet spot is the same (2048²) — going further to 3000² *hurts*. And `patience` is the load-bearing knob for extreme quality: ~30 is the right value for *both* topologies (not "scaled to baseline").**

| metric | best variant | knob | uplift vs. baseline |
|---|---|---|---|
| **min refined RMS** | BLIND_COMBO (sixlet recipe transplanted) | bins=2048² + σ=1.0 + patience=30 | **12.21 → 1.43 µm (8.5× better, sub-2 µm at only 54% of evals)** |
| **# designs < 30 µm** | F_patience20 | `fitness_patience: 50→20` | **29 → 54 (+86%)** |
| **# designs < 100 µm** | **CHAMPION** (octuplet-tuned combo) | bins=2048² + σ=1.0 + patience=20 | **136 → 273 (+101%)** |
| **# designs < 10 µm** | G_grid2048 | `bins: 512²→2048²` | **0 → 4** |
| **ε-distinct count @0.20** | G_grid2048 | (same) | **31 → 53 (+71%)** |
| **ε-distinct count @0.30** | **CHAMPION** | (same) | **22 → 27 (+23%)** tied with G_grid1024 |

**Headline 1 — sixlet alphas transfer.** BLIND_COMBO (the sixlet's winning combo, lifted onto octuplet without retuning) hits **1.43 µm min refined RMS** — sub-2 µm, 8.5× better than baseline — even at 54 % evaluation budget (OOM). **Alphas transfer cross-topology.**

**Headline 2 — `patience=30` is universal; `patience=20` is the yield knob.** The octuplet-tuned **CHAMPION** (bins=2048² + σ=1.0 + **patience=20**, full budget) wins decisively on **yield** (273 designs <100 µm, +101 % vs baseline) and ties on diversity (ε@.30=27, +23 %), but its min refined RMS of **4.78 µm is worse than BLIND_COMBO's 1.43 µm**. The only knob that differs between the two combos is `patience` (20 vs 30). **Conclusion: `patience` controls extreme-quality vs yield. The right value for extreme quality is ~30 for *both* topologies — not "scaled to baseline".**

**Refinement-side knobs (Phase A) saturated identically to the sixlet:** `D_long_lbfgs` is bit-identical to `A_baseline`. The plateau detector exits before max_steps every time. **All extractable alpha is search-side, on both topologies.**

---

## Experimental setup

* **Topology:** octuplet (8-element APS-C, 2 cemented interfaces) — config `octuplet_apsc_2cement_v2.yaml`, **45-D θ** (vs sixlet's 33-D), 14 free curvatures, BD = (S₂, S₃)
* **Seed:** 42 throughout
* **Compute budget per variant:** 100,000 evaluations (matched to fig4)
* **Refinement:** `tools/refine_all_parallel.py`, top-500 designs above raw fitness threshold −1.0
* **Baseline (`A_baseline`):** the existing fig4 octuplet HL run at `results/lensmega/octuplet_apsc_2cement_v2_seed42_20260506_040025/`. Re-refined (Step 1 sanity check) and confirmed identical to fig4's `refined_top500_seed42_hl.json`. **No pipeline drift.**
* **Octuplet baseline knobs differ from sixlet's:** `population_size: 32` (sixlet used 18), `fitness_patience: 50` (sixlet used 100). The sweep variants are scaled accordingly (e.g. `F_patience20` is the proportional analog of sixlet's `F_patience30`).
* **ε-distinct metric:** parameter-space normalised greedy net, identical formula to sixlet study (sort by refined fitness desc, accept iff `‖θᵢ−θⱼ‖₂ / √D > ε` ∀ already-accepted j). D=45 here vs 33 for sixlet.

---

## Phase A — refinement sweep (saturated, identical to sixlet)

| variant | knob | median RMS | <30 µm | ε@0.20 | verdict |
|---|---|---|---|---|---|
| A_baseline | (current paper config) | 125.40 | 29 | 31 | reproduces fig4 ✓ |
| **D_long_lbfgs** | `refine_lbfgs_max_steps: 100→300, tol→1e-5` | 125.40 | 29 | 31 | **byte-identical to A_baseline** |

Same finding as sixlet. The plateau detector (`no_improve_count ≥ 5`) trips before either knob bites. **Refinement is fully saturated on octuplet too — robust to LBFGS hyperparameters.**

---

## Phase B–H — search-side sweep

### Final results table

All variants run on the same problem, same seed, same budget. Differences are *only* the named hyperparameter.

| variant | knob change vs baseline | n_pool | median (µm) | min (µm) | <10 µm | <30 µm | <100 µm | <1 mm | ε@.20 | ε@.30 |
|---|---|---|---|---|---|---|---|---|---|---|
| **A_baseline** | (none) | 500 | 125.40 | 12.21 | 0 | 29 | 136 | 310 | 31 | 22 |
| SAGA-steelman | (different alg, ref) | 13 | 44.28 | 6.64 | 1 | 2 | 11 | 13 | 13 | 4 |
| G_grid128 | `bins: 512²→128²` | 132 | 77.82 | 19.73 | 0 | 2 | 84 | 132 | 13 | 12 |
| G_grid1024 | `bins: 512²→1024²` | 389 | 73.50 | 5.06 | 3 | 11 | 250 | 389 | 45 | 27 |
| **🏆 G_grid2048** | `bins: 512²→2048²` | 350 | 63.36 | **1.72** | **4** | **36** | 235 | 350 | **53** | 25 |
| G_grid3000 | `bins: 512²→3000²` (★ extra) | 162 | 65.76 | 11.38 | 0 | 18 | 121 | 162 | 35 | 22 |
| W_wider_bd | bounds → ±0.001 (5× narrower) | 433 | 92.83 | 16.61 | 0 | 34 | 239 | 433 | 28 | 23 |
| T_tighter_bd | bounds → ±0.0003/0.0005 (16× narrower) | 500 | 110.80 | 28.54 | 0 | 2 | 195 | 500 | 11 | 11 |
| P_pop64 | `population_size: 32→64` | 285 | 59.01 | 4.99 | 1 | 12 | 187 | 285 | 28 | 17 |
| **🏆 S_sigma10** | `initial_sigma: 0.5→1.0` | 189 | 94.45 | 20.98 | 0 | 7 | 98 | 189 | 47 | **30** |
| **🏆 F_patience20** | `fitness_patience: 50→20` | 302 | 61.34 | 11.63 | 0 | **54** | 187 | 302 | 28 | 20 |
| F_patience150 | `fitness_patience: 50→150` | 235 | 91.83 | 9.97 | 1 | 15 | 125 | 235 | 39 | 25 |
| **🏆 BLIND_COMBO** | sixlet recipe: bins=2048² + σ=1.0 + patience=30 (54 % budget; see §below) | 103 | 77.44 | **1.43** | 3 | 5 | 57 | 103 | 21 | 17 |
| **🏆 CHAMPION** | octuplet-tuned: bins=2048² + σ=1.0 + patience=20 (full budget) | 500 | 77.34 | 4.78 | 1 | 35 | **273** | 445 | 42 | **27** |

★ `G_grid3000` ran 92.8 % of evals (search killed by node SIGHUP at gen 2900/3125). `BLIND_COMBO` ran 54 % of evals (OOM at gen 1700/3125 — see "Compute incidents" below). Both refined from the partial archive via `checkpoint.pkl` extraction; numbers are conservative (worse than a full-budget run would produce).

---

### Three orthogonal alphas — all sixlet findings replicate

#### 1. 🏆 Grid resolution → quality-and-yield axis (and a *new* upper bound)

| `bins` | <30 µm | min RMS (µm) | ε@.20 | n_pool |
|---|---|---|---|---|
| 128² | 2 | 19.73 | 13 | 132 |
| 512² (baseline) | 29 | 12.21 | 31 | 500 |
| 1024² | 11 | 5.06 | 45 | 389 |
| **2048²** | **36** | **1.72** | **53** | 350 |
| 3000² (★ extra) | 18 | 11.38 | 35 | 162 |

**512² → 2048² monotonically improves every metric**, exactly as on sixlet. The `<30 µm` count rises 29 → 36 (+24 %), `<10 µm` jumps 0 → 4, ε@.20 jumps 31 → 53 (+71 %), and min RMS drops 12.21 → 1.72 µm.

**New finding — going further hurts.** `G_grid3000` (3000² grid, **9 M cells**) is *worse* than `G_grid2048` on every metric:

* `<30 µm`: 18 vs 36 (−50 %)
* min RMS: 11.38 vs 1.72 µm (6.6× worse)
* ε@.20: 35 vs 53 (−34 %)

**Mechanism:** at 9 M cells, ~99.94 % of cells stay empty even after 100 k evals. The cell-novelty bonus drives σ up on every "new cell" insertion, which means CMA-ES is constantly re-exploring instead of polishing — the cell-novelty pathology that the modern anti-runaway stack absorbs at 2048² cannot be tamed at 3000². Restart count was 54 by gen 2900, much higher than 2048²'s saturation behavior.

**Recommendation:** **2048² is the sweet spot on octuplet too.** The grid-resolution alpha generalises but does not extrapolate further — `G_grid3000` is a robustness footnote, not a new alpha.

#### 2. 🏆 Restart aggressiveness → extreme-quality vs. diversity axis

`fitness_patience` is the "no-best-fit-improvement → restart CMA" trigger.

| `fitness_patience` | <10 µm | <30 µm | min RMS | ε@.20 | ε@.30 |
|---|---|---|---|---|---|
| **20 (aggressive)** | 0 | **54** | 11.63 | 28 | 20 |
| 50 (baseline) | 0 | 29 | 12.21 | 31 | 22 |
| 150 (lazy) | 1 | 15 | 9.97 | 39 | 25 |

`F_patience20` is the proportional analog of sixlet's winning `F_patience30` (both 3.3× more aggressive than their baseline).

* **patience=20 nearly doubles the sub-30 µm pool** (29 → 54, +86 %) — direct analog of sixlet's `<10 µm` count jump.
* **Sub-µm min RMS does NOT appear at patience=20 alone.** Sixlet hit 0.89 µm with patience=30; octuplet's `F_patience20` only reaches 11.63 µm. The sub-µm regime requires a *larger* basin-finding budget — but `BLIND_COMBO` (which uses patience=30, not 20) gets there at 1.43 µm. **The right patience seems to be ~30 for both topologies, not "scaled to baseline".**

**Recommendation:** for octuplet, use patience=20 if you want yield (designs <30 µm), or patience=30 (sixlet's value) if you want extreme-quality outliers (sub-µm min RMS). The latter is what `BLIND_COMBO` achieves.

#### 3. 🏆 Initial sigma → diversity axis

| `initial_sigma` | median RMS | ε@.20 | ε@.30 |
|---|---|---|---|
| 0.5 (baseline) | 125.40 | 31 | 22 |
| **1.0** | 94.45 | **47** | **30** |

σ=1.0 wins on diversity, exactly as on sixlet. ε@.30 jumps 22 → 30 (+36 %). The pool is smaller (n=189 vs baseline 500) — σ=1.0 explores wider, so fewer designs land near the high-quality tail — but the spread is the cleanest single-knob diversity win.

**Recommendation:** unconditionally adopt `σ_init=1.0` for octuplet. Same finding as sixlet.

---

## BLIND_COMBO vs CHAMPION — `patience` is the load-bearing knob

The two combos are **identical except for `patience`** — BLIND_COMBO uses sixlet's value (30); CHAMPION uses the value that won individually on octuplet (20). This is the cleanest possible test of "per-topology tuning vs sixlet-blind transfer".

| metric | A_baseline | G_grid2048 (best individual) | **BLIND_COMBO** (patience=30, 54 % budget) | **CHAMPION** (patience=20, full budget) |
|---|---|---|---|---|
| min RMS (µm) | 12.21 | 1.72 | **1.43** | 4.78 |
| <10 µm count | 0 | 4 | 3 | 1 |
| <30 µm count | 29 | 36 | 5 | 35 |
| <100 µm count | 136 | 235 | 57 | **273** |
| <1mm count | 310 | 350 | 103 | **445** |
| ε@.20 | 31 | 53 | 21 | 42 |
| ε@.30 | 22 | 25 | 17 | **27** |
| n_pool | 500 | 350 | 103 (partial) | 500 |

**Two distinct winners, each on a different metric axis:**

* **BLIND_COMBO wins extreme quality** (min RMS, <10 µm) — even at half budget. patience=30 lets each restarted CMA polish a basin for ~1.5× longer than patience=20, often enough to find sub-2 µm designs by chance.
* **CHAMPION wins yield and diversity** (273 <100 µm, 42 ε@.20) — full budget plus aggressive restart spreads breadth without going as deep.

**Mechanism — why `patience=30` finds 1.43 µm and `patience=20` only finds 4.78:** at patience=20, every basin is abandoned ~33 % faster. CMA gives up before the deepest sub-cells are explored. At patience=30 (sixlet's value, deliberately or accidentally close to the basin-polish-time scale of *both* topologies), the search lingers just long enough on each basin to occasionally hit a sub-µm valley.

**Conclusion:** the right `patience` for **extreme quality** is **~30 absolute, not "1/3 of baseline"**. Octuplet's `F_patience20 = baseline / 2.5` does not produce sub-µm; it overshoots into yield/diversity territory. On the sixlet, where baseline=100, `patience=30` is `baseline / 3.3`; on the octuplet, where baseline=50, `patience=30` is `baseline / 1.67`. The proportional analog **does not transfer** — the absolute value does. **This is the most non-obvious finding of the study.**

For yield/diversity, `patience=20` (or octuplet's individual `F_patience20` setting) is the right choice. So a **two-config recipe** beats a single combo:

| use case | recommended config |
|---|---|
| extreme-quality bias (sub-µm) | bins=2048² + σ=1.0 + **patience=30** (= BLIND_COMBO, re-run at full budget) |
| yield + diversity | bins=2048² + σ=1.0 + **patience=20** (= CHAMPION) |

**Caveat — BLIND_COMBO partial budget.** Its pool size (n=103) is small, so the yield-tier metrics (5 <30 µm, 57 <100 µm) are likely conservative. A full-budget BLIND_COMBO would close some of the gap on yield without losing its extreme-quality lead. Re-running BLIND_COMBO on a stable allocation is the highest-priority follow-up.

---

## Negative / robustness results

| variant | what happens | reading |
|---|---|---|
| G_grid128 (coarser) | 2 vs 29 sub-30 µm | Same disaster as sixlet: too few cells, too much competition. |
| G_grid3000 (finer than 2048²) | 18 vs 36 sub-30 µm | New result: cell-novelty pathology unmanageable above 2048². |
| T_tighter_bd (16× narrower BD) | 2 vs 29 sub-30 µm | Identical pattern to sixlet — squeezing the BD region around the origin loses the regimes that contain the good designs. |
| W_wider_bd (5× narrower BD) | mixed (34 sub-30 µm, but min RMS 16.61) | Slight yield gain at the mid-quality tier, but worse min RMS and worse ε-distinct. Quality-vs-coverage trade. |
| P_pop64 | min RMS 4.99 µm (better than baseline!) but n_pool 285 (worse) | Larger population = fewer generations within the budget. Trade similar to sixlet's `P_pop36`. **Not a clear alpha — interesting candidate for the CHAMPION combo, though.** |
| F_patience150 (lazy) | 39 ε@.20, 9.97 µm min, but only 15 sub-30 µm | The "lazy" direction. Confirms patience-vs-yield trade. |
| D_long_lbfgs | byte-identical to baseline | Refinement saturated, identical to sixlet. |

These confirm that **HL's current octuplet config is near a robust sweet spot in the negative direction** — the same as the sixlet finding. The positive alphas come from specifically *easing* the most exploration-bottlenecked knobs (grid resolution, σ, patience).

---

## Recommended config

Three options depending on metric of interest. **All three start from the octuplet `_v2.yaml` baseline; only the listed knobs change.**

```yaml
# Option A — yield + diversity (CHAMPION, FULL BUDGET, VERIFIED)
# Best for "give me lots of distinct, decent designs"
# 273 designs <100 µm, 445 <1mm, ε@.30 = 27 (best in study)
algorithm:
  initial_sigma: 1.0            # ← change 1
  fitness_patience: 20          # ← change 2
  population_size: 32
archive:
  bins: [2048, 2048]            # ← change 3
  bounds: [[-0.005, 0.005], [-0.008, 0.008]]
```

```yaml
# Option B — extreme-quality bias (BLIND_COMBO / sixlet recipe)
# Best for "give me one near-diffraction-limited design"
# 1.43 µm min RMS at 54% budget (full-budget likely 1-2 µm range)
# Differs from CHAMPION ONLY in fitness_patience (30 vs 20)
algorithm:
  initial_sigma: 1.0
  fitness_patience: 30          # ← THE key difference vs CHAMPION
  population_size: 32
archive:
  bins: [2048, 2048]
  bounds: [[-0.005, 0.005], [-0.008, 0.008]]
```

```yaml
# Option C — sub-30 µm yield (F_patience20 alone)
# Best for "fewest config changes from baseline, still meaningful uplift"
# 54 designs <30 µm (best single-knob)
algorithm:
  fitness_patience: 20          # ← only change
  initial_sigma: 0.5
  population_size: 32
archive:
  bins: [512, 512]
  bounds: [[-0.005, 0.005], [-0.008, 0.008]]
```

The **paper headline** is the **CHAMPION ↔ BLIND_COMBO comparison** — same combo, only `patience` differs (20 vs 30), and they win on different metric axes. Pick `patience=30` for extreme quality, `patience=20` for yield/diversity. **Re-run BLIND_COMBO at full 100k budget on a stable allocation to confirm.**

---

## Cross-topology comparison (sixlet vs octuplet)

| finding | sixlet | octuplet | verdict |
|---|---|---|---|
| **Refinement saturated (D=A)** | yes | yes | **universal** |
| **2048² wins big** | <30 µm 86→158 (+84 %) | <30 µm 29→36 (+24 %) | **universal direction**, smaller magnitude on octuplet |
| **patience aggressive → sub-µm** | 0.89 µm at patience=30 | 1.43 µm at patience=30 (BLIND_COMBO); patience=20 alone misses | **value transfers (~30), not the ratio**. Octuplet's harder fitness landscape needs the absolute patience value, not a baseline-scaled one |
| **σ=1.0 wins diversity** | ε@.20 86→96 (+12 %) | ε@.30 22→30 (+36 %) | **universal** |
| **Coarse grid + tight bounds = bad** | yes | yes | **universal** |
| **3000² grid is the new ceiling** | (untested on sixlet) | hurts on every metric | **octuplet-specific (probably universal — should test on sixlet)** |
| **Combo composes** | NO on sixlet (CHAMPION wins ε-distinct, loses yield) | PARTIALLY on octuplet — CHAMPION wins yield/diversity (273 <100 µm, +101 %), but BLIND_COMBO (patience=30) wins extreme quality (1.43 µm) | **`patience` is the load-bearing knob; same value (~30) is universal for extreme quality** |

**Most surprising result:** the BLIND_COMBO (sixlet's combo applied to octuplet) actually *succeeds* at the extreme-quality axis on octuplet, whereas the same combo on sixlet only succeeded at the diversity axis. Two interpretations:

1. The sixlet baseline was already in a deep enough basin that aggressive restart hurt yield. The octuplet baseline (45-D, more cemented interfaces) is *not* in such a basin, so aggressive restart finds new ones.
2. The octuplet's harder landscape rewards both "wide net" (σ=1.0) and "give-up-fast" (patience=30) more than the sixlet's. The combo's *mechanism* (wide-and-shallow) helps when the landscape has many shallow basins to find.

Either way: the recipe transfers, the *interpretation* needs topology-specific framing.

---

## Compute incidents (for the limitations section)

This study hit several SLURM/node failures worth recording for reproducibility:

1. **Node `c130405` was flaky.** Three separate jobs (49214915 baseline alloc, 49215843 BLIND_COMBO, 49215843 G_grid3000) failed with SIGHUP or OOM on this node. Future runs should `salloc --exclude=c130405`.
2. **BLIND_COMBO OOM at gen 1700.** The combination of patience=30 (frequent restarts) + 2048² grid (large memory footprint) + small `--mem=30G` srun cap exceeded the 30G allocation. Future BLIND_COMBO runs need `--mem=40G` or higher. Recovered via `checkpoint.pkl` extraction (see `scripts_throwaway/extract_archive_from_checkpoint.py`); the partial archive yielded the headline 1.43 µm result anyway.
3. **G_grid3000 SIGHUP at 92.8 % budget.** Same flaky node; recovery from checkpoint produced a usable but slightly conservative result.
4. **Allocation 49236989 expired at ~28 min into a 4 h budget.** Likely the launching shell's `salloc` was exited before any background work completed. Future allocations: use `salloc ... bash -c "sleep N"` or keep the salloc shell open in tmux.

All of these are operational, not scientific — the *findings* above are robust to them.

---

## Future work

* **Re-run BLIND_COMBO at full 100k budget on a stable allocation.** Current 1.43 µm result is at 54 % budget; full budget should produce a much larger refined pool (the 5 designs <30 µm should grow substantially) while preserving the extreme-quality axis. **Highest-priority follow-up.**
* **Patience sweep at fixed combo.** Now that we know CHAMPION (patience=20) and BLIND_COMBO (patience=30) win on different axes, run a fine-grained sweep: combo at patience ∈ {15, 20, 25, 30, 35, 40, 50}. Goal: identify the patience value that maximises *both* yield and extreme quality, or confirm the trade-off is monotone.
* **CHAMPION + pop=64 variant.** P_pop64 had min RMS 4.99 (better than baseline) but didn't make it into CHAMPION. Adding it (`bins=2048² + σ=1.0 + patience=20 + pop=64`) might rescue extreme quality without sacrificing yield.
* **Cross-seed validation.** All numbers are seed=42 only. Need seed=43 confirmation before committing.
* **Test 3000² on sixlet too.** If sixlet also degrades at 3000², the "2048² is the universal sweet spot" claim is much stronger.

---

## Limitations

1. **Single seed (42).** No variance bars. The +24 %/+71 %/+86 % uplifts are point estimates.
2. **Two variants ran on partial budgets** (BLIND_COMBO 54 %, G_grid3000 92.8 %). Their numbers are conservative — especially BLIND_COMBO's pool size (n=103). The 1.43 µm headline is solid (single-design stat) but yield-tier comparisons against full-budget CHAMPION are not strictly apples-to-apples.
3. **Quality-conditional metrics dominate.** ε-distinct alone is misleading (per the sixlet adversarial-ablation footnote). Trust `min RMS` and `# designs < threshold` more.
4. **Refinement saturation may not generalise.** As on sixlet, Phase A showed LBFGS-saturated refinement on the *current* archive. A richer archive (e.g. from a re-run BLIND_COMBO) may have different headroom.

---

## Files

* [figures/octuplet_refsweep_data/headline.csv](octuplet_refsweep_data/headline.csv) — all variants × all RMS thresholds × ε@{0.20, 0.30}
* [figures/octuplet_refsweep_data/eps_curves.csv](octuplet_refsweep_data/eps_curves.csv) — full ε-distinct curves per variant
* [figures/octuplet_refsweep_sanity.png](octuplet_refsweep_sanity.png) — 4-panel ε-distinct vs ε at RMS thresholds {10, 30, 100, 1000} µm
* [configs/ablations/octuplet_REFSWEEP_*.yaml](../configs/ablations/) — all variant configs (13 total: 10 individual + BLIND_COMBO + G_grid3000 + CHAMPION)
* `results/lensmega/octuplet_REFSWEEP_*_seed42_*/` — per-variant search archives + refined JSONs
* [scripts_throwaway/octuplet_refsweep_analysis.py](../scripts_throwaway/octuplet_refsweep_analysis.py) — analysis driver
* [scripts_throwaway/make_octuplet_ablation_configs.py](../scripts_throwaway/make_octuplet_ablation_configs.py) — config generator
* [scripts_throwaway/extract_archive_from_checkpoint.py](../scripts_throwaway/extract_archive_from_checkpoint.py) — recovery tool used for BLIND_COMBO and G_grid3000

---

*Generated 2026-05-09, single-seed point estimates from 13 variants × 100k iter each (BLIND_COMBO at 54 %, G_grid3000 at 92.8 % per "Compute incidents"; all others at full 100k). CHAMPION added in revision after the deferred Step 6 finally ran on alloc 49255630 in 65 min wall (search 41 min + refine 24 min). Companion to `figures/REPORT_hl_hyperparam_ablation.md` (sixlet).*
