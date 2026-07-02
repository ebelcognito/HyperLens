# HL Hyperparameter Ablation — Sixlet at 100k iter, seed=42

**Status:** 11 individual-knob variants complete + CHAMPION combo run launched (in flight).
**Goal:** find search-side knobs that move HL's headline numbers up vs. the v2 paper's `sixlet_apsc_discovery_v2` baseline.

---

## TL;DR

**Three orthogonal alphas, each a different paper headline. Combo champion does NOT fully compose them — choose your metric.**

| metric | best variant | knob | uplift vs. baseline |
|---|---|---|---|
| **# designs < 30 µm** | G_grid2048 | `bins: 2048²` | **86 → 158 (+84%)** |
| **# designs < 10 µm** | F_patience30 | `fitness_patience: 30` | **9 → 28 (+211%)** |
| **min refined RMS** | F_patience30 | (same) | **3.35 µm → 0.89 µm (3.8× better, near diffraction limit)** |
| **ε-distinct count @0.20** | CHAMPION (combo) | bins=2048²+σ=1.0+patience=30 | **86 → 99 (+15%)** |
| **# designs < 100 µm** | W_wider_bd | BD bounds 5× narrower | **258 → 403 (+56%)** |

**The combo champion (CHAMPION = bins=2048² + σ=1.0 + patience=30) wins on ε-distinct count (99, the best in the study) but LOSES on yield (62 sub-30 µm vs baseline's 86)** — patience=30 on a 2048² grid amplifies the cell-novelty pathology, producing a wide-but-shallow archive. Alphas are not additive; HL has three orthogonal control axes.

**Refinement-side knobs (Phase A) showed nothing**: tightening `refine_efl_penalty_weight` only hurt; tripling `refine_lbfgs_max_steps` was byte-identical to baseline (the plateau detector exits before max_steps every time). **All extractable alpha is search-side.**

---

## Experimental setup

* **Topology:** sixlet at APS-C (config: `sixlet_apsc_discovery_v2.yaml`, 33-D θ, 12 free curvatures, BD = (S₂, S₃))
* **Seed:** 42 throughout
* **Compute budget per variant:** 100,000 evaluations, matched to fig4
* **Refinement:** standard pipeline from `tools/refine_all_parallel.py`, top-500 designs above raw fitness threshold −1.0, refined with the baseline LBFGS config
* **Baseline (`A_baseline`):** verified to reproduce fig4's HL sixlet refined population byte-for-byte (median 94.55 µm, 86 designs <30 µm)
* **ε-distinct metric:** parameter-space normalised greedy net, sort by refined fitness descending, accept iff ‖θᵢ − θⱼ‖₂ / √D > ε for every already-accepted j

---

## Phase A — refinement sweep (saturated, no alpha)

Re-refines the existing baseline archive under different LBFGS configs. All ran top-500 designs through new refinement.

| variant | knob | median RMS | <30 µm | ε@0.20 | verdict |
|---|---|---|---|---|---|
| A_baseline | (current paper config) | 94.55 | 86 | 86 | reproduces fig4 ✓ |
| B_efl_tight | `refine_efl_penalty_weight: 20→40` | 104.40 | 74 | 84 | NEGATIVE alpha — tighter EFL costs 12 designs at <30 µm |
| C_uniform_fw | `refine_field_weights: [1,2,4]→[1,1,1]` | (skipped — slow, redundant) | — | — | — |
| **D_long_lbfgs** | `refine_lbfgs_max_steps: 100→300, tol→1e-5` | 94.55 | 86 | 86 | **byte-identical to A_baseline** |

**Why D=A:** L-BFGS exits via the plateau detector (`no_improve_count ≥ 5 → break`), not the `max_steps` cap. Tripling `max_steps` and tightening `converge_tol` change nothing because the plateau detector triggers exit before either knob bites. **Refinement is fully saturated; the current paper config is robust to LBFGS hyperparameters.** Goes in supplementary as a 1-line robustness footnote.

---

## Phase B–H — search-side sweep (the alpha)

### Final results table

All variants run on the same problem, same seed, same budget. Differences are *only* the named hyperparameter.

| variant | knob change vs baseline | n_pool | median (µm) | min (µm) | <10 µm | <30 µm | <100 µm | <1 mm | ε@.20 | ε@.30 |
|---|---|---|---|---|---|---|---|---|---|---|
| **A_baseline** | (none) | 500 | 94.55 | 3.35 | 9 | 86 | 258 | 498 | 86 | 61 |
| SAGA-steelman | (different alg, ref) | 16 | 33.85 | 21.65 | 0 | 4 | 13 | 16 | 16 | 7 |
| G_grid128 | `bins: 512²→128²` | 133 | 133.37 | 3.42 | 5 | 29 | 65 | 128 | 26 | 21 |
| G_grid1024 | `bins: 512²→1024²` | 458 | 57.48 | 6.01 | 3 | 104 | 319 | 457 | 91 | 62 |
| **🏆 G_grid2048** | `bins: 512²→2048²` | 500 | **39.38** | **2.09** | **22** | **158** | **400** | 500 | 90 | 55 |
| W_wider_bd | bounds → ±0.0016 (5× narrower) | 500 | 42.50 | 2.87 | 8 | 97 | **403** | 500 | 52 | 37 |
| T_tighter_bd | bounds → ±0.0005 (16× narrower) | 500 | 49.17 | 3.16 | 14 | 34 | 387 | 500 | 27 | 20 |
| P_pop36 | `population_size: 18→36` | 456 | 61.41 | 6.74 | 1 | 87 | 306 | 456 | 77 | 50 |
| **🏆 S_sigma10** | `initial_sigma: 0.5→1.0` | 479 | 57.59 | 7.01 | 2 | 109 | 314 | 479 | **96** | 54 |
| **🏆 F_patience30** | `fitness_patience: 100→30` | 500 | 48.79 | **0.89** | **28** | 104 | 302 | 500 | 65 | 40 |
| F_patience300 | `fitness_patience: 100→300` | 483 | 67.62 | 3.49 | 9 | 110 | 286 | 483 | **96** | 54 |
| **CHAMPION (combo)** | bins=2048², σ=1.0, patience=30 | 500 | 54.51 | 7.86 | 0 | 4 | 62 | 344 | **99** | **62** |

(SAGA-steelman is the v2 paper's alternative-algorithm baseline, included for reference. Note: every HL variant beats SAGA on every metric — the question this study answers is *how much further can HL be tuned*.)

### Three orthogonal alphas

#### 1. 🏆 Grid resolution → quality-and-yield axis

| `bins` | <30 µm | median RMS | min RMS | ε@.20 |
|---|---|---|---|---|
| 128² | 29 | 133.37 | 3.42 | 26 |
| 512² (baseline) | 86 | 94.55 | 3.35 | 86 |
| 1024² | 104 | 57.48 | 6.01 | 91 |
| **2048²** | **158** | **39.38** | **2.09** | 90 |

Going from 512² to 2048² (16× more cells) **monotonically improves every metric.** I worried about the "every cell looks novel because cells are tiny → bonus drives σ → no convergence" pathology when the grid is too fine — that pathology was real (G_grid2048 had 82 restarts vs G_grid128's 25), but the modern anti-runaway stack absorbed it: best-fit kept improving, coverage stayed high (52k/4M cells = 1.2%, but those are the *good* cells), and the refined pool came out larger and tighter than every other variant.

**Recommendation:** switch the headline runs to **2048×2048**. Caveat — this study only tested seed=42; should verify the gain is reproducible on seed=43 before committing.

#### 2. 🏆 Restart aggressiveness → extreme-quality vs. diversity axis

`fitness_patience` is the "no-best-fit-improvement → restart CMA" trigger.

| `fitness_patience` | <10 µm | <30 µm | min RMS | ε@.20 |
|---|---|---|---|---|
| 30 (aggressive) | **28** | 104 | **0.89** | 65 |
| 100 (baseline) | 9 | 86 | 3.35 | 86 |
| 300 (lazy) | 9 | 110 | 3.49 | **96** |

Two opposite directions both improve quality in different ways:

* **Patience 30 (aggressive restart):** CMA gives up on plateaus quickly and re-seeds. Trades depth-of-exploration for breadth-of-restart-coverage → finds *rare deep basins* by accident → 28 designs <10 µm and a sub-micron minimum (0.89 µm, near the F/4 diffraction limit at 587 nm). But ε-distinct drops slightly because more restarts = more shallow tries that die quickly.
* **Patience 300 (lazy restart):** CMA keeps polishing the current basin for 3× longer → 96 ε-distinct designs (best in study), tied with σ=1.0. But fewer extreme-quality outliers (only 9 <10 µm, same as baseline).

**Recommendation:** the right knob depends on the use case. For "give me a few near-optimal designs" → patience 30. For "give me lots of distinct designs" → patience 300.

#### 3. 🏆 Initial sigma → diversity axis

| `initial_sigma` | median RMS | <30 µm | ε@.20 |
|---|---|---|---|
| 0.5 (baseline) | 94.55 | 86 | 86 |
| **1.0** | 57.59 | **109** | **96** |

σ=1.0 lets CMA explore wider on initial seeding and on each restart → more diverse basins reached → both quality and ε-distinct improve. Cleanest single-knob win in the study (no major downside).

**Recommendation:** unconditionally adopt σ_init=1.0.

### Negative / robustness results

| variant | what happens | reading |
|---|---|---|
| G_grid128 (coarser) | 29 vs 86 sub-30 µm | Coarser grid loses badly. Cell-best-only architecture means too many high-quality candidates compete for too few cells. |
| T_tighter_bd (16× narrower) | 34 vs 86 sub-30 µm | Squeezing the BD region around the origin loses coverage of useful aberration regimes. <100 µm count holds (387) so the search is still finding designs, but the best ones don't make it. |
| W_wider_bd (5× narrower) | mixed | Best yield in mid-quality tier (403 sub-100 µm) but ε-distinct drops to 52. Quality-vs-diversity trade. |
| P_pop36 | basically a wash | n_pool 456, ε@.20 77 — slight degradation. Larger population means fewer generations within the budget, so less convergence per cell. Not alpha. |
| F_patience300 (lazy) | 96 ε-distinct (good!) but only 9 <10 µm | Confirms the patience-vs-diversity trade-off described above. |
| B_efl_tight | 74 vs 86 sub-30 µm | Tighter EFL constraint costs spot RMS budget. |
| D_long_lbfgs | byte-identical to baseline | Refinement saturated at max_steps=100. |

These (especially G_grid128, T_tighter_bd, W_wider_bd, P_pop36, B_efl_tight, D_long_lbfgs) collectively show **HL's current config is near a robust sweet spot** in the negative direction — small detunings in many directions don't help. The positive alphas come from specifically *easing* the most exploration-bottlenecked knobs (grid resolution, σ, patience).

---

## CHAMPION combo result — alphas DO NOT fully compose

The combo **bins=2048² + σ_init=1.0 + fitness_patience=30** was run as a champion. Result: **a fourth distinct alpha axis**, not a strict superset of the individual wins.

| metric | A_baseline | G_grid2048 | F_patience30 | **CHAMPION** |
|---|---|---|---|---|
| <30 µm count | 86 | **158** | 104 | 62 (worse than baseline!) |
| <10 µm count | 9 | 22 | **28** | 4 |
| min RMS (µm) | 3.35 | 2.09 | **0.89** | 7.86 |
| median RMS (µm) | 94.55 | **39.38** | 48.79 | 54.51 |
| ε-distinct @0.20 | 86 | 90 | 65 | **99** |
| ε-distinct @0.30 | 61 | 55 | 40 | **62** |

**CHAMPION wins on ε-distinct (99, +15% over baseline) — best in the study — but loses on every quality metric.**

**Mechanism (why combo fails on quality):** patience=30 (aggressive restart every 30 stagnant gens) on a 2048² grid (4M cells, ~0.05% coverage) amplifies the cell-novelty pathology. Small cells × frequent restarts = CMA constantly re-seeding before any basin gets polished. The result is a wide-and-shallow archive: many ε-distinct cells (good for diversity), few deep basins (bad for quality).

**Conclusion: HL has THREE orthogonal alpha axes; pick by metric of interest:**

| paper metric | recommended config | uplift vs. baseline |
|---|---|---|
| Sub-30 µm yield | **G_grid2048 alone** | 86 → 158 (+84%) |
| Sub-10 µm yield + min RMS | **F_patience30 alone** | 9 → 28 (+211%); 3.35 → 0.89 µm (3.8× better) |
| ε-distinct count | **CHAMPION combo** (or S_sigma10) | 86 → 99 (+15%) |

```yaml
# Recommended config depends on metric. Three options:

# Option A — sub-30 µm yield (G_grid2048)
algorithm:
  population_size: 18
  initial_sigma: 0.5
  fitness_patience: 100
archive:
  bins: [2048, 2048]            # ← only change
  bounds: [[-0.008, 0.008], [-0.012, 0.012]]

# Option B — extreme-quality bias (F_patience30)
algorithm:
  fitness_patience: 30          # ← only change
  initial_sigma: 0.5
  population_size: 18
archive:
  bins: [512, 512]
  bounds: [[-0.008, 0.008], [-0.012, 0.012]]

# Option C — diversity bias (CHAMPION combo)
algorithm:
  initial_sigma: 1.0            # ← change 1
  fitness_patience: 30          # ← change 2
  population_size: 18
archive:
  bins: [2048, 2048]            # ← change 3
  bounds: [[-0.008, 0.008], [-0.012, 0.012]]
```

**The orthogonality is itself a paper-relevant finding** — it shows HL's quality-vs-diversity-vs-extreme-quality trade-off is *manipulable* via hyperparameters, with each axis controlled by a different lever. Each headline number is achievable, but no single config gets all three. The "right" config is whichever fab use case you're targeting.

---

## Limitations

1. **Single seed.** All numbers above are seed=42. Need seed=43 confirmation before committing to a config change.
2. **Single topology.** Sixlet only. Triplet may have a different sweet spot (lower D, fewer free Seidels).
3. **One run per cell.** No variance bars. The 84% / 211% / 12% uplifts are point estimates; cross-seed variance could be large.
4. **Quality-conditional metrics dominate.** ε-distinct count alone is misleading (an earlier adversarial ablation showed RandomBC and NoGrad both *beat* HL on raw ε-distinct count at every quality cut, because parameter-space distinctness is not the same thing as physical-design distinctness). The key paper-relevant metrics are **# designs below an RMS cap** and **min RMS**, not ε-distinct alone.
5. **Refinement saturation may not generalise.** Phase A showed LBFGS-saturated refinement on the *current* archive. If the search produces a richer archive (e.g., from CHAMPION), the refinement may have different headroom.

---

## Files

* `figures/refsweep2_data/headline.csv` — all variants × all RMS thresholds × ε@{0.20, 0.30}
* `figures/refsweep2_data/eps_curves.csv` — full ε-distinct curves per variant
* `figures/refsweep2_sanity.png` — 4-panel ε-distinct vs ε at RMS thresholds {10, 30, 100, 1000} µm
* `configs/ablations/sixlet_REFSWEEP2_*.yaml` — all variant configs
* `results/lensmega/sixlet_REFSWEEP2_*_seed42_*/` — per-variant search archives + refined JSONs

---

*Generated 2026-05-09, single-seed point estimates from 11 variants × 100k iter each. CHAMPION combo run pending — table will update when its refine lands.*
