# Discussion / Limitations / Future Work — rewrite plan + 3-day experiment digest

**Date:** 2026-05-10
**Purpose:** consolidate everything we've actually run and measured over the last three days (May 8–10) so the §5 Discussion, §5.4 Limitations, and §6 Future Work in the HyperLens manuscript can be rewritten from scratch against what currently exists, rather than against the older Family-A six-element narrative the current §5.1 was drafted around.

This is a planning document, not paper prose. Section 1 catalogues what we ran. Section 2 names what's in the current draft and judges keep/cut/edit. Section 3 lists what's missing. Section 4 sketches a new section structure.

---

## 1. What we actually have on disk as of 2026-05-10

All numbers single-seed (42), matched 100 000-evaluation budget against SA+GA, refinement = L-BFGS-B top-500 with the same plateau-triggered exit.

### 1.1 Octuplet CHAMPION (the new headline)

* **Run:** [`results/lensmega/octuplet_REFSWEEP_CHAMPION_seed42_20260509_145427/`](results/lensmega/octuplet_REFSWEEP_CHAMPION_seed42_20260509_145427/)
* **Config:** [`configs/ablations/octuplet_REFSWEEP_CHAMPION.yaml`](configs/ablations/octuplet_REFSWEEP_CHAMPION.yaml) — `bins=2048² + σ_init=1.0 + fitness_patience=20` (other knobs inherit from `octuplet_apsc_2cement_v2.yaml`)
* **Problem:** 8-element APS-C, 2 cemented interfaces, 45-D θ, Seidel (S₂, S₃) BC
* **Wall-clock:** 41 min search (32-core node) + 24 min top-500 refinement
* **Archive coverage:** 56 788 cells / 4 194 304 = 1.35 %, 150 CMA restarts
* **Headline counts (CHAMPION vs SA+GA-steelman Pool B):**
  | metric | CHAMPION | SA+GA Pool B | ratio |
  |---|---|---|---|
  | refined pool size (fit > −1) | 500 | 13 | 38× |
  | # designs < 100 µm | 273 | 11 | 24.8× |
  | # designs < 30 µm | 35 | 2 | 17.5× |
  | min refined RMS (µm) | 4.78 | 6.64 | 1.39× tighter |
  | ε-distinct @0.30 (full pool) | 27 | 4 | 6.75× |
  | ε-distinct @0.30 (<50 µm) | 15 | 2 | 7.5× |
* **Behaviour-space hull (matched fitness gates):** 3–17× larger than SA+GA's, and at the SA+GA `tracker_threshold=−10` gate **contains all 70** SA+GA points. Two SA+GA outliers fall outside HL hull at tighter gates: one represents an SA+GA-favoured mid-fitness basin at (S₂≈−0.36e-3, S₃≈+1.6e-3) where their GA+ADAM polishes deeper than CHAMPION's `patience=20` re-seeds away from; the other is at S₃=+1.02e-2, outside HL archive bounds (representation artifact).
* **Spotlight designs:** four ε-distinct in 45-D θ-space at ε ≥ 0.365 — three HL-exclusive (`d1, d2, d3` outside SA+GA matched hull), one shared (`common`). Per-design panels: `figures/champion_spotlight_panels/champion_{d1,d2,d3,common}_{lens,seidel,robust}.png`
* **Source doc:** [`figures/CHAMPION_results_for_paper.md`](figures/CHAMPION_results_for_paper.md) (this is the canonical paper-ready data dump)

### 1.2 Octuplet BLIND_COMBO (sixlet recipe on octuplet, 54 % budget)

* **Run:** `results/lensmega/octuplet_REFSWEEP_BLIND_COMBO_seed42_20260509_074619/` (OOM-killed at gen 1700)
* **Config:** identical to CHAMPION except `fitness_patience: 30 vs 20`
* **Result:** min refined RMS = **1.43 µm** (single-design), 103-design refined pool
* **Reading:** `fitness_patience` is the load-bearing knob for *extreme quality* on octuplet; CHAMPION's `patience=20` trades min-RMS for archive spread.
* **Caveat:** 54 % budget — full-budget BLIND_COMBO has not run. Listed as highest-priority follow-up.

### 1.3 Sixlet CHAMPION + per-knob study

* **Run:** [`results/lensmega/sixlet_REFSWEEP2_CHAMPION_seed42_20260509_080537/`](results/lensmega/sixlet_REFSWEEP2_CHAMPION_seed42_20260509_080537/)
* **Config:** `bins=2048² + σ_init=1.0 + fitness_patience=30` (note: `patience=30`, **not 20** — sixlet's optimum differs from octuplet's)
* **Problem:** 6-element APS-C, 33-D θ
* **Pool:** 500 refined, min RMS 7.86 µm, 62 < 30 µm, 344 < 100 µm, ε@.30=62 (best in study)
* **Three orthogonal alphas from the 11-variant ablation** ([`figures/REPORT_hl_hyperparam_ablation.md`](figures/REPORT_hl_hyperparam_ablation.md)):
  | alpha axis | best variant | single knob | uplift |
  |---|---|---|---|
  | Yield (# <30 µm) | `G_grid2048` | `bins: 512² → 2048²` | 86 → 158 (+84 %) |
  | Extreme quality (min RMS) | `F_patience30` | `fitness_patience: 100 → 30` | 3.35 → **0.89 µm** (near F/4 diffraction limit) |
  | Diversity (ε@.20) | CHAMPION combo | three knobs layered | 86 → 99 (+15 %) |
* **The non-obvious finding:** the three alphas **do not compose additively on the sixlet** — layering them produces a *diversity* champion that *loses* on yield (62 sub-30 µm vs G_grid2048's 158). On the octuplet they *do* compose (CHAMPION wins yield + diversity over fig4 baseline).

### 1.4 SA+GA baseline details

Two SA+GA octuplet pools exist; choice affects which numbers are quoted:
* **Pool A** (`results/diversity_run_20260506_033447/octuplet/seed42/sa_ga/`, `ga_diversity_min_distance=0.05`): 68 refined, min 1.23 µm — but at refined RMS <30 µm, all 45 designs collapse to **a single ε@0.30 cluster** (i.e. polish a single basin 45 ways).
* **Pool B** (`results/steelman_div020_20260506_123801/octuplet/seed42/sa_ga/`, `ga_diversity_min_distance=0.20` — QGSO default): 13 refined, min 6.64 µm. CHAMPION beats this unambiguously on every metric.

We should report Pool B as the primary baseline (it uses the QGSO published default) and acknowledge Pool A in a footnote / limitations bullet — it's the only configuration on which SA+GA beats HL on *one* metric (min refined RMS on the octuplet), and the cost is producing 1 distinct lens form vs HL's 15 at the fab-grade quality bar.

### 1.5 Gradient-alignment runs on the APS-C champion configs

New as of 2026-05-10. Source: [`experiments/gradient_alignment/RESULTS_2026-05-10_apsc_champions.md`](experiments/gradient_alignment/RESULTS_2026-05-10_apsc_champions.md).

Two 50 k-evaluation logging runs (CHAMPION configs but with iteration cap and `log_gradients=true`):
* `results/lensmega/gradient_alignment_sixlet_apsc_champion_20260510_034307/`
* `results/lensmega/gradient_alignment_octuplet_apsc_champion_20260510_034318/`

**Mean |cos(−∇L, ∇M_j)| over all CMA-MEGA generations:**
| topology | θ-dim | n gens | S2 | S3 | random null mean | random max(200) | frac random ≥ Seidel |
|---|---|---|---|---|---|---|---|
| triplet (reproduced) | 19 | 1389 | 0.616 | 0.701 | 0.183 | 0.488 | 0/200 |
| sixlet (old, small sensor, reproduced) | 37 | 2778 | 0.859 | 0.697 | 0.137 | 0.401 | 0/200 |
| septuplet APS-C (reproduced) | 41 | 2084 | 0.45 | 0.47 | 0.125 | 0.245 | 0/200 |
| **sixlet APS-C champion (new)** | 37 | 2084 | **0.501** | **0.478** | 0.136 | 0.271 | 0/200 |
| **octuplet APS-C champion (new)** | 45 | 1563 | **0.414** | **0.423** | 0.121 | 0.232 | 0/200 |

**5-Seidel post-hoc replay** (same gradient log, recompute ∇S_k for k=1..5):
* Sixlet APS-C: range 0.42–0.53, S4 (Petzval) top, S5 (distortion) bottom
* Octuplet APS-C: range 0.35–0.47, same shape

**Reading:** the partial-alignment claim holds across the new APS-C champion topologies. S₂/S₃ sit at mean |cos| ≈ 0.41–0.50, ≈3–4× the random null, and 0/200 random projections reach Seidel values. No single Seidel coefficient dominates — the choice of (S₂, S₃) is justified physically (off-axis aberrations, separable field dependence), not by an alignment-magnitude argument.

This data is what Table 2 in the current §4.2 is derived from — those numbers are current.

### 1.6 Persistence demo (seventuplet `-+--+-` chain)

Already integrated into §4.3 of the current draft. Source: [`persistence_argument_export/`](persistence_argument_export/) (`DISCUSSION.md`, `PERSISTENCE_DEMO.md`, `PAPER_DEMO_TEXT.md`).

* **Run:** `results/lensmega/septlet_F3_20260425_224405/` (this is an older April run, but its `lineage.jsonl` is the only one with full parent-chain pointers; no re-run needed)
* **The chain:** seed (gen 0, RMS 13 937 µm) → gen 31 (RMS 13 128 µm, NEW cell) → gen 271 (RMS 13 035 µm, REPLACE) → gen 317 (RMS 12 576 µm, NEW) → gen 531 (RMS 13 381 µm, NEW) → gen 824 (RMS 925 µm, REPLACE — branch point) → gen 1454 elite_0020 (RMS 443 µm raw) → L-BFGS polish (RMS 2.7 µm)
* **Three siblings of elite_0020 from the same branch:** elite_0006 → 7.5 µm, elite_0019 → 2.8 µm, elite_0020 → 2.7 µm
* **Gradient alignment at gen 824:** |cos(−∇L, ∇M_S₂)| = 0.38, |cos(−∇L, ∇M_S₃)| = 0.43 (43-D null = 0.13). Subspace projection of Δθ from gen 824 → elite_0020 onto `span{∇L, ∇M_S₂, ∇M_S₃}` = 1.000 (random 3-D null mean = 0.249). Empirical mean of the three descent-aligned descriptor gradients has cos = **0.99** with −∇L.
* **Caveat:** autograd vs FD sign mismatch on ∇S₃ at this θ (magnitudes agree, cos(autograd, FD) = 0.57). All |cos| numbers in the paper are correct; signed walks were re-checked via FD.

### 1.7 Other recent runs (negative / robustness results)

From the sixlet ablation: G_grid128, T_tighter_bd (16× narrower), P_pop36, W_wider_bd, F_patience300, B_efl_tight, D_long_lbfgs. From the octuplet ablation: G_grid3000 (`bins=3000²`).

Three are paper-relevant as honest negative controls:
* **D_long_lbfgs (sixlet) — byte-identical to baseline.** Tripling `refine_lbfgs_max_steps` (100 → 300) and tightening `converge_tol` changes nothing because the plateau detector exits before `max_steps` ever bites. **All extractable HyperLens alpha is search-side, not refinement-side.** Load-bearing for the methodology claim.
* **G_grid3000 (octuplet) — hurts every metric.** At 9 M cells (~99.94 % empty after 100 k evals), cell-novelty bonus drives σ unstably. There is a ceiling on archive resolution.
* **T_tighter_bd / G_grid128 — both hurt.** Squeezing BD bounds or coarsening the grid both lose useful aberration regimes. The current `bins=2048² + standard BD bounds` is a robust sweet spot.

---

## 2. Audit of the current §5 / §5.4 / §6 — keep / cut / edit

Section numbers reference the manuscript as quoted in the PDF.

### §5.1 "Why the descriptor partitioning works"

**Status:** mostly fine in spirit, factually misaligned with §4.3.

* **Keep:** the central insight that "spot RMS is a scalar projection" and the Seidel descriptor "lifts this scalar back into the structured space it came from". This is the right mechanistic framing. The "different decompositions of the same scalar" → "different cells with different fates" line is good.
* **Cut:** the Family-A six-element paragraph ("Family A in the six-element configuration was discovered three times across the run, at separate emitter lives. The raw spot radii at all three discoveries hovered in the 250 to 300 µm range… 28 steps to 12.7 µm, against 8 steps to 179.6 µm on the first life"). This narrative is from an earlier sixlet run that we no longer reference anywhere in §4. §4.3 uses the seventuplet `-+--+-` chain as the canonical example, and the discussion should reference *that* example, not re-import a different one.
* **Cut:** the "CMA step size shrank from roughly 6 down to roughly 1" data — this is from the Family-A run and we don't currently report it as evidence.
* **Edit/replace:** if we want a concrete instantiation, point back to §4.3's seventuplet chain. The three sibling refinements (7.5 / 2.8 / 2.7 µm) all descending from the same gen-824 ancestor is the same "same archive region revisited under progressively tighter sampling" mechanism, and we already have the data and figures for it.
* **Cut:** the last paragraph ("The mechanism is therefore not what it might first look like… each cell of the archive is a parallel hypothesis…"). It restates §5.1's opening twice with slightly different metaphors. Shrink to a single sentence.

### §5.2 "The descriptor determines whether the archive can refine" (descriptor-choice ablation)

**Status:** keep, mostly intact. Honest, clean experiment with a sharp finding.

* **Keep:** Table 4 (Seidel vs Random projection vs Single-Obj on Cooke triplet at 50 k evals) — this is the clean ablation that justifies the descriptor choice as physical, not arbitrary.
* **Keep:** "coverage is not the issue, depth is" framing (Random fills 40.7 % vs Seidel 24.1 %, but every percentile of refined-RMS is worse).
* **Keep:** "The basins are deeper, not narrower" finding (improvement ratios 7.1× vs 3.9×; perturbation-stability 24.3 % vs 29.0 % shows neither is fragile vs sturdy).
* **Cut / shorten:** "The CMA dynamics confirm the descriptor is doing work" paragraph (σ blowing up to 10⁸–10⁹ in Random). Useful but secondary to the headline; keep at most one sentence.
* **Edit:** Single-Obj column belongs in this table — currently §5.3 walks through it separately. Consolidating to one ablation table reads cleaner.

### §5.3 "The archive enables stepping-stone refinement"

**Status:** mostly redundant with §5.1 + §5.2.

* **Keep:** the single sharp finding — Single-Obj reaches comparable best raw fitness (−0.0413 vs Seidel's −0.0453) but preserves only one design and that design does not refine to any usable quality. This is the *cleanest* answer to the "why not just use CMA-ES?" reviewer question.
* **Keep:** the median-improvement-ratio bullet (Seidel 7.1×, Random 3.9×, Single-Obj near 1.0× because each preserved point is already a local current-best at every restart).
* **Cut:** the long paragraph re-arguing that "the archive is not a post-hoc diversity filter on a single-trajectory optimizer" — we've made this point twice already in §5.1 and §5.2.
* **Edit:** can fold this into §5.2 as the third column. Single section about ablations, not two.

### §5.4 Limitations

**Status:** partially current, several entries are out of date or describe paper-internal placeholders.

* **Keep:** descriptor space is 2-D (we use S₂, S₃), uniform grid scales poorly past 4 D, CVT is the natural extension.
* **Keep:** Seidel formulation assumes spherical surfaces; aspherics/freeforms/diffractives need higher-order theory.
* **Keep (rephrase):** fixed-element-count topology — this naturally hands off to §6.
* **Cut:** "The decuplet run in §?? was checkpointed at 46 % of its iteration budget…". We have no decuplet headline in this draft; this is a leftover.
* **Cut:** "The seven-element sign-pattern logging used a six-character format inherited from earlier runs, so the 'more than eight families' count in §?? is a lower bound." Same — there is no eight-families claim in the current §4.
* **Cut:** "All experiments use a fixed sensor radius of 6.5 mm" — defensible but mid-paper detail, not a limitation.
* **Edit/replace:** the random-projection-single-seed bullet — should be expanded into a general single-seed limitation that covers *all* our headline numbers (the 17×, 7.5×, 24×, 38× multiplicative claims).
* **Add (load-bearing, missing):**
  * **Single-seed (42) throughout.** Every multiplicative comparison is a point estimate. Cross-seed variance not yet measured. This needs to be the *first* limitation bullet — it's the most likely reviewer objection.
  * **ε-distinct is a θ-space metric, not a physical-design distinctness metric.** A prior adversarial-ablation control (RandomBC, NoGrad) beats HyperLens on raw ε-distinct count at every quality cut, because parameter-space distinctness is not the same as physical-design distinctness. The metrics we win on most decisively are **# designs below an RMS cap** and **min refined RMS**; the ε-distinct numbers are quality-conditioned (gated by RMS < threshold) precisely to mitigate this.
  * **SA+GA Pool A vs Pool B caveat.** The SA+GA baseline depends on the `ga_diversity_min_distance` setting. At Pool A's setting (0.05), SA+GA reaches 1.23 µm on the octuplet by polishing a single basin 45 ways (1 ε-distinct lens form at the fab-grade quality bar). At Pool B's setting (0.20, QGSO default), SA+GA reaches 6.64 µm and CHAMPION beats it unambiguously on every metric. We should report both interpretations honestly.
  * **BLIND_COMBO ran at 54 % budget.** The 1.43 µm headline is a single-design metric on solid ground, but the BLIND_COMBO pool counts (n=103) are conservative for yield comparisons against full-budget CHAMPION. Re-running on a stable allocation is the highest-priority follow-up.
  * **Refinement saturation may not generalise.** D_long_lbfgs being byte-identical to baseline establishes saturation on the *baseline* archive. A richer archive (e.g., a longer search run) may have different L-BFGS headroom.
  * **The patience-vs-quality trade-off is single-knob.** We have one CHAMPION + one BLIND_COMBO data point on the octuplet. A full `fitness_patience` sweep at fixed (bins=2048², σ=1.0, values ∈ {15, 20, 25, 30, 35, 40, 50}) would tighten the claim; current data points are sparse but consistent with patience ≈ 30 being the sweet spot for extreme quality, patience ≈ 20 for yield.

### §6 Future work

**Status:** the two existing bullets (variable element count, higher-D descriptor space) are still right. Two new ones to add:

* **Keep:** fixed-topology K → integration with transdimensional methods (Teh et al. 2025).
* **Keep:** descriptor-dimensionality extension (CVT past 4-D, chromatic / distortion descriptors).
* **Add:** **CMA-MEGA emitter is the current wall-clock bottleneck.** The archive populates well before the coefficient-space CMA adaptation exhausts its useful gradient signal. A faster emitter would tighten the runtime without changing what the archive recovers.
* **Add:** **patience sweep + cross-seed validation.** The cleanest immediate experimental follow-up is a fixed-config patience sweep on both sixlet and octuplet with 3–5 seeds, to pin down the patience-vs-quality trade-off and bound the variance on every multiplicative uplift.

---

## 3. What is missing from §5 that recent work supports adding

If we want the discussion to reflect what the experiments now actually say:

1. **The patience-vs-yield-vs-min-RMS trade-off is mechanistically clean.** The CHAMPION/BLIND_COMBO octuplet pair (identical except `fitness_patience: 20 vs 30`) gives us one of the cleanest knob-ablations in the paper. We should either work it into §5 or move it from §4 if it's not there yet. It's currently in [`figures/CHAMPION_results_for_paper.md`](figures/CHAMPION_results_for_paper.md) §7 but not in the manuscript.

2. **Three orthogonal hyperparameter alphas.** The sixlet ablation revealed that bins / σ_init / patience each control a different axis (yield / diversity / extreme quality), and the combo composes differently on the two topologies. This is a *paper-relevant finding*: HyperLens is not "one config, one result", it exposes a *configurable* quality-vs-diversity-vs-extreme-quality trade-off. Whether to put this in Results or Discussion is a structural decision.

3. **Refinement saturation as a load-bearing methodology claim.** D_long_lbfgs byte-identical to baseline means "all our wins are search-side, not L-BFGS-tuning artefacts". This is the cleanest answer to a likely reviewer objection ("you just tuned L-BFGS"). Worth one line in §5 or as a footnote.

4. **Cross-topology generalisation of alphas.** The same three knobs (bins, σ, patience) move the headline metrics in the same direction on both 33-D sixlet and 45-D octuplet. The non-obvious part is that the *value* `patience ≈ 30` transfers across topologies, but the *ratio to baseline* does not — sixlet baseline is 100, octuplet baseline is 50. This is a generalisability point.

5. **The "SA+GA wins min-RMS on one variant of one topology by giving up diversity" reading.** This belongs in either §5 or §5.4. It's the honest reading of the Pool A comparison.

---

## 4. Proposed new structure for §5–§6

A leaner Discussion + Limitations + Future Work, in the order they should appear:

### §5 Discussion

**§5.1 — Mechanism: optical-behaviour partitioning of a scalar objective.**
One paragraph: spot RMS is a scalar projection of a structured aberration field; the Seidel descriptor restores that structure and partitions the archive into cells with separate fates. Point to §4.3 (seventuplet chain) as the empirical instantiation. **Length: ~150 words.**

**§5.2 — Ablation: descriptor choice and archive structure.**
Merge current §5.2 + §5.3 into one section. Single Cooke-triplet ablation table (Seidel / Random / Single-Obj). Headline: descriptor structure determines basin depth, not coverage; the archive enables stepping-stone refinement. **Length: ~400 words + Table 4 as-is.**

**§5.3 — A configurable trade-off, exposed by three orthogonal hyperparameters.**
*New section.* Reports the bins / σ / patience axes and the CHAMPION-vs-BLIND_COMBO octuplet finding. Headline: HyperLens does not produce a single answer — it produces a configurable trade-off between yield, diversity, and extreme quality, each controlled by a different knob. Patience ≈ 30 (absolute, not ratio) is the load-bearing knob for sub-µm RMS; bins = 2048² is the universal sweet spot; σ = 1.0 is an unconditional diversity gain. **Length: ~300 words + one small table.**

**§5.4 — Refinement is saturated; the wins are upstream.**
*New short subsection or a paragraph in §5.3.* D_long_lbfgs byte-identical to baseline. **Length: ~80 words.**

### §5.5 Limitations

Bulleted, in priority order:

1. Single seed (42) throughout — every multiplicative number is a point estimate.
2. ε-distinct is a θ-space metric, not physical-design distinctness; we report quality-conditioned ε-distinct (gated by RMS) for this reason; the load-bearing metrics are # designs below an RMS cap and min RMS.
3. The SA+GA baseline depends on `ga_diversity_min_distance`; we report both Pool A (0.05) and Pool B (0.20 QGSO default) interpretations.
4. BLIND_COMBO ran at 54 % budget; the 1.43 µm headline is a single-design metric on solid ground but pool counts are conservative.
5. Descriptor space is 2-D; uniform-grid archives don't scale cleanly past 4-D (CVT is the natural extension).
6. Seidel formulation assumes spherical surfaces; aspherics / freeforms / diffractives need higher-order aberration theory or a different descriptor.
7. Fixed-element-count topology (handed off to §6).
8. CMA-MEGA emitter is the current wall-clock bottleneck (handed off to §6).
9. Patience-vs-quality trade-off is currently a 2-point characterisation; a full sweep with multi-seed validation is the highest-priority follow-up.
10. Refinement saturation established on the baseline archive only; richer archives may have headroom.

### §6 Future work

1. Variable element count K via transdimensional QD outer loop (Teh et al. 2025 integration).
2. Higher-D descriptor space (chromatic, distortion, field-dependent), CVT past 4-D (Vassiliades et al. 2018).
3. Faster CMA-MEGA emitter — current wall-clock bottleneck.
4. Patience sweep + multi-seed cross-validation for the §5.3 trade-off claims.
5. Aspheric / freeform / diffractive descriptors via higher-order Welford or Zernike-fitted residuals.

---

## 5. Things to delete from the current draft outright

Concrete strings of text in the manuscript PDF that should not survive into the next revision:

* §3.1: `"we need a clear method overview fig. the algo can also be pushed near 3.1"` — author note, delete.
* §3.2: `"mentioning a section raises expectations that you will show some results. if you have results maybe put in suppl."` — author note, delete.
* §3.2: `"Correct this figure to use 'M' instead of 'S'"` — author note, delete; fix Fig. 3 to use M_j labels matching §3.1's behaviour-map notation.
* §4.4: `"[inline]Octuplet archive-hull figure placeholder."` and `"Octuplet archive-hull figure placeholder."` — replace with the actual octuplet hull figure (`figures/octuplet_hull_clean_HL_vs_SAGA.png` exists as of 2026-05-10) or its CHAMPION counterpart (`results/figure_archive_demo_octuplet_CHAMPION/v16oct_main_hull-fit_gt_m10_sol-150_m4_SPREAD.png`).
* §5.4: the decuplet-46% and "more than eight families" bullets — neither claim appears anywhere in §4 of the current draft; delete or restore the missing experiments.
* §5.4: "Section ?? shows raw fitness is a poor predictor of refined depth" — broken cross-reference, either resolve or delete.

---

## 6. Reading order for the inheriting agent / writing pass

If you're picking this up cold:

1. [`figures/CHAMPION_results_for_paper.md`](figures/CHAMPION_results_for_paper.md) — octuplet CHAMPION data dump, the single most paper-relevant document.
2. [`figures/CHAMPION_sixlet_results_for_paper.md`](figures/CHAMPION_sixlet_results_for_paper.md) — sixlet CHAMPION + three-axis finding.
3. [`figures/RESULTS_summary_for_paper.md`](figures/RESULTS_summary_for_paper.md) — cross-topology summary; same numbers as the two above, indexed differently.
4. [`figures/REPORT_hl_hyperparam_ablation.md`](figures/REPORT_hl_hyperparam_ablation.md) — sixlet per-knob ablation in full.
5. [`figures/REPORT_octuplet_hl_ablation.md`](figures/REPORT_octuplet_hl_ablation.md) — octuplet per-knob ablation in full.
6. [`experiments/gradient_alignment/RESULTS_2026-05-10_apsc_champions.md`](experiments/gradient_alignment/RESULTS_2026-05-10_apsc_champions.md) — APS-C-champion gradient-alignment numbers used in §4.2.
7. [`persistence_argument_export/DISCUSSION.md`](persistence_argument_export/DISCUSSION.md) + [`persistence_argument_export/PAPER_DEMO_TEXT.md`](persistence_argument_export/PAPER_DEMO_TEXT.md) — seventuplet `-+--+-` persistence demo used in §4.3.

---

*Generated 2026-05-10 from the experiments executed 2026-05-08 through 2026-05-10 on the longleaf cluster. All numbers are single-seed (42), matched 100 k-evaluation budget, refined with the L-BFGS-B top-500 plateau-triggered exit pipeline unless explicitly noted otherwise.*
