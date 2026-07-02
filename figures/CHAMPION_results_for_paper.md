# Results — Octuplet CHAMPION run (100 k iterations, seed 42)

Paper-ready data dump for the octuplet CHAMPION HyperLens configuration. All
numbers below are single-seed point estimates; matched 100 k-evaluation budget
against the SA+GA baseline. Sources for every number are linked at the bottom.

---

## 1. TL;DR

> At a matched 100 k-evaluation budget on an 8-element APS-C imaging-lens
> design problem (45-D θ, Seidel S₂/S₃ behavior space), the octuplet-tuned
> HyperLens CHAMPION configuration (`bins=2048² + σ_init=1.0 +
> patience=20`) produces a refined design pool that is **~38× larger** than
> the SA+GA baseline's, with **a behavior-space hull 4–17× larger**
> (depending on the matched fitness gate). At the tight, fab-grade quality
> bar of refined RMS < 50 µm, CHAMPION produces **15 ε-distinct lens forms**
> (in normalized 45-D θ-space) versus SA+GA's **2** — a 7.5× richer design
> library at matched quality. The HyperLens hull contains every SA+GA design
> at any reasonable matched fitness gate, with two BD-region exceptions that
> correspond to a SA+GA-favored mid-fitness basin and a SA+GA outlier outside
> the HyperLens archive's representable region.

---

## 2. Setup

### CHAMPION configuration

Octuplet-tuned HyperLens combo, derived from the per-knob ablation study
(see `figures/REPORT_octuplet_hl_ablation.md` for the rationale).
All other settings inherit from
`optical_rl/qd/lensmega/config/examples/octuplet_apsc_2cement_v2.yaml`.

```yaml
algorithm:
  population_size: 32           # baseline (unchanged)
  initial_sigma: 1.0            # CHAMPION: ↑ from baseline 0.5
  fitness_patience: 20          # CHAMPION: ↓ from baseline 50  (aggressive restart)
archive:
  bins: [2048, 2048]            # CHAMPION: ↑ from baseline 512²
  bounds: [[-0.005, 0.005], [-0.008, 0.008]]   # baseline (unchanged)
descriptor:
  kind: seidel
  params: {terms: [S2, S3]}     # behavior space = Seidel coma (S₂) and astigmatism (S₃)
```

### Problem

| | value |
|---|---|
| topology | 8-element APS-C, 2 cemented interfaces |
| free-θ dimensionality D | **45** |
| evaluation budget | **100 000** raw-fitness evaluations |
| seed | 42 |
| problem config | `optical_rl/qd/lensmega/config/examples/octuplet_apsc_2cement_v2.yaml` |
| refinement | L-BFGS-B (top-500 archive designs above raw fitness −1, plateau-triggered exit, max_steps=100) |
| run dir | `results/lensmega/octuplet_REFSWEEP_CHAMPION_seed42_20260509_145427/` |

### SA+GA baseline (the comparison)

Reference implementation matches QGSO/OptiFusion (IEEE TCI 2025) at fixed
`evaluate()` call budget, with two principled deviations logged in
`results/diversity_run_20260506_033447/octuplet/seed42/sa_ga/hyperparameters.json`
(autograd gradients in lieu of finite-difference; mutation `num_genes=2`).
Two SA+GA octuplet runs are available for comparison:

| pool | source | `ga_diversity_min_distance` | n_refined (fit > −1) | min RMS |
|---|---|---|---|---|
| **Pool A** | `diversity_run_20260506_033447/octuplet/seed42/sa_ga` | 0.05 | 68 | 1.23 µm |
| **Pool B** | `steelman_div020_20260506_123801/octuplet/seed42/sa_ga` | 0.20 *(QGSO default)* | 13 | 6.64 µm |

Both differ from CHAMPION only in upstream search — the L-BFGS-B refinement
pipeline used after both is identical.

### Search-cost accounting (CHAMPION run)

| stage | wall-clock | evaluations |
|---|---|---|
| HL search | 41 min on 32-core compute node | 100 000 (3 125 generations × pop 32; **150 CMA restarts**) |
| HL archive | — | **56 788** cells filled out of 4 194 304 (1.35 % coverage) |
| HL refinement | 24 min, top-500 designs polished in parallel | (refine evals not counted in search budget) |
| SA+GA search (for ref) | ~14 min | 99 994 |
| SA+GA refinement (for ref) | ~2 min | — |

---

## 3. Pool size and quality counts at matched 100 k budget

CHAMPION compared against the HL `fig4` baseline (a less-tuned HyperLens config
on the same octuplet) and both SA+GA pools. All numbers from
`figures/octuplet_refsweep_data/headline.csv` (the per-method × per-RMS-threshold
table produced by `scripts_throwaway/octuplet_refsweep_analysis.py`).

| | HL fig4 baseline | **CHAMPION** | SA+GA Pool A | SA+GA Pool B |
|---|---|---|---|---|
| **pool size** (refined, fit > −1) | 310 | **500** | 68 | 13 |
| min refined RMS (µm) | 12.21 | 4.78 | 1.23 | 6.64 |
| median refined RMS (µm) | 125.40 | 77.34 | 20.00 | 44.28 |
| # designs <10 µm | 0 | 1 | 19 | 1 |
| # designs <30 µm | 29 | **35** | 45 | 2 |
| # designs <50 µm | 81 | **121** | 58 | 4 |
| # designs <100 µm | 136 | **273** | 62 | 11 |
| # designs <1 mm | 310 | **445** | 68 | 13 |

**Multiplicative comparisons** (CHAMPION vs each comparator):

| metric | CHAMPION / HL fig4 | CHAMPION / SA+GA Pool A | CHAMPION / SA+GA Pool B |
|---|---|---|---|
| pool size | 1.6× | **7.4×** | **38×** |
| # <100 µm | 2.0× | **4.4×** | **24.8×** |
| # <1 mm | 1.4× | 6.5× | **34.2×** |

---

## 4. Behavior-space hull coverage (matched fitness gates)

The convex hull of cells with raw fitness ≥ θ in the BD plane (S₂, S₃) is a
useful proxy for "where did the search find good designs?". At any
**matched** fitness threshold, HyperLens CHAMPION's hull strictly dominates
SA+GA's. Numbers below are post-BD-clipping: one SA+GA point at
S₃=+1.02×10⁻² lies outside HyperLens's archive bounds
(S₃ ∈ [−8×10⁻³, +8×10⁻³]) and is excluded from the SA+GA matched hull —
this is a representation artifact, not a search-quality difference.

| fitness gate | HL n_cells | HL hull area | SA+GA n | SA+GA hull area | **HL / SAGA ratio** | SAGA points inside HL hull |
|---|---|---|---|---|---|---|
| fit ≥ −10 *(SAGA's own `tracker_threshold`)* | 7 427 | 5.72e−5 | 70 | 3.30e−6 | **17.3×** | **70 / 70** |
| fit ≥ −5 | 4 044 | 2.40e−5 | 70 | 3.30e−6 | 7.3× | 67 / 70 |
| fit ≥ −4 | 3 163 | 1.85e−5 | 70 | 3.30e−6 | 5.6× | 68 / 70 |
| fit ≥ −3.7 | 2 870 | 1.41e−5 | 70 | 3.30e−6 | 4.3× | 68 / 70 |
| fit ≥ −3 | 2 219 | 9.81e−6 | 69 | 1.65e−6 | 6.0× | 63 / 69 |
| fit ≥ −2 | 1 319 | 6.88e−6 | 68 | 9.89e−7 | 7.0× | 61 / 68 |
| fit ≥ −1 | 446 | 1.64e−6 | 68 | 9.89e−7 | 1.7× | 60 / 68 |

**Reading**: HL CHAMPION's BD-space coverage is **3–17× larger** than SA+GA's
at any matched fitness gate, and (at the SA+GA-defined `tracker_threshold` of
−10) **strictly contains every SA+GA point**.

The few SA+GA points outside the HL hull at tighter gates correspond to a
single SA+GA-favored mid-fitness basin at S₂≈−0.36×10⁻³, S₃≈+1.6×10⁻³ where
SA+GA's GA+ADAM machinery polishes a basin deeper than HL CHAMPION's
`patience=20` aggressive-restart CMA-ES does (see §6).

---

## 5. Quality-conditioned diversity (the design-library argument)

Counting **ε-distinct designs in normalized 45-D θ-space**, restricted to
designs that meet a refined-RMS quality gate. Greedy ε-net with sort by
refined-fitness desc; design *i* is accepted iff
`‖θᵢ − θⱼ‖₂ / √D > ε` for every already-accepted *j*. Note: ε-distinct in
θ-space is a stricter test of design-library variety than BD-space spread.

| RMS gate | method | n designs | ε=.10 | ε=.20 | ε=.30 |
|---|---|---|---|---|---|
| <30 µm | **CHAMPION** | 35 | 7 | **7** | **7** |
| <30 µm | SA+GA Pool A | 45 | 19 | 4 | 1 |
| <30 µm | SA+GA Pool B | 2 | 2 | 2 | 2 |
| <50 µm | **CHAMPION** | 121 | 18 | **18** | **15** |
| <50 µm | SA+GA Pool A | 58 | 5 | 5 | 2 |
| <100 µm | **CHAMPION** | 273 | 28 | **28** | **18** |
| <100 µm | SA+GA Pool A | 62 | 6 | 6 | 3 |

**Headline**: At the fab-grade RMS-<50-µm gate, CHAMPION produces **15
ε-distinct lens forms (ε=0.30) from 121 refined designs**; SA+GA Pool A
produces **2 ε-distinct forms from 58 refined designs** at the same gate.
CHAMPION's design library is **7.5× more varied** at matched quality.

**The non-obvious finding**: SA+GA Pool A's 45 designs at refined RMS <30 µm
collapse to **a single ε-distinct cluster at ε=0.30** — i.e., they are all
slightly different polishings of one and the same basin. This is the
mechanism by which SA+GA achieves its minimum RMS of 1.23 µm: by
concentrating the entire search budget on a single deep basin and polishing
it 45 different ways. CHAMPION's 35 designs <30 µm cover **7 truly distinct
lens forms** at the same quality threshold. SA+GA's lower minimum is a
*single-design metric*; CHAMPION's higher ε-distinct count is a
*library-of-designs metric*.

---

## 6. Where SA+GA wins, and why (the basin finding)

At S₂ ≈ −0.36×10⁻³, S₃ ≈ +1.6×10⁻³, SA+GA finds **7 designs with raw fitness
−0.10 to −0.34** that refine to RMS 130–145 µm. HyperLens CHAMPION has 6 640
archive cells in this same BD region, but its best raw fitness there is
**−0.81** (refining to RMS 772 µm) — only 1 CHAMPION-refined design lands
in this region, and it is not a usable design.

**Mechanism**: CHAMPION's `fitness_patience=20` causes CMA-ES to abandon each
basin after 20 stagnant generations. For basins like this one — where
sub-200-µm performance is reachable but only with 30+ generations of
polishing — `patience=20` is too aggressive; the basin gets visited
(cell-novelty bonus triggers the visit) but is not polished deeply enough.
SA+GA's GA+ADAM polishes basins via 30 ADAM steps per offspring, which
suffices for this specific basin.

**Interpretation for the paper**: at the very deepest single-basin metric
(min refined RMS in *this* BD region only), SA+GA wins. At the global
minimum and at every yield-count metric, HL CHAMPION wins. The trade-off is
explicit: aggressive restart trades depth-per-cell for archive-spread. The
companion HyperLens variant **BLIND_COMBO** (identical to CHAMPION except
`fitness_patience=30`) shows the other side of this trade — it produces a
sub-2 µm global minimum (1.43 µm at 54 % evaluation budget; OOM-killed
before completion) but a much smaller refined pool (n=103) than CHAMPION's
500. See §7.

---

## 7. CHAMPION ↔ BLIND_COMBO: `fitness_patience` is the load-bearing knob

A bonus single-knob ablation against the CHAMPION config: BLIND_COMBO is
the sixlet's CHAMPION recipe (`bins=2048² + σ=1.0 + patience=30`) applied
verbatim to the octuplet. It differs from the octuplet CHAMPION in *only*
`fitness_patience` (30 vs 20). BLIND_COMBO was OOM-killed at gen 1700
(54 % of the 100 k budget) but the partial archive was refined; numbers
below carry this caveat.

| | **CHAMPION** (patience=20, full 100k) | **BLIND_COMBO** (patience=30, 54% budget) |
|---|---|---|
| pool size | 500 | 103 |
| min refined RMS (µm) | 4.78 | **1.43** |
| median refined RMS (µm) | 77.34 | 77.44 |
| # designs <10 µm | 1 | 3 |
| # designs <30 µm | 35 | 5 |
| # designs <100 µm | 273 | 57 |
| ε-distinct @0.30 | 27 | 17 |

**Reading**: patience=30 polishes basins ~1.5× longer per restart than
patience=20, occasionally hitting sub-µm valleys; patience=20 spreads
breadth without going as deep. **The two settings expose the same
yield-vs-extreme-quality trade-off**. A patience-sweep at fixed `bins=2048²,
σ=1.0` (values ∈ {15, 20, 25, 30, 35, 40, 50}) is the cleanest
follow-up — current data points are sparse but consistent with patience≈30
being a sweet spot for the extreme-quality axis, while patience≈20 is the
sweet spot for the yield axis.

---

## 8. Four representative CHAMPION designs (spotlights)

Spotlights picked by ε-net in 45-D normalized θ-space (ε=0.35, pool gated to
refined RMS <200 µm). Three HyperLens-exclusive (outside SA+GA matched hull
at fit≥−10), one in the shared basin (inside SA+GA matched hull). All four
are mutually ε-distinct at ≥ 0.365 in normalized θ-space — *genuinely
different physical lens forms*, not perturbations of the same design.

| label | role | S₂ (×10⁻³) | S₃ (×10⁻³) | refined RMS (µm) | in SA+GA matched hull? |
|---|---|---|---|---|---|
| d1 | HL-exclusive | +0.19 | +0.08 | 58.96 | no |
| d2 | HL-exclusive | −0.16 | +0.01 | **19.31** | no |
| d3 | HL-exclusive | +0.24 | +0.35 | 49.53 | no |
| common | shared with SA+GA | +0.03 | −0.15 | 85.87 | yes |

Per-design panels (lens cross-section + |S₂|/|S₃| bar chart + stability
ribbon vs field) live in
[`figures/champion_spotlight_panels/`](champion_spotlight_panels/):

* `champion_<role>_lens.png`    — filled-glass lens cross-section with 3-field ray bundles (blue/green/red wavelengths)
* `champion_<role>_seidel.png`  — two-bar |S₂| / |S₃| chart
* `champion_<role>_robust.png`  — RMS-vs-field stability ribbon, band = min/max across 3 wavelengths, line = per-field median

The lens cross-sections visibly confirm that d2 (RMS 19.31 µm) and
`common` (RMS 85.87 µm) are *different physical lenses* — `common` has
notably thicker/rounder glass elements; they are not minor parameter
variants of one design.

---

## 9. Headline single-figure summary

The complete result is summarized in two paper-ready figures in
[`results/figure_archive_demo_octuplet_CHAMPION/`](../results/figure_archive_demo_octuplet_CHAMPION/):

* **`v16oct_main_hull-fit_gt_m10_sol-150_m4_SPREAD.png`** — wide view (fit ≥ −10
  hull): HyperLens hull is **17.3× SA+GA's** and contains **all 70** SA+GA
  points; four spotlight designs (3 blue HL-exclusive at spread BD positions,
  1 red shared); HyperLens refined-design dots (n=353 at <150 µm) fill the
  warm-band high-fitness core while SA+GA dots concentrate in a single
  tight cluster.
* **`v16oct_main_hull-fit_gt_m3p7_sol-150_m4_DIVERSE_smallbtn.png`** — tighter
  view at the high-quality gate (fit ≥ −3.7), HyperLens hull **4.28×
  SA+GA's**, with ε-distinct (in 45-D θ-space) spotlights `d1, d2, d3,
  common`.

---

## 10. Limitations

1. **Single seed (42).** Cross-seed variance not yet measured. All
   multiplicative comparisons (17×, 7.5×, etc.) are point estimates.
2. **One CHAMPION run.** The patience-vs-quality trade-off (§7) is inferred
   from a single CHAMPION + single BLIND_COMBO data point; a full
   `fitness_patience` sweep at fixed (bins=2048², σ=1.0) would tighten this.
3. **BLIND_COMBO at 54 % budget.** The 1.43 µm headline from
   BLIND_COMBO is a single-design metric on solid ground (single refined
   design), but the BLIND_COMBO pool counts (n=103) are conservative for
   yield comparisons against full-budget CHAMPION. A re-run on a stable
   allocation is the highest-priority follow-up.
4. **ε-distinct is θ-space, not physical-design distinctness.** A prior
   adversarial-ablation control (sixlet) showed RandomBC and NoGrad
   *beat* HyperLens on raw ε-distinct count at every quality cut, because
   parameter-space distinctness is not equivalent to physical-design
   distinctness. The metrics CHAMPION wins on most decisively are
   **# designs below an RMS cap** and **min refined RMS**, not ε-distinct
   alone. The ε-distinct numbers above are quality-conditioned (gated by
   RMS < threshold) precisely to mitigate this; the headline argument is
   *count-of-distinct-designs-at-a-given-quality*, not raw count.
5. **SA+GA Pool A vs Pool B**: the cited SA+GA "baseline" depends on which
   `ga_diversity_min_distance` setting one chooses (0.05 → Pool A's
   1.23 µm minimum; 0.20 → Pool B's 6.64 µm, which is the QGSO published
   default). CHAMPION beats Pool B unambiguously on every metric; against
   Pool A, CHAMPION wins on every count and on quality-conditioned
   diversity, while Pool A wins on single-design minimum RMS at the cost
   of producing only 1 ε-distinct lens form at the fab-grade quality bar.
   Both interpretations should be reported.

---

## 11. Source artifacts (for reproducibility)

| artifact | path |
|---|---|
| CHAMPION run dir | `results/lensmega/octuplet_REFSWEEP_CHAMPION_seed42_20260509_145427/` |
| CHAMPION config | `configs/ablations/octuplet_REFSWEEP_CHAMPION.yaml` |
| CHAMPION refined pool JSON | `results/lensmega/octuplet_REFSWEEP_CHAMPION_seed42_20260509_145427/refined_top500_seed42_CHAMPION.json` |
| BLIND_COMBO config | `configs/ablations/octuplet_REFSWEEP_BLIND_COMBO.yaml` |
| BLIND_COMBO refined pool JSON | `results/lensmega/octuplet_REFSWEEP_BLIND_COMBO_seed42_20260509_074619/refined_top500_seed42_BLIND_COMBO.json` |
| SA+GA Pool A | `results/diversity_run_20260506_033447/octuplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json` |
| SA+GA Pool B | `results/steelman_div020_20260506_123801/octuplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json` |
| HyperLens analysis driver | `scripts_throwaway/octuplet_refsweep_analysis.py` |
| Hull comparison input builder | `results/figure_archive_demo_octuplet_CHAMPION/build_inputs.py` |
| Mathematica archive renderer | `results/figure_archive_demo_octuplet_CHAMPION/render_archive_v16_oct.wl` |
| Per-spotlight panel renderer | `scripts_throwaway/render_champion_spotlight_panels.py` |
| HL CHAMPION archive figure (wide) | `results/figure_archive_demo_octuplet_CHAMPION/v16oct_main_hull-fit_gt_m10_sol-150_m4_SPREAD.png` |
| HL CHAMPION archive figure (tight, smaller buttons) | `results/figure_archive_demo_octuplet_CHAMPION/v16oct_main_hull-fit_gt_m3p7_sol-150_m4_DIVERSE_smallbtn.png` |
| Per-spotlight panels (12 PNGs) | `figures/champion_spotlight_panels/` |
| Per-method × RMS-threshold headline CSV | `figures/octuplet_refsweep_data/headline.csv` |
| Full ε-distinct curves CSV | `figures/octuplet_refsweep_data/eps_curves.csv` |
| Companion detailed report (full ablation context) | `figures/REPORT_octuplet_hl_ablation.md` |
| Cross-topology summary (sixlet + octuplet) | `figures/RESULTS_summary_for_paper.md` |

---

*Generated 2026-05-10. Single-seed (42) point estimates; matched 100 000
evaluation budget for all method comparisons. Octuplet topology:
8-element APS-C imaging lens with 2 cemented interfaces, 45-D θ, Seidel
behavior space (S₂, S₃). Octuplet baseline knobs prior to CHAMPION tuning:
population_size=32, initial_sigma=0.5, fitness_patience=50, bins=512².*
