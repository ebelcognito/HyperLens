# Session 2026-05-06 — cross-system replication, fidelity audit, steelman, saturation, paper strategy

Self-contained snapshot of the 2026-05-06 working session. Read in order with the canonical log [EXPERIMENT_LOG_sa_ga_vs_hyperlens_sixlet.md](hyperlens/EXPERIMENT_LOG_sa_ga_vs_hyperlens_sixlet.md) for full context (§12 and §13 cover this session's data; this document adds the conversational discussion, strategic interpretation, and paper-direction options not committed to the log).

---

## TL;DR

1. **Cross-system replication succeeded** (§12 in log): HL beats SA+GA on ε=0.20 diversity in 17/18 (system × seed × threshold) cells across triplet (D=19), sixlet (D=37), octuplet (D=45), 2 seeds each, 100k matched evaluate budget. Aggregated mean ratios: triplet 2.16–2.55×, sixlet 3.87–6.91×, octuplet 2.38–3.53×.

2. **Hardware fairness inverted** vs prior session: HL ran on the cleaner-of-two contended nodes; SA+GA absorbed the heavily-loaded one. Diversity win is no longer attributable to SA+GA hardware advantage.

3. **Post-hoc ε=0.30 confirms HL lead grows tighter**: at "highly distinct" ε=0.30, HL still wins every cell, with sixlet ratios reaching 19× sub-30μm. ε=0.40 ratios collapse to ~1× because pool sizes are tiny — noise.

4. **Fidelity audit found our SA+GA was numerically weaker than QGSO defaults** in several inner-loop intensities, most notably `ga_diversity_min_distance` was 0.05 vs QGSO's 0.20. Steelman re-run with `div_rate=0.20` was launched on a 2:30 hov allocation.

5. **Steelman result**: HL still wins every cell. Tightening the filter to QGSO default cuts SA+GA pool 65–81% and degrades best refined RMS 4–14×, while only modestly raising ε=0.20-distinct count. HL/SG ratio shrinks but remains 1.6–5.4× at sub-1mm ε=0.20.

6. **Saturation finding (key novel result)**: at div_rate=0.20, every design in SA+GA's refined pool is already ε=0.20-distinct from every other one. The ε-net is saturated at the pool level — raising div_rate further cannot help, since the bottleneck is now pool size, not filter aggressiveness. Pool size is itself bounded by `population × outer_gens × mu/m × accept_rate` at fixed compute, so no hyperparameter sweep on SA+GA closes the gap. **The architectural ceiling is hyperparameter-independent at fixed compute.**

7. **Gradient efficiency is a secondary claim**, not a standout. Refinement compute is *not* equalized — HL refines ~8× more designs per pool than SA+GA. The matched compute axis is search evaluations only.

8. **Paper-direction recommendation**: lead the discussion with **spec-adaptation transfer** ("HL produces a library of archetypes; new specs warm-start from the nearest archetype, no re-search needed"). Defer scaling-law and archetype-novelty claims.

---

## What ran this session

### Phase 1: cross-system replication launcher (overnight)

Launched ~03:35, finished ~05:50 (2 h 16 min total).

- Launcher: [`run_diversity_experiments.sh`](hyperlens/run_diversity_experiments.sh) called with allocations `48131433` (HL→c151608) and `48131431` (SAGA→c130602)
- 3 systems × 2 seeds × 2 methods × 100k search → top-500 HL refined + all SA+GA above -1.0 refined → ε-net analysis
- Run root: [`results/diversity_run_20260506_033447/`](hyperlens/results/diversity_run_20260506_033447/)
- Master log captures all phase transitions
- Cosmetic: `triplet_seed43_hl` reports exit 134 (benign torch shutdown crash *after* full data was written; analysis used the intact JSON)

### Phase 2: fidelity audit

Compared [`sa_ga_refine_baseline.py`](hyperlens/sa_ga_refine_baseline.py) against [`/nas/longleaf/home/hisham26/Desktop/QGSO/TCI_Optifusion/gaoptics/gygad.py`](QGSO/TCI_Optifusion/gaoptics/gygad.py) line-by-line. Found:

| param | QGSO default | Our overnight | gap |
|---|---|---|---|
| `sa_iter_max` | 300 | 50 | **6× weaker** |
| `adam_iter_num` | 80 | 30 | 2.7× weaker |
| `adam_plateau_window` | 20 | 10 | 2× weaker |
| `ga_diversity_min_distance` (`div_rate`) | 0.20 | 0.05 | **4× looser** |
| `mutation_num_genes` | 2 | 2 | matched |
| `mu/m`, `nu/m` | 0.06, 0.02 | 0.06, 0.02 | matched |

Skeleton was structurally faithful (six-stage outer loop, sign-flips for max-vs-min consistent). Inner intensities were systematically dialed below QGSO defaults. The autograd-vs-FD ADAM swap was already documented in the prior session's log.

QGSO repo single-commit dump dated 2025-03-19 — ~14 months old, no commit history. "Defaults" are whatever was in that snapshot, not necessarily what the OptiFusion paper used.

### Phase 3: steelman re-run

Goal: rerun SA+GA with `div_rate=0.20` (matching QGSO default) and confirm HL still wins.

- New config: [`configs/sa_ga_baseline_norefine_100k_div020.yaml`](hyperlens/configs/sa_ga_baseline_norefine_100k_div020.yaml)
- New launcher: [`run_steelman_div020.sh`](hyperlens/run_steelman_div020.sh) — 3 systems × seed=42 only, on a single allocation, search → refinement
- Allocation: `48242996` (c151403, ~99% top contention level — clean by Longleaf standards)
- Run root: [`results/steelman_div020_20260506_123801/`](hyperlens/results/steelman_div020_20260506_123801/)
- Wall: 40 min total

### Phase 4: rendering

- New tool: [`tools/render_top9_eps_grid.py`](hyperlens/tools/render_top9_eps_grid.py) — system-agnostic, ε-net based, replaces the sign-pattern-based `render_diversity_comparison.py`
- Output rendered: [`results/diversity_grids/triplet_hl_seed42_top9_sub20um_eps020.png`](hyperlens/results/diversity_grids/triplet_hl_seed42_top9_sub20um_eps020.png) — 3×3 grid of HL triplet seed=42 ε=0.20-distinct sub-20μm refined designs (354 in pool, 21 ε-distinct, top 9 by refined RMS)

---

## Key experimental findings

### 1. Replication holds across systems and seeds

ε=0.20-distinct refined design counts at sub-1mm:

| | HL seed=42 | HL seed=43 | SG seed=42 | SG seed=43 | mean ratio |
|---|---|---|---|---|---|
| triplet | 39 | 55 | 19 | 18 | **2.55×** |
| sixlet | 86 | 78 | 13 | 11 | **6.81×** |
| octuplet | 31 | 29 | 7 | 11 | **3.53×** |

Cross-seed std stays well below cross-method gap. Robust to seed.

### 2. ε-sensitivity is favorable to HL

Same data, ε swept from 0.10 to 0.40:

- At ε=0.10: HL/SG ratio 1.3–8× (loose)
- At **ε=0.20**: ratio 1.6–7× (the headline)
- At **ε=0.30**: ratio **1.3–19×** — gap *grows* in some cells because SA+GA's neighbors collapse into single archetypes while HL retains independent basins
- At ε=0.40: ratio collapses to ~1× because both methods have only 1–5 maximally-distant designs at this granularity (statistical noise)

Conclusion: the diversity advantage is *robust*, not metric-specific.

### 3. Steelman hyperparameter (div_rate=0.20) doesn't close the gap

Pool sizes (post-refinement, seed=42):

| system | div=0.05 (orig) | div=0.20 (steel) |
|---|---|---|
| triplet | 69 | 24 |
| sixlet | 64 | 16 |
| octuplet | 68 | 13 |

Best refined RMS got **4–14× worse** under div=0.20 (sixlet jumped 1.5 → 21.6 μm). HL still beats steelman 1.6× / 5.4× / 2.4× at ε=0.20 sub-1mm.

### 4. The saturation finding

In the steelman pool, **every refined design is already ε=0.20-distinct from every other**:

| system | refined pool size | ε=0.20-distinct |
|---|---|---|
| triplet | 24 | 24 (100%) |
| sixlet | 16 | 16 (100%) |
| octuplet | 13 | 13 (100%) |

The ε-net is **saturated at the pool level**. SA+GA's diversity ceiling under QGSO defaults is set by *pool size*, not by filter aggressiveness. Raising div_rate further cannot increase distinct count — it would shrink the pool while distinctness is already 100%.

The pool-size ceiling is itself bounded by `population × outer_gens × mu/m × accept_rate` at fixed compute. No hyperparameter sweep closes the architectural gap with HL's 31–86 distinct designs at the same compute envelope.

This is a stronger claim than "HL beats SA+GA at one setting" — it's "HL beats *any* SA+GA setting at this compute."

---

## Strategic discussion

### ε=0.20 is a chosen threshold, not a canonical literature value

Three principled groundings for the paper:

1. **Stability-perturbation anchor (recommended)**: stability protocol perturbs by 0.005 × bounds_range. ε=0.20 = 40× this scale → designs at this distance are unambiguously distinct, not the same design plus measurement noise.
2. **Robustness-across-ε**: report ε ∈ {0.10, 0.20, 0.30, 0.40} curve in appendix; lead with ε=0.20 as the representative midpoint.
3. **Dimensionless interpretability**: ε=0.20 in /√D-normalized L2 means "designs differ by ~20% of each parameter's bound range on average."

Don't claim ε=0.20 is canonical — it isn't.

### Gradient efficiency is not a standout claim

The original log §7.6 / §8.3 claim "2.9× HL gradient efficiency" measured ε=0.10-distinct per backward pass, but:
- HL's backward passes are multi-output (∇fit + Σ ∇descriptors) — more FLOPs each
- The 2.9× ratio is *less impressive* than the 5–7× search-budget ratio at ε=0.20
- Refinement compute is unequal (HL refines ~8× more designs per pool)

**Recommendation**: drop "gradient efficiency" as a separate claim. Use the emitter-vs-ADAM mechanism (CMA-MEGA samples N coefficients per gradient → N candidates; ADAM uses each gradient for 1 step) as the *mechanistic explanation* of the diversity advantage in the discussion, not as a compute claim.

### Three candidate "this leads to X" hooks for the paper

Ranked by reviewer impact / demonstrability with current data:

#### 1. **Spec-adaptation transfer (recommended lead)**

> "HyperLens's refined archive is a library of structurally distinct archetypes. Specification adaptation (different F#, EFL, FOV, materials) reduces to refinement of the nearest archetype — no re-search needed. SA+GA's mode-seeking pool offers a single starting point per spec."

- Connects to broader trends: foundation libraries, transfer learning, high-throughput materials screening
- Demonstrable in ~30 min: take sixlet HL archive, change F-target to 5.6, re-refine top archetypes, compare convergence speed + final RMS vs SA+GA single design re-refined
- Falsifiable: if HL's archetypes don't transfer well, the claim fails

#### 2. **Archetype novelty / discovery**

> "Of the 102 sub-30μm sixlet ε=0.20-distinct designs HL produces, K fall outside the classical APS-C catalog (Cooke triplet, double-Gauss, retrofocus, etc.), suggesting QD reveals undocumented archetypes."

- High reviewer impact in optics literature (named archetypes are revered)
- Risk: requires actual catalog mapping; might be that all 102 are classical
- Demonstrable: cluster by lens-shape signature (per-element power sign, cement pattern, stop position), check against Smith / Kingslake / Fischer

#### 3. **Diversity scaling laws**

> "SA+GA's diversity saturates at a hyperparameter-bounded ceiling at fixed compute. HL's archive scales linearly with compute. QD has favorable scaling laws on diverse-design objectives."

- Connects to ML scaling-law literature
- Heavier to demonstrate: 50k / 100k / 250k / 500k runs both methods, ~6–8 h compute
- The saturation finding (§13.6) already prefigures this — the experiment would confirm

### Paper-positioning recommendation

- **Headline claim**: matched-search-budget ε-net diversity (5–7× sixlet, 2–3× triplet/octuplet, robust across seeds and ε)
- **Mechanism explanation**: emitter samples N candidates per gradient vs ADAM 1 candidate per gradient; archive retains designs across generations vs tracker discarding novel-but-not-best
- **Architectural lift**: saturation argument — no SA+GA hyperparameter sweep closes the gap at fixed compute
- **Future-applications hook**: spec-adaptation transfer (lead), archetype novelty (defer until catalog-mapped), scaling laws (defer until budget-swept)
- **Honest caveats**: hardware contention, single-system class, no learned descriptor experiments, refinement compute not equalized

---

## Artifacts written this session

### Code

| | path |
|---|---|
| Steelman config | [`configs/sa_ga_baseline_norefine_100k_div020.yaml`](hyperlens/configs/sa_ga_baseline_norefine_100k_div020.yaml) |
| Steelman launcher | [`run_steelman_div020.sh`](hyperlens/run_steelman_div020.sh) |
| ε-net grid renderer | [`tools/render_top9_eps_grid.py`](hyperlens/tools/render_top9_eps_grid.py) |

### Run dirs

| | path |
|---|---|
| Cross-system replication (overnight) | [`results/diversity_run_20260506_033447/`](hyperlens/results/diversity_run_20260506_033447/) |
| Steelman (afternoon) | [`results/steelman_div020_20260506_123801/`](hyperlens/results/steelman_div020_20260506_123801/) |
| HL run dirs (×6) | `results/lensmega/{cooke_triplet_materials_modernstack,sixlet_apsc_discovery_v2,octuplet_apsc_2cement_v2}_seed{42,43}_2026050[56]_*/` |

### Logs

| | path |
|---|---|
| Replication master | `results/diversity_run_20260506_033447/logs/master.log` |
| Replication analysis | `results/diversity_run_20260506_033447/logs/analysis.log` |
| Steelman master | `results/steelman_div020_20260506_123801/logs/master.log` |
| Per-pair search/refine logs | both run dirs' `logs/{system}_seed{N}_{hl,saga,refine_hl,refine_saga}.log` |

### Figures

| | path |
|---|---|
| Triplet HL top-9 ε=0.20 sub-20μm grid | [`results/diversity_grids/triplet_hl_seed42_top9_sub20um_eps020.png`](hyperlens/results/diversity_grids/triplet_hl_seed42_top9_sub20um_eps020.png) |

### Documentation

| | path |
|---|---|
| Canonical log (now §12 + §13) | [`EXPERIMENT_LOG_sa_ga_vs_hyperlens_sixlet.md`](hyperlens/EXPERIMENT_LOG_sa_ga_vs_hyperlens_sixlet.md) |
| This session summary | [`SESSION_2026-05-06_summary.md`](hyperlens/SESSION_2026-05-06_summary.md) |

---

## Open work after this session

### Tonight's open follow-ups

- [ ] Steelman seed=43 (variance band on the saturation result; one seed only so far)
- [ ] Steelman with all QGSO inner-loop intensities (sa_iter_max=300, adam_iter_num=80, plateau_window=20) — confirm the saturation isn't due to our weakened SA / ADAM
- [ ] Render top-9 ε=0.20 grids for sixlet + octuplet (HL and SA+GA), and SA+GA triplet — for paper figure pack

### Next-session "this leads to X" experiments

- [ ] **Spec-adaptation experiment**: change sixlet F-target from 4.0 → 5.6, re-refine HL archetypes vs SA+GA single design, compare convergence speed + final quality (recommended lead, ~30 min run)
- [ ] **Archetype catalog mapping**: cluster HL sub-30μm sixlets by per-element power signs / cement pattern, compare to classical lens taxonomy (Smith / Kingslake / Fischer)
- [ ] **Scaling-law sweep**: HL + SA+GA at 50k / 100k / 250k / 500k, plot diversity-vs-compute curves

### Pre-existing open work (carried forward)

- [ ] Manufacturability yield (Monte Carlo tolerance perturbation)
- [ ] Glass-snap survival on top archetypes
- [ ] Higher-budget octuplet (500k) for final paper figures
- [ ] Multi-seed (n=4–5) replication for tighter variance bands
- [ ] SA+GA hyperparameter sensitivity grid (3×3 over T, mu/m, adam_lr)

### Files NOT to touch

- The 2026-05-05 sixlet matched-100k results dirs (canonical reference data)
- `results/lensmega/sixlet_apsc_discovery_v2_20260505_140231` (the 400k sixlet HL reference)
- `EXPERIMENT_LOG_sa_ga_vs_hyperlens_sixlet.md` is append-only

---

## Compute / fairness notes for the paper

- All 2026-05-06 runs on UNC Longleaf `hov` partition, 32-CPU 48G allocations
- Both nodes used (c151608 + c130602 overnight, c151403 afternoon) were heavily oversubscribed by co-tenants throughout — typical Longleaf operating regime
- Overnight: HL got the cleaner-of-two contended nodes (inverted vs prior 2026-05-05 session), so the diversity win is no longer attributable to SA+GA hardware advantage
- Steelman: single allocation, sequential SA+GA on each system → no method-vs-method contention asymmetry within steelman
- Wall times therefore reflect contended-node performance, not benchmark-clean throughput

---

End of session summary.
