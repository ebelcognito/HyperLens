# LensMEGA — Session context export, 2026-04-22

A compact summary of everything we worked through today so a fresh conversation
can pick up cleanly.

---

## Project in one paragraph

LensMEGA = MAP-Elites with the **CMA-MEGA emitter** for lens design.  Behavioural
descriptors are **Seidel aberration coefficients** (S1 spherical, S2 coma,
S3 astigmatism, S4 Petzval, S5 distortion), computed from a paraxial ray trace
(differentiable via torch).  The archive is a 2-D grid keyed by two of the
five Seidel sums.  Offspring are produced by linearly combining the fitness
gradient with the descriptor gradients.  Thesis goal: show that Seidel-based
descriptors make exploration over lens designs dramatically more efficient
than either parameter-space-random BCs or non-MAP-Elites baselines.

The repo already has a working code path under `optical_rl/qd/lensmega/`.

---

## What we built today

New top-level scripts (all under `/home/hisham26/Desktop/RL/`):

| File | Purpose |
|:--|:--|
| `benchmark_metrics.py` | Unified metrics for comparing methods: best fitness, # elites above thresholds, # families by curvature-sign pattern and by θ-clustering, useful efficiency.  Accepts archive_cells.json, a run directory (reads refined elite_designs), or our method-output JSON. |
| `m3_nsga2.py` | NSGA-II baseline (OptiFusion-style EA with θ-space kNN-distance diversity as second objective).  Uses pymoo. |
| `m4_multistart_lbfgs.py` | Multi-start L-BFGS from uniform-random θ seeds.  Originally pitched as LensNet-style; we later admitted this is just "classical random-restart refinement", not LensNet. |
| `lineage_analysis.py` | Reconstruct ancestry chains from lineage.jsonl (parent → child hashes) and plot per-step Seidel + fitness + BC path. |
| `render_lineage.py` | Render lens layout + spot diagram + Seidel bar chart at each step of a lineage exemplar. |
| `seidel_coverage.py` | "Option A" — 2-D hull area, median \|S\|, entropy per RMS-tier on an archive. |
| `seidel_coverage_nd.py` | N-D extension (handles 5-D Seidel BC archives too). |
| `seidel_coverage_5d.py` | Alternate 5-D implementation (superseded by `_nd`). |
| `seidel_discrimination.py` | R² of `log RMS = β·log\|S\|` stratified by RMS tier. |
| `seidel_fingerprint.py` | Stratified Pearson r of (d_θ, d_Seidel) — tests "does Seidel fingerprint design identity". |
| `seidel_pair_scan.py` | Post-hoc virtual-archive scan over all 10 BC-pair choices on a lineage. |
| `augment_archive_seidel.py` | Recompute all 5 Seidel for cells from older runs that only stored (S2, S3). |

Infra changes inside the lensmega package:
- **`core/lineage.py`** (new) — `LineageLogger` class writing per-emission JSONL rows with parent_theta_hash, theta_hash, measures, all-5-Seidel, rms_spot, invalid_fraction, etc.
- **`core/seeder.py`** — wired the lineage logger into seed + calibration; **added strict filter** to calibration so pathological random seeds (BFL-negative, pupil-broken, invalid_fraction > 0.2) don't inflate archive bounds.
- **`core/runner.py`** — parent_theta_hash tracking per generation, lineage logging for every emission.
- **`descriptors/seidel.py`** — expose full 5-Seidel via `self._last_all_seidel` side-channel.
- **`problem/lens_problem.py`, `problem/full_dof.py`, `problem/power_budget.py`** — write `seidel_S1..S5` into metrics dict so they flow through the pool.
- **`io/layout.py`** — new `lineage_jsonl` path.

New config files in `optical_rl/qd/lensmega/config/examples/`:
- `cooke_triplet_seidel5d.yaml` — 5-D Seidel BC experiment
- `cooke_triplet_s3s4.yaml` — 2-D (S3, S4) BC on Cooke
- `sixlet_discovery_s3s4.yaml` — 2-D (S3, S4) BC on sixlet (no material variability)

---

## Key findings (with caveats)

### 1. The "Option A" coverage result is real, but effect size depends on architecture

Archive elites in the worst RMS tier occupy more of the BC grid than well-corrected elites:

| Architecture | BC | valid cells | Hull(t4)/Hull(t1) | Median \|S\| ratio |
|:--|:--|:-:|:-:|:-:|
| Cooke triplet + materials (10k-iter) | (S₂, S₃) | 2,906 | **1.78×** | 3.5× |
| Sixlet, fixed glass (36k-cell run) | (S₂, S₃) | 22,912 | **1.26×** | 1.8× |

Effect is monotonic with non-overlapping CIs between tiers 1 and 4 — but smaller
on sixlets (more DOF = more ways to achieve any given S₂/S₃, so well-corrected
tier *already* spreads across 78% of the BC grid).

### 2. The "fingerprint" thesis does NOT hold as originally framed

The user's original thesis was: "Seidel fingerprints designs when spots are
sub-optimal."  Operationalised as correlation of pairwise distances
`r(d_θ, d_{|S|})` per RMS tier.  What we found:

| Tier | r (Cooke S₂S₃ archive, clean) |
|:-:|:-:|
| 1 well-corrected | **0.12** (highest) |
| 2 | 0.07 |
| 3 | 0.06 |
| 4 worst | 0.07 |

The inversion persists on the 5-D full-Seidel test as well (tier 1 r = 0.53).
**Physical explanation**: well-corrected designs live on a smooth sub-manifold
where the θ → Seidel map is near-linear; bad lenses sit off that manifold in
nonlinear territory where the same θ step can cause chaotic Seidel jumps.
This is classical perturbation theory, not a failure of the paper.

**The paper should NOT claim "Seidel fingerprints designs better in sub-optimal
regime"**.  It should claim "Seidel coefficients spread elites across more of
the BC grid in sub-optimal tiers — which is what MAP-Elites consumes as
diversity signal."

### 3. The R² test is the better thesis support

Stratified R² of `log RMS = β · log|S|` on the Cooke archive:

| Tier | R² | 95% CI |
|:-:|:-:|:-:|
| 1 (well-corrected, 0.02–1.28 mm) | **0.030** | [0.014, 0.065] |
| 2 (1.28–3.26 mm) | 0.134 | [0.100, 0.188] |
| 3 (3.26–5.36 mm) | 0.119 | [0.081, 0.170] |
| 4 (5.36–22.5 mm) | 0.084 | [0.049, 0.138] |

Tier 1's CI doesn't overlap tier 2+ — this is a clean "Seidel goes dark in
the well-corrected regime" finding.  Classical third-order/higher-order
argument:

> Among well-corrected lenses, residual performance is dominated by 5th+
> order aberrations that Seidel doesn't see.  Among aberrated lenses, third-
> order dominates and Seidel captures the physics faithfully.

### 4. Per-coefficient predictive contribution

From the same archive:
- S4 (Petzval): r = +0.59 (dominant predictor of RMS)
- S5 (distortion): r = +0.36
- S3 (astigmatism): r = +0.34
- S2 (coma): r = +0.24
- S1 (spherical): r = +0.08 (nearly uncorrelated with RMS on Cooke)

### 5. Pair-scan ranking turned out to be a confound

I ran a virtual archive over every (Sᵢ, Sⱼ) pair using lineage data from the
5-D Cooke run.  That scan ranked (S₃, S₄) best with a 6× coverage ratio,
vs 1.8× for (S₂, S₃).  **This ranking is suspect** because the 5-D Cooke run
was a failed search dominated by broken designs — see finding 6.

### 6. **THE BIG BUG: auto-calibration + random broken seeds = inflated bounds**

The MAP-Elites archive's BC bounds were auto-calibrated from 1000–3000 random
θ samples.  Random θ includes many broken lenses (BFL-negative, pupil
anomalies) whose Seidel values are pathological (sixlet random seeds had
S₃ values as large as **−53**, a 30,000× overshoot vs realistic lenses).
Auto-calibration took the 2–98 percentile of this polluted distribution and
stretched archive bounds to swallow the outliers.  Consequence: well-behaved
lenses squashed into a sub-pixel region of the 256×256 grid, archive had no
resolution on the manifold of valid designs, CMA-MEGA couldn't steer.

| Run | Bounds | Calibration source | % valid | Outcome |
|:--|:--|:--|:-:|:--|
| Cooke triplet main 10k | hardcoded [-0.001, +0.001] | — | 94% | successful |
| Sixlet 36k cells (S₂, S₃) | hardcoded [-0.001, +0.001] | — | 62% | successful |
| Cooke 5-D Seidel | null (auto) | S₃ range 0.59, S₅ range 4.6 | **1.4%** | **failed** |
| Cooke (S₃, S₄) | null (auto) | S₃ range 0.90 | **13%** | **failed** |
| Sixlet (S₃, S₄) | null (auto) | S₃ range 58.04 | **0.1%** | **failed** |

**All runs with `bounds: null` failed catastrophically.  All with hardcoded
sensible bounds worked.**  I've fixed calibration in `seeder.py` with a
stricter filter (rejected=0 AND rejected_bfl=0 AND rejected_pupil=0 AND
invalid_fraction < 0.2 AND finite measures).  Verified: on the failed sixlet
(S₃, S₄) seeds, new filter yields S₃ range 0.012 vs old 45.84 — 3900× more
sensible.  **All the "failed" runs need re-running with the fix.**

### 7. LensMEGA refines its elites

I initially read raw archive best fitness (−0.04, ~22 µm RMS) and claimed
that as LensMEGA's output.  Wrong — LensMEGA runs an L-BFGS refinement pass
on archive cells above a tracker threshold, stored in
`<run_dir>/elite_designs/*.json` under a nested `refined` key.  Actual best
refined Cooke lens: **2.8 µm** (from `cooke_triplet_materials_20260422_163857
/elite_designs/elite_0004_5400k.json`).  `benchmark_metrics.py` has been
updated to auto-load refined elites when given a run directory.

### 8. M4 (random-restart L-BFGS) is not LensNet

I pitched M4 as a LensNet-equivalent baseline.  That's wrong — LensNet uses
a *learned* prior (NN trained on existing lens designs), not uniform random.
M4 is more honestly "classical random-restart L-BFGS", a generic baseline.
It has a valid place in the benchmark but doesn't stand in for LensNet.
A real LensNet analog would need to train an NN on a corpus first.

---

## Paper-narrative advice given today

User pushed for a first-principles pitch that would make reviewers think
"damn, this is obvious why haven't people done this before".  The argument:

> *"Optical design has been organised around the five Seidel coefficients for
> 130 years.  They aren't parameters — they are the discipline's native
> coordinate system for optical behaviour.  Every diverse-search method over
> lens designs to date operates in parameter space (NSGA-II, OptiFusion) or
> learned-prior space (LensNet) — not in behaviour space with a physics-given
> coordinate system.  This paper is the first to navigate the aberration
> manifold directly."*

The four properties of Seidel no competing method exploits:
1. **Physics-meaningful** — each axis is a distinct optical phenomenon.
2. **Differentiable** — closed-form paraxial, gradients via autograd.
3. **Architecture-agnostic** — S₂ axis of triplet = S₂ axis of sixlet.
4. **Interpretable to designers** — an expert can read the archive by eye.

Reviewer-objection rebuttals drafted:
- "oh but deep learning" → NN bounded by training data; Seidel captures physics of *new* regimes.
- "oh but curriculum learning" → curricula shape time; Seidel BCs shape space.
- "oh but NSGA-II / population search" → those preserve param- or objective-
  space diversity, not behaviour-space.  Two lenses with identical spot RMS
  can have radically different aberration profiles and futures.
- "oh but Novelty Search" → NS needs a behaviour characterisation; we supply
  the discipline's canonical one.
- "oh but random MAP-Elites" → random BCs partition θ-space arbitrarily,
  Seidel partitions by aberration regime = optimization-basin boundaries.

**Demonstration recommendations (not classical ablations)**:
1. *"The archive is physically readable"* — render 4 corner cells with labelled aberration regimes.
2. *"Cross-architecture transfer"* — show that the same (S₂, S₃) cell across triplet/sixlet/quadruplet contains lenses with similar optical behaviour despite different numbers of elements.  **This is the killer demo and doesn't exist yet.**
3. *"Seidel gradients are physically meaningful steering directions"* — perturb a parent along ∇S₂ vs ∇S₃; show the spot diagrams respond exactly as classical optics predicts.
4. *BC-vs-BC ablation* (not method-vs-method): Seidel vs random linear projection vs (c₀, c₁) vs spot-RMS-at-two-fields.  Only Seidel should give archive-readable, interpretable results.

---

## What was running when the session ended, and what to restart

**All processes were killed at user's request.**  Nothing is running now.

Baselines in progress (never completed, outputs lost):
- M3 NSGA-II (~30 min into 42-min run)
- M4 Multi-start L-BFGS (~255/800 starts, best −0.0033 = 3.3 µm, would have taken ~80 min total)

Sixlet (S₃, S₄) run completed but failed due to calibration bug.

---

## Recommended next steps (ordered)

1. **Re-run the Cooke (S₃, S₄) experiment with fixed calibration**  
   The pair-scan's 6× coverage claim needs clean data.  Use the new
   `seeder.calibrate` filter (already committed).
2. **Build the cross-architecture transfer demo** (the paper's visual
   centerpiece).  Use the hardcoded-bounds (S₂, S₃) runs on triplet +
   sixlet that already succeeded.  Pick matching BC cells, render the
   lenses, compare spot diagrams.
3. **Rework the benchmark around BC-vs-BC**, not method-vs-method.
   Random linear projection + (c₀, c₁) parameter-space + spot-RMS
   behaviour-but-not-Seidel are the key ablations.
4. **Defer NSGA-II / L-BFGS comparisons** to a separate "diverse-search
   methods at equal budget" table — don't mix them with the BC-choice
   argument.
5. **Decide whether to re-run the 5-D Seidel experiment** with the
   calibration fix.  If 5-D MAP-Elites works with clean bounds, that's a
   very different story (and a nicer paper).

---

## Data artefacts worth keeping

- `cooke_triplet_materials_20260422_163857/` — successful 10k-iter (S₂, S₃)
  Cooke run with material variability.  Best refined lens: 2.8 µm.
- `cooke_triplet_materials_20260414_173935/` — earlier (S₂, S₃) Cooke run,
  n=38,504 cells, used for the larger-archive version of Option A.
- `sixlet_discovery_20260421_154452/` — successful (S₂, S₃) sixlet with
  36k cells, 62% valid, 69 elites above −0.2 fitness.
- All failed runs (cooke_triplet_seidel5d_20260422_180914, 
  cooke_triplet_s3s4_20260422_183508, sixlet_discovery_s3s4_20260422_210826)
  — kept for lineage logs; archives themselves are mostly broken designs.

---

## The calibration fix, in one sentence

`seeder.calibrate` now excludes BFL-failed, pupil-broken, and
high-invalid-fraction seeds from the percentile calculation, preventing
auto-calibration from stretching archive bounds to swallow pathological
random seeds.  Committed in `optical_rl/qd/lensmega/core/seeder.py`.
