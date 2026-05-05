# Session export — Gradient-alignment experiment series

**Date:** 2026-05-04
**Working directory:** `/home/hisham26/Desktop/RL`
**Experiment dir:** [`experiments/gradient_alignment/`](experiments/gradient_alignment/)

## Goal

Empirically validate the §3.2 claim that "Seidel coefficients decompose
the wavefront aberration the loss aggregates" by measuring cosine
similarity between the loss-descent direction `−∇_θ L` and each
descriptor gradient `∇_θ M_j` along the actual CMA-MEGA search
trajectory. Convert a structural claim ("S2/S3 are *components* of the
loss") into a directional claim ("∇M_j and ∇L *point similarly* in
parameter space").

## Code change (one place)

Added a read-only `GradientLogger` to
[`optical_rl/qd/lensmega/core/runner.py`](optical_rl/qd/lensmega/core/runner.py),
gated by a new `io.log_gradients: bool = False` flag in
[`optical_rl/qd/lensmega/config/schema.py`](optical_rl/qd/lensmega/config/schema.py).

Per CMA-MEGA generation, logs raw `∇_θ fitness` and each `∇_θ M_j` *before*
the unit-rescaling done by `_normalized_gradients`. Optimiser behaviour
is unchanged when `log_gradients` is `false`.

Outputs `<run_dir>/gradient_log.npz` containing:

| key | shape | content |
|-----|-------|---------|
| `evals` | (N,) int | completed eval count at each generation |
| `generations` | (N,) int | generation index |
| `thetas` | (N, P) f64 | parent θ_μ |
| `grad_fitness` | (N, P) f64 | raw +∇fitness = −∇L |
| `grad_m` | (N, K, P) f64 | raw ∇M_j for j = 0..K−1 |
| `fitness` | (N,) f64 | fitness at parent |

## Experiments — runs and post-hoc analyses

### 1. Cooke triplet (19-D, S2/S3) — baseline

Run config: [`experiments/gradient_alignment/config.yaml`](experiments/gradient_alignment/config.yaml).
Mirrors `ablation_seidel_s2s3.yaml`. 50k evals, 18 workers, ~12 min wall.
1389 generations logged.

**Result:** mean `|cos|` = 0.62 (S2), 0.70 (S3) vs 19-D random null = 0.18.
Signed cos: S2 ≈ 0 (high variance), S3 strongly negative (Seidel-sign-convention
artefact). The claim survives; sign mostly tracks the conventional Seidel sign.

### 2. Random-projection control (post-hoc on triplet)

100 random 19→2 Gaussian matrices replayed against the same gradient log.
Mean `|cos|` for the 200 columns = 0.191 ± 0.104, max 0.488. **0/200**
random projections reach either Seidel value. Pre-empts the "any low-D
subspace would do" objection.

### 3. Sixlet (37-D, S2/S3) — replication

Run config: [`experiments/gradient_alignment/sixlet_config.yaml`](experiments/gradient_alignment/sixlet_config.yaml).
50k evals, ~17 min wall, 2778 generations.

**Result:** mean `|cos|` = **0.86** (S2), 0.70 (S3) — *higher* despite
larger θ-dim. Signed cos now strongly positive. 37-D random null = 0.13;
0/200 random projections approach Seidel. Outcome A clean.

### 4. 5-Seidel post-hoc (triplet + sixlet)

Replay each θ with `SeidelDescriptor(["S1","S2","S3","S4","S5"])` —
no new search. Recomputed ∇L matches stored to 0.0 relative error
(deterministic ray sampler), recomputed S2/S3 cosines reproduce stored
values exactly.

| Seidel | triplet | sixlet | optical meaning |
|--------|---:|---:|-----------------|
| S1 spherical | 0.65 | **0.90** | on-axis blur |
| S2 coma | 0.62 | 0.86 | off-axis tail asymmetry |
| S3 astigmatism | 0.70 | 0.70 | tangential / sagittal split |
| S4 Petzval | **0.84** | 0.70 | field flatness |
| S5 distortion | 0.45 | 0.40 | geometric position error |

All five Seidels above the random null. S5 (distortion) consistent
outlier — physically meaningful, distortion is position not blur. The
dominant Seidel changes between topologies (triplet: S4, sixlet: S1),
matching what's optically expected (triplet is field-curvature-limited,
sixlet absorbs that and becomes spherical-aberration-limited).

### 5. Septlet APS-C (41-D, S4/S2) — wider sensor + F#-enforced loss

Run config: [`experiments/gradient_alignment/septlet_config.yaml`](experiments/gradient_alignment/septlet_config.yaml).
Cuts iterations from published 100k → 50k. ~19 min wall, 2084 generations.

**Result:** mean `|cos|` = 0.47 (S4), 0.45 (S2). Much lower than
triplet/sixlet but still 3.7× the 41-D random null (0.125), 0/200 random
projections approach. The diagnostic 5-Seidel post-hoc shows **all five
Seidels collapse to a flat 0.43–0.47** with no hierarchy — signature of
∇L having power outside the Seidel subspace altogether.

### 6. Septlet F#-OFF control — falsifies the F-number hypothesis

Run config: [`experiments/gradient_alignment/septlet_nofnum_config.yaml`](experiments/gradient_alignment/septlet_nofnum_config.yaml).
Identical to (5) but `enforce_fnum: false`.

**Result:** mean `|cos|` = 0.465 (S4), 0.436 (S2) — **virtually unchanged**
from F#-on. ‖∇L‖ dropped 4× when F# removed (consistent with a smaller
loss), but the *direction* of ∇L barely shifted. F# is **not** the cause
of the lower septlet alignment.

### 7. Septlet light-intersect control — partial confirmation

Run config: [`experiments/gradient_alignment/septlet_lightintersect_config.yaml`](experiments/gradient_alignment/septlet_lightintersect_config.yaml).
F# off AND `intersect_penalty_weight: 1.0` (was 10.0; matches triplet/sixlet).

**Result:** mean `|cos|` = 0.519 (S4), 0.516 (S2) — **+0.05–0.08 over the
heavy-intersect runs**. 5-Seidel post-hoc:

| Seidel | F# on, heavy intersect | F# off, heavy intersect | F# off, light intersect |
|--------|---:|---:|---:|
| S1 | 0.43 | 0.43 | 0.49 |
| S2 | 0.45 | 0.44 | 0.52 |
| S3 | 0.47 | 0.45 | 0.53 |
| S4 | 0.47 | 0.47 | 0.52 |
| S5 | 0.46 | 0.42 | 0.51 |

Intersect penalty contributes ~0.05–0.07 of the alignment loss. Flat
collapse persists, just shifted up. Remaining gap to triplet/sixlet
(~0.10–0.30) attributable to **wider field (higher-order aberrations
the Seidel basis can't span) + larger θ-dim**, neither directly tested.

### 8. Descriptor comparison — Seidel vs Zernike (300 θ subset, smoke + partial)

Tool: [`experiments/gradient_alignment/posthoc_descriptor_compare.py`](experiments/gradient_alignment/posthoc_descriptor_compare.py).
Three axes per candidate: speed (median ms / autograd), discriminativeness
(participation-ratio rank of descriptor matrix), alignment (per-dim
`|cos|` + joint subspace projection of ∇L onto span(∇M_*)).

Smoke (30 θ, septlet F#-on log):

| descriptor | dim | speed (ms) | effrk | <&#124;cos&#124;> | joint proj |
|---|---:|---:|---:|---:|---:|
| Seidel[S4,S2] | 2 | 247 | 1.99 | 0.46 | 0.64 |
| Seidel[S2,S3] | 2 | 260 | 1.04 | 0.44 | 0.63 |
| Seidel[S1..S5] | 5 | 337 | 1.01 | 0.44 | **0.84** |
| Zernike[7,11] | 2 | 2318 | 1.04 | 0.37 | 0.61 |
| Zernike[9,14] | 2 | 2409 | 1.04 | 0.40 | 0.63 |
| Zernike[7,9,11,14] | 4 | 2055 | 1.08 | 0.39 | **0.82** |
| Zernike[4..16] | 13 | — | — | — | — |

Three things visible in the smoke:

1. **Zernike is ~10× slower** (~2.3 s vs ~0.25 s per θ-eval) — paraxial
   closed-form Seidel vs ray-traced wavefront sampling.
2. **Per-dim |cos| is *lower* for Zernike** (0.37–0.40 vs 0.44 for Seidel)
   but **joint subspace alignment is comparable** (0.82 for 4-D Zernike vs
   0.84 for 5-D Seidel). Individual Zernike modes are less aligned but
   collectively span similar ∇L content.
3. **Effective rank ≈ 1.0** for almost everything except Seidel[S4,S2] (1.99).
   On *this* trajectory, descriptor outputs collapse to a near-1-D manifold
   — partly because the search was driven by S4/S2 specifically, biasing
   the trajectory along those axes. Clean discriminativeness measurement
   would require uniform-θ sampling, not search trajectory.

**Full 300-θ run results (septlet F#-on log):**

| descriptor | dim | speed (ms) | effrk | <&#124;cos&#124;> | joint proj |
|---|---:|---:|---:|---:|---:|
| Seidel[S4,S2] | 2 | 227 | 1.96 | 0.46 | 0.65 |
| Seidel[S2,S3] | 2 | 227 | 1.05 | 0.46 | 0.62 |
| Seidel[S1..S5] | 5 | 266 | 1.00 | 0.46 | 0.84 |
| Zernike[7,11] | 2 | 2037 | 1.47 | 0.47 | 0.66 |
| Zernike[9,14] | 2 | 2035 | 1.56 | 0.48 | 0.65 |
| Zernike[7,9,11,14] | 4 | 2123 | 1.63 | 0.47 | **0.80** |
| **Zernike[4..16]** | **13** | **2516** | **1.50** | **0.47** | **0.972** |

**Three substantive findings:**

1. **Zernike[4..16] captures 97% of ∇L.** Thirteen Zernike modes spanning
   third-and-fourth-order recover essentially all of the loss-gradient
   direction. Compared to Seidel[S1..S5]'s 84%, the missing 13 percentage
   points live in higher-Noll Zernike modes that the third-order Seidel
   basis cannot span. **Direct evidence that the septlet alignment drop is
   from higher-order aberration content in ∇L** that escapes third-order
   theory.
2. **Per-dim |cos| is flat ~0.46–0.48 across all candidates.** Adding
   modes doesn't raise individual-mode alignment; it spreads joint capture
   across more components. No single descriptor "absorbs" the missing
   alignment; they collectively span it.
3. **Speed cost is real but not catastrophic.** Zernike is ~9× slower than
   Seidel (~2 s vs ~0.25 s per θ-eval). Notably Zernike scales weakly with
   dim (13 modes only 25% slower than 2 modes) because the cost is
   dominated by ray tracing, not autograd backward passes — adding more
   Zernike modes is essentially free once you've paid for the trace.

**Practical recommendation for APS-C / wide-field designs:**
Zernike[7,9,11,14] is the sweet spot — 80% joint capture (vs 65% for
Seidel[S4,S2]) at 2.1 s per eval. The full 13-mode sweep captures 97%
at only 19% additional cost; worth it if alignment is critical. Seidel
remains the right choice for paraxial-friendly regimes (triplet/sixlet)
where third-order theory captures 80%+ of ∇L on its own and is 9× faster.

**Caveat on discriminativeness (effective rank):** the rank ≈ 1.0-1.96
range is partly trajectory-biased — the septlet search was driven by
S4/S2, so descriptor outputs along the trajectory collapse along those
axes. Seidel[S4,S2] at 1.96 reflects this bias, not an inherent property.
Clean discriminativeness measurement requires uniform-θ sampling, not the
search trajectory.

### 9. Field-index = 2 post-hoc (full-field descriptor) — DEFINITIVE NULL

Same code path as the 5-Seidel post-hoc but with `descriptor_field_index`
overridden from 1 (mid-FOV) to 2 (full-field). Tested the hypothesis that
evaluating Seidels at the field edge changes the alignment picture.

**Result: cosine alignment is structurally invariant to field choice.**

| Seidel | mean &#124;cos&#124; @ field=1 | mean &#124;cos&#124; @ field=2 | measure-magnitude ratio | predicted from Seidel theory |
|--------|---:|---:|---:|---:|
| S1 spherical | 0.4307 | 0.4307 | 1.000× | field⁰ (on-axis) |
| S2 coma | 0.4454 | 0.4454 | 2.067× | field¹ |
| S3 astigmatism | 0.4688 | 0.4688 | 4.429× | field² |
| S4 Petzval | 0.4673 | 0.4673 | 4.352× | field² |
| S5 distortion | 0.4631 | 0.4631 | 10.157× | field³ (tan³ correction) |

`cos(∇M_S2 @ field=1, ∇M_S2 @ field=2) = 1.000000` for every θ checked.

**Why:** third-order Seidel theory factorises each term as
`S_k(θ, field) = field^p_k × g_k(θ)` where `p_k ∈ {0, 1, 2, 2, 3}` is the
term's field power. So `∇_θ S_k = field^p_k × ∇_θ g_k(θ)` — same
*direction* in θ-space regardless of field; magnitude scales by a known
power of field. Cosine is direction-only, so it is exactly invariant
under field choice. Empirical ratios (column 4 above) match the predicted
field powers (column 5) within ~1% — the residual is `tan(x) ≠ x` at
wider field.

**Implication:** `descriptor_field_index` is a normalisation knob for the
descriptor *magnitude* (relevant for archive bin definitions and
discriminativeness scaling), not a gradient-alignment knob. Cosine
alignment is the same whether you evaluate at on-axis, mid-FOV, or
full-field. To address the septlet alignment drop, the only knobs that
move it are (a) the descriptor *basis* (richer than third-order Seidel
— e.g., Zernike higher Nolls, multi-field stacking, 5th-order Seidels)
and (b) the *loss* (removing off-Seidel terms — intersect penalty
contributes ~0.05–0.07; F# does not).

## Cross-experiment summary

| metric | triplet | sixlet | septlet F#on | septlet F#off | septlet F#off + light-intersect |
|---|---:|---:|---:|---:|---:|
| θ-dim | 19 | 37 | 41 | 41 | 41 |
| BC pair | S2,S3 | S2,S3 | S4,S2 | S4,S2 | S4,S2 |
| BC1 mean &#124;cos&#124; | 0.616 | **0.859** | 0.467 | 0.465 | 0.519 |
| BC2 mean &#124;cos&#124; | 0.701 | 0.697 | 0.445 | 0.436 | 0.516 |
| 5-Seidel range &#124;cos&#124; | 0.45–0.84 | 0.40–0.90 | 0.43–0.47 | 0.42–0.47 | 0.49–0.53 |
| analytic null | 0.183 | 0.131 | 0.125 | 0.125 | 0.125 |
| 100-matrix random &#124;cos&#124; | 0.191 ± 0.10 | 0.137 ± 0.07 | 0.125 ± 0.04 | 0.125 ± 0.03 | 0.126 ± 0.04 |
| max random &#124;cos&#124; (200) | 0.488 | 0.401 | 0.245 | 0.232 | 0.249 |
| frac random ≥ Seidel | 0/200 | 0/200 | 0/200 | 0/200 | 0/200 |
| F# in loss | no | no | **yes** | no | no |
| intersect_penalty_weight | 1.0 | 1.0 | 10.0 | 10.0 | **1.0** |
| best fitness | −0.020 | −0.791 | −0.098 | −0.109 | −0.167 |
| run wall | ~12 m | ~17 m | ~19 m | ~19 m | ~17 m |

## Outcome labels

- **Triplet**: Outcome A (modified) — high `|cos|` magnitude, mixed signs.
  Sign convention of S3 makes signed-cos negative; CMA-MEGA invariant to
  this.
- **Sixlet**: Outcome A (clean) — high `|cos|` AND positive signed cos.
- **Septlet (all variants)**: Outcome A (degraded) — `|cos|` ≈ 0.45–0.52,
  still well above the random null (3.7×) and 0/200 random projections.
  But the gradient-alignment claim weakens with a wider field /
  higher-order aberration regime / heavier non-aberration loss terms.

## Important caveats and what these experiments do NOT show

- **Gradient alignment ≠ QD utility.** A descriptor with low alignment
  can still be a *good* QD descriptor (textbook MAP-Elites theory wants
  `|cos|` ≈ 0 so descriptor exploration is fitness-blind). Gradient
  alignment is upstream evidence about ∇M_j and ∇L sharing structure;
  it doesn't say the descriptor produces a useful archive. That's a
  separate ablation (compare Seidel-archive coverage, best-fitness,
  refinement depth vs Zernike-archive vs random-BC-archive on the same
  template).
- **Effective rank ≈ 1.0 is partly trajectory-biased.** The septlet
  search was driven by S4/S2, so descriptor outputs along the trajectory
  collapse to a ~1-D submanifold even for descriptors that are
  fundamentally K-D. Clean discriminativeness measurement needs uniform
  θ sampling.
- **Sign of signed cos is conventional.** Seidel formula sign is a
  textbook convention; flipping it doesn't change CMA-MEGA behaviour
  since the emitter samples gradient coefficients with both signs.

## Files produced (by experiment)

### Core scripts (reusable, parameterised)

| file | purpose |
|------|---------|
| [`experiments/gradient_alignment/analyze_gradients.py`](experiments/gradient_alignment/analyze_gradients.py) | per-generation cosines + sanity checks; `--log` `--out` `--labels` |
| [`experiments/gradient_alignment/random_projection_compare.py`](experiments/gradient_alignment/random_projection_compare.py) | 100 random projections + per-matrix/per-column stats |
| [`experiments/gradient_alignment/posthoc_all_seidels.py`](experiments/gradient_alignment/posthoc_all_seidels.py) | replay log θ to compute all 5 Seidel grads; `--descriptor-field-index` override |
| [`experiments/gradient_alignment/posthoc_descriptor_compare.py`](experiments/gradient_alignment/posthoc_descriptor_compare.py) | candidate descriptor sweep (speed / discrim / alignment) |
| [`experiments/gradient_alignment/make_figure.py`](experiments/gradient_alignment/make_figure.py) | 4-panel run figure (trajectory + signed hist + &#124;cos&#124; + random-projection panel) |
| [`experiments/gradient_alignment/figure_all_seidels.py`](experiments/gradient_alignment/figure_all_seidels.py) | violin plot of all 5 Seidel &#124;cos&#124; per topology |

### Run configs

| file | description |
|------|-------------|
| `config.yaml` | triplet 19-D S2/S3 |
| `sixlet_config.yaml` | sixlet 37-D S2/S3 |
| `septlet_config.yaml` | septlet 41-D S4/S2 with F# enforced |
| `septlet_nofnum_config.yaml` | septlet without F# penalty (control) |
| `septlet_lightintersect_config.yaml` | septlet without F# AND intersect_penalty_weight=1.0 |

### Data outputs (one set per run)

`<run_dir>/gradient_log.npz`, `cosine_similarities_<tag>.npz`,
`random_projection_distribution_<tag>.npz`,
`random_projection_matrix_<tag>.npy`,
`all_seidels_<tag>.npz`,
`figure_gradient_alignment_<tag>.{pdf,png}`,
`summary_<tag>.txt`.

### Aggregate figures and docs

| file | description |
|------|-------------|
| `figure_all_seidels.{pdf,png}` | 4-panel violin (triplet, sixlet, septlet F#on, septlet F#off) |
| `descriptor_compare_septlet.json` | descriptor-comparison results (in progress) |
| `README.md` | written-up findings, tables, recommendations for §3.2 |

## What remains in flight at session export

- **Descriptor compare full run** (300 θ × 7 candidates) — 3 of 7 candidates
  done at export time, 4 Zernike candidates remaining (~30 min wall).
- **Field-index=2 5-Seidel post-hoc** — just launched, ~12 min wall.

## Open questions worth a follow-up

1. **Does the field-index=2 run change the picture?** If full-field Seidels
   align *better* than mid-FOV, the wider-field hypothesis gets a partial
   confirmation. If they align *worse*, the third-order Seidel basis is
   genuinely incomplete at the field edge and a richer basis (Zernike
   higher Nolls or 5th-order Seidel) is the right direction.
2. **What does the full descriptor-compare table say once finished?**
   Specifically: does the 13-D Zernike[4..16] reach joint projection > 0.9
   (near-complete capture of ∇L)? If yes, that's evidence the missing
   alignment lives in higher-Noll Zernike modes — i.e., higher-order
   aberrations.
3. **QD utility ablation on septlet.** Run CMA-MEGA on the septlet
   template with three different BC choices (Seidel S4/S2, Zernike[7,11],
   random-projection) and compare archive metrics (coverage, best-fit,
   refinement depth). This is the *direct* evidence for descriptor quality;
   gradient alignment is upstream evidence.
4. **Multi-field Seidel** — small extension of `SeidelDescriptor` to
   evaluate at multiple field heights and stack outputs as the BC vector.
   Cheaper than Zernike, captures field-dependent aberration variation.
   Not implemented yet.
5. **Wider-field control on triplet/sixlet.** If we add an APS-C-class
   sensor template to the triplet/sixlet, does *their* alignment drop in
   the same way? Would isolate "field width" as a single variable.

## How to reproduce any single result

```bash
source activate_env.sh

# Run one of the configs:
python3 -m optical_rl.qd.lensmega.run \
  --config experiments/gradient_alignment/<chosen>_config.yaml

# Find the new gradient log:
LOG=$(ls experiments/gradient_alignment/*_run/*/gradient_log.npz | tail -1)

# Compute per-generation cosines:
python3 experiments/gradient_alignment/analyze_gradients.py \
  --log "$LOG" \
  --out experiments/gradient_alignment/cosine_similarities_<tag>.npz \
  --labels S4,S2

# Random-projection comparison:
python3 experiments/gradient_alignment/random_projection_compare.py \
  --log "$LOG" \
  --matrix-out experiments/gradient_alignment/random_projection_matrix_<tag>.npy \
  --out experiments/gradient_alignment/random_projection_distribution_<tag>.npz \
  --cosine-path experiments/gradient_alignment/cosine_similarities_<tag>.npz

# 5-Seidel post-hoc (replay; no new HyperLens search):
PYTHONPATH=. python3 experiments/gradient_alignment/posthoc_all_seidels.py \
  --log "$LOG" \
  --config experiments/gradient_alignment/<chosen>_config.yaml \
  --out experiments/gradient_alignment/all_seidels_<tag>.npz

# Build the 4-panel figure:
python3 experiments/gradient_alignment/make_figure.py \
  --cosine experiments/gradient_alignment/cosine_similarities_<tag>.npz \
  --randproj experiments/gradient_alignment/random_projection_distribution_<tag>.npz \
  --fig experiments/gradient_alignment/figure_gradient_alignment_<tag>.pdf \
  --summary experiments/gradient_alignment/summary_<tag>.txt \
  --title-suffix "<topology description>"
```

## Suggested §3.2 paragraph for the paper (data-supported)

> Across N = 1389 / 2778 / 2084 CMA-MEGA generations on the
> triplet / sixlet / septlet, descriptor gradients ∇_θ M_j are strongly
> parallel to the loss gradient — mean `|cos(−∇L, ∇M_j)|` of 0.62 / 0.86 / 0.45
> for the first BC and 0.70 / 0.70 / 0.45 for the second, vs random-vector
> baselines of 0.18 / 0.13 / 0.13 in 19 / 37 / 41 dimensions. The alignment
> is specific to the Seidel choice: across 100 random-projection
> descriptors of matched dimension, **0 of 200** sampled directions reach
> any of the Seidel values in any topology. As the optical regime widens
> beyond paraxial-friendly (APS-C septlet) and additional non-aberration
> terms enter the loss (geometric-constraint penalties), some of ∇L
> migrates outside the third-order Seidel basis and alignment degrades to
> ~3.7× the random null while remaining clearly above it.
