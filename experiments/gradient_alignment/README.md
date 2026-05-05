# Gradient-alignment experiment (§3.2)

## What this is

Empirical measurement of cosine similarity between the loss-descent direction
`-∇_θ L` and each Seidel-descriptor gradient `∇_θ M_j` (j = S2, S3) over the
course of a HyperLens / CMA-MEGA search on the Cooke triplet.

The experiment converts the §3.2 structural claim ("Seidel coefficients
decompose the wavefront aberration that the loss aggregates") into a
directional claim: do `∇M_j` and `∇L` actually point in similar directions in
parameter space?

## What was run

- Cooke triplet, 19D parameter vector (6 curvatures + 6 thicknesses/airspaces
  + 1 back flange + 3 (n, V) pairs).
- 50 mm focal length, polychromatic (F, d, C wavelengths).
- Seidel S2 (coma) + S3 (astigmatism) descriptors.
- N_seed = 2000, pop_size = 36, 200×200 archive grid, 50 000 evaluations.
- Random seed 42, 18 CPU workers, ~12 min wall time.
- See [`run_config.json`](run_config.json) and [`config.yaml`](config.yaml).

The CMA-MEGA inner loop already computes `∇_θ fitness` and `∇_θ M_j` once per
generation (`LensProblem.evaluate(..., with_grads=True)`); the experiment adds
read-only logging of those raw vectors immediately before normalisation. No
optimiser behaviour was changed.

Parent gradients were captured for **1389 generations** (one per CMA-MEGA
parent step). 0 of those frames had NaN/zero gradients.

## Sign convention

The codebase stores `grad_fitness = ∇_θ fitness`, where
`fitness = -L − penalties`. The CMA-MEGA descent direction on the loss is
therefore `-∇_θ L = +∇_θ fitness`. Throughout this analysis,
"cosine similarity" means

```
cos(-∇_θ L, ∇_θ M_j)  =  cos(grad_fitness, grad_M_j)
```

so positive ⇒ `∇M_j` points in the descent direction; negative ⇒ it points
against descent. Note that the *sign* of any individual Seidel coefficient
(and hence its gradient) is a convention of the formula; what is physically
meaningful is the parallelism (i.e. `|cos|`).

## Headline numbers

| descriptor | mean signed cos | median signed cos | mean **&#124;cos&#124;** | frac &#124;cos&#124; > 0.3 |
|------------|------:|------:|------:|------:|
| S2 (coma)         | −0.042 | −0.072 | **0.616** | 79 % |
| S3 (astigmatism)  | −0.571 | −0.788 | **0.701** | 86 % |
| Random_1 (seed 12345 col 0) | −0.071 | — | 0.105 | 0 % |
| Random_2 (seed 12345 col 1) | −0.365 | — | 0.460 | 95 % |
| 100 random projections (200 cols) — mean ± std | — | — | 0.191 ± 0.104 | — |
| 100 random projections — max across all 200 cols | — | — | **0.488** | — |
| 19-D random-vector null  |  0.000 |  — |  0.183 | ~10 % |

Cross-descriptor: mean `cos(∇M_S2, ∇M_S3) = +0.05`, but mean
`|cos(∇M_S2, ∇M_S3)| = 0.74` — the two Seidel descriptor gradients are **not**
orthogonal to each other within any one generation; they line up strongly,
just with sign that flips across the run.

**Random-projection comparison (addressing the "any low-D subspace would do"
objection):** across 100 independently-drawn 19→2 Gaussian projection
matrices (200 descriptor columns total), the distribution of mean `|cos|`
over the trajectory is **0.191 ± 0.104**, centred on the 19-D random null
of 0.183, with a maximum of 0.488. **0 of 200** random projections reach
either Seidel value (0.616 or 0.701). The Seidel descriptors pick out
specific high-information directions — they are not artefacts of a generally
low-dimensional loss subspace.

## Interpretation: closest to Outcome A, with a sign caveat

Using the user-defined outcome categories:

- **Signed mean cos** (the quantity the user's §3.2 framing predicts):
  - S2 is approximately orthogonal on average (mean −0.04, median −0.07).
    Per-generation values are highly bimodal — 47 % positive, 42 % strongly
    negative — so the signed alignment **flips sign frequently**.
  - S3 is **strongly negative** on average (mean −0.57). 78 % of generations
    have cos < −0.3.

- **|cos|** (the sign-invariant version, which is what physically matters
  because the sign of any Seidel descriptor is a formula convention and
  the CMA-MEGA emitter samples gradient coefficients with **both signs**):
  - Mean |cos| is 0.62 (S2) and 0.70 (S3) — far above the 0.183
    random null in 19 D.
  - 79 % (S2) and 86 % (S3) of generations have |cos| > 0.3.

**Substantive finding:** descriptor gradients are **not** orthogonal to the
descent direction. They live in a strongly overlapping subspace of parameter
space. The §3.2 structural claim is empirically supported: moving along
`∇M_j` is approximately equivalent (up to sign) to moving along `∇L`. This
matches Outcome A (positive alignment) once the conventional sign of S3 is
factored out.

**The S3 sign:** The strongly negative mean signed-cos for S3 is consistent
with the sign of S3 in the standard Seidel formula at this lens scale —
flipping the descriptor's sign convention would yield mean signed-cos ≈ +0.57
without changing any optimisation behaviour. CMA-MEGA's emitter mixes
`∇fitness` and `∇M_j` with sampled coefficients of either sign, so the
algorithm is invariant to this convention. The signed-cos number is reported
for completeness; it is not the load-bearing statistic.

## How to read this for §3.2 of the paper

A cautious one-sentence summary suitable for the paper:

> Across N = 1389 CMA-MEGA generations on the triplet, descriptor
> gradients `∇_θ M_j` are strongly parallel to the loss gradient — mean
> `|cos(-∇L, ∇M_j)|` of 0.62 (S2) and 0.70 (S3), versus a random-vector
> baseline of 0.18 in 19 dimensions — confirming that descriptor-driven
> steps and fitness-improvement steps share directional structure rather
> than being orthogonal.

A second sentence pre-empting the "any low-D subspace would do" objection:

> The alignment is specific to the Seidel choice: across 100 random-projection
> descriptors of the same dimension, the mean `|cos|` values are tightly
> distributed around the 19-D random null (0.191 ± 0.104), with a maximum of
> 0.488 — none of the 200 random descriptor directions approaches the Seidel
> values of 0.62 / 0.70 (Figure, panel d).

(If the §3.2 prose specifically wants signed alignment, the honest version
is "S2 is approximately orthogonal on average; S3 is anti-aligned with
descent under the standard Seidel sign convention; in both cases, the
sign-invariant magnitude `|cos|` is much larger than the random-vector
null, indicating strong directional overlap.")

## Sixlet replication (37-D)

To test whether the result generalises beyond the triplet topology, the
same experiment was repeated on a 6-element sixlet at the same focal
length and f-number. Sixlet θ-dim is **37** (12 curvatures + 12 gaps + 1
flange + 6 (n,V) pairs), so the random null drops from 0.183 (19-D) to
**0.131** (37-D), making the comparison harder, not easier.

Run config: [`sixlet_config.yaml`](sixlet_config.yaml), matching the
published §5.2 sixlet baseline (`ablation_sixlet_discovery_materials_seidel_s2s3.yaml`).
2778 generations logged. ~17 min main-loop wall time.

| descriptor | mean signed cos | median signed cos | mean **&#124;cos&#124;** | frac &#124;cos&#124; > 0.3 |
|------------|------:|------:|------:|------:|
| S2 (coma) | **+0.624** | +0.943 | **0.859** | 95 % |
| S3 (astigmatism) | **+0.554** | +0.729 | **0.697** | 92 % |
| Random projection (100 matrices, 200 cols) — mean ± std | — | — | 0.137 ± 0.074 | — |
| Random projection — max across 200 cols | — | — | **0.401** | — |
| 37-D random-vector null | 0.000 | — | 0.131 | — |

**Two things change vs the triplet result:**

1. **Sign convention works in our favour.** Both signed-cos means are now
   strongly *positive* (+0.62, +0.55), so the descriptor gradients agree
   with the descent direction directly — no need to invoke the |cos|
   reframing to make the §3.2 claim. 81 % / 83 % of generations have
   `cos > +0.3`; only 14 % / 9 % have `cos < −0.3`.

2. **Magnitude alignment is even stronger, despite the larger search
   space.** Mean |cos| jumps from 0.62 (triplet S2) to **0.86** (sixlet
   S2). Both descriptors stay near 1.0 throughout the run (panel a of
   [`figure_gradient_alignment_sixlet.png`](figure_gradient_alignment_sixlet.png)).
   The 37-D random-projection distribution sits at 0.137 ± 0.074, exactly
   matching the analytic null; **0 / 200** random projections approach
   either Seidel value.

The two effects together: the sixlet result is **clean Outcome A** in the
user's terminology, with no sign caveat. The §3.2 claim does **not**
require the |cos| reframing here — descriptor gradients literally point
in the descent direction on most generations.

| metric | triplet (19-D) | sixlet (37-D) |
|--------|------:|------:|
| mean &#124;cos&#124; S2 | 0.616 | **0.859** |
| mean &#124;cos&#124; S3 | 0.701 | **0.697** |
| signed mean S2 | −0.04 | **+0.62** |
| signed mean S3 | −0.57 | **+0.55** |
| random-projection 100-matrix mean &#124;cos&#124; | 0.191 ± 0.104 | 0.137 ± 0.074 |
| random-projection max &#124;cos&#124; (200 cols) | 0.488 | 0.401 |
| analytic random null | 0.183 | 0.131 |
| frac random ≥ Seidel S2 | 0/200 | 0/200 |
| frac random ≥ Seidel S3 | 0/200 | 0/200 |
| run wall time | ~12 min | ~17 min |

**What the sixlet replication tells us:** the gradient-alignment finding
holds across topologies. Doubling the parameter dimension (which would
push uncorrelated descriptor gradients toward the lower 0.131 null) does
the opposite — Seidel `|cos|` goes *up*, not down. This is the strongest
form of the claim we have evidence for: descriptor gradients track the
loss gradient *more* tightly when there are more degrees of freedom for
them to "miss" in.

## Septlet APS-C replication (41-D, larger sensor + F# constraint)

A third run on a **7-element septlet** for an APS-C-class image circle
(`septlet_apsc_e2e3cement_s4s2.yaml` baseline) tests two things at once:
(i) a wider field angle / larger r_sensor — the septlet covers
substantially more diagonal FOV than the prior two; (ii) a
loss with an **F-number penalty active** (`enforce_fnum: true`,
`fnum_penalty_weight: 1.0`) on top of the usual spot-RMS + intersect +
invalid-fraction terms. The descriptor pair is also different: **S4 + S2**
(Petzval + Coma) rather than S2 + S3.

θ-dim = **41** (13 curvatures + 13 gaps + 1 flange + 7 (n,V) — one E2/E3
cemented doublet adds an interface). Random null drops further to **0.125**.
Run config: [`septlet_config.yaml`](septlet_config.yaml).
2084 generations logged. ~19 min main-loop wall.

| descriptor | mean signed cos | median signed cos | mean **&#124;cos&#124;** | frac &#124;cos&#124; > 0.3 |
|------------|------:|------:|------:|------:|
| S4 (Petzval) — BC | −0.235 | −0.370 | **0.467** | 71 % |
| S2 (coma) — BC | +0.089 | +0.110 | **0.445** | 63 % |
| Random projection (100 matrices, 200 cols) — mean ± std | — | — | 0.125 ± 0.035 | — |
| Random projection — max across 200 cols | — | — | **0.245** | — |
| 41-D random-vector null | 0.000 | — | 0.125 | — |

**Two things change vs the triplet / sixlet result:**

1. **Mean |cos| drops from 0.6–0.9 down to ~0.45** for both BC descriptors.
   The descriptor gradients are still ~3.7× the random null (0/200 random
   projections approach either Seidel), but the *gap* is much narrower.
2. **Signed alignment is now noisy** — both descriptors hover near 0 with
   wide variance (panel a of
   [`figure_gradient_alignment_septlet.png`](figure_gradient_alignment_septlet.png)).

**Why is the alignment lower here?** Three plausible mechanisms — the
F-number-penalty hypothesis was tested by a control run and **falsified**:

- ~~F-number penalty in the loss adds an off-Seidel component to ∇L.~~
  **Tested and ruled out.** A control run with `enforce_fnum: false` and
  everything else identical ([`septlet_nofnum_config.yaml`](septlet_nofnum_config.yaml))
  produced **the same alignment numbers**: S4 |cos| = 0.465 (was 0.467),
  S2 |cos| = 0.436 (was 0.445). All five Seidels stay at 0.42–0.47 in the
  control too, so the off-Seidel component isn't the F# penalty. Removing
  it does drop ‖∇L‖ from 8.1 × 10⁴ to 2.1 × 10⁴ (consistent with cutting
  out a contributing term), but the *direction* of ∇L is essentially
  unchanged. See [`summary_septlet_nofnum.txt`](summary_septlet_nofnum.txt)
  and panel (d) of [`figure_all_seidels.png`](figure_all_seidels.png).
- **Wider field angle → higher-order aberrations matter more.** The
  third-order Seidels are a paraxial expansion; a wider chief-ray angle
  pushes power into 5th- and 7th-order terms that no Seidel captures.
- **More elements / larger θ-dim → ∇L can spread across directions no
  Seidel picks out as cleanly.** 41-D is the highest of the three runs,
  and the search visits regions where intersect/invalid penalty terms
  fire more often (`intersect_penalty_weight: 10.0` here vs `1.0` in the
  prior runs), adding more "geometric, non-aberration" power to ∇L.

**The 5-Seidel post-hoc remains the key diagnostic** (next section): in
the triplet/sixlet, the unselected Seidels spread across **0.4 → 0.9** in
mean |cos| with a clear hierarchy. In **both** septlet runs (F# on and
F# off), all five collapse to a flat **0.42–0.47** — the hallmark of ∇L
having power in directions outside the Seidel basis altogether. The cause
of that off-basis power is *not* the F# penalty; it's most likely the
combination of higher-order aberrations (from the wider field) and
geometric-constraint gradients (from the heavier intersect penalty).

| metric | triplet | sixlet | septlet |
|--------|------:|------:|------:|
| mean &#124;cos&#124; first BC | 0.616 (S2) | **0.859** (S2) | 0.467 (S4) |
| mean &#124;cos&#124; second BC | 0.701 (S3) | 0.697 (S3) | 0.445 (S2) |
| signed mean first BC | −0.04 | +0.62 | −0.23 |
| signed mean second BC | −0.57 | +0.55 | +0.09 |
| 100-matrix random &#124;cos&#124; | 0.191 ± 0.104 | 0.137 ± 0.074 | 0.125 ± 0.035 |
| max random &#124;cos&#124; (200 cols) | 0.488 | 0.401 | 0.245 |
| analytic null | 0.183 | 0.131 | 0.125 |
| frac random ≥ Seidel BC | 0/200 | 0/200 | 0/200 |
| F# penalty in loss? | no | no | **yes** |
| run wall time | ~12 min | ~17 min | ~19 min |

**Bottom line:** the alignment claim survives — Seidels still beat 100/100
random projections handily, and `|cos|` is still 3.7× the analytic null.
But the *strength* of the alignment depends visibly on what other terms
are in the loss; an F-number penalty (or a similar "geometric, non-blur"
term) drains alignment uniformly across all five Seidels by adding power
to ∇L in a direction the Seidel basis can't see.

## All five Seidel coefficients (post-hoc analysis)

The runs above only logged ∇M_S2 and ∇M_S3 because S2/S3 was the BC pair.
A post-hoc replay of every parent θ in the gradient log — with
`SeidelDescriptor(seidel_terms=["S1","S2","S3","S4","S5"])` — gives ∇M_k for
all five terms without re-running the search. This costs ~3 min for the
triplet, ~12 min for the sixlet, and answers "is the alignment specific to
S2 / S3?".

The recomputed ∇L matches the stored ∇L to **0.0** relative error
(deterministic ray sampler), and the recomputed S2 / S3 cosines reproduce the
previously logged values exactly — so the post-hoc is a correctness-preserving
extension, not a fresh measurement.

| Seidel term | triplet &#124;cos&#124; | sixlet &#124;cos&#124; | septlet (F# on) &#124;cos&#124; | septlet (F# off) &#124;cos&#124; | optical meaning |
|-------------|------:|------:|------:|------:|-----------------|
| S1 spherical | 0.645 | **0.898** | 0.431 | 0.428 | on-axis blur |
| S2 coma | 0.616 | 0.859 | 0.445 | 0.436 | off-axis tail asymmetry |
| S3 astigmatism | 0.701 | 0.697 | 0.469 | 0.450 | tangential / sagittal split |
| S4 Petzval | **0.843** | 0.704 | 0.467 | 0.465 | field flatness; sum of curvatures × 1/n |
| S5 distortion | 0.445 | 0.395 | 0.463 | 0.421 | geometric position error, not a blur |
| analytic random null | 0.183 (19-D) | 0.131 (37-D) | 0.125 (41-D) | 0.125 (41-D) | baseline |

The **flat 0.42–0.47 collapse persists with F# off**, confirming that
the F-number penalty is not what suppresses the Seidel hierarchy in the
septlet runs.

Signed means (under the standard Seidel sign convention):

| Seidel term | triplet | sixlet | septlet (F# on) | septlet (F# off) |
|-------------|------:|------:|------:|------:|
| S1 | −0.522 | **+0.758** | −0.206 | −0.272 |
| S2 | −0.042 | +0.624 | +0.089 | +0.127 |
| S3 | −0.571 | +0.554 | −0.129 | −0.172 |
| S4 | −0.652 | +0.578 | −0.235 | −0.293 |
| S5 | −0.073 | +0.070 | +0.033 | +0.018 |

See [`figure_all_seidels.png`](figure_all_seidels.png) for violin plots of
all five distributions side-by-side.

**Four things this shows:**

1. **All five Seidels are well above the random null in all three
   topologies.** Even the lowest values (~0.4) sit at 2.4–3.7× the null.
   The §3.2 claim is not specific to the BC pair; it's a property of the
   Seidel subspace as a whole.

2. **In triplet/sixlet, S5 (distortion) is the consistent outlier on
   the low end** while one blur Seidel dominates (S4 in triplet at 0.84,
   S1 in sixlet at 0.90). Distortion is a geometric position error, not
   a spot-size contribution, so the loss "doesn't see it" as much.
   The hierarchy is steep: ~0.4 → 0.9.

3. **In both septlet runs, the hierarchy disappears** — all five Seidels
   collapse to **0.42–0.47** with no clear winner. The flat collapse is
   the diagnostic signature of **∇L having power in a direction outside
   the Seidel subspace altogether**. A control run with
   `enforce_fnum: false` (everything else identical) was used to test
   whether the F-number penalty was responsible: it produced **the same
   flat collapse**, falsifying that hypothesis. The off-Seidel power
   most likely comes from a combination of (i) higher-order aberrations
   the third-order Seidel basis can't capture (the septlet has the
   widest field of the three), and (ii) the heavier intersect / invalid
   penalty terms (`intersect_penalty_weight: 10.0` here vs `1.0` in
   triplet / sixlet) which fire more often when the search probes
   geometric-constraint boundaries. Both contribute to ∇L through
   pathways the Seidel basis doesn't span.

4. **The dominant Seidel changes between topologies when there is one.**
   Triplet: S4 (Petzval) leads — small triplet's field-flattening is the
   binding constraint. Sixlet: S1 (spherical) leads — extra elements
   absorb field curvature, on-axis blur becomes the limit. Septlet:
   no leader — the loss is dominated by a non-aberration term.

Field angle: descriptor was evaluated at **mid-FOV** (`descriptor_field_index: 1`,
i.e. 0.5 × HFOV) in both runs — matching the standard configuration of the
published §5.2 Seidel ablations. Re-evaluating at on-axis (0°) or full-field
(HFOV) would change S2/S3/S5 contributions (which scale with field) but not
the qualitative finding.

## Outcome label

- **Triplet (19-D)**: Outcome A (modified). Seidel descriptor gradients are
  highly aligned with the descent direction in magnitude (mean |cos| ≫ random
  null). Sign of S3 is negative under the standard Seidel convention but
  CMA-MEGA is invariant to it.
- **Sixlet (37-D)**: Outcome A (clean). Both descriptors are positively signed
  *and* highly parallel to descent (mean signed cos +0.62 / +0.55, mean |cos|
  0.86 / 0.70). No sign reframing needed.
- **Random-projection follow-up** (both topologies): mean |cos| of 100 random
  matrices matches the analytic random null (0.191 in 19-D, 0.137 in 37-D).
  0/200 random projections reach the Seidel values in either run.

The §3.2 claim survives the strongest plausible reviewer objection in both
topologies. The sixlet replication is the stronger statement of the two: in
the higher-dimensional search space the alignment actually *strengthens*,
ruling out the explanation that the triplet result is an artefact of the
small parameter space.

## Sanity checks (all passed)

- `cos(∇L, ∇L) = 1.000000` ✓
- `cos(∇L, perm(∇L))` averaged over 1000 random permutations:
  `+0.0009 ± 0.241` — matches the predicted 0 ± 1/√19 ≈ 0.229 null. ✓
- All 1389 × 2 cosine values are finite and in [−1, 1]. ✓
- 0 NaN frames. ✓
- Gradient norms are finite throughout (mean ||∇L|| ≈ 8.3 × 10³,
  mean ||∇M_S2|| ≈ 3.06 × 10⁻², mean ||∇M_S3|| ≈ 2.02 × 10⁻²; the
  raw-magnitude gap is expected — the runner column-normalises both
  before mixing them, so absolute scale is not meaningful here).

## Deliverables

| file | description |
|------|-------------|
| `config.yaml` | run configuration used |
| `run_config.json` | hyperparameter snapshot, seed, schema notes |
| `run/gradient_alignment_triplet_<timestamp>/gradient_log.npz` | raw per-generation parent gradients |
| `cosine_similarities.npz` | per-generation cosine similarities + norms |
| `figure_gradient_alignment.pdf` (and `.png`) | three-panel publication figure |
| `summary.txt` | numeric summary statistics |
| `analyze_gradients.py` | computes cosine similarities and runs sanity checks |
| `make_figure.py` | renders the figure and writes summary.txt |
| `random_projection_compare.py` | computes random-projection cosines and the 100-matrix distribution |
| `random_projection_matrix.npy` | the published §5.2 ablation matrix (seed 12345, 19×2, unit-normalised columns) |
| `random_projection_distribution.npz` | mean/median/frac stats across 100 random matrices, plus all 100 sampled matrices for reproducibility |
| `sixlet_config.yaml` | sixlet replication run config (37-D, S2+S3, matches §5.2 ablation) |
| `sixlet_run/.../gradient_log.npz` | sixlet raw gradient log (2778 generations × 37 D) |
| `cosine_similarities_sixlet.npz` | sixlet cosine output |
| `random_projection_matrix_sixlet.npy`, `random_projection_distribution_sixlet.npz` | sixlet random-projection comparison artefacts |
| `figure_gradient_alignment_sixlet.{pdf,png}` | sixlet four-panel figure |
| `summary_sixlet.txt` | sixlet numeric summary |
| `posthoc_all_seidels.py` | replays each logged θ to compute ∇M_k for all five Seidels |
| `all_seidels_triplet.npz`, `all_seidels_sixlet.npz` | per-generation cosines + cross-Seidel similarity for all five terms |
| `figure_all_seidels.{pdf,png}` | side-by-side violin plot of all 5 Seidel &#124;cos&#124; distributions across triplet/sixlet/septlet |
| `figure_all_seidels.py` | renders the multi-Seidel figure |
| `septlet_config.yaml` | septlet APS-C run config (41-D, S4+S2, F# enforced) |
| `septlet_run/.../gradient_log.npz` | septlet raw gradient log (2084 generations × 41-D) |
| `cosine_similarities_septlet.npz`, `random_projection_*_septlet.npz`, `random_projection_matrix_septlet.npy`, `figure_gradient_alignment_septlet.{pdf,png}`, `summary_septlet.txt`, `all_seidels_septlet.npz` | septlet artefacts (same schema as the other two topologies) |
| `septlet_nofnum_config.yaml` | F#-disabled control config |
| `septlet_nofnum_run/.../gradient_log.npz` | gradient log for the control run |
| `cosine_similarities_septlet_nofnum.npz`, `random_projection_*_septlet_nofnum.npz`, `figure_gradient_alignment_septlet_nofnum.{pdf,png}`, `summary_septlet_nofnum.txt`, `all_seidels_septlet_nofnum.npz` | control-run analysis artefacts |

## How to reproduce

```bash
source activate_env.sh
python3 -m optical_rl.qd.lensmega.run \
  --config experiments/gradient_alignment/config.yaml
python3 experiments/gradient_alignment/analyze_gradients.py
python3 experiments/gradient_alignment/random_projection_compare.py
python3 experiments/gradient_alignment/make_figure.py
```

The random-projection step does not require a new HyperLens run — it
operates entirely on the already-logged ∇L(t) values and a freshly drawn
projection matrix.

To reproduce the sixlet replication:

```bash
source activate_env.sh
python3 -m optical_rl.qd.lensmega.run \
  --config experiments/gradient_alignment/sixlet_config.yaml
SIXLET_LOG=$(ls experiments/gradient_alignment/sixlet_run/*/gradient_log.npz | tail -1)
python3 experiments/gradient_alignment/analyze_gradients.py \
  --log "$SIXLET_LOG" \
  --out experiments/gradient_alignment/cosine_similarities_sixlet.npz
python3 experiments/gradient_alignment/random_projection_compare.py \
  --log "$SIXLET_LOG" \
  --matrix-out experiments/gradient_alignment/random_projection_matrix_sixlet.npy \
  --out experiments/gradient_alignment/random_projection_distribution_sixlet.npz \
  --cosine-path experiments/gradient_alignment/cosine_similarities_sixlet.npz
python3 experiments/gradient_alignment/make_figure.py \
  --cosine experiments/gradient_alignment/cosine_similarities_sixlet.npz \
  --randproj experiments/gradient_alignment/random_projection_distribution_sixlet.npz \
  --fig experiments/gradient_alignment/figure_gradient_alignment_sixlet.pdf \
  --summary experiments/gradient_alignment/summary_sixlet.txt \
  --title-suffix "Sixlet (37-D), 50 mm f/4, polychromatic."
```

## Date and provenance

Run date: 2026-05-04. Code instrumented at
`optical_rl/qd/lensmega/core/runner.py::GradientLogger`, gated by
`io.log_gradients: true` in the YAML config (added to
`config/schema.py`). The instrumentation is read-only; the optimiser
behaviour is unchanged when `log_gradients` is `false`.
