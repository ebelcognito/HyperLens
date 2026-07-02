# Gradient-alignment results — APS-C champion topologies

**Date:** 2026-05-10
**Allocation:** SLURM job 49681495, 64 CPU, c151405
**Working dir:** `/nas/longleaf/home/hisham26/Desktop/hyperlens`

## What was run

Two new CMA-MEGA gradient-alignment runs on the current APS-C champion configs:

| run | config (50k evals, log_gradients=true) | gradient_log |
|---|---|---|
| sixlet APS-C champion | `experiments/gradient_alignment/sixlet_apsc_champion_config.yaml` | `results/lensmega/gradient_alignment_sixlet_apsc_champion_20260510_034307/gradient_log.npz` |
| octuplet APS-C champion | `experiments/gradient_alignment/octuplet_apsc_champion_config.yaml` | `results/lensmega/gradient_alignment_octuplet_apsc_champion_20260510_034318/gradient_log.npz` |

Both seeded by their respective REFSWEEP CHAMPION configs
(`configs/ablations/sixlet_REFSWEEP2_CHAMPION.yaml`,
`configs/ablations/octuplet_REFSWEEP_CHAMPION.yaml`),
with iterations cut to 50k and gradient logging enabled.
Population size 24 / 32; descriptor `Seidel[S2, S3]`; APS-C image circle;
F# enforcement off; intersect penalty weight 8.0.

## Headline numbers (S2/S3, mean |cos(−∇L, ∇M_j)|)

| topology | sensor | θ-dim | n gens | mean &#124;cos&#124; S2 | mean &#124;cos&#124; S3 | random null mean | random max (200) | frac random ≥ Seidel |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| triplet | small | 19 | 1389 | 0.616 | 0.701 | 0.183 | 0.488 | 0/200 |
| sixlet (old, small sensor) | small | 37 | 2778 | 0.859 | 0.697 | 0.137 | 0.401 | 0/200 |
| septuplet APS-C | APS-C | 41 | 2084 | 0.45 | 0.47 | 0.125 | 0.245 | 0/200 |
| **sixlet champion** | **APS-C** | **37** | **2084** | **0.501** | **0.478** | **0.136** | **0.271** | **0/200** |
| **octuplet champion** | **APS-C** | **45** | **1563** | **0.414** | **0.423** | **0.121** | **0.232** | **0/200** |

(Triplet / old sixlet / septuplet rows are reproduced from the
2026-05-04 session export; the bottom two rows are new this session.)

## 5-Seidel post-hoc (replay; same gradient log)

`posthoc_all_seidels.py` replays each logged θ and computes
`∇_θ S_k` for k=1..5 against the *stored* `∇_θ L`. Recomputed ∇L
matches stored to ≤ 3e-4 relative error (deterministic ray sampler);
recomputed S2/S3 reproduce the analyse-pass values exactly.

### Sixlet APS-C champion (n=2084, θ-dim=37)

| Seidel | mean signed cos | median signed cos | mean &#124;cos&#124; | median &#124;cos&#124; | frac &#124;cos&#124;>0.3 |
|--------|---:|---:|---:|---:|---:|
| S1 spherical | −0.332 | −0.447 | **0.514** | 0.524 | 0.753 |
| S2 coma | −0.005 | −0.043 | **0.501** | 0.496 | 0.723 |
| S3 astigmatism | −0.259 | −0.322 | **0.478** | 0.458 | 0.660 |
| S4 Petzval | −0.295 | −0.464 | **0.532** | 0.537 | 0.772 |
| S5 distortion | +0.040 | +0.117 | **0.422** | 0.394 | 0.609 |

Range 0.42–0.53. S4 marginally on top, S5 (distortion) lowest.

### Octuplet APS-C champion (n=1563, θ-dim=45)

| Seidel | mean signed cos | median signed cos | mean &#124;cos&#124; | median &#124;cos&#124; | frac &#124;cos&#124;>0.3 |
|--------|---:|---:|---:|---:|---:|
| S1 spherical | −0.286 | −0.313 | **0.408** | 0.381 | 0.597 |
| S2 coma | +0.001 | +0.010 | **0.414** | 0.380 | 0.605 |
| S3 astigmatism | −0.229 | −0.267 | **0.423** | 0.382 | 0.602 |
| S4 Petzval | −0.267 | −0.410 | **0.468** | 0.476 | 0.714 |
| S5 distortion | +0.033 | +0.063 | **0.353** | 0.311 | 0.512 |

Range 0.35–0.47. Same shape as sixlet — S4 highest, S5 lowest, S2/S3 mid-pack.

## Random-projection control (200 directions of matched dim)

| topology | random null mean ± std | 95% range | random max (200) | Seidel S2 | Seidel S3 | frac ≥ S2 | frac ≥ S3 |
|---|---|---|---:|---:|---:|---:|---:|
| sixlet APS-C champion | 0.136 ± 0.043 | [0.068, 0.236] | 0.271 | 0.501 | 0.478 | 0/200 | 0/200 |
| octuplet APS-C champion | 0.121 ± 0.035 | [0.062, 0.200] | 0.232 | 0.414 | 0.423 | 0/200 | 0/200 |

Same conclusion as prior runs — Seidel alignment is highly specific,
no random low-D subspace approaches it.

## Sign pattern

- **S2** mean signed cos ≈ 0 (mixed) on both APS-C topologies — coma
  changes can move with or against ∇L depending on the local geometry.
- **S3** strongly negative mean signed cos (−0.23 to −0.26) on both —
  same Seidel-sign-convention artefact as in the prior runs (CMA-MEGA
  invariant; the emitter samples gradient coefficients with both
  signs).
- S1 and S4 also lean negative; S5 leans slightly positive.

## What this says for §3.2 of the paper

**1. The "partial alignment" claim is the right framing.**
APS-C sixlet/octuplet sit at S2/S3 mean |cos| ≈ 0.41–0.50 — 3.4× and
3.7× the random null (and 0/200 random projections approach), but
nowhere near unity. Exactly the regime the paragraph argues for:
descriptor exploration tends to keep candidates physically useful
(non-zero alignment with ∇L) while preserving genuine
descriptor-driven diversity (alignment well below 1.0). The pattern
is consistent across triplet → sixlet (small) → septuplet APS-C →
sixlet APS-C → octuplet APS-C.

**2. Justification for choosing S2/S3.**
The 5-Seidel breakdown on the APS-C topologies is relatively flat
(range 0.35–0.53 sixlet, 0.35–0.47 octuplet). All five Seidels are
above the random null and well within a "partial alignment" band; no
single coefficient dominates by a margin that would force the choice.
S2 (coma) and S3 (astigmatism) are mid-pack — neither the most nor
least aligned — which puts them squarely in the regime the §3.2
narrative wants:

- aligned enough that descriptor moves stay on the manifold of
  physically useful designs, and
- decoupled enough that they expose distinct off-axis aberration
  profiles (1st- vs 2nd-order field dependence — different optical
  signatures, not just rescalings of one another).

S1 (spherical) tracks ∇L most closely on the small-sensor topologies
(triplet 0.65, sixlet-small 0.90), so using it as a BC would buy less
behavioural diversity. S5 (distortion) is consistently the least
aligned (0.40–0.45) and is geometric position rather than blur — a
poor proxy for image-quality-relevant variation. S4 (Petzval / field
flatness) is competitive, especially for the wider field — a
reasonable alternative future direction. **For S2/S3 the soft
justification is: physically meaningful, separable (different field
powers), and partially aligned — exactly the property §3.2 builds on.**
The numbers don't force S2/S3 over every alternative, and the
paper doesn't need them to — that's the honest reading.

## Reproduce

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hyperlens
cd /nas/longleaf/home/hisham26/Desktop/hyperlens

# 1. Run (each ~10–12 min on 32 CPU)
python3 -m optical_rl.qd.lensmega.run \
  --config experiments/gradient_alignment/sixlet_apsc_champion_config.yaml
python3 -m optical_rl.qd.lensmega.run \
  --config experiments/gradient_alignment/octuplet_apsc_champion_config.yaml

# 2. Per-generation cosines
python3 experiments/gradient_alignment/analyze_gradients.py \
  --log <run_dir>/gradient_log.npz \
  --out experiments/gradient_alignment/cosine_similarities_<tag>.npz \
  --labels S2,S3

# 3. Random-projection null
python3 experiments/gradient_alignment/random_projection_compare.py \
  --log <run_dir>/gradient_log.npz \
  --matrix-out experiments/gradient_alignment/random_projection_matrix_<tag>.npy \
  --out experiments/gradient_alignment/random_projection_distribution_<tag>.npz \
  --cosine-path experiments/gradient_alignment/cosine_similarities_<tag>.npz

# 4. 5-Seidel post-hoc (no new search)
PYTHONPATH=.:DeepLens python3 experiments/gradient_alignment/posthoc_all_seidels.py \
  --log <run_dir>/gradient_log.npz \
  --config experiments/gradient_alignment/<topology>_apsc_champion_config.yaml \
  --out experiments/gradient_alignment/all_seidels_<tag>.npz
```

## Files produced

| file | content |
|---|---|
| `sixlet_apsc_champion_config.yaml`, `octuplet_apsc_champion_config.yaml` | run configs (50k evals, log_gradients) |
| `cosine_similarities_sixlet_apsc_champion.npz`, `…octuplet…` | per-generation S2/S3 cosines |
| `random_projection_matrix_*.npy`, `random_projection_distribution_*.npz` | published random matrix + 200-direction null |
| `all_seidels_sixlet_apsc_champion.npz`, `…octuplet…` | 5-Seidel post-hoc cosines |
| `results/lensmega/gradient_alignment_{sixlet,octuplet}_apsc_champion_20260510_034*/gradient_log.npz` | raw θ + gradient logs |
