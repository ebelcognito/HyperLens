# Supplementary S?: Why the f-number target is left unconstrained

## Setup — a matched A/B on the constraint

To justify leaving the f-number unconstrained in the main archive, we run a
controlled ablation in which the *only* change is the f-number constraint. For
both the sixlet and octuplet systems we take the exact discovery configuration
used in the main text — same starting lens template, same full-DOF
parameterization, identical Seidel $(S_2,S_3)$ archive (sixlet $2048^2$,
octuplet $512^2$) with the same calibrated bounds, the same 100k-evaluation
CMA-MEGA budget, seed 42 — and re-run it with `enforce_fnum: true`. With
`enforce_fnum: false` the f-number penalty is **gated off and absent from the
objective** (so `f_number_target`/`fnum_penalty_weight` are inert; f/# is a free
byproduct of RMS minimization at fixed EFL). The enforced arms add a quadratic
relative-error penalty at weight $w_{f/\#}\in\{1,10\}$, where $w{=}10$ matches
the (hard-held) EFL penalty weight. Each archive's top-100-by-fitness designs
are refined to convergence with L-BFGS, exactly as in the main pipeline.

Three arms per system: **none** (`enforce_fnum:false`, the main-text pool),
**enforced $w{=}1$**, **enforced $w{=}10$**.

## What enforcement does, and what it costs (top-100 by fitness)

| system | constraint | median f/# | % in [3.6,4.4] | median RMS (µm) | **best RMS (µm)** | sharp $n_{<20\mu m}$ | ε-distinct$_{<20\mu m}$ |
|---|---|---|---|---|---|---|---|
| **sixlet** | none | 5.18 | 7% | 19.0 | **2.1** | **54** | **10** |
| | enforced $w{=}1$ | 4.09 | 75% | 61.6 | 10.2 | 1 | 1 |
| | enforced $w{=}10$ | 3.98 | **86%** | 47.8 | 16.0 | 1 | 1 |
| **octuplet** | none | 4.09 | 48% | 33.9 | **12.2** | **5** | **2** |
| | enforced $w{=}1$ | 4.17 | 70% | 79.0 | 21.3 | 0 | 0 |
| | enforced $w{=}10$ | 4.04 | **82%** | 97.0 | 20.2 | 0 | 0 |

*On-spec band f/# ∈ [3.6, 4.4]; sharp ≡ RMS < 20 µm. $n_{<20\mu m}$ = number of
top-100 designs under 20 µm; ε-distinct$_{<20\mu m}$ = those mutually distinct at
ε = 0.30 in normalized θ (diverse **and** sharp).*

Three observations:

1. **Enforcement pins the aperture (it works).** The in-band fraction rises from
   7%→86% (sixlet) and 48%→82% (octuplet); the median f/# moves onto f/4.
   (`analysis/fig_fnum_hist.png`.)

2. **The sharpness cost is steep.** Pinning f/4 roughly doubles the best
   achievable polychromatic RMS (sixlet 2.1→16 µm; octuplet 12→20 µm) and
   triples the median. The optimizer's genuinely sharp basins sit at *slower*
   apertures; a hard f/4 constraint forecloses them. (`analysis/fig_fnum_vs_rms.png`:
   the unconstrained sixlet pool is slow-and-sharp, f/4.5–6 down to 2 µm; the
   enforced pool is on-spec but lifted above the 20 µm line.)

3. **Diverse, sharp designs collapse.** Among the top-100, the number under
   20 µm falls 54→1 (sixlet) and 5→0 (octuplet), and the count of *mutually
   distinct* sub-20 µm designs falls 10→1 and 2→0. *(Caveat on metric: the
   ε-distinct count must be taken over designs that clear the RMS bar. Ungated,
   the enforced arms show more ε-distinct designs — 17→20–28 — but that merely
   reflects parameter-space scatter of optically poor designs under the penalty;
   it is not useful diversity. We therefore gate on RMS < 20 µm. See
   `diversity_gated.py` for the threshold sweep, which confirms the direction at
   every bar.)*

## Takeaway sentence (for the main text)

> We leave the f-number unconstrained during discovery (`enforce_fnum:false`;
> the penalty is absent from the objective). A matched ablation that turns it on
> (weights 1 and 10) shows enforcement does pin the aperture — the fraction of
> refined designs within f/4 ± 0.4 rises from 7% to 86% for the sixlet — but at
> the cost of both sharpness (best polychromatic RMS roughly doubles, 2.1→16 µm
> sixlet; 12→20 µm octuplet) and diverse-sharp yield (the number of mutually
> distinct sub-20 µm designs falls 10→1 sixlet, 2→0 octuplet). Leaving f/#
> unconstrained and re-filtering the archive post-hoc therefore recovers more
> spec-compliant, sub-20 µm designs at better sharpness, which is why we adopt it
> (§S9).

## Provenance / reproduce

```
PY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
$PY experiments/fnum_enforcement/analyze.py          # table + f/# & RMS figs
$PY experiments/fnum_enforcement/diversity_gated.py  # quality-gated ε-distinct
```
- Enforced runs: `results/lensmega/{sixlet_REFSWEEP2_G_grid2048_seed42,octuplet_REFSWEEP_D_long_lbfgs_seed42}_ENF_W{1,10}_<ts>/`
  (search archives + `refined_top100_*.json`)
- Unconstrained baselines: `results/lensmega/sixlet_REFSWEEP2_G_grid2048_seed42_20260509_043849`,
  `results/lensmega/octuplet_REFSWEEP_D_long_lbfgs_seed42_20260509_121606`
- Note: each top-100 L-BFGS refine took ~25 min; a top-500 refine (≈2 h) would
  not fit a 2 h node — top-100 is the right granularity for this ablation.
```
