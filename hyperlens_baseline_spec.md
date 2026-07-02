# HyperLens SA+GA+Refine Baseline: Implementation Specification

## Purpose

This document specifies the SA+GA+Refine baseline experiment for HyperLens. It is intended for an implementation agent to execute the experiment and produce structured results that a separate writing agent will later consume for paper prose. The implementation agent should NOT write paper prose.

## 1. Strategic Frame (Context)

The baseline tests one claim: physically grounded differentiable behavior-space search (HyperLens) produces more diverse high-quality designs at fixed per-design compute than parameter-space stochastic search with gradient refinement on loss alone (the SA+GA+Refine method class).

The baseline is a faithful representative of the parameter-space stochastic search method class. It is NOT a reimplementation of OptiFusion. OptiFusion did not release code, and the claim is about the method class regardless of scheduling details. The implementation agent should configure the baseline as a strong reasonable instance of the class, not as the weakest possible version (no strawmanning) and not as a precise replica of OptiFusion (out of scope).

LDG-EA is not implemented. The descriptor-quality argument against LDG-EA is handled by HyperLens's existing §3.5 alignment experiment and §5.2 random-projection ablation. No work needed on that axis.

Curriculum learning is deferred. The implementation agent should NOT do any curriculum learning work in this run.

## 2. SA+GA+Refine Algorithm

```
Inputs: parameter dimension n, population size m, generations G,
        SA temperature schedule (T_0, T_final), SA step size sigma,
        GA selection ratio mu/m, GA elite ratio nu/m, mutation fraction p,
        ADAM steps per refinement K_adam, ADAM learning rate alpha,
        diversity filter distance d_min, total ray-trace budget B_total

Initialize: population P of m random parameter vectors in feasible domain Theta
Track: total_ray_traces = 0

For generation g = 1 to G:
    # Stage 1: SA pass
    For each individual theta_i in P:
        Propose theta_i' = theta_i + sigma * eps_g * normal(0, I)
        Clip to bounds
        Evaluate L(theta_i') and increment total_ray_traces
        Compute Delta_L = L(theta_i') - L(theta_i)
        Accept theta_i' with probability min(1, exp(-Delta_L / T_g))
    
    # Stage 2: GA selection with diversity filter
    Sort P by fitness ascending
    Greedy-select Parents: take best, then iteratively add next-best whose
        Euclidean distance to all already-selected exceeds d_min, until mu chosen
    
    # Stage 3: ADAM refinement on Parents (gradient on L only, no descriptor gradients)
    For each Parent theta_j:
        For step k = 1 to K_adam:
            Compute gradient g_L = grad_theta L(theta_j) and increment ray traces
            Apply ADAM update with lr alpha
    
    # Stage 4: Elite selection
    Sort Parents by post-ADAM fitness
    Elites = top nu individuals
    
    # Stage 5: Mutation
    For each non-Elite Parent, randomly perturb fraction p of variables
        within bounds
    
    # Stage 6: Merge for next generation
    P_next = Elites + Mutated_Parents + random_fill_to_m()
    P = P_next
    
    # Stop if budget exhausted
    If total_ray_traces >= B_total: break

# Final refinement (matched to HyperLens's procedure)
For each individual in final P with L below refinement threshold:
    Apply L-BFGS-B polish using HyperLens's exact refinement code

Return: refined design set R
```

Critical constraints:

- The ADAM refinement uses gradients of L only. No descriptor gradients. No Seidel computation inside the baseline.
- The L-BFGS-B final polish must use HyperLens's exact refinement code (same procedure as §3.4 of the HyperLens paper). This ensures both methods get the same refinement stage; any quality difference comes from the upstream search machinery.
- The diversity filter operates in normalized parameter space, not behavior space.

## 3. Configuration

The user has a config file with hyperparameter values. Consume that config rather than hardcoding values. Required fields:

- `population_size` (m)
- `generations_max` (G, with early termination on budget)
- `sa_temperature_initial` (T_0)
- `sa_temperature_final` (T_final)
- `sa_temperature_schedule` (geometric or linear)
- `sa_step_size_initial` (sigma)
- `sa_step_size_decay` (multiplicative per generation)
- `ga_selection_ratio` (mu / m)
- `ga_elite_ratio` (nu / m)
- `ga_diversity_min_distance` (d_min, normalized parameter space)
- `mutation_fraction` (p)
- `adam_steps_per_refinement` (K_adam)
- `adam_learning_rate` (alpha)
- `total_ray_trace_budget` (B_total, MUST match HyperLens triplet run)
- `refinement_fitness_threshold` (same as HyperLens, e.g. -0.2)

If a value is missing from the config, use OptiFusion's reported value where applicable (m=4000 scaled to budget, mu/m=0.06, nu/m=0.02, p=0.3) and log every default used. Do not silently override the config.

## 4. Compute Matching Protocol

The baseline must run at the same total ray-trace budget as the HyperLens triplet run reported in §5.2 of the paper. Confirm the budget by inspecting the HyperLens triplet run logs before starting (do not assume; check). The current §5.2 ablation uses 50,000 evaluations at 256 rays per evaluation, giving 12,800,000 ray traces, but verify this against the actual run.

Required:

1. Instrument every ray trace (forward fitness eval, gradient eval that internally traces rays, refinement traces).
2. Terminate the outer loop as soon as `total_ray_traces >= B_total`.
3. Run the same final L-BFGS-B refinement that HyperLens uses on the resulting candidates. Track refinement ray traces separately and confirm both methods spend comparable amounts on refinement.

Run the experiment on the **triplet system** first (19-D parameter vector, same problem instance as §5.2). Use the same target spec (50 mm EFL, f/4, 11.8° HFOV, polychromatic), same physical bounds, same seed batch size.

Do NOT run on the six-element or seven-element system in this pass. Triplet first, validate, then scaling experiments come later.

## 5. Metrics

Compute the same metrics that appear in HyperLens Table 3 (the §5.2 ablation), so the SA+GA+Refine column can be added directly. All metrics are method-agnostic: they survive the question "would this metric mean the same thing if applied to OptiFusion's output?"

Required metrics:

- Best raw fitness
- Refined RMS distribution: best, 10th percentile, median, mean, worst
- Count of elites below thresholds: <10 µm, <5 µm, <3 µm
- Improvement ratio: median and mean (raw / refined fitness)
- Structural diversity: unique sign-pattern families, total tracked elites
- Wall-clock time

Do NOT compute or report:

- Behavior-cell coverage
- Anything that uses Seidel coordinates
- CMA restart count (HyperLens-specific)

## 6. Results Handoff Format

Produce results in `/mnt/user-data/outputs/sa_ga_refine_results/` with the following files:

**`metrics.json`** — All metrics from §5 in a structured JSON keyed by metric name. Include both HyperLens triplet results (re-extracted from existing logs for direct comparison) and SA+GA+Refine results. Schema:

```json
{
  "experiment": "sa_ga_refine_vs_hyperlens_triplet",
  "compute_budget": {
    "ray_traces_target": 12800000,
    "ray_traces_actual_hyperlens": ...,
    "ray_traces_actual_sa_ga": ...
  },
  "hyperlens": {
    "best_raw_fitness": ...,
    "refined_rms_best_um": ...,
    "refined_rms_p10_um": ...,
    "refined_rms_median_um": ...,
    "refined_rms_mean_um": ...,
    "refined_rms_worst_um": ...,
    "count_below_10um": ...,
    "count_below_5um": ...,
    "count_below_3um": ...,
    "improvement_ratio_median": ...,
    "improvement_ratio_mean": ...,
    "unique_sign_families": ...,
    "tracked_elites": ...,
    "wall_clock_seconds": ...
  },
  "sa_ga_refine": { ... same fields as hyperlens ... }
}
```

**`refined_designs_sa_ga.json`** — Parameter vectors for all refined SA+GA+Refine designs above the fitness threshold, with their fitness, sign-pattern family, and refined RMS. The writing agent will pick illustrative examples from this.

**`refined_designs_hyperlens.json`** — Same format for HyperLens triplet refined designs, re-extracted for direct side-by-side use.

**`hyperparameters.json`** — Full hyperparameter dump for the SA+GA+Refine run (every value from the config plus any defaults applied). The writing agent will need this for an appendix table.

**`timing.log`** — Wall-clock timing for the SA+GA+Refine run, broken down by stage (SA, ADAM, GA selection, mutation, refinement).

**`README.md`** — Short summary of what each file contains and any caveats from the run (e.g., "ADAM step count was reduced from config value 100 to 50 due to budget constraints"). No paper prose, just an honest run report.

## 7. Risk Mitigation (Experimental Design)

These are about getting the experiment right, not about defending it in prose.

**Avoid strawmanning the baseline.** Hyperparameters should be reasonable representatives of the method class, drawn from OptiFusion's reported values where applicable. If the SA temperature schedule looks broken (e.g., temperature drops too fast and SA accepts almost nothing after generation 2), tune it on a small held-out problem before the main run. Log any tuning done.

**Method-agnostic metrics.** All §5 metrics must be computable from parameter vectors and raw fitness alone. If you find yourself needing Seidel coordinates or archive structure to compute a metric, that metric is wrong for this comparison.

**Optional sensitivity analysis.** If time permits after the main run, vary three key hyperparameters (population size, mutation fraction, ADAM learning rate) over a 3x3 grid and confirm the gap (whatever it is) persists across reasonable settings. This is optional but produces strong defensive evidence. If you do this, save results in a `sensitivity/` subdirectory with the same JSON schema.

## 8. Action Items

In order:

1. Read this document fully.
2. Read the user's SA+GA+Refine config file (path to be provided by user).
3. Verify the HyperLens triplet ray-trace budget by inspecting actual run logs. Do not assume.
4. Implement the SA+GA+Refine baseline per §2 of this document, using the user's config values.
5. Instrument ray-trace counting per §4.
6. Run the baseline on the triplet system at matched compute.
7. Compute the metrics in §5.
8. Produce the handoff artifacts in `/mnt/user-data/outputs/sa_ga_refine_results/` per §6.
9. (Optional) Run sensitivity analysis per §7.
10. Stop. Hand off to the writing agent.

## 9. Things NOT to Do

- Do NOT reimplement OptiFusion's specific schedule.
- Do NOT use Seidel coordinates, archive structure, or any HyperLens-specific machinery inside the baseline.
- Do NOT compute behavior-cell coverage or any descriptor-aware metric.
- Do NOT write paper prose. Results go in the handoff artifacts only.
- Do NOT silently override config values. Log every default applied.
- Do NOT run on six-element or seven-element systems in this pass. Triplet only.
- Do NOT touch curriculum learning. Deferred.
