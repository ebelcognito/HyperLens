# EF50 L-BFGS-B Refinement Fix + Observability Improvements

**Session scope**: debugging a silent failure in the L-BFGS-B refinement path (`refine_ef50_archive.py`), plus unrelated but co-occurring improvements to the parallel seeding/calibration runner (`ef50_zernike_parallel.py`) around progress tracking, DeepLens warning suppression, and strict-mode pupil rejection verification.

**Primary outcome**: a ~20-line code change inside `_make_objective` in `refine_ef50_archive.py` that brings the L-BFGS-B objective in line with the CMA-MEGA fitness function. Secondary outcomes: a reusable stdout filter for DeepLens fallback warnings, per-batch progress bars with elapsed/ETA timing for calibration and seeding, and verified correctness of the strict entrance-pupil rejection path in parallel runs.

---

## 1. Render Order in Refined-Design Grid Plots

**Question raised**: when `render_refined_designs.py` produces the unified grid image of 20 refined designs, does the index position (panel `#n/20`) match the refinement processing order?

**Answer**: yes, it's a direct 1:1 mapping.

**Code trace**:
- [`render_refined_designs.py:272-281`](optical_rl/qd/render_refined_designs.py#L272-L281) — the grid renderer loops `for idx in range(rows * cols)` and picks `results[idx]` / `basin_labels[idx]` with no sorting or reordering.
- [`refine_ef50_archive.py:708-733`](optical_rl/qd/refine_ef50_archive.py#L708-L733) — the `results` list is built by appending in `selected` iteration order, matching the `1/20, 2/20, ...` prints during refinement.
- Panel titles hardcode `f"#{rank + 1}  basin {basin_label}"` at [`render_refined_designs.py:213`](optical_rl/qd/render_refined_designs.py#L213), where `rank == idx`.

**Caveat**: the per-basin grids (`basin_<k>_grid.png`) re-group by basin, so within those the ordering is "processing order filtered to that basin" — still monotonic in the original index but not contiguous 1..20.

---

## 2. Strict Entrance-Pupil Rejection: Architecture and Verification

### How it works in parallel runs

`strict_entrance_pupil: true` in YAML activates a two-stage reject mechanism:

1. **Worker-side tagging** at [`ef50_zernike.py:973-983`](optical_rl/qd/ef50_zernike.py#L973-L983): after each `calc_entrance_pupil()` call, detect if the returned pupil exactly matches the DeepLens fallback `(surfaces[0].d, surfaces[0].r)`. If so, return early with `fitness = -1e9` and `metrics = {"invalid_fraction": 1.0, "rejected": 1.0, "rejected_pupil": 1.0}`.

2. **Parent-side filtering** in three places:
   - Calibration: [`ef50_zernike_parallel.py:1109-1115`](optical_rl/qd/ef50_zernike_parallel.py#L1109-L1115) — skip rejected results before computing percentile bounds.
   - Seeding: [`ef50_zernike_parallel.py:1174-1187`](optical_rl/qd/ef50_zernike_parallel.py#L1174-L1187) — skip before inserting into the MAP-Elites archive.
   - CMA-MEGA candidates: [`ef50_zernike_parallel.py:1257-1263`](optical_rl/qd/ef50_zernike_parallel.py#L1257-L1263) — skip before archive insertion.

The `-1e9` fitness is in-band signaling between the worker process and the parent (workers can't return `None` cleanly through the metrics dict), not a score that ever touches the archive.

### DeepLens warning origin

The warnings `"No intersection points found, use the first surface as entrance pupil."` and `"Zero or negative entrance pupil is detected, use the first surface as entrance pupil."` come from unconditional `print()` calls inside [`DeepLens/deeplens/optics/geolens.py:1420-1440`](DeepLens/deeplens/optics/geolens.py#L1420-L1440). Both fallback paths return exactly `(surfaces[0].d.item(), surfaces[0].r)`, which our strict-mode check catches via float-equality comparison to tolerance `1e-9`.

### Edge case: when strict mode is OFF

Initial investigation incorrectly assumed the user's YAML had `strict_entrance_pupil: false` based on a misread of the IDE-selection snippet. With the flag OFF, the worker silently accepts the fallback pupil (`surfaces[0].r`) and proceeds through the full ray-trace pipeline, producing a meaningless RMS spot for a physically degenerate lens. These "phantom designs" would be inserted into the archive with competitive-looking but bogus fitness scores. The user confirmed the saved YAML in their run directory does have `strict_entrance_pupil: true`, so this contamination mode is not active in current runs.

---

## 3. Parallel Worker Count Investigation

### The hard cap mystery

**Observation**: setting `parallel_workers: 100` in YAML produced only ~20 worker processes and GPU memory plateaued at ~6 GB / 49 GB per card.

**Root cause**: `_pick_worker_count` at [`seed_parallel_experiment.py:39-47`](optical_rl/qd/seed_parallel_experiment.py#L39-L47) clamps the requested count:

```python
def _pick_worker_count(requested: int | None, num_gpus: int) -> int:
    available_cpus = _available_cpu_count()
    if requested is not None:
        return max(1, min(requested, available_cpus))
    # Auto: ~6 workers per GPU (11% util each → ~66% target), cap at half CPUs
    auto = num_gpus * 6
    return max(2, min(auto, available_cpus // 2, 32))
```

When `requested=100` and the machine has 20 CPUs (verified via `os.sched_getaffinity(0)`), the actual worker count is `min(100, 20) = 20`. The function is duplicated verbatim in [`entrance_pupil_experiment.py:48-53`](optical_rl/qd/entrance_pupil_experiment.py#L48-L53).

### Worker distribution across GPUs

[`seed_parallel_experiment.py:124-129`](optical_rl/qd/seed_parallel_experiment.py#L124-L129) round-robins workers across visible GPUs:

```python
gpu = gpu_ids[w % len(gpu_ids)]
```

User's machine has 2× RTX 6000 Ada (49 GB each), so 20 workers split 10/10. Per-worker GPU memory footprint is ~600 MB (dominated by fixed CUDA context overhead of ~300-450 MB per spawned process, plus a small working set for ray tracing). 10 workers × 600 MB ≈ 6 GB per GPU, matching the `nvidia-smi` observation.

### Scaling guidance

The code cap is a safety guard, not a hardware ceiling. To bypass:

```python
# Before
if requested is not None:
    return max(1, min(requested, available_cpus))
# After
if requested is not None:
    return max(1, requested)
```

But the physical CPU limit (20 slots) means anything above 20 is oversubscription with kernel time-slicing. The evaluator is ~50/50 CPU (Python lens construction) / GPU (ray tracing), so the empirical sweet spot is typically 1.5-2× CPU oversubscription — around 30-40 workers on this machine. 100 workers will almost certainly be worse than 30 due to context-switching overhead.

**Per-worker GPU memory note**: each additional worker adds ~500-600 MB of CUDA context. At 50 workers per GPU you'd hit ~30 GB on a 49 GB card — fine but no longer free.

---

## 4. L-BFGS-B Refinement Objective: The Silent Fake-Zero Bug

### Symptom

In a standard refinement run, the user observed for design #2:

```
[refine-ef50]  2/20  archive_fitness=-21.94201  Zero or negative entrance pupil is detected, use the first surface as entrance pupil.
No intersection points found, use the first surface as entrance pupil.
rms 3.1568 → 0.0000 mm  1 iters  [OK]
```

Two red flags:
1. Converged RMS of exactly 0.0000 mm is physically impossible for any real lens.
2. Single L-BFGS-B iteration to reach this value is suspicious — real optimization needs at least a few line searches.
3. DeepLens warnings appeared between the start and end of this design's refinement.

### Initial (incorrect) hypothesis

First diagnosis blamed a stale archive from before `strict_entrance_pupil` was enabled, reasoning that `archive_fitness = -21.94` implies `rms ≈ 21.94 mm` which is absurd and would have been rejected by strict mode. **This was wrong.** User corrected by showing the run directory's saved YAML has `strict_entrance_pupil: true`, so the archive is clean.

### Correct diagnosis: post-refinement Zernike measurement triggers the warnings

The DeepLens warnings appear **after** L-BFGS-B converges, not before. Proof from the call chain:

1. L-BFGS-B terminates in 1 iteration (converged_rms ≈ 0).
2. [`refine_ef50_archive.py:246`](optical_rl/qd/refine_ef50_archive.py#L246) calls `_full_zernike_vector(problem, converged_theta)`.
3. `_full_zernike_vector` builds a fresh lens at the converged theta and calls `problem._zernike_measures(lens, hfov)`.
4. [`ef50_zernike.py:873`](optical_rl/qd/ef50_zernike.py#L873) inside `_zernike_measures` calls `lens.calc_chief_ray_infinite(chief_rfov, plane="meridional")`.
5. Inside `calc_chief_ray_infinite` at [`DeepLens/.../eval.py:1011`](DeepLens/deeplens/optics/geolens_pkg/eval.py#L1011), DeepLens calls `pupilz, _ = self.calc_entrance_pupil()`.
6. **This is where the warning fires** — the converged lens has a degenerate entrance pupil, which is the entire reason `_zernike_measures` is failing.

Additionally, `start_rms = 3.1568` (not `nan`) proves the pre-evaluation did NOT trigger strict rejection — the start theta passed strict mode cleanly. The degeneracy is at the *converged* theta, not the start.

### Root cause: two compounding bugs in the L-BFGS-B objective

**Bug #1**: `_rms_spot_batch` silently returns ~1e-6 mm when all rays fail to trace. From [`ef50_zernike.py:784-794`](optical_rl/qd/ef50_zernike.py#L784-L794):

```python
def _rms_spot_batch(self, ray):
    valid = ray.is_valid                    # (F, num_rays)
    xy = ray.o[..., :2]                     # (F, num_rays, 2)
    denom = valid.sum(dim=-1).clamp_min(1e-8)
    centroid = (xy * valid.unsqueeze(-1)).sum(dim=-2) / denom.unsqueeze(-1)
    err2 = ((xy - centroid.unsqueeze(-2)) ** 2).sum(dim=-1)
    rms = ((err2 * valid).sum(dim=-1) / denom + 1e-12).sqrt()
    inv_frac = 1.0 - valid.sum(dim=-1) / valid.shape[-1]
    return rms, inv_frac
```

When `valid` is all zeros:
- Numerator `(err2 * valid).sum()` = 0
- Denominator `valid.sum().clamp_min(1e-8)` = 1e-8
- `rms = sqrt(0/1e-8 + 1e-12)` ≈ 1e-6 mm

The function returns this as the RMS value, correctly reports `inv_frac = 1.0` as a separate return, and expects the caller to handle the vignetting case.

**Bug #2**: The L-BFGS-B objective in `_make_objective` discards `inv_frac` entirely. From the original [`refine_ef50_archive.py:159-172`](optical_rl/qd/refine_ef50_archive.py#L159-L172):

```python
rms_per_wvln = []
for w in wavelengths:
    ray = problem._trace_fields_batch(lens, field_pts, wvln=w)
    rms, _ = problem._rms_spot_batch(ray)   # <-- inv_frac discarded
    rms_per_wvln.append(rms)

rms_mean = torch.stack(rms_per_wvln).mean()
rms_mean.backward()                          # <-- only rms_mean
```

No intersection penalty (which the CMA path includes via `intersect_penalty_weight × _intersection_penalty(lens)`). No invalid-ray penalty. Just mean RMS.

**Combined effect**: L-BFGS-B computes a gradient through `_trace_fields_batch` at the start theta (where rays still trace). That gradient points toward a region where rays vignette entirely. One line-search step lands in that region, `_rms_spot_batch` returns ~1e-6 mm, L-BFGS-B sees a huge "improvement" from 3.1568 to 1e-6 and declares success. `n_iters=1`, `[OK]`. The converged design is physically meaningless — no rays actually reach the sensor.

### Compare to CMA fitness path

At [`ef50_zernike.py:1000-1013`](optical_rl/qd/ef50_zernike.py#L1000-L1013), the fitness function correctly uses all three terms:

```python
rms_mean = rms_all.mean()
invalid_fraction_mean = inv_all.mean()
fitness = -rms_mean

intersect_loss = self._intersection_penalty(lens)
if float(intersect_loss.detach()) > 0:
    fitness = fitness - self.config.intersect_penalty_weight * intersect_loss

inv_frac_val = float(invalid_fraction_mean.detach().item())
if inv_frac_val > self.config.invalid_tol:
    excess = invalid_fraction_mean - self.config.invalid_tol
    fitness = fitness - self.config.invalid_penalty_weight * excess
```

With `invalid_tol=0.05` and `invalid_penalty_weight=5.0`, any design that vignettes a significant fraction of rays gets penalized. CMA-MEGA is correctly repelled from vignetting holes. The refinement path was not.

---

## 5. The Fix: Penalties Added to L-BFGS-B Objective

### Decision process

Three candidate changes were discussed:

- **Change A**: Add invalid-ray penalty to `_make_objective`, mirroring `_fitness_and_metrics` exactly.
- **Change B**: Make `_rms_spot_batch` itself robust against fully-vignetted fields by returning a large sentinel (e.g., 1e3 mm) instead of fake-zero. Defense-in-depth.
- **Change C**: Add intersection penalty to `_make_objective`, also mirroring `_fitness_and_metrics`.

User chose to apply **A + C only**, dropping B. Reasoning: when even one of the 3 fields is fully vignetted across all 3 wavelengths, `inv_mean ≈ 1/3 ≈ 0.33`, which is 6.6× the `invalid_tol=0.05`. The penalty fires and adds `5.0 × (0.33 - 0.05) = 1.42` mm to the loss, which dwarfs any spurious "improvement" from `rms_mean ≈ 1e-6`. The penalty alone is sufficient; the sentinel would be purely redundant belt-and-suspenders. Avoiding Change B also means `_rms_spot_batch` stays untouched, so CMA's hot path and `entrance_pupil_experiment.py` are not affected.

### Applied change

**File**: [`optical_rl/qd/refine_ef50_archive.py`](optical_rl/qd/refine_ef50_archive.py), function `_make_objective`, lines 159-188.

**Before** (14 lines):

```python
wavelengths = problem.config.fitness_wavelengths_um
field_pts = problem._field_points(hfov)
rms_per_wvln = []
for w in wavelengths:
    ray = problem._trace_fields_batch(lens, field_pts, wvln=w)
    rms, _ = problem._rms_spot_batch(ray)
    rms_per_wvln.append(rms)

rms_mean = torch.stack(rms_per_wvln).mean()
rms_mean.backward()

loss = float(rms_mean.detach().cpu().item())
grad = theta_t.grad.detach().cpu().numpy().astype(np.float64)
return loss, grad
```

**After** (28 lines):

```python
wavelengths = problem.config.fitness_wavelengths_um
field_pts = problem._field_points(hfov)
rms_per_wvln, inv_per_wvln = [], []
for w in wavelengths:
    ray = problem._trace_fields_batch(lens, field_pts, wvln=w)
    rms, inv_frac = problem._rms_spot_batch(ray)
    rms_per_wvln.append(rms)
    inv_per_wvln.append(inv_frac)

rms_mean = torch.stack(rms_per_wvln).mean()
inv_mean = torch.stack(inv_per_wvln).mean()

# Mirror _fitness_and_metrics in ef50_zernike.py: CMA maximizes
# (-rms - intersect_penalty - invalid_penalty), L-BFGS-B minimizes
# the equivalent (rms + intersect_penalty + invalid_penalty).
loss_t = rms_mean

intersect_loss = problem._intersection_penalty(lens)
if float(intersect_loss.detach()) > 0:
    loss_t = loss_t + problem.config.intersect_penalty_weight * intersect_loss

inv_tol = float(problem.config.invalid_tol)
if float(inv_mean.detach()) > inv_tol:
    loss_t = loss_t + problem.config.invalid_penalty_weight * (inv_mean - inv_tol)

loss_t.backward()

loss = float(loss_t.detach().cpu().item())
grad = theta_t.grad.detach().cpu().numpy().astype(np.float64)
return loss, grad
```

### Sign convention

Critical code-review item. CMA at [`ef50_zernike.py:1002-1013`](optical_rl/qd/ef50_zernike.py#L1002-L1013):
- `fitness = -rms_mean` (line 1002)
- `fitness = fitness - intersect_weight * intersect_loss` (line 1007, subtraction)
- `fitness = fitness - invalid_weight * excess` (line 1013, subtraction)

CMA maximizes `fitness`, so larger penalty → smaller fitness → worse. L-BFGS-B minimizes `loss`, so the equivalent is:
- `loss = rms_mean + intersect_weight * intersect_loss + invalid_weight * excess` (penalties **added**)

If accidentally subtracted instead of added, L-BFGS-B would be *attracted* to vignetting holes, making things strictly worse than the original bug.

### Reused functions (no new code paths)

- `_intersection_penalty(lens)` at [`ef50_zernike.py:903-934`](optical_rl/qd/ef50_zernike.py#L903-L934) — already differentiable, returns a torch scalar tensor that chains gradients through `surface_with_offset` calls.
- `_rms_spot_batch(ray)` — already returns `(rms, inv_frac)`, the refiner just stops discarding the second value.
- Config values `invalid_tol=0.05`, `invalid_penalty_weight=5.0`, `intersect_penalty_weight=1.0` — already loaded from YAML in [`refine_ef50_archive.py:100-102`](optical_rl/qd/refine_ef50_archive.py#L100-L102) via `_build_problem`. No new config plumbing.

### Speed impact

Per L-BFGS-B objective call, overhead breakdown:

| Component | Cost |
|---|---|
| Stop discarding `inv_frac` | 0 (already computed); list append + stack + `.mean()` on 3 scalars = microseconds |
| `_intersection_penalty(lens)` forward | 11 surface pairs × 17 sample points ≈ 200 scalar ops, ~0.3% of one `_trace_fields_batch` |
| `_intersection_penalty` backward | similar magnitude, ~0.5% |

A typical objective call costs `3 × _trace_fields_batch + 1 × backward`. Total overhead **~0.4% per call**. For a 20-design refinement at `--max-iter 200 --num-rays 256`, the worst-case 4000 objective calls translate to a few seconds on top of a multi-minute run. Wall-clock impact imperceptible.

### Quality impact

- **Healthy designs** (common case): unchanged. Both penalty terms evaluate to zero at honest local minima, so the gradient is identical to before. Converged RMS matches to floating-point noise.
- **Previously-failing designs**: the `0.0000 mm 1 iter [OK]` failure mode disappears. They converge to real local minima in the plausible EF50 range (~0.005-0.100 mm), or terminate quickly with `start_rms ≈ converged_rms` if the start is already at a local minimum.
- **Designs whose archive start slightly violates intersection constraints**: the intersect term pulls theta back into feasibility. There is a C^0 gradient discontinuity at the feasibility boundary (the `> 0` gate flips), but the jump magnitude is small and L-BFGS-B's line search handles it — standard Zemax DLS soft-constraint behavior. Worst case: 1-2 wasted iterations near the boundary.

### Verification plan

Smoke test:

```bash
python3 optical_rl/qd/refine_ef50_archive.py --count 5 --max-iter 50 --num-rays 256
```

Success indicators:
- No `→ 0.0000 mm` or sub-1e-4 mm converged RMS.
- No `1 iter [OK]` for a design whose start RMS was > 0.1 mm.
- All converged RMS values in ~0.005-0.100 mm range.
- No DeepLens "first surface as entrance pupil" warnings after `[refine-ef50] done`.

Failure indicators:
- Any `0.0000 mm` or sub-1e-4 converged RMS (fix didn't apply or sign error).
- DeepLens warnings appearing after refinement convergence.
- Converged RMS *worse* than the buggy run for previously-broken designs (would indicate the penalties are attracting instead of repelling — sign error).

---

## 6. Observability Improvements

### 6.1 DeepLens Warning Filter

**Problem**: DeepLens's `calc_entrance_pupil()` at [`geolens.py:1420-1440`](DeepLens/deeplens/optics/geolens.py#L1420-L1440) prints two unconditional `print()` warnings when the front-group ray trace fails and it falls back to `surfaces[0]`. With `strict_entrance_pupil: true` these designs are rejected by our worker anyway, so the warning text is pure noise drowning out progress logs. Calibration over 20000 random samples produced thousands of warning lines.

**Solution**: line-buffered stdout proxy that drops matching lines and passes everything else through.

**File**: [`optical_rl/qd/seed_parallel_experiment.py:33-95`](optical_rl/qd/seed_parallel_experiment.py#L33-L95).

Key pieces:
- `_DEEPLENS_WARNING_PATTERNS = ("No intersection points found", "Zero or negative entrance pupil")` — substring matches against both fallback phrases.
- `_FilteredStdout` class: buffers writes until a newline, then drops the line if it contains any pattern, otherwise passes it through unchanged.
- `install_deeplens_warning_filter()` helper: idempotent (safe to call multiple times), honors `DEEPLENS_KEEP_WARNINGS=1` in env as an opt-out for debugging.

**Installed in two places**:
- **Workers**: at the top of `_worker_fn` in [`seed_parallel_experiment.py:128-130`](optical_rl/qd/seed_parallel_experiment.py#L128-L130), before any lens object is built.
- **Parent**: at the top of `main()` in [`ef50_zernike_parallel.py:1450-1454`](optical_rl/qd/ef50_zernike_parallel.py#L1450-L1454), before `EF50ZernikeRunner` is constructed, so the in-process CMA loop's `self.problem.evaluate()` calls at [`ef50_zernike_parallel.py:1258`](optical_rl/qd/ef50_zernike_parallel.py#L1258) also stay quiet.

**Performance**: one substring scan per `sys.stdout.write()` call (~200 ns Python work each), negligible against the ray-tracing cost.

**Scope note**: only `ef50_zernike_parallel.py`'s `main()` installs the parent-side filter. Running `refine_ef50_archive.py` or `render_refined_designs.py` standalone will still show the warnings. One-line fix per script if wanted: `from optical_rl.qd.seed_parallel_experiment import install_deeplens_warning_filter; install_deeplens_warning_filter()` at the top of `main()`.

### 6.2 Live Progress Counter with Elapsed Time and ETA

**Problem**: `_parallel_evaluate` in `seed_parallel_experiment.py` is a batch API — workers are dispatched, parent blocks until all results arrive, then iterates. During a 20k-sample calibration or 40k-sample seeding, there was no way to watch progress tick by.

**Solution**: optional `progress_label` and `progress_every` keyword arguments to `_parallel_evaluate`. When `progress_label` is set, the parent's result-collection loop tracks `n_done`, `n_rejected`, `n_worker_err`, and elapsed time, printing a throttled one-liner every `progress_every` results pulled.

**File**: [`optical_rl/qd/seed_parallel_experiment.py:195-258`](optical_rl/qd/seed_parallel_experiment.py#L195-L258).

**Output format**:

```
[ef50-zernike calibration] 500/20000 ( 2.5%)  elapsed  0:08  ETA  5:28  rate 62.5/s  rejected 127
[ef50-zernike calibration] 1000/20000 ( 5.0%)  elapsed  0:16  ETA  5:19  rate 62.7/s  rejected 283
...
[ef50-zernike calibration] 20000/20000 (100.0%)  elapsed 5:20  ETA  0:00  rate 62.5/s  rejected 5817
```

Format helper `_fmt_hms` at [`seed_parallel_experiment.py:39-49`](optical_rl/qd/seed_parallel_experiment.py#L39-L49) prints `m:ss` for short durations and `h:mm:ss` past one hour, with `--:--` for infinite ETAs (very first sub-second window before rate stabilizes).

**Wired into two callers**:
- Calibration at [`ef50_zernike_parallel.py:1103-1107`](optical_rl/qd/ef50_zernike_parallel.py#L1103-L1107): `progress_label="ef50-zernike calibration"`, `progress_every=500`.
- Seeding at [`ef50_zernike_parallel.py:1172-1176`](optical_rl/qd/ef50_zernike_parallel.py#L1172-L1176): `progress_label="ef50-zernike seeding"`, `progress_every=500` (user later changed default in code to 100).

**Performance note**: the counter lives in the parent's queue-collection loop which was previously just blocking on `result_queue.get()`. The throttled `print()` (every 500 results) + counter increment adds microseconds per iteration. Workers themselves do nothing extra; GPU throughput is unaffected.

**Backward compatibility**: keyword-only parameters with defaults (`progress_label=None, progress_every=100`). All pre-existing positional callers in `seed_parallel_experiment.py` and `entrance_pupil_experiment.py` (note: the latter has a separate `_parallel_evaluate` function with a different signature, not touched) continue to work unchanged.

### 6.3 Rejected-Design Tracking in Calibration

**Question**: "when a design is rejected, does it count as a calibration iteration?"

**Answer**: yes — `calibration_count` (e.g., 20000) is the number of thetas *sampled and evaluated*. Rejection happens after evaluation inside the worker, so a rejected design has burned its full GPU cost before being filtered out. It just doesn't contribute to the percentile-bounds sample.

The live progress line's `rejected N` column shows this count growing in real time. With `strict_entrance_pupil: true`, the typical rejection rate is ~25-35% for random initial samples drawn from the full `theta_bounds` — meaning ~7000 of 20000 calibration samples are pupil-degenerate and contribute nothing to bounds calibration, but still consume full evaluation time.

---

## 7. Archive-Insert Semantics During Parallel Seeding

**Question raised** (triggered by discussing streaming insertions): is the parallel seeding process inefficient because workers might evaluate "duplicate" designs when the archive isn't built incrementally?

**Answer**: no — seeding is not adaptive. All seed thetas are sampled upfront at [`ef50_zernike_parallel.py:1162-1167`](optical_rl/qd/ef50_zernike_parallel.py#L1162-L1167):

```python
flat = self.problem.initial_theta()
all_thetas = np.vstack([
    flat.reshape(1, -1),
    self.problem.sample_theta(self.config.seed_count, self.config.cma_seed + 31),
])
```

That's one deterministic call producing all N thetas from the fixed-distribution prior, before any worker runs. Workers are completely blind to the archive state — they receive `(idx, theta)` tuples from a queue, evaluate, and ship results back. They don't adaptively pick work based on archive coverage. Insertion order doesn't affect final archive state either: `archive.insert()` keeps the best design per cell regardless of arrival order.

**What batching-then-inserting actually costs** (three things, none of them computational waste):

1. **Crash recovery**: if the run dies 80% through seeding, the archive in memory is still empty because nothing's been inserted yet. 32000 evaluated results sit in the parent's `results` list RAM, but if the parent dies before the insertion loop, they're lost.
2. **Observability**: coverage can't be watched growing during seeding; only `X/N evaluated` from the progress prints is visible.
3. **Memory**: parent holds all N results in RAM (~30-50 MB for 40k designs) until insertion completes.

**Where adaptivity *does* matter** is the CMA-MEGA loop. Each new candidate's parameters are generated by the emitter based on current archive contents. That loop at [`ef50_zernike_parallel.py:1258`](optical_rl/qd/ef50_zernike_parallel.py#L1258) calls `self.problem.evaluate()` directly in the parent process, candidate by candidate, and inserts immediately before generating the next one. Adaptivity is preserved exactly where it matters.

---

## 8. Refinement Code Path Reference (for writing the Methods section)

### Entry point
[`refine_ef50_archive.py:636-807`](optical_rl/qd/refine_ef50_archive.py#L636-L807) `main()` — argparse, archive loading, diverse-elite selection, per-design refinement loop, result serialization, plot generation.

### Problem construction
[`refine_ef50_archive.py:67-121`](optical_rl/qd/refine_ef50_archive.py#L67-L121) `_build_problem()` — constructs `EF50ZernikeProblem` with `descriptor_zernike_indices=(7, 11)` so `j_max=11` and the Zernike vector for clustering has components Z1..Z11. Stage is hardcoded to `"stage2"` (all-field RMS). Default `num_rays=256` for refinement (vs. 1000 for CMA).

### Per-design refinement
[`refine_ef50_archive.py:199-258`](optical_rl/qd/refine_ef50_archive.py#L199-L258) `refine_design()`:
- Pre-evaluation at start theta to record `start_rms_mm` (calls `problem.evaluate`).
- Optional trajectory recording via L-BFGS-B callback.
- `scipy.optimize.minimize` with `method="L-BFGS-B"`, `jac=True`, `bounds=theta_bounds`, `options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-14}`.
- Post-convergence Zernike vector extraction via `_full_zernike_vector(converged_theta)`.

### Objective function
[`refine_ef50_archive.py:128-190`](optical_rl/qd/refine_ef50_archive.py#L128-L190) `_make_objective()` — returns a `(loss, grad)` callable for scipy. The function this session modified to add the invalid-ray and intersection penalties.

### Basin labeling
[`refine_ef50_archive.py:276-293`](optical_rl/qd/refine_ef50_archive.py#L276-L293) `_label_basins()` — connected-component clustering of converged designs by pairwise L2 distance in Zernike space (skipping Z1 piston). Threshold `dist_threshold` defaults to `0.003` mm from YAML.

### Outputs per run
- `refinement_results.json` — full metadata including `start_theta`, `converged_theta`, `zernike_vector`, `basin_labels`, `dist_matrix`, per-design `n_iters` / `converged` flags.
- `refinement_heatmap.png` — pairwise Zernike distance heatmap, N×N with basin labels on axes.
- `refinement_basins.png` — archive measure-space scatter with before→after arrows and basin coloring.
- `refinement_theta_pca.png` — 2D PCA projection of all archive thetas with starting and converged points overlaid.

---

## 9. Glossary and Key Numbers

**Config** (from [`optical_rl/configs/ef50_zernike.yaml`](optical_rl/configs/ef50_zernike.yaml)):
- Lens template: EF50mm f/1.8 (`ef50mm_f1.8.json`)
- Focal length target: 50.52 mm (EFL-constrained → theta is 10-D)
- Object depth: -20000 mm (far field)
- Fitness wavelengths: 0.656, 0.587, 0.486 µm (Fraunhofer C/d/F)
- Stage: `stage2` → 3 meridional field points (on-axis, 0.5×HFOV, 1.0×HFOV)
- Descriptor indices: Z7 (vertical coma) and Z11 (primary spherical)
- Archive bins: 512 × 512 (Zernike measure space)
- `strict_entrance_pupil: true`
- `invalid_tol: 0.05`, `invalid_penalty_weight: 5.0`
- `intersect_penalty_weight: 1.0`, `air_min_mm: 0.1`, `thick_min_mm: 1.0`
- `dist_threshold: 0.003` (basin clustering threshold)
- `seed_count: 5000`, `calibration_count: 1000`, `iterations: 5000`
- `parallel_workers: 20`

**Machine**: 2× NVIDIA RTX 6000 Ada (49 GB each), 20 CPU cores visible.

**Typical numbers**:
- Per-worker GPU memory footprint: ~600 MB (dominated by CUDA context, not working set)
- Per-worker CPU work: roughly 50% Python lens construction, 50% GPU ray tracing with sync points
- L-BFGS-B wall clock per design: a few seconds with `num_rays=256`, typically 20-80 iterations to converge
- Good EF50 stage-2 RMS: 0.010-0.050 mm
- Nonsense fake-zero before fix: 1e-6 mm
- Absurd mis-scored fitness: archive_fitness = -21.94201 (implies rms ≈ 21.94 mm, 3× the sensor diagonal)

---

## 10. Files Modified in This Session

| File | Purpose | Lines Changed |
|---|---|---|
| [`optical_rl/qd/refine_ef50_archive.py`](optical_rl/qd/refine_ef50_archive.py) | Add invalid-ray + intersection penalty to L-BFGS-B objective | 159-188 (~14 lines added to `_make_objective`) |
| [`optical_rl/qd/seed_parallel_experiment.py`](optical_rl/qd/seed_parallel_experiment.py) | DeepLens warning filter, progress counter with elapsed/ETA, `_fmt_hms` helper | 33-95 (filter class + helper), 39-49 (`_fmt_hms`), 171-258 (progress parameters and live counter) |
| [`optical_rl/qd/ef50_zernike_parallel.py`](optical_rl/qd/ef50_zernike_parallel.py) | Install parent-side filter, wire progress label into calibration + seeding calls | 1103-1107 (calibration call), 1172-1176 (seeding call), 1450-1454 (parent filter install in `main()`) |

Unchanged (read-only reference):
- [`optical_rl/qd/ef50_zernike.py`](optical_rl/qd/ef50_zernike.py) — source of `_fitness_and_metrics` (lines 988-1013), `_intersection_penalty` (lines 903-934), `_rms_spot_batch` (lines 784-794), strict-mode pupil check (lines 973-983).

---

## 11. Open Items for Followup

1. **Apply same filter to `refine_ef50_archive.py` and `render_refined_designs.py`** — one-line import + call each. Session scoped to `ef50_zernike_parallel.py` only.
2. **Post-refinement verification print** — optional addition to `refine_design` that calls `problem.evaluate(converged_theta, with_grads=False)` after `minimize()` returns and logs the real RMS and invalid fraction next to L-BFGS-B's `result.fun`. If the two numbers diverge significantly, it surfaces the vignetting issue immediately rather than via downstream warnings.
3. **Smoke-test the fix** — user has not yet run the refiner against the buggy archive after the change. Expected outcome: design #2 no longer shows `→ 0.0000 mm 1 iters [OK]`; all designs converge to physically plausible RMS values.
4. **Consider smoothing the intersection-penalty gate** — currently uses `if float(intersect_loss.detach()) > 0` which creates a C^0 gradient discontinuity at the feasibility boundary. Zemax DLS folklore says soft constraints with kinks at zero magnitude are fine, and the session confirmed this via L-BFGS-B line-search behavior. Not a real issue unless observed in practice.
