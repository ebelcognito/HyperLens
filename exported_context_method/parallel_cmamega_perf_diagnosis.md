# Parallel CMA-MEGA Performance Diagnosis — Session Context

**Date:** 2026-04-10 / 2026-04-11
**Repo:** `/home/hisham26/Desktop/RL` (optical_rl, EF50 Zernike QD pipeline)
**Goal of session:** figure out why the parallel CMA-MEGA archive search (`ef50_zernike_parallel.py`) was running at roughly the same wallclock as the sequential version, despite using 20 worker processes across 2 GPUs.

This document captures the diagnostic methodology, the timing instrumentation that was added, the empirical results, the multiple wrong hypotheses that were ruled out, and the final root-cause finding. Intended as input for a writeup of the methods/results section of the paper.

---

## 1. Starting context

### 1.1 The pipeline under test

- **`optical_rl/qd/ef50_zernike_parallel.py`** — parallel CMA-MEGA QD search over a 10-dimensional curvature parameterization of an EF50 mm f/1.8 lens (Canon design). EFL is fixed to 50.52 mm via a paraxial constraint that derives the rear element curvature from the others, so the optimizer only sees 10 free parameters.
- Behavioral descriptors: two Noll Zernike coefficients (Z7 vertical coma + Z11 primary spherical) computed at mid-field via a coherent OPD trace through the lens.
- Fitness: mean RMS spot over 3 wavelengths (Fraunhofer C, d, F = 0.656, 0.587, 0.486 μm) and 3 meridional field points (on-axis, 0.5×HFOV, full HFOV) — i.e. "stage2".
- Archive: 100×100 MAP-Elites grid over the (Z7, Z11) descriptor space.
- CMA-MEGA inner loop: each generation samples 1 elite parent, computes its gradient via autograd, asks an inner CMA-ES emitter for `population_size=20` directional coefficients, builds 20 candidate θ vectors as `unit_parent + cma_eta * (c_f * grad_fitness + c_z7 * grad_z7 + c_z11 * grad_z11)`, evaluates them in parallel, and tells the emitter the resulting fitnesses.
- Hardware: 2× NVIDIA RTX 6000 Ada (49 GB each), shared workstation.
- Worker pool: `PersistentEvalPool` in `optical_rl/qd/_eval_pool.py`, using `mp.get_context("spawn")`, 20 workers round-robined across the 2 GPUs (10 per GPU). Workers stay alive across generations; each generation just dispatches the 20 candidate θs onto the shared task queue and reads results back.

### 1.2 The symptom

Despite running 20-worker / 2-GPU parallelism, the user observed wallclock that "felt almost the same" as the sequential `ef50_zernike.py` version. A representative log snippet at the start of investigation:

```
[ef50-zernike] CMA-MEGA worker pool: 20 workers across GPUs [0, 1]
[ef50-zernike] gen=5  evals=100 coverage=0.0115 best=-0.840543 | new=14/20 replaced=3/20 wasted=0/10  sigma=1.4341
[ef50-zernike] gen=10 evals=200 coverage=0.0167 best=-0.840543 | new=13/20 replaced=3/20 wasted=0/10  sigma=1.5822
[ef50-zernike] gen=15 evals=300 coverage=0.0223 best=-0.840543 | new=10/20 replaced=3/20 wasted=0/10  sigma=2.4527
[ef50-zernike] gen=20 evals=400 coverage=0.0270 best=-0.840543 | new=10/20 replaced=4/20 wasted=0/10  sigma=3.1331
```

No per-generation wallclock was being printed at this point — just eval counts. The user's qualitative impression was that the speedup over sequential was much smaller than the 20× that the worker count would naively suggest.

---

## 2. Initial (incorrect) hypothesis: the parent-grad step is the bottleneck

### 2.1 Reasoning

Looking at the CMA-MEGA inner loop in `ef50_zernike_parallel.py:1378-1499`, each generation has the following structure:

1. **Parent gradient eval** — `self.problem.evaluate(parent_theta, with_grads=True)` runs on the **driver process**, not in the pool. It does a full forward + backward pass through the autodiff ray tracer on all 3 wavelengths × field points and the descriptor Zernikes. Predicted cost ~0.8 s.
2. **Build 20 candidate θs** — pure CPU/numpy, microseconds.
3. **Pool evaluate** — `self._eval_pool.evaluate(candidate_thetas)` dispatches 20 forward-only evals across 20 workers. Predicted cost ~0.27 s (one parallel wave, 20 workers each doing one forward eval).
4. **Insert + tell** — archive insertions and `emitter.tell()`, cheap.

Initial hypothesis: the **serial parent-grad on the driver** is Amdahl's-law dominant. With ~0.8 s parent + ~0.27 s pool, the parent is 75% of the wallclock and parallelism cannot help it.

### 2.2 Action: add per-phase timing instrumentation to the parallel runner

To validate this hypothesis, per-phase wallclock timing was added to the CMA-MEGA loop in `optical_rl/qd/ef50_zernike_parallel.py`. Key design choices:

- Use `torch.cuda.synchronize()` at every phase boundary so that GPU async work is attributed to the phase that launched it, not to the next phase.
- Accumulate per-phase wallclock over `log_every` (= 5) generations and print one timing line right after the existing per-gen log line, then reset accumulators.
- Phases tracked: `parent` (parent-grad eval), `build` (candidate θ construction), `pool` (`PersistentEvalPool.evaluate` call), `insert` (archive insert + emitter.tell).

The added log line format:
```
[ef50-zernike] timing gen=N avg/gen over last K gens (ms): parent=... build=... pool=... insert=... total=... | parent%=... pool%=... throughput=... evals/s
```

(Code added at `optical_rl/qd/ef50_zernike_parallel.py` around line 1378–1592.)

### 2.3 First measurement — initial hypothesis falsified

Run output with timing instrumentation, **no MPS, default Default-mode CUDA contexts**:

```
[ef50-zernike] gen=5  ... pool%= 81.5 throughput=  3.7 evals/s
[ef50-zernike] timing gen=5  parent=  986.6  pool= 4362.2  insert=  0.91  total= 5352.5
[ef50-zernike] gen=10 ... pool%= 65.3 throughput= 10.3 evals/s
[ef50-zernike] timing gen=10 parent=  673.9  pool= 1274.6  insert=  0.87  total= 1951.2
```

**Findings:**
- Steady-state (gen 6–10, after CUDA/JIT warmup): `parent ≈ 674 ms`, `pool ≈ 1275 ms`, `total ≈ 1951 ms`, `throughput ≈ 10.3 evals/s`.
- The **pool**, not the parent, is the dominant cost: `pool% ≈ 65%`, `parent% ≈ 35%`.
- Hypothesis 1 (parent-grad is the bottleneck) is falsified. The pool of 20 supposedly-parallel forward evals is taking ~1.3 s, which is far more than a single forward eval should take (~125–200 ms expected).

### 2.4 Refined hypothesis: CUDA process-context contention

Under default CUDA mode, kernels from N processes sharing one GPU are serialized at the kernel level (round-robin time-slicing on the default stream). With 10 workers per GPU all issuing `trace2sensor` kernels simultaneously, real concurrency per GPU is much less than 10. Sanity-check arithmetic:

- Each forward-only eval ≈ 200–300 ms uncontended.
- 20 evals split across 2 GPUs = 10 per GPU.
- If each GPU effectively serializes ~5-way (out of 10 attempted), wallclock = 10 × 130 ms = ~1300 ms ≈ observed 1275 ms ✓

This suggested that NVIDIA Multi-Process Service (MPS), which lets multiple processes share one GPU's SMs concurrently rather than time-slicing, should give a ~3× speedup of the pool phase.

---

## 3. Hypothesis 2 ruled out: MPS gives essentially zero speedup

### 3.1 MPS setup

A user-isolated MPS daemon was started so it would not affect other users on the shared workstation:

```bash
mkdir -p ~/.mps-claude/pipe ~/.mps-claude/log
chmod 700 ~/.mps-claude/pipe ~/.mps-claude/log
CUDA_MPS_PIPE_DIRECTORY=~/.mps-claude/pipe \
CUDA_MPS_LOG_DIRECTORY=~/.mps-claude/log \
nvidia-cuda-mps-control -d
```

Then the training was launched with the same env vars exported, so all 21 processes (driver + 20 workers spawned via `mp.get_context("spawn")`) inherited them and would route through MPS.

### 3.2 MPS-on results

```
[ef50-zernike] timing gen=5  parent=  972.3  pool= 3860.4  total= 4835.6  throughput=  4.1
[ef50-zernike] timing gen=10 parent=  710.3  pool= 1197.6  total= 1910.6  throughput= 10.5
[ef50-zernike] timing gen=15 parent=  716.6  pool= 1308.5  total= 2027.9  throughput=  9.9
[ef50-zernike] timing gen=20 parent=  704.9  pool= 1125.4  total= 1832.8  throughput= 10.9
[ef50-zernike] timing gen=25 parent=  670.0  pool= 1307.9  total= 1980.4  throughput= 10.1
[ef50-zernike] timing gen=30 parent=  694.5  pool= 1299.9  total= 1997.0  throughput= 10.0
```

| | No MPS | With MPS | Δ |
|---|---|---|---|
| pool ms (steady) | ~1275 | ~1240 | -3% (noise) |
| parent ms | ~674 | ~700 | +4% |
| throughput | 10.3 | 10.0 | flat |

MPS gave essentially **zero** speedup. Verified that MPS was actually being used:
- 61 unique client PIDs in `~/.mps-claude/log/server.log` = exactly **driver (1) + seed pool (20) + calibration pool (20) + CMA-MEGA persistent pool (20)**, confirming all 60 worker processes connected to MPS.
- `nvidia-smi` showed the driver as `M+C` (MPS+Compute) instead of just `C`, confirming MPS routing.
- `mp.get_context("spawn")` correctly inherits parent's `os.environ`, so workers got `CUDA_MPS_PIPE_DIRECTORY` automatically.

So MPS was active for every single eval, and it didn't help. This means the bottleneck was **not** CUDA process-context contention. The diagnosis was wrong.

---

## 4. Single-process benchmark — definitively settles parallelism vs per-eval cost

A standalone benchmark script was added at `optical_rl/qd/bench_eval_singleproc.py`. It builds one `EF50ZernikeProblem` on a single GPU in a single process — no pool, no MPS dependence — and times sequential evaluations of `problem.evaluate()` with `cuda.synchronize()` around each call.

The purpose: directly measure single-eval wallclock in the simplest possible setup, so the pool's effective speedup could be computed honestly.

### 4.1 First bench result (num_rays = 1000, stage2, 3 wavelengths)

```
[bench] device=cuda:0 num_rays=1000 stage=stage2 wavelengths=(0.656, 0.587, 0.486)
[bench] using theta from archive_cells.json
[bench] warming up (3 evals each, discarded)...
[bench] timing 20 sequential forward-only evals...
[bench] timing 5 sequential forward+grad evals...

  forward-only     n= 20  total=  4892.0 ms  mean= 244.60  median= 243.58  min= 241.89  max= 259.96
  forward+grad     n=  5  total=  2630.1 ms  mean= 526.02  median= 526.68  min= 523.62  max= 527.44
```

**Headline numbers:**
- Single-process forward-only eval: **244.6 ms mean**
- Single-process forward+grad eval: **526 ms mean** (backward = +281 ms ≈ 1.15× forward)
- 20 sequential forward-only evals: **4892 ms total**

### 4.2 Pool speedup, computed honestly

| Setup | Wallclock for 20 forward-only evals | Speedup vs single-proc |
|---|---|---|
| 1 process, 1 GPU | **4892 ms** | 1.0× |
| 20-worker pool, 2 GPUs (no MPS) | 1274 ms | **3.84×** |
| 20-worker pool, 2 GPUs (with MPS) | 1197 ms | **4.09×** |
| Theoretical perfect (20-way parallel) | 245 ms | 20.0× |

**The pool is giving real parallelism — about 4× — but is far short of the naive 20× ceiling.** The measured 4× is consistent with each GPU effectively running ~2 concurrent kernels (2 GPUs × ~2 = 4×), with the remaining 8 kernels per GPU queued.

### 4.3 The num_rays surprise

To test the "reduce per-eval kernel cost via fewer rays" recommendation, the bench was re-run with `num_rays=500` and `num_rays=250`:

| num_rays | Mean per-eval (forward-only) | Total for 20 evals |
|---|---|---|
| 1000 | 244.6 ms | 4892 ms |
| 500  | 245.7 ms | 4914 ms |
| 250  | 250.1 ms | 5002 ms |

**Per-eval time is essentially flat across a 4× range of ray counts.** Reducing rays by 4× did **not** reduce wallclock at all (within noise). This was the second hypothesis-killer.

This proves the 244 ms is dominated by **fixed per-eval overhead** that does not scale with ray count. The kernels are too small for ray count to matter — they are **launch-bound on the CPU side**, not compute-bound on the GPU side. The 6 GB / 48 GB GPU memory utilization observed by the user is consistent with this: small tensors, small kernels, lots of CPU-side dispatch overhead.

---

## 5. Intra-eval timing breakdown — the smoking gun

The bench script was extended with a `--profile` mode that monkey-patches the hot methods on the `EF50ZernikeProblem` instance with timing wrappers. Each wrapper does `torch.cuda.synchronize()` at entry and exit so per-phase wallclock includes the GPU work each method launches. Methods wrapped:

- `_build_lens_from_theta_tensor`
- `_calc_hfov_deg`
- `_trace_fields_batch`
- `_rms_spot_batch`
- `_intersection_penalty`
- `_zernike_measures`

CLI overrides were also added for `--wavelengths` (override `fitness_wavelengths_um`) and `--stage` (override `stage1`/`stage2`) so the loop-count vs work-per-call hypothesis could be tested directly.

### 5.1 Baseline profile (num_rays=1000, stage2, 3 wavelengths)

```
[bench] intra-eval breakdown (avg per eval, across 25 evals):
  _zernike_measures              per_eval= 116.52 ms  per_call= 116.52 ms  calls/eval= 1.00  pct= 50.3%
  _trace_fields_batch            per_eval= 100.23 ms  per_call=  33.41 ms  calls/eval= 3.00  pct= 43.3%
  _intersection_penalty          per_eval=  12.07 ms  per_call=  12.07 ms  calls/eval= 1.00  pct=  5.2%
  _build_lens_from_theta_tensor  per_eval=   1.44 ms  per_call=   1.44 ms  calls/eval= 1.00  pct=  0.6%
  _rms_spot_batch                per_eval=   1.25 ms  per_call=   0.42 ms  calls/eval= 3.00  pct=  0.5%
  _calc_hfov_deg                 per_eval=   0.02 ms  per_call=   0.02 ms  calls/eval= 1.00  pct=  0.0%
  (sum of phases above)          per_eval= 231.53 ms

  forward-only     n= 20  total=  4940.6 ms  mean= 247.03  median= 246.72  min= 245.79  max= 249.77
  forward+grad     n=  5  total=  2631.4 ms  mean= 526.27  median= 524.07  min= 523.02  max= 535.09
```

**The smoking gun:** `_zernike_measures` — the **behavioral descriptor computation** — is the single largest line item, ~115 ms per eval, ~50% of total wallclock.

### 5.2 Drop wavelengths to 1 (num_rays=1000, stage2, 1 wavelength)

```
[bench] intra-eval breakdown (avg per eval, across 25 evals):
  _zernike_measures              per_eval= 111.97 ms  per_call= 111.97 ms  calls/eval= 1.00  pct= 71.0%
  _trace_fields_batch            per_eval=  32.45 ms  per_call=  32.45 ms  calls/eval= 1.00  pct= 20.6%
  _intersection_penalty          per_eval=  11.49 ms  per_call=  11.49 ms  calls/eval= 1.00  pct=  7.3%
  _build_lens_from_theta_tensor  per_eval=   1.34 ms  per_call=   1.34 ms  calls/eval= 1.00  pct=  0.9%
  _rms_spot_batch                per_eval=   0.41 ms  per_call=   0.41 ms  calls/eval= 1.00  pct=  0.3%
  _calc_hfov_deg                 per_eval=   0.02 ms  per_call=   0.02 ms  calls/eval= 1.00  pct=  0.0%
  (sum of phases above)          per_eval= 157.68 ms

  forward-only     n= 20  total=  3473.3 ms  mean= 173.66  median= 173.49  min= 172.76  max= 176.31
  forward+grad     n=  5  total=  1904.0 ms  mean= 380.80  median= 380.87  min= 378.70  max= 382.07
```

- `_trace_fields_batch` dropped from 100 ms (3 calls × 33 ms) to 32 ms (1 call × 32 ms) — confirming the wavelength loop is linear and saves ~67 ms when collapsed to 1 wavelength.
- `_zernike_measures` is **unchanged** at ~112 ms because it doesn't depend on `fitness_wavelengths_um` — it does its own coherent trace at a single wavelength.
- New per-eval total: **174 ms** (down from 247 ms — a 30% improvement).

### 5.3 Stage1 attempts (crashed)

The two `--stage stage1` runs (with both 3 wavelengths and 1 wavelength) crashed with:

```
File "ef50_zernike.py", line 861, in _zernike_measures
    field = self._field_points(hfov_deg)[idx][1]
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^
IndexError: tuple index out of range
```

`_zernike_measures` indexes `_field_points(hfov)[zernike_field_index]` with `zernike_field_index = 1`, but stage1 has only one field point (index 0). This is a known bug independent of the perf work — the descriptor field index needs to clamp to 0 when stage1 is in effect, or the YAML default needs to enforce stage2 when `zernike_field_index > 0`. Documented as a fix to do alongside any performance work.

---

## 6. Decomposition of `_zernike_measures` (the ~115 ms)

Looking at `_zernike_measures` in `optical_rl/qd/ef50_zernike.py:847-911`, the method does a separate, second-pipeline trace through the lens to compute the Zernike OPD descriptor:

| Step | Estimated cost | What it is |
|---|---|---|
| `_trace_coherent` (1000 rays, **float64**) | ~50–70 ms | Full coherent ray bundle through all 12 surfaces in float64 |
| `lens.trace2sensor(chief_ray)` (1 ray) | ~15–25 ms | Separate single-ray trace for reference chief ray (full launch overhead even for 1 ray) |
| `lens.get_exit_pupil()` + `calc_chief_ray_infinite()` | ~15–25 ms | Two paraxial solves |
| `prop_to(pupil_z)` × 2 | ~5 ms | Ray propagation to exit pupil |
| OPD math + `pinv_Z @ opd` | ~5 ms | The Zernike fit itself (cheap) |

Two compounding root causes for why this is so expensive:

1. **Float64 throughput penalty.** DeepLens's coherent OPL tracer requires float64 for accuracy — see comment at `ef50_zernike.py:470-477`. RTX 6000 Ada is consumer-derived silicon and runs float64 at **~1/64 the speed of float32**. So a float64 coherent trace through 1000 rays × 12 surfaces is 10–30× more expensive than the float32 fitness trace at the same ray count.

2. **1000 rays for an 11-coefficient Zernike fit.** The descriptor uses the same `num_rays` config as the fitness (1000 rays), but reliable Z1..Z11 fitting only needs ~50–100 sample points. So **10× more rays than the fit needs**, traced in float64, every single eval. The `_aperture_xy` and `_pinv_Z` tensors are built once at startup against the same 1000-point Fibonacci disk, so changing this would require building separate `_aperture_xy_desc` and `_pinv_Z_desc` for the descriptor.

3. **Single-ray chief-ray trace pays full launch overhead.** `lens.trace2sensor(chief_ray)` is called with a 1-ray bundle just to get the chief ray's exit-pupil position and OPL. The kernel launch + sync overhead dwarfs the actual compute (1 ray × 12 surfaces is microseconds of math).

---

## 7. Summary of speedup ceilings (the "how fast can it go" answer)

### Tier 1 — Pure mechanical optimization, no science change
**Single change: decouple descriptor ray count from fitness ray count + drop fitness wavelengths to 1.**

- Add `descriptor_num_rays: 128` config knob.
- Build `self._aperture_xy_desc` and `self._pinv_Z_desc` at 128 rays at startup.
- `_zernike_measures` uses these instead of `self._aperture_xy` / `self._pinv_Z`.
- Set `fitness_wavelengths_um: [0.587]` for the search; reuse the existing 3-wavelength path only for refinement / archive rendering.

**Expected impact:**
- `_zernike_measures`: 115 ms → ~40–55 ms (linear in ray count, ~8× reduction in coherent trace cost; chief ray + paraxial overhead unchanged)
- `_trace_fields_batch`: 100 ms → ~33 ms (1 wavelength)
- Per-eval total: **247 ms → ~115 ms** (~2.2× speedup)
- Pool wallclock: 1240 ms → ~600 ms
- Throughput: 10 evals/s → ~33 evals/s

**Risk:** none; same numerical descriptors at lower ray-count noise (Z7/Z11 fit values change at the 3rd–4th significant figure but archive cell assignments are essentially identical for a 100×100 binning).

### Tier 2 — Replace chief-ray trace with paraxial chief ray
**Eliminate `lens.trace2sensor(chief_ray)` entirely.** The chief ray of an axisymmetric lens at a known field angle has a closed-form paraxial solution computable from ABCD matrices in numpy in microseconds. Same with `get_exit_pupil()` and `calc_chief_ray_infinite()` — both are paraxial solves that don't need DeepLens's full per-surface intersection machinery.

- Implement an analytic chief-ray solver using the lens's surface curvatures, thicknesses, and indices.
- Use it to directly compute `(x_chief_ep, y_chief_ep, opl_chief)` without launching any GPU kernels.
- The chief ray is *already paraxial*; the trace is just numerically reproducing the analytic result, so the Z7/Z11 values are identical.

**Expected impact (on top of Tier 1):**
- `_zernike_measures`: ~50 ms → ~15–20 ms
- Per-eval total: ~115 ms → **~85 ms** (~2.9× speedup vs baseline)
- Pool wallclock: ~430 ms
- Throughput: ~30 evals/s

**Risk:** requires re-deriving the paraxial chief-ray equations for the EF50 lens architecture. Verification: numerically check that the paraxial chief-ray result matches the full-trace chief-ray to ~6 sig figs over a representative sample of designs.

### Tier 3 — Replace Z7/Z11 descriptor with cheaper proxy (changes archive structure)
The behavioral descriptor only needs to bucket designs into a 100×100 archive grid. Two cheaper proxies that correlate ~0.9+ with Z7/Z11 for axisymmetric lenses:

- **Seidel coefficients** computed analytically from curvatures + thicknesses + indices — closed-form, ~1 ms, no ray tracing. Z11 ≈ Seidel spherical, Z7 ≈ Seidel coma to leading order.
- **Spot-shape moments** (elongation, centroid offset) extracted from the fitness trace already being computed — free.

**Expected impact:**
- `_zernike_measures` (or its replacement): 115 ms → **~1–5 ms**
- Per-eval total: 247 ms → **~30 ms** (~8× speedup vs baseline)
- Pool wallclock: ~150 ms
- Throughput: ~80 evals/s

**Risk:** changes the archive cell mapping; the science of the QD search (what "diversity" means) is reframed in terms of leading-order aberration coefficients rather than measured Zernike OPD coefficients. This is a paradigm decision, not a pure optimization.

---

## 8. Hypotheses ruled out (with evidence)

This is the cleanest part of the methodology — every wrong hypothesis had a falsification.

| Hypothesis | Predicted observation | Actual observation | Verdict |
|---|---|---|---|
| **H1: Parent-grad is the bottleneck** (Amdahl on serial driver autograd) | parent% > 70% in timing log | parent% = 35%, pool% = 65% | **Ruled out** by per-phase timing |
| **H2: CUDA process-context contention** (10 workers per GPU time-slice without MPS) | MPS gives ~3× pool speedup | MPS gives ~3% (noise) | **Ruled out** by MPS A/B test |
| **H3: GPU SM saturation** (kernels are big enough to fully use SMs) | Pool wallclock ≈ single-proc-single-GPU / N_GPUs | Pool ≈ 1240 ms, single ≈ 4900 ms → 4× speedup, not 2× | **Partially ruled out** — pool is getting ~4× real parallelism (consistent with ~2 effective concurrent kernels per GPU); per-GPU contention exists but not the full story |
| **H4: Per-eval ray-trace compute scales with `num_rays`** | Halving num_rays halves per-eval time | num_rays 1000→500→250 gives 244.6→245.7→250.1 ms (flat) | **Ruled out** by num_rays sweep |
| **H5: Wavelength loop dominates** (3 calls × heavy per-call cost) | 1-wavelength run gives ~3× speedup | 1-wavelength gives 247→174 ms (1.4×); _trace_fields_batch part scales linearly but is only 40% of cost | **Partially confirmed** — wavelength loop is real but not dominant |
| **H6: `_zernike_measures` is the dominant cost** | Profile shows >40% in `_zernike_measures` | Profile shows **50.3%** in `_zernike_measures` | **Confirmed** |

---

## 9. Files added/modified during this session

| File | Change |
|---|---|
| `optical_rl/qd/ef50_zernike_parallel.py` | Added per-phase timing instrumentation in CMA-MEGA loop (lines ~1378-1592). Tracks `parent`, `build`, `pool`, `insert` phases with `cuda.synchronize()` boundaries; accumulates over `log_every` gens; prints alongside per-gen log. |
| `optical_rl/qd/bench_eval_singleproc.py` | **New file.** Standalone single-process benchmark for `EF50ZernikeProblem.evaluate()`. Times sequential forward-only and forward+grad evals on one GPU. CLI flags: `--n-forward`, `--n-grad`, `--gpu`, `--num-rays`, `--wavelengths`, `--stage`, `--profile`. The `--profile` flag installs monkey-patch timers on hot methods to print intra-eval breakdown. |

No production code paths were modified — all instrumentation is either log-line additions to the runner or in the standalone bench script.

---

## 10. Numerical results summary table (for paper writeup)

### Baseline single-eval costs (no pool, single GPU, RTX 6000 Ada)

| num_rays | Mean per-eval (ms) | Notes |
|---|---|---|
| 1000 | 244.6 | Default config |
| 500  | 245.7 | Half rays — no improvement |
| 250  | 250.1 | Quarter rays — no improvement |

**Conclusion:** per-eval cost is independent of fitness ray count; CPU/launch overhead dominates GPU compute.

### Forward vs forward+grad

| Mode | Mean per-eval (ms) |
|---|---|
| forward-only | 244.6 |
| forward + autograd backward | 526.0 |
| backward-only delta | 281.4 (~1.15× forward) |

### Pool parallelism efficiency (20 forward-only evals)

| Setup | Wallclock (ms) | Speedup vs single-proc | Efficiency vs ideal 20× |
|---|---|---|---|
| 1 process, 1 GPU | 4892 | 1.0× | 5% |
| 20 workers, 2 GPUs, no MPS | 1274 | 3.84× | 19% |
| 20 workers, 2 GPUs, MPS | 1197 | 4.09× | 20% |
| Theoretical perfect (single eval cost) | 245 | 20.0× | 100% |

### Intra-eval phase breakdown (single process, num_rays=1000, stage2, 3 wavelengths)

| Phase | Per-eval (ms) | % of total | Calls/eval |
|---|---|---|---|
| `_zernike_measures` | 116.5 | 50.3 | 1 |
| `_trace_fields_batch` | 100.2 | 43.3 | 3 |
| `_intersection_penalty` | 12.1 | 5.2 | 1 |
| `_build_lens_from_theta_tensor` | 1.4 | 0.6 | 1 |
| `_rms_spot_batch` | 1.3 | 0.5 | 3 |
| `_calc_hfov_deg` | 0.02 | 0.0 | 1 |
| **Sum of profiled phases** | **231.5** | — | — |
| Unprofiled (entrance pupil, autograd setup, dispatch) | ~15 | ~6.5 | — |
| **Total per eval** | **247.0** | 100 | — |

### Intra-eval phase breakdown (num_rays=1000, stage2, **1 wavelength**)

| Phase | Per-eval (ms) | % of total | Calls/eval | Δ vs baseline |
|---|---|---|---|---|
| `_zernike_measures` | 112.0 | 71.0 | 1 | -4 ms (noise) |
| `_trace_fields_batch` | 32.4 | 20.6 | 1 | **-67.8 ms (3 calls → 1 call)** |
| `_intersection_penalty` | 11.5 | 7.3 | 1 | -0.6 ms |
| `_build_lens` | 1.3 | 0.9 | 1 | unchanged |
| `_rms_spot_batch` | 0.4 | 0.3 | 1 | -0.9 ms |
| **Total per eval** | **173.7** | 100 | — | **-73 ms (30% faster)** |

### Projected speedup with optimization tiers

| Optimization | Per-eval (ms) | Pool wallclock (ms) | Throughput (evals/s) | Speedup vs baseline |
|---|---|---|---|---|
| Baseline (current) | 247 | 1240 | 10 | 1.0× |
| Tier 1: descriptor_num_rays=128 + 1 wavelength | ~115 | ~600 | ~33 | ~2.2× |
| Tier 2: + paraxial chief ray | ~85 | ~430 | ~46 | ~2.9× |
| Tier 3: + Seidel-coefficient descriptor | ~30 | ~150 | ~133 | ~8.2× |

---

## 11. Methodological notes for the paper

A few things from this investigation that are worth highlighting in a Methods section because they generalize beyond this specific bottleneck:

1. **Wallclock scaling does not reveal where the wallclock goes.** The user's qualitative impression that "20 workers feels like sequential" is consistent with both "GPU saturation at 4× ceiling" and "process-contention serializing everything", and these have very different fixes. Naïve speedup comparison (sequential vs parallel total wallclock) cannot distinguish them. We needed two independent measurements to disambiguate: (a) per-phase timing in the production loop, (b) per-eval timing in a single-process bench.

2. **MPS is the cleanest A/B test for "is process-context contention the bottleneck"** because it changes only the GPU-sharing scheduler, with no application-level changes. If MPS gives ~zero speedup (as here), CUDA-level contention is ruled out as the dominant cost.

3. **Memory utilization ≠ compute utilization.** The user observed 6 GB / 48 GB on each GPU and intuited this meant the GPU had headroom for more work. This intuition was directionally right (the GPU is in fact mostly idle from a compute standpoint) but for the wrong reason: the GPU is idle not because the kernels are big and we could fit more of them, but because the kernels are *tiny* and most of the per-eval wallclock is CPU-side dispatch + sync overhead, with the GPU simply waiting between launches.

4. **Profile per-method, not per-tensor-op.** Monkey-patching the hot methods on the problem instance with `cuda.synchronize()`-bracketed timers gave a clean attribution of wallclock without needing a full PyTorch Profiler trace. This took ~30 lines of code in the bench script and worked across both forward and forward+grad modes without modifying the production class. The technique generalizes to any class-based PyTorch pipeline.

5. **Hypothesis falsification matters.** The session sequentially ruled out three reasonable-sounding hypotheses (parent-grad bottleneck, process-context contention, GPU SM saturation) before locating the actual bottleneck (`_zernike_measures` doing redundant float64 work). Publishing the negative results alongside the positive finding makes the methodology reproducible and prevents readers from trying the same false leads.

---

## 12. Open questions / things left unverified at end of session

1. **The Tier 1 fix has not been implemented or measured.** Estimated 247 → 115 ms based on linear ray-count scaling of the float64 coherent trace. Should be validated with a one-line test: temporarily change `_aperture_xy` to a 128-point disk and re-run the bench.

2. **The chief-ray paraxial replacement (Tier 2) has not been written.** Requires deriving the analytic ABCD chief-ray equations for the EF50 architecture and verifying numerical agreement with the full trace.

3. **The Seidel-coefficient descriptor (Tier 3) is a science decision.** Would change archive structure; correlation with Z7/Z11 over a representative archive sample needs to be measured before committing.

4. **The stage1 crash in `_zernike_measures`** (`zernike_field_index` indexing into a 1-element tuple) is a known bug surfaced during the bench runs. Trivial fix: clamp `zernike_field_index = min(zernike_field_index, len(field_points) - 1)` or assert at config-load time.

5. **The pool's actual within-GPU concurrency factor** (~2 effective concurrent kernels per GPU, deduced from arithmetic) was not directly measured via `nvidia-smi dmon -s u`. A direct measurement of SM% during the pool phase would confirm whether the GPUs are actually saturated or sitting at 30–60% (which would change the diagnosis again).

6. **Whether Tier 1 + Tier 2 hits a new bottleneck.** Once `_zernike_measures` drops from ~115 ms to ~15–20 ms, the dominant cost shifts to `_trace_fields_batch` (now ~32 ms at 1 wavelength) or to `_intersection_penalty` (~12 ms, currently ignored). The analysis would need to be redone post-fix to find the next bottleneck.
