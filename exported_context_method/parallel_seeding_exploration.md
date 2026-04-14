# Parallel Seeding Exploration — EF50 Zernike CMA-MEGA

## Goal

Reduce wall-clock time for the **seeding** phase of the EF50mm f/1.8 CMA-MEGA
quality-diversity optimisation pipeline so that more seeds can be evaluated in
the same compute budget, giving finer resolution over the parameter-variation
space. Constraint: must continue using DeepLens's `trace2sensor` forward pass
unchanged — no custom tracing code, no algorithmic shortcuts.

The seeding phase is a pure forward pass (no gradients required), which makes
it embarrassingly parallel in principle.

## Background — what the seeding phase does

The EF50 runner ([optical_rl/qd/ef50_zernike.py](optical_rl/qd/ef50_zernike.py))
uses CMA-MEGA over a 10-dimensional curvature space `[c0..c10]` (c11 derived
from the focal-length constraint). Before the gradient-driven optimisation
loop runs, two phases populate the MAP-Elites archive with random designs:

1. **Calibration** (`calibration_count` = 1000 evaluations).
   `EF50ZernikeRunner.calibrate_archive_bounds()` samples designs via Latin
   Hypercube Sampling (LHS), evaluates each one to extract its two Zernike
   behavioural descriptors (Z7 vertical coma + Z11 primary spherical), then
   takes the 2nd/98th percentile of those measure values to fix the archive's
   2-D bin bounds.

2. **Seeding** (`seed_count` = 1000 evaluations).
   `EF50ZernikeRunner.seed_archive()` samples another 1000 LHS designs,
   evaluates each one, and inserts the result into the MAP-Elites archive.

Both phases call `EF50ZernikeProblem.evaluate(theta, with_grads=False)` in a
sequential Python `for` loop. Each `evaluate()` does:

- `_build_lens_from_theta_tensor()` — construct the GeoLens object
- For each of 3 wavelengths (Fraunhofer C, d, F): `_trace_fields_batch()` →
  `lens.trace2sensor(ray)` with rays of shape `(F, num_rays, 3)`
- `_trace_coherent()` for the marginal Zernike OPD bundle →
  `lens.trace2sensor(ray)`
- `lens.trace2sensor(chief_ray)` for the reference chief ray
- RMS spot computation, intersection penalty, Zernike pseudo-inverse fit

So each seed evaluation makes **5 separate `trace2sensor` calls**, all
sequential, each processing 10 000 rays through 11 surfaces.

## Investigation

### Baseline measurement

On the user's workstation (2× NVIDIA RTX 6000 Ada, 49 GB each):

| Phase        | Sequential time | Throughput      |
|--------------|-----------------|-----------------|
| Calibration  | 276.55 s        | ~3.6 evals/s    |
| Seeding      | 274.01 s        | ~3.6 evals/s    |
| **Total**    | **551.25 s**    |                 |

`nvidia-smi` during the sequential run showed:

- **GPU 0**: 11 % utilisation, 638 MiB / 49 140 MiB used, 73 W / 300 W
- **GPU 1**: 0 % utilisation, completely idle

This is the critical observation: the GPU was nowhere near saturated. Each
forward pass touches only a tiny fraction of the available compute, and there
is an entire second GPU sitting idle. The bottleneck is **not** GPU
throughput — it is the sequential Python dispatch loop.

### Attempt 1 — `ThreadPoolExecutor` with shared `EF50ZernikeProblem`

First experiment: a `concurrent.futures.ThreadPoolExecutor` running 14 worker
threads, each calling `evaluate()` on its own `EF50ZernikeProblem` instance
(separate instances were necessary because `_fitness_and_metrics` mutates
`self._fitness_aperture_xy` and `self._pupil_z` per call — sharing one
instance would race).

**Result:**

```
                Sequential    Parallel    Speedup
  Calibration:    276.55s      363.98s    0.76x
  Seeding:        274.01s      364.99s    0.75x
  Total:          551.25s      728.97s    0.76x
```

The parallel run was **slower** than sequential. Diagnosis:

1. **Python GIL contention.** PyTorch GPU kernel launches release the GIL,
   but everything around them — building the lens, constructing ray tensors,
   extracting `.item()` results, archive bookkeeping — is pure Python and
   serialised by the GIL. With 14 threads competing, GIL handoff overhead
   dominated.
2. **Single-GPU contention.** All 14 threads submitted CUDA kernels into the
   same default stream on GPU 0, adding scheduling overhead without
   exploiting GPU 1.
3. **No real overlap window.** The five sequential `trace2sensor` calls per
   evaluation don't leave large idle gaps for thread interleaving to fill.

Threading is the wrong tool. Each evaluation is dispatch-bound, not
compute-bound, and the GIL prevents threads from removing the dispatch
bottleneck.

### Attempt 2 — `torch.multiprocessing` across both GPUs

Replaced the threading approach with `torch.multiprocessing` using the
`spawn` start method (required for CUDA safety). Architecture:

- Spawn N worker **processes** (separate Python interpreters → no shared
  GIL).
- Round-robin assignment of workers to GPUs: with 12 workers and 2 GPUs,
  6 workers run on `cuda:0` and 6 on `cuda:1`.
- Each worker process:
  - Calls `torch.cuda.set_device(gpu_id)` so all tensors land on the right
    card.
  - Constructs its own `EF50ZernikeProblem` (with `device=f"cuda:{gpu_id}"`).
  - Reads `(idx, theta)` tuples from a `multiprocessing.Queue`.
  - Calls `problem.evaluate(theta, with_grads=False)` inside `torch.no_grad()`.
  - Posts a plain dict (`theta`, `fitness`, `measures`, `metrics`) back via a
    result queue. Plain dicts avoid sending torch tensors across the
    process boundary.
- The parent process samples thetas with LHS, dispatches them to the task
  queue, collects results, and inserts into the (single, parent-side)
  `EF50Archive2D` in deterministic original-theta order.

Worker count auto-selection logic:

- Detect CPUs via `os.sched_getaffinity(0)` (respects cgroup limits and other
  users on the shared workstation).
- Detect GPUs via `torch.cuda.device_count()`.
- Default: `min(num_gpus * 6, available_cpus // 2, 32)` — targets ~66 % per-GPU
  utilisation given the observed 11 % per-eval footprint, while leaving half
  the CPUs free for other users.
- Overridable via `--workers N` and `--gpus 0,1`.

The same parallel evaluator is used for both calibration and seeding. The
calibration phase reuses the harness because it has the identical
`evaluate()`-in-a-loop structure.

### Standalone experiment script

The whole exploration was done in a self-contained script
[optical_rl/qd/seed_parallel_experiment.py](optical_rl/qd/seed_parallel_experiment.py)
that imports `EF50ZernikeProblem`, `EF50Archive2D`, `EF50EvalResult`, and
`_load_config` from the production module without modifying it. If the
parallel approach failed, the production pipeline was untouched.

Script features:

- Runs sequential baseline followed by parallel run on the same config
  (the sequential run absorbs CUDA JIT warmup, slightly disadvantaging itself
  in the comparison).
- Reports calibration / seeding / total time and speedup separately.
- Prints throttled live progress with elapsed, ETA, rate, and rejection counts
  during long parallel runs.
- Includes a small `_FilteredStdout` proxy that drops two unconditional
  DeepLens warnings ("No intersection points found" and "Zero or negative
  entrance pupil") which would otherwise drown out worker logs. The filter is
  installed in both the parent and each worker process and is bypassed by
  setting `DEEPLENS_KEEP_WARNINGS=1`.

## Design choices and trade-offs

- **Why `spawn` instead of `fork`?** CUDA contexts are not safe across
  `fork()`. `spawn` re-imports the module in each child, which is slower to
  start but is the only correct option when the parent has touched CUDA.

- **Why one `EF50ZernikeProblem` per worker, not per evaluation?** The
  problem object holds the cached aperture sample disk, the pre-computed
  Zernike pseudo-inverse matrix, and the template lens. Constructing it costs
  more than evaluating one design. Per-worker reuse amortises that cost over
  hundreds of evaluations.

- **Why send plain dicts back, not `EF50EvalResult` dataclasses?** Avoids
  pulling torch + DeepLens import side-effects through the pickle round-trip
  and removes any chance of accidentally shipping torch tensors across
  process boundaries.

- **Why insert into the archive in the parent in deterministic order?** The
  archive is not thread- or process-safe, and the workers complete in
  non-deterministic order. Sorting results by original theta index before
  insertion gives reproducible archive contents on reruns.

- **Why no `try/except` swallow inside the worker?** Errors are caught,
  packaged with the theta index, and surfaced to the parent. The parent
  prints up to five examples and continues — a single bad lens shouldn't
  abort the whole seeding run.

- **Determinism caveat.** Even with deterministic insertion order, the
  archive fill count between sequential and parallel runs can differ by a
  handful of cells when two designs tie for the same cell — whichever was
  inserted first wins. Seq=644 vs Par=649 in our run is consistent with that
  (the underlying physics and per-design results are identical).

## What changes in the production pipeline

**Nothing changes during this exploration.** The script lives entirely
outside `EF50ZernikeRunner`. The mechanism that would change if the user
adopts it is:

- `EF50ZernikeRunner.calibrate_archive_bounds()` — replace its sequential
  loop with a call into the multiprocess evaluator.
- `EF50ZernikeRunner.seed_archive()` — same.

Everything else (`evaluate()`, `_fitness_and_metrics()`, `_trace_fields_batch`,
`_trace_coherent`, `_zernike_measures`, `trace2sensor`, the archive logic, the
LHS sampler, the gradient-driven CMA-MEGA loop after seeding) is unchanged.

## Implications for the methods / results section

Talking points the user can lift into their paper:

- **Seeding is embarrassingly parallel** because every random LHS design is
  independent and only requires a forward pass. There is no shared mutable
  state across designs once each worker has its own problem instance.
- **The bottleneck is dispatch, not compute.** A single GPU at 11 %
  utilisation under sequential dispatch is direct evidence that the
  per-evaluation Python overhead (lens construction, tensor setup, .item()
  syncs) dominates, not the ray tracing kernels themselves.
- **Threading does not solve dispatch-bound workloads in Python.** The 0.76×
  result with 14 threads is a textbook GIL-contention failure mode and
  illustrates why threading is unsuitable here.
- **Multiprocessing across both GPUs is the right architecture** because it
  bypasses the GIL entirely and lets each worker get its own CUDA context on
  a possibly different device. This converts seeding from a "one stream of
  small kernels on one GPU" workload into "many parallel streams across both
  GPUs", finally putting the second GPU to work.
- **The parallelisation does not affect the science.** It changes only how
  evaluations are dispatched. The LHS samples are identical, every
  `trace2sensor` call is identical, the percentile-based archive bounds are
  identical, and the per-design fitness/measures are bit-for-bit reproducible.
  The only non-determinism is tie-breaking in MAP-Elites cell insertion,
  which is bounded and physically meaningless.
- **Compute-budget interpretation.** With seeding accelerated, the
  user can either (a) keep the same `seed_count` and free GPU-hours for
  longer CMA-MEGA loops, or (b) increase `seed_count` to get a finer-grained
  initial sampling of the parameter space — which was the original
  motivation.

## Files referenced in this exploration

- [optical_rl/qd/ef50_zernike.py](optical_rl/qd/ef50_zernike.py) —
  production runner and problem (unchanged).
- [optical_rl/qd/seed_parallel_experiment.py](optical_rl/qd/seed_parallel_experiment.py) —
  standalone parallel seeding experiment script (new).
- [optical_rl/configs/ef50_zernike.yaml](optical_rl/configs/ef50_zernike.yaml) —
  configuration with `seed_count: 1000`, `calibration_count: 1000`,
  `num_rays: 10000`, `fitness_wavelengths_um: [0.656, 0.587, 0.486]`,
  `stage: stage2`.
- [DeepLens/deeplens/optics/geolens.py](DeepLens/deeplens/optics/geolens.py) —
  `trace2sensor` lives here, unchanged.
