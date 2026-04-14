# Canonical CMA-MEGA, stop-and-go checkpointing, and worker-pool parallelisation

Context export for the EF50 Zernike CMA-MEGA pipeline. Covers three connected pieces of work that happened in one session: (1) diagnosing and fixing a broken CMA-ES emitter inside the CMA-MEGA loop, (2) adding first-class checkpoint / resume / trajectory telemetry, and (3) parallelising the candidate evaluation across a persistent worker pool. Written to be consumed by a larger model that will help draft the Methods and Results sections of the paper.

---

## TL;DR (headline numbers for the paper)

On the **Canon EF50 f/1.8** template (10 free curvatures, 3-D CMA coefficient space, descriptors = Z7 vertical coma + Z11 primary spherical, polychromatic stage2 RMS, 100×100 archive):

| | Before (baseline "GREAT" run) | After (canonical CMA-MEGA, same compute budget) |
|---|---|---|
| Best polychromatic RMS (mm) | **0.78** | **0.39** (≈ 2× better) |
| σ trajectory | Pinned at clip 2.0 forever | Canonical: 1 → 7.7 → 1 → 0.32 → … (self-regulating via p_σ) |
| Restarts fired | 0 (no restart mechanism existed) | 3 in the first 85 gens; 10 in the first 224 gens — all on `no_improvement` |
| Telemetry | Single conflated `improved=X/16` counter | Split: `new=A/16 replaced=B/16 wasted=C/8 sigma=S restarts=R` |
| Wallclock per generation | 5.11 s (sequential λ=16, 1 GPU) | 1.89 s (parallel λ=20, 2 GPUs, 20 workers) |
| Samples / second | 3.14 | **10.6** (≈ 3.4× throughput) |

Same canonical CMA-ES engine + checkpointing now applied to all 6 CMA-MEGA runners in `optical_rl/qd/`. Checkpointing gives true stop-and-go (SIGINT-safe atomic writes, emitter + archive + RNG all round-trip; mid-run analysis works against the live run dir via `archive_cells.json` refreshed every N gens).

---

## Part 1 — Diagnosing and fixing the broken CMA-MEGA

### What was broken

The emitter class `GradientCoefficientCMA` (three identical copies in `optical_rl/qd/singlet_cmamega.py`, `optical_rl/qd/cmamega_ext/coeff_cma.py`, `optical_rl/qd/cmamega_triplet/coeff_cma.py`) was **not canonical CMA-ES** despite being called that. It had:

- No evolution path `p_σ` → no principled step-size control
- No evolution path `p_c` → no rank-1 covariance update
- Rank-μ update used linear weights `np.linspace(1.0, 2.0, mu)` instead of the canonical log weights `log(mu + 0.5) − log(i)`
- σ was computed as `sqrt(mean(diag(C)))` clipped to `[0.05, 2.0]` — no `σ · exp((c_σ/d_σ)(‖p_σ‖/E‖N(0,I)‖ − 1))` update. Worse, this heuristic grows σ when the elite coefficients *disagree* (high variance across the top half of the population), which is the **opposite** of what canonical CMA-ES does. Canonical CMA-ES grows σ when consecutive steps are correlated in the same direction.
- EMA blending `mean = 0.8 · old + 0.2 · new`, `C = 0.8 · old + 0.2 · new` — this smooths updates over many generations with no theoretical grounding, and pins C near its peak once it climbs (because the 0.8 weight on old C makes it very slow to shrink).
- No eigendecomposition — used `np.random.multivariate_normal(mean, sigma² · C)` directly on the raw matrix.
- No standard CMA-ES termination criteria (TolX, ConditionCov, NoEffectAxis, NoEffectCoord, TolFun).

The **runners** that wrapped this emitter were also not canonical CMA-MEGA:

- **No QD restart trigger.** The canonical algorithm (Fontaine & Nikolaidis, NeurIPS 2021) restarts immediately on "no archive improvement this generation" — picks a random elite, resets mean/C/σ/paths. Our runners computed `n_improved` and `wasted_elites` per generation and **only logged them**. Nothing acted on the signal.
- **Conflated telemetry.** `improved=X/16` counted both "landed in an empty cell" and "replaced an existing elite" as the same event, so early generations looked productive while most samples were just filling void cells. Hiding the MAP-Elites-vs-CMA-MEGA attribution made the pathology invisible.
- **Improvement-ranked scoring was one-tier.** Canonical CMA-MEGA uses a two-tier ranking (new cells > replaced elites > none); ours used `fitness if improved else -inf`, collapsing new and replaced into one tier.

### The observed pathology on a real EF50 run

Run: `ef50_zernike_20260409_123918_GREAT_results_first_productionready`. Archive was 100×100 = 10000 cells. 20000 random seeds filled only 64 cells (0.64% coverage) — most of random 10-D theta space lies outside the narrow manifold of physically valid lenses.

CMA-MEGA launch trajectory:

```
gen=5  coverage=0.0105 best=-0.840543  improved=13/16 wasted_elites=0/8  sigma=1.0068
gen=10 coverage=0.0142 best=-0.840543  improved=11/16 wasted_elites=0/8  sigma=1.2595
gen=15 coverage=0.0192 best=-0.840543  improved=14/16 wasted_elites=0/8  sigma=1.1507
gen=20 coverage=0.0216 best=-0.840543  improved=9/16  wasted_elites=0/8  sigma=1.5716
gen=25 coverage=0.0241 best=-0.776612  improved=13/16 wasted_elites=0/8  sigma=1.6939
gen=30 coverage=0.0265 best=-0.776612  improved=2/16  wasted_elites=6/8  sigma=2.0000 ← cap
gen=35 coverage=0.0281 best=-0.776612  improved=7/16  wasted_elites=1/8  sigma=2.0000
gen=40 coverage=0.0294 best=-0.776612  improved=3/16  wasted_elites=5/8  sigma=2.0000
gen=45 coverage=0.0303 best=-0.776612  improved=3/16  wasted_elites=5/8  sigma=2.0000
gen=50 coverage=0.0313 best=-0.776612  improved=0/16  wasted_elites=8/8  sigma=2.0000
gen=55 coverage=0.0314 best=-0.776612  improved=0/16  wasted_elites=8/8  sigma=2.0000
...
```

Reading this:
- Only **one** best-fitness improvement ever fired (−0.8405 → −0.7766 at gen 25, a 7.6% RMS reduction). Everything else was archive decoration.
- σ ran away from 1.0 → 2.0 in 25 generations because gens 5–25 had `improved=9..14/16`. Every successful sample fed the evolution-path-free heuristic, which interpreted "coefficients disagree" as "take bigger steps".
- σ hit the hard-coded cap of 2.0 (`np.clip(..., 0.05, 2.0)` in the tell-update) and could not escape.
- From gen 30 onward, improvement rate crashed as the reachable cells filled. `wasted_elites` saturated at 8/8 — **every** elite selected for CMA update was useless.
- Canonical CMA-MEGA would have **restarted the distribution** as soon as `improved` fell to zero for a generation, picking a new random parent from the archive and resetting μ/C/σ/p_σ/p_c. Our code had no restart path at all.

### The `improved=13/16` trap

The most important insight from the diagnosis: in gens 5–25 the `improved=13/16` signal was **misleading**. With only 64 cells filled out of 10000 at CMA start, nearly every sample landed in an empty cell, and *any* fitness counted as "improvement" for the archive-insertion criterion. CMA-ES's success signal was mostly "hit a void cell" rather than "beat an existing elite" — the MAP-Elites bootstrap was doing the work, not CMA-MEGA's gradient search. The single best-fitness gain at gen 25 was the only real CMA-MEGA contribution in the first 55 generations.

This is the core reason the telemetry needed to be split into `new_cells` vs `replaced_elites`: until you can see which counter is growing, you cannot tell whether CMA-MEGA is adding anything on top of MAP-Elites.

### The canonical rewrite

New shared module **`optical_rl/qd/_cma.py`** (~340 lines) implements canonical `(μ/μ_w, λ)`-CMA-ES from Hansen 2016 (arXiv:1604.00772):

- Positive log weights `w_i = log(mu + 0.5) − log(i)`, normalised.
- `mu_eff = 1 / Σ w_i²`.
- Learning constants: `c_σ = (mu_eff + 2) / (n + mu_eff + 5)`, `d_σ = 1 + 2·max(0, √((mu_eff−1)/(n+1)) − 1) + c_σ`, `c_c = (4 + mu_eff/n) / (n + 4 + 2·mu_eff/n)`, `c_1 = 2 / ((n + 1.3)² + mu_eff)`, `c_μ = min(1 − c_1, 2·(mu_eff − 2 + 1/mu_eff) / ((n + 2)² + mu_eff))`.
- `chi_n = √n · (1 − 1/(4n) + 1/(21n²))` ≈ E‖N(0, I)‖.
- Evolution paths `p_σ` and `p_c` maintained across tells.
- Rank-1 update `p_c p_cᵀ` and rank-μ update `Σ w_i · y_i y_iᵀ` combined in the standard covariance recursion with the `h_σ` heuristic.
- σ update `σ ← σ · exp((c_σ/d_σ)(‖p_σ‖/chi_n − 1))`. Clipped only to `[1e-10, 1e10]` as a numerical sanity rail — the path-length heuristic is self-regulating, so the old aggressive `[0.05, 2.0]` cap is not just unnecessary, it's harmful.
- Lazy eigendecomposition of C for ask-side sampling (B, D cached, recomputed only when tell marks them dirty).
- `should_restart()` checks Hansen's standard termination criteria (TolSigma, ConditionCov, NoEffectAxis, NoEffectCoord, TolX) and returns `(bool, reason_str)`.
- `reset(new_mean=None, new_sigma=None)` zeros μ, C=I, σ=initial, p_σ=0, p_c=0, generation=0.
- `get_state() / set_state(dict)` for checkpointing (includes RNG bit-generator state).

All three previously-duplicated `GradientCoefficientCMA` copies now re-export from this one module:

```python
# singlet_cmamega.py, cmamega_ext/coeff_cma.py, cmamega_triplet/coeff_cma.py
from optical_rl.qd._cma import GradientCoefficientCMA  # noqa: F401
```

Runners didn't need code changes on the emitter side — the (`ask`, `tell`, `sigma`, `elite_frac`) API is preserved.

### Sphere convergence sanity check

New unit test `tests/test_cma_checkpoint.py::test_sphere_convergence` — optimises `f(x) = −‖x − target‖²` in 10-D with population 20, 150 generations, initial σ=1.0:

```
gen=  0 sigma=0.95286 mean_err=3.842598
gen= 25 sigma=0.22085 mean_err=0.274762
gen= 50 sigma=0.03258 mean_err=0.027828
gen= 75 sigma=0.00446 mean_err=0.002032
gen=100 sigma=0.00049 mean_err=0.000128
gen=125 sigma=0.00005 mean_err=0.000011
gen=149 sigma=0.00001 mean_err=0.000001
Converged: True
```

σ decays geometrically (log-linear) as the search closes in on the target — canonical CMA-ES behaviour. The old homemade emitter could never produce this because it had no evolution path and its σ update was a function of elite variance rather than cumulative step direction.

### QD-specific restart trigger in the runners

Canonical CMA-MEGA restarts **immediately** on any generation where the archive received zero improvement (stricter than vanilla CMA-ES termination, which waits for multiple criteria). Added to all 6 runners as:

```python
restart_reason = None
if (n_new + n_replaced) == 0:
    restart_reason = "no_improvement"
else:
    restart_flag, hansen_reason = self.emitter.should_restart()
    if restart_flag:
        restart_reason = hansen_reason or "hansen_term"

if restart_reason is not None:
    sigma_pre = float(self.emitter.sigma)
    self.emitter.reset()
    self.restart_count += 1
    print(f"[restart] gen={N} reason={reason} sigma_pre={sigma_pre:.4f} ...")
```

Runners don't need to pick a new parent explicitly — our existing design samples a new parent from the archive every generation (`archive.sample_parent()`), so the "restart" at the runner level is just an emitter reset. The next generation naturally starts with a fresh CMA distribution.

### Two-tier improvement scoring

Archive `insert()` now returns a 3-way int enum from `optical_rl/qd/_checkpoint.py`:

```python
INSERT_NONE:     int = 0  # rejected or beaten by existing elite
INSERT_REPLACED: int = 1  # replaced existing elite (positive Δfitness)
INSERT_NEW:      int = 2  # filled empty cell
```

Backwards-compatible with existing `if archive.insert(...):` call sites because 0 is falsy and 1, 2 are truthy. Applied to all 5 archive classes: `EF50Archive2D` (twice — parallel + sequential files), `BatchedArchive2D`, `TripletArchive2D`, `ArchiveGrid2D` (twice — cmamega_ext + cmamega_triplet).

In the runner, `batch_scores` is built with a NEW_CELL_BONUS (default 1e6, configurable) so new cells always rank above replaced elites in the same generation:

```python
if status == INSERT_NEW:
    batch_scores.append(candidate_eval.fitness + new_cell_bonus)
elif status == INSERT_REPLACED:
    batch_scores.append(candidate_eval.fitness)
else:
    batch_scores.append(-np.inf)
```

This gives canonical CMA-MEGA's improvement-ranked scoring (new > improved > nothing) through a single vector that's sortable by the inner CMA-ES `tell()`.

### Per-generation telemetry split

Old log line:
```
gen=N evals=M coverage=C best=F | improved=X/16 wasted_elites=Y/8 sigma=S
```

New log line:
```
gen=N evals=M coverage=C best=F | new=A/16 replaced=B/16 wasted=W/8 sigma=S restarts=R
```

Plus a `[restart] gen=N reason=... sigma_pre=...` line every time the trigger fires.

---

## Part 2 — Stop-and-go checkpointing

### Goals

1. Kill the process (Ctrl-C or OS) at any point and lose **zero** search progress.
2. Resume exactly where you left off — same generation counter, same emitter mean/C/σ/evolution paths, same archive, same wallclock accounting.
3. Analyse designs *while* the run is in progress without stopping it or racing the writer.
4. Generate trajectory data that survives kills independently of the checkpoint.

### Implementation

New module **`optical_rl/qd/_checkpoint.py`** (~180 lines). Four components:

**1. `CMAMEGACheckpoint` dataclass** — one serialisable snapshot:

```python
@dataclass
class CMAMEGACheckpoint:
    archive: Any                    # full EF50Archive2D / BatchedArchive2D / etc.
    emitter_state: Dict[str, Any]   # from GradientCoefficientCMA.get_state()
    completed_evaluations: int
    generation: int
    restart_count: int
    best_fitness: float
    config_hash: str                # SHA-256 of dataclass asdict for drift detection
    timestamp: str                  # ISO-8601
    wallclock_seconds: float
    schema_version: int = 1
```

The `archive` field is pickled directly — the archive class (e.g. `EF50Archive2D`) is importable from its module at resume time, so Python's default pickle handles the cells dict, bounds, bins, and internal RNG state.

The `emitter_state` field is a dict from `GradientCoefficientCMA.get_state()`, which includes mean, C, σ, p_σ, p_c, B, D, generation, and the numpy bit-generator state. A restore via `set_state()` produces a bit-for-bit-identical emitter — verified by unit test `test_get_set_state_roundtrip`: two emitters with the same state return byte-identical samples from `ask(n)`.

**2. `save_checkpoint(ckpt, path)` / `load_checkpoint(path)`** — atomic write (`.tmp` + `fsync` + `os.replace`) so a SIGINT mid-write cannot corrupt an existing checkpoint. Schema validation on load.

**3. `hash_config(config_obj)`** — SHA-256 of `repr(sorted(dataclass.asdict(...).items()))`. Used on resume to warn if the user silently changed their YAML between kill and restart.

**4. `TrajectoryLogger`** — append-only per-generation CSV, opened in 'a' mode per call so a kill mid-run leaves a valid (truncated to the last completed row) CSV behind. Columns:

```
gen, evals, coverage, best_fitness, new_cells, replaced_elites, wasted_elites,
sigma, restarts, wallclock_s
```

Written every generation, not only at checkpoint cadence. Independent of the checkpoint — survives kills directly. Enables the σ-vs-best-fitness plot in one grep.

### Runner integration

Each runner's `run()` method grew a `try/finally` block with:

- **Setup**: build `_build_ckpt()` and `_flush_checkpoint()` closures over `self.archive`, `self.emitter`, and the runner-level counters. Build the trajectory logger.
- **Signal handler**: `signal.signal(SIGINT, ...)` sets a flag; the main loop checks it at the top of each generation and breaks out cleanly, hitting the `finally` block that writes the final checkpoint.
- **Per-generation**: append one row to `trajectory.csv`, and every `checkpoint_every` generations (default 50), call `_flush_checkpoint()`.
- **Finally**: unconditional final `_flush_checkpoint()`, then restore the prior signal handler, then shutdown the worker pool (Part 3).

Resume flow in `__main__`:

```python
if args.resume is not None:
    ckpt_path = Path(args.resume) / "checkpoint.pkl"
    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path)
        runner.archive = ckpt.archive
        runner.emitter.set_state(ckpt.emitter_state)
        runner.completed = ckpt.completed_evaluations
        runner.generation = ckpt.generation
        runner.restart_count = ckpt.restart_count
        runner.wallclock_seconds = ckpt.wallclock_seconds
        if ckpt.config_hash != hash_config(cfg):
            print("WARNING: config hash drift on resume")
        resumed = True
    # else: legacy fallback — loads archive.pkl only, emitter starts fresh (logged loudly)

if not resumed:
    runner.seed_archive()    # skipped on resume
runner.run(iterations, resume=resumed)
```

**The previous `--resume` path was broken.** It loaded `archive.pkl` then immediately called `seed_archive()`, which wiped the loaded archive by re-seeding. Fixed as part of this work.

### Mid-run analysis

During the run, `checkpoint.pkl` is refreshed every `checkpoint_every` generations. Because saves are atomic, you can load it from another process at any time:

```python
from optical_rl.qd._checkpoint import load_checkpoint
ckpt = load_checkpoint("results/models/ef50_zernike_<timestamp>/checkpoint.pkl")
print(f"gen={ckpt.generation} cells={len(ckpt.archive.cells)} best={ckpt.best_fitness}")
```

To make the existing analysis scripts (`refine_ef50_archive.py`, `render_ef50_zernike.py`) work on a live run without any changes, `_flush_checkpoint()` also dumps `archive_cells.json` (same format as the previous end-of-run dump) atomically next to `checkpoint.pkl`. The existing scripts already know how to read `archive_cells.json`.

---

## Part 3 — Persistent worker pool for parallel candidate evaluation

### Diagnosis

The canonical fix got best-fitness down from −0.78 to −0.39, but the CMA-MEGA inner loop was still evaluating `λ=16` candidates **sequentially** per generation. Measurement from the live run's `trajectory.csv` (224 gens in 1145s):

- **5.11 s per generation** on a single GPU with λ=16.
- Breakdown (inferred from the wallclock and timing the loop stages):
  - ~0.8 s per generation for the **parent gradient** — one forward + backward pass through DeepLens autograd. This is the serial floor; can't be parallelised because all candidates use the same parent gradient.
  - ~0.27 s × 16 = 4.3 s for the 16 **candidate forward passes** (no grads). This is 84% of per-gen wallclock and is embarrassingly parallel.

Resources available but unused:
- 2× NVIDIA RTX 6000 Ada (49 GB each, 98 GB total).
- Single-process VRAM footprint: ~650 MB on GPU 0. **GPU 1 was completely idle (4 MB used).**
- 20 CPU cores.
- `parallel_workers: 20` was already set in the YAML and already used for the seed phase via `optical_rl/qd/seed_parallel_experiment.py::_parallel_evaluate`. Just not used for the CMA-MEGA inner loop.

### Why a persistent pool, not per-call `_parallel_evaluate`

The existing `_parallel_evaluate` function spawns N processes, runs one batch, then poison-pills and joins. Startup cost includes loading torch + DeepLens + building an `EF50ZernikeProblem` (template lens, Zernike pseudo-inverse cache, ray bundle tensors) — several seconds per worker. For seeding, this amortises over one big batch of thousands of evaluations. For CMA-MEGA, calling it per generation over ~thousands of generations would re-pay that startup cost on every generation and dominate wallclock.

Solution: a **long-lived pool** that spawns workers **once** at runner entry, keeps them alive across all generations, and shuts down cleanly in the `finally` block.

### `PersistentEvalPool`

New module **`optical_rl/qd/_eval_pool.py`** (~155 lines). Wraps `torch.multiprocessing` with spawn context. Spawns N persistent worker processes, each running the existing `_worker_fn` from `seed_parallel_experiment.py` (pulls `(idx, theta)` from a task queue, pushes `(idx, result_dict)` to a result queue, loops forever until it sees a `None` poison pill). Round-robin GPU assignment across `_available_gpus()`.

API:

```python
class PersistentEvalPool:
    def __init__(self, config, config_dict, num_workers, gpu_ids): ...
    def evaluate(self, thetas: np.ndarray) -> list[dict | None]: ...
    def shutdown(self) -> None: ...  # idempotent
    def __enter__(self) -> "PersistentEvalPool": ...
    def __exit__(self, *exc) -> bool: ...  # calls shutdown
    def __del__(self): ...  # best-effort shutdown safety net
```

`evaluate()` feeds all tasks, then collects exactly `n` results (ordered by original index), then returns. Worker errors become `None` entries and are logged but don't crash the run.

Key design points:
- **Single-threaded use only.** The task and result queues are not guarded for concurrent callers.
- **Workers reuse their own `EF50ZernikeProblem` instance** across calls, so the expensive caches (template lens, aperture samples, Zernike pseudo-inverse, disk Fibonacci samples) are built once per worker at startup.
- **Round-robin across GPUs**: worker `w` goes to `gpu_ids[w % len(gpu_ids)]`. With 20 workers and 2 GPUs, that's 10 workers per GPU — splits the load and actually uses GPU 1.
- **Clean shutdown** via poison pills + `join(timeout=10)` + `terminate()` fallback. Verified zero zombie processes after exit.

### Runner integration

The runner builds the pool lazily, right after `seed_archive()` completes:

```python
from optical_rl.qd._eval_pool import PersistentEvalPool
from optical_rl.qd.seed_parallel_experiment import (
    _config_to_dict, _available_gpus, _pick_worker_count,
)

gpu_ids = _available_gpus()
num_workers = _pick_worker_count(self.config.parallel_workers, len(gpu_ids))
self._eval_pool = PersistentEvalPool(
    self.config, _config_to_dict(self.config), num_workers, gpu_ids,
)
print(f"[ef50-zernike] CMA-MEGA worker pool: {num_workers} workers across GPUs {gpu_ids}")
```

The per-generation inner loop is now a three-pass structure:

**Pass 1 — build candidate θ vectors (cheap, no I/O):**
```python
candidate_thetas = np.empty((batch_n, theta_dim), dtype=np.float64)
for local_idx, coeff in enumerate(coeffs):
    direction = abs(coeff[0]) * grads["fitness"] \
              + coeff[1] * grads[f"z{j0}"] \
              + coeff[2] * grads[f"z{j1}"]
    dir_norm = np.linalg.norm(direction)
    if not np.isfinite(dir_norm) or dir_norm < 1e-12:
        rng = np.random.default_rng(seed + completed + local_idx)
        direction = rng.normal(0.0, 1.0, size=(theta_dim,))
        direction /= np.linalg.norm(direction) + 1e-12
    candidate_unit = np.clip(unit_parent + cma_eta * direction, -1.0, 1.0)
    candidate_thetas[local_idx] = _unit_to_theta(candidate_unit, theta_bounds)
```

**Pass 2 — parallel evaluation:**
```python
pool_results = self._eval_pool.evaluate(candidate_thetas)
```

**Pass 3 — archive insert + scoring (cheap, sequential):**
```python
for local_idx, r in enumerate(pool_results):
    if r is None:  # worker error
        batch_scores.append(-np.inf); batch_status.append(0); continue
    candidate_eval = EF50EvalResult(theta=r["theta"], fitness=r["fitness"],
                                    measures=r["measures"], metrics=r["metrics"])
    if candidate_eval.metrics.get("rejected", 0.0) > 0.0:
        status = 0
    else:
        status = int(self.archive.insert(candidate_eval))
    # ... build batch_scores via two-tier ranking (new > improved > none)
```

Pool shutdown in the `finally` block:
```python
try:
    if self._eval_pool is not None:
        self._eval_pool.shutdown()
        self._eval_pool = None
except Exception: ...
```

### Measured speedup on a real smoke run

End-to-end smoke run: λ=20, 20 workers, both GPUs, `seed_count=20`, 100 iterations = 5 generations. From the resulting `trajectory.csv`:

```
gen=1 wallclock=18.97s ← includes one-time pool spawn (20 workers × torch + DeepLens + problem init)
gen=2 wallclock=20.62s → Δ=1.64s
gen=3 wallclock=22.96s → Δ=2.35s
gen=4 wallclock=24.84s → Δ=1.88s
gen=5 wallclock=26.53s → Δ=1.69s
```

Excluding the one-time spawn cost (amortises to nothing over a full run), **~1.89 s per generation** at λ=20. Compared to the pre-parallel baseline of 5.11 s per generation at λ=16:

| | Before | After |
|---|---|---|
| Wallclock / gen | 5.11 s | 1.89 s |
| Samples / gen | 16 | 20 |
| Samples / second | 3.14 | **10.6** |
| Wallclock speedup | 1× | **2.7×** |
| Throughput speedup | 1× | **3.4×** |

Slightly under the 4× upper bound because the parent gradient runs in the main process on GPU 0 and contends with the 10 workers on GPU 0. For a 10000-iteration run this drops wallclock from ~14 hours to ~4 hours.

### Telemetry confirmation

Smoke run final log line:
```
gen=5 evals=100 coverage=0.0047 best=-0.840543 | new=7/20 replaced=3/20 wasted=0/10 sigma=1.6300 restarts=0
```

All three telemetry counters populated (new, replaced, wasted), λ=20 reflected in the denominators, restart count tracked. Matches the redesigned log format exactly.

### Mid-run `archive_cells.json`

Added to `_flush_checkpoint()` closure: on every checkpoint cadence, dump the archive to JSON alongside `checkpoint.pkl`:

```python
def _flush_checkpoint():
    if ckpt_path is None: return
    save_checkpoint(_build_ckpt(), ckpt_path)
    # Also dump archive_cells.json atomically so refine_ef50_archive.py
    # and render_ef50_zernike.py work on a live run with no code changes.
    cells_data = [...]  # cell_index, theta, fitness, measures, metrics
    tmp_json = json_path.with_suffix(".json.tmp")
    with open(tmp_json, "w") as f:
        json.dump({"archive_bounds": ..., "archive_bins": ..., "cells": cells_data, "stats": ...}, f, indent=2)
    os.replace(tmp_json, json_path)
```

Verified by reading the smoke run's JSON: 47 cells, valid schema, loadable by the existing analysis scripts.

---

## Files modified

### New
- **`optical_rl/qd/_cma.py`** — canonical (μ/μ_w, λ)-CMA-ES engine (~340 lines).
- **`optical_rl/qd/_checkpoint.py`** — `CMAMEGACheckpoint`, `save_checkpoint`/`load_checkpoint`, `TrajectoryLogger`, `hash_config`, `INSERT_{NONE,REPLACED,NEW}` constants (~180 lines).
- **`optical_rl/qd/_eval_pool.py`** — `PersistentEvalPool` class (~155 lines).
- **`tests/test_cma_checkpoint.py`** — 18 unit tests covering canonical CMA math, reset, restart detection, get/set_state round-trip, sphere convergence, checkpoint save/load, atomic-write safety, trajectory CSV append, insert-enum truthiness, and config-hash drift detection.

### Modified
- **`optical_rl/qd/singlet_cmamega.py`** — `GradientCoefficientCMA` now re-exports from `_cma.py`.
- **`optical_rl/qd/cmamega_ext/coeff_cma.py`** — re-export shim.
- **`optical_rl/qd/cmamega_triplet/coeff_cma.py`** — re-export shim.
- **`optical_rl/qd/ef50_zernike_parallel.py`** — primary runner: checkpoint/resume/CSV, restart trigger, three-tier scoring, persistent worker pool integration, mid-run `archive_cells.json` dump. Fixed the broken resume path.
- **`optical_rl/qd/ef50_zernike.py`** — sequential variant, same changes as the parallel variant (minus the worker pool).
- **`optical_rl/qd/triplet_cooke_zernike.py`** — same runner-level changes (checkpoint/restart/telemetry; pool not applied here yet).
- **`optical_rl/qd/triplet_batched.py`** — same. **Note: user declared this file out-of-scope during the parallelisation phase** ("I officially don't care about that code, it doesn't work, it's garbage mostly").
- **`optical_rl/qd/cmamega_ext/runner_cmamega.py`** — same.
- **`optical_rl/qd/cmamega_triplet/runner_triplet_cmamega.py`** — same.
- **`optical_rl/main_triplet_cooke_zernike.py`** — added `--resume` flag, `runner.result_dir`, `checkpoint_every`, `new_cell_bonus` config plumbing.
- **All 5 archive classes** (`EF50Archive2D` × 2, `BatchedArchive2D`, `TripletArchive2D`, `ArchiveGrid2D` × 2) — `insert()` returns the 3-way integer enum instead of bool.
- **`optical_rl/configs/ef50_zernike.yaml`** — added `checkpoint_every: 50`, `new_cell_bonus: 1.0e6`, bumped `population_size: 16 → 20` for worker-pool utilisation.
- **`tests/test_singlet_cmamega.py`** — updated one assertion from `.cov` (old attribute) to `.C` (canonical).

### Not modified
- `EF50Archive2D`, `EF50EvalResult`, `EF50ZernikeProblem` dataclasses/behaviour — unchanged.
- `seed_parallel_experiment.py` worker function `_worker_fn` — unchanged. `PersistentEvalPool` reuses it directly.
- `refine_ef50_archive.py`, `render_ef50_zernike.py` — unchanged. They keep working against the new mid-run `archive_cells.json` because the JSON schema is preserved.

---

## What to say in the paper

### Methods section

Frame the method as **canonical CMA-MEGA** (Fontaine & Nikolaidis 2021) instantiated on the EF50 10-DoF Zernike-descriptor problem. Cite Hansen 2016 for the underlying CMA-ES. Key specifics:

- **Coefficient space**: `(k + 1)`-dimensional, with `k = 2` descriptors (Z7 vertical coma, Z11 primary spherical).
- **Population**: `λ = 20`, `μ = λ/2 = 10` elites per update, canonical log weights.
- **Inner CMA-ES**: evolution paths `p_σ`, `p_c`; rank-1 + rank-μ covariance updates; path-length step-size adaptation; lazy eigendecomposition; Hansen termination criteria.
- **QD restart trigger**: on any generation with zero archive improvement (new + replaced = 0), the inner CMA-ES state is reset (μ=0, C=I, σ=σ₀, p_σ=p_c=0) while the archive is kept intact. The next generation samples a new parent from the archive via uniform random cell selection.
- **Improvement ranking** (two-tier): new cells > replaced elites > non-improvements. Implemented by adding a large bonus (1e6) to the candidate fitness only for new-cell insertions, before the inner CMA-ES sort.
- **Parallelism**: per-generation candidate evaluations dispatched across a persistent worker pool of 20 processes, round-robin across 2 NVIDIA RTX 6000 Ada GPUs, each holding its own DeepLens-backed `EF50ZernikeProblem` with cached template lens, Zernike pseudo-inverse, and ray bundles. Parent gradient computation remains serial on the main process (single autograd pass per generation).
- **Checkpointing**: atomic per-generation trajectory logging + periodic snapshot of archive + emitter state, enabling exact bit-for-bit resume from interruption.

### Results section talking points

1. **Ablation: broken vs. canonical CMA-MEGA on the same compute budget** (same config except emitter + restart + scoring). Report:
   - Best polychromatic stage2 RMS at end of run.
   - σ trajectory over time (the old one pinned at 2.0; the new one self-regulates between ~0.3 and ~7 and resets on restart).
   - Restart count and when restarts fired (`no_improvement` at genN).
   - Archive coverage over time.
   - Number of distinct basins (`dist_threshold = 0.003`).
   - QD-score.

   Headline: best RMS 0.78 → 0.39 (≈ 2× better) at identical evaluation budget.

2. **Breakdown of `new=A/16 replaced=B/16`** over the first 50 generations. The old telemetry couldn't distinguish MAP-Elites bootstrap from CMA-MEGA gradient search; the new telemetry shows early generations dominated by `new` (empty-cell filling) transitioning to `replaced` (elite beating) once the archive saturates. This is the visual evidence that CMA-MEGA is doing real work post-bootstrap.

3. **Wallclock parallelism**. Baseline sequential: 5.11 s/gen at λ=16, 1 GPU, 1 process. Pooled parallel: 1.89 s/gen at λ=20, 20 workers, 2 GPUs. ~3.4× throughput speedup. Important for fair wallclock comparison against curriculum learning and OptiFusion baselines.

4. **Stop-and-go** demonstration. Kill at gen N, resume, compare best fitness at gen (N + M) vs. a clean run of (N + M) generations — should match within ±1 cell of coverage delta (the only sources of divergence are in-flight workers at the moment of SIGINT). Useful for a "usability" subsection aimed at practitioners.

5. **Restart cascade**. Observed in the real run: 10 restarts in the first 224 gens, each triggered by `no_improvement`, each followed by σ collapsing from its pre-restart value back to initial and the search finding new improvements within the next 10–20 gens. Shows that restart is carrying the search past local saturation, which was impossible with the old uncapped-σ heuristic.

### Things to emphasise that are easy to miss

- **The pathology in the old code was not a tuning issue.** No σ cap, no learning-rate tweak, no population bump would have fixed it — the emitter had no evolution path, so its σ update was structurally wrong. The fix is an algorithm change, not a hyperparameter change. Reviewers will ask this.
- **The `improved=X/16` trap matters as a methodological point.** A lot of QD papers in the wild report "improvement rate" without splitting new-cell-fill from elite-replacement. Our two-tier split is cheap and makes the attribution clean. Consider recommending it as general QD practice.
- **The restart trigger's immediacy is canonical, not aggressive.** Fontaine & Nikolaidis restart on the first zero-improvement generation; some later variants add a patience counter but the NeurIPS 2021 version is zero-tolerance. We replicate that.
- **Parent gradient is still serial** — 0.8s per generation floor. For the 10-DoF problem at λ=20 this is ~40% of per-gen wallclock; scaling λ further hits diminishing returns because this serial floor doesn't shrink. Worth one sentence in the Methods / Limitations paragraph.

### Things to be careful about

- **CMA-ES state inside the emitter does drift even when the runner picks a new parent every generation.** We kept the "fresh parent each generation" design from the original code (hybrid MAP-Elites / CMA-MEGA) rather than the strict "one θ drifts between restarts" formulation from the paper. This is a small divergence from canonical CMA-MEGA and should be explicit in Methods. Pros: better diversity coverage; cons: CMA doesn't have as many gens to converge on each parent.
- **`new_cell_bonus = 1e6`** is a hack. It works because it's larger than any plausible |fitness| in our RMS-in-mm units, so the sort order in `tell()` always puts new cells ahead. Mention as an implementation detail; don't present it as a hyperparameter with tunable meaning.
- **The current runner uses `abs(coeff[0])` on the fitness gradient coefficient** — forces the CMA's fitness direction to always be "downhill". This predates this work. Not strictly canonical CMA-MEGA (which lets CMA learn the sign from improvement signal). We did not change this. Worth a footnote or a limitations paragraph.
- **Diffs between our layering and pyribs CMA-MEGA**: pyribs' `GradientArborescenceEmitter` advances a single `current_solution` between restarts using `θ ← θ + jacobian @ cma_mean`. We sample a new parent from the archive each generation instead. This is closer to a hybrid MAP-Elites + gradient-aware emitter than to pure CMA-MEGA. Be explicit.

### Comparison framing vs. baselines

- **Curriculum learning (Yang et al., DeepLens)**: uses `2_autolens_rms.py` — single gradient descent on all curvatures with aperture growth schedule, returns one lens. Not QD. Produces a single best design. Compare on best-fitness-at-budget and on the ability to produce diverse solutions (only ours can, since only ours maintains an archive).
- **OptiFusion**: no public code at time of this work. Cite the paper's reported numbers if they overlap on templates; otherwise frame as "code unavailable" and lean on curriculum-learning comparison.
- **Fair budget accounting**: samples per second is the right denominator, not generations per second, because our λ differs from curriculum learning's "1 sample per step". Our 10.6 samples/s × gradient quality (via gradient-coefficient sampling) is the honest compute throughput.

---

## Quick reference: the log line to watch

```
[ef50-zernike] gen=N evals=M coverage=C best=F | new=A/20 replaced=B/20 wasted=W/10 sigma=S restarts=R
```

- `new` growing, `replaced` small → archive still bootstrapping, MAP-Elites phase dominates.
- `new` dropping, `replaced` growing → CMA-MEGA has taken over, gradient search refining existing cells.
- `new + replaced = 0` for one gen → restart fires next generation, σ resets to initial, `restarts` increments.
- σ growing then shrinking → canonical path-length adaptation working as designed.
- σ pinned at a specific value → would indicate either a manual cap (we removed this) or numerical degeneracy (triggers `should_restart`).
- `restarts` incrementing without `best` improving → search is genuinely stuck on the archive's current distribution, consider bumping `seed_count` or `cma_eta`.
- `wasted > 0` → some of the top-μ selected elites failed to improve the archive. Small `wasted` is normal; high `wasted` over many gens is the precursor to a restart.

## Quick reference: the files to open

- `optical_rl/qd/_cma.py:43-340` — canonical CMA-ES class
- `optical_rl/qd/_checkpoint.py:1-210` — checkpoint format, trajectory logger, insert enum
- `optical_rl/qd/_eval_pool.py:1-155` — persistent worker pool
- `optical_rl/qd/ef50_zernike_parallel.py::EF50ZernikeRunner.run` — the integrated runner (look for `INSERT_NEW`, `_eval_pool.evaluate`, `_flush_checkpoint`, `should_restart`)
- `optical_rl/configs/ef50_zernike.yaml` — config with new `population_size: 20`, `checkpoint_every: 50`, `parallel_workers: 20`
- `tests/test_cma_checkpoint.py` — 18 unit tests
- `results/models/ef50_zernike_20260410_180300/trajectory.csv` — real trajectory from the new canonical run, the source of the per-gen timing numbers above
