# Exported Chat Context — EF50 Zernike Parallel Seeding: Out-of-Bounds Measures

**Date of conversation:** 2026-04-11
**Repo root:** `/home/hisham26/Desktop/RL`
**Primary file investigated:** `optical_rl/qd/ef50_zernike_parallel.py`
**IDE-opened file (incidental, unrelated):** `DeepLens/2_autolens_rms.py`

---

## User's question

> What happens, in the seeding part of the EF50 Zernike parallel process, if one of the randomly sampled points has measures *outside* the measure range?

The user is writing the Results section of a paper and wanted to understand exactly how the seeding pipeline handles a sampled design whose behavioral descriptors (Zernike-based measures) fall outside the configured `archive_bounds` of the MAP-Elites archive.

---

## What I investigated

I traced the seeding code path in `optical_rl/qd/ef50_zernike_parallel.py`. The relevant pieces:

### 1. `EF50ZernikeRunner.seed_archive()` — lines 1143–1192

PARALLEL variant. Dispatches all seeding evaluations to the multi-GPU multiprocessing pipeline from `seed_parallel_experiment`. Builds `all_thetas` by stacking the initial theta with `seed_count` Latin-hypercube samples drawn via `self.problem.sample_theta(...)`. Calls `_parallel_evaluate(...)` and then iterates over results:

```python
for r in results:
    if r is None:
        continue
    # Skip rejected designs entirely — do not insert into the archive.
    if r["metrics"].get("rejected", 0.0) > 0.0:
        continue
    eval_result = EF50EvalResult(
        theta=r["theta"],
        fitness=r["fitness"],
        measures=r["measures"],
        metrics=r["metrics"],
        grads=None,
    )
    self._maybe_log_seed_accept(self.archive.insert(eval_result), eval_result)
return self.archive.stats()
```

Two filters happen here:
- **Filter A (explicit):** designs whose `metrics["rejected"] > 0` are dropped (e.g. ray-trace failures, geometric violations).
- **Filter B (implicit, downstream in `insert`):** valid designs whose measures fall outside `archive_bounds`.

### 2. `EF50Archive2D.insert()` — lines 256–265

```python
def insert(self, result: EF50EvalResult) -> bool:
    idx = self._to_index(result.measures)
    if idx is None:
        return False
    current = self._cells.get(idx)
    self._best_fitness = max(self._best_fitness, float(result.fitness))
    if current is None or float(result.fitness) > float(current.fitness):
        self._cells[idx] = result
        return True
    return False
```

Key observation: when `_to_index` returns `None`, the function returns `False` *before* `self._best_fitness` is updated. So an out-of-bounds design has zero effect on archive state.

### 3. `EF50Archive2D._to_index()` — lines 247–254

```python
def _to_index(self, measures):
    (x_lo, x_hi), (y_lo, y_hi) = self.bounds
    x, y = float(measures[0]), float(measures[1])
    if not (x_lo <= x <= x_hi and y_lo <= y <= y_hi):
        return None
    def _bin(v, lo, hi, n):
        return int(np.clip(np.floor((v - lo) / (hi - lo) * n), 0, n - 1))
    return (_bin(x, x_lo, x_hi, self.bins[0]), _bin(y, y_lo, y_hi, self.bins[1]))
```

Strict closed-interval bounds check. No clipping, no nearest-cell projection — out-of-bounds → `None`.

### 4. `_maybe_log_seed_accept()` — lines 1131–1141

```python
def _maybe_log_seed_accept(self, inserted: bool, result: EF50EvalResult) -> None:
    if not inserted or not self.config.log_seed_accepts:
        return
    ...
```

If `insert` returned `False` (which is the case for out-of-bounds), the logger early-returns. No print, no counter.

---

## Final answer given to the user

When a seeding sample's measures fall outside `archive_bounds`:

1. The design **is silently dropped**.
2. It is **not** placed in any archive cell.
3. `_best_fitness` is **not** updated (the bounds check happens before that line).
4. `_maybe_log_seed_accept` early-returns with `inserted=False` — **no log line, no warning, no counter**.
5. The only externally visible trace is that `seed_stats["filled"]` (returned by `archive.stats()`) ends up smaller than `seed_count`. There is no separate "out-of-bounds" tally distinct from "rejected".

### Important distinction for the paper

There are **two** different ways a seed can fail to enter the archive, and they are handled at different points in the pipeline but produce indistinguishable summary statistics:

| Failure mode | Where it's filtered | What it means |
|---|---|---|
| `metrics["rejected"] > 0` | `seed_archive`, line 1182 | Geometrically/optically invalid design (e.g. ray-trace failure, constraint violation) |
| `_to_index` returns `None` | `EF50Archive2D.insert`, line 258 | Valid design, but its Zernike behavioral descriptors land outside `archive_bounds` |

A consequence worth flagging in the Results section: **if `archive_bounds` are set too tight, low coverage will look the same as low yield from rejection** — there is no logging that distinguishes "the lens design failed" from "the lens design was fine but its descriptor was out of the user-specified MAP-Elites region". If you want to report or discuss seeding yield in the paper, this is a measurement gap.

---

## File/line references (for the downstream agent)

- `optical_rl/qd/ef50_zernike_parallel.py:1143–1192` — `seed_archive()` parallel variant
- `optical_rl/qd/ef50_zernike_parallel.py:1178–1191` — result loop with rejection filter and insert call
- `optical_rl/qd/ef50_zernike_parallel.py:256–265` — `EF50Archive2D.insert`
- `optical_rl/qd/ef50_zernike_parallel.py:247–254` — `EF50Archive2D._to_index` (the bounds check)
- `optical_rl/qd/ef50_zernike_parallel.py:1131–1141` — `_maybe_log_seed_accept`
- `optical_rl/qd/ef50_zernike_parallel.py:1163–1167` — how `all_thetas` is constructed (initial theta + Latin-hypercube samples)
- `optical_rl/qd/ef50_zernike_parallel.py:707–709` — `sample_theta` (Latin-hypercube unit sampling)
- `optical_rl/qd/ef50_zernike_parallel.py:95–96` — `_latin_hypercube` helper

## Config knobs relevant to this behavior

- `seed_count` (default 128) — number of Latin-hypercube seeds
- `archive_bins` — 2D MAP-Elites grid resolution
- `archive_bounds` — `((x_lo, x_hi), (y_lo, y_hi))` for the two Zernike descriptors; **this is the range that determines the silent-drop behavior**
- `descriptor_zernike_indices` — which two Zernike coefficients are used as behavioral descriptors
- `log_seed_accepts` (default True) — only logs *accepted* inserts; does not log drops

## Note on what was NOT investigated in this chat

- Whether `archive_bounds` is auto-calibrated by `calibrate_archive_bounds()` (referenced at line 1161 but not read in this conversation) — if calibration uses the same Latin-hypercube samples, the practical out-of-bounds rate might be small by construction. Worth checking before making strong claims in the paper.
- Whether the sequential variant `ef50_zernike.py` differs in this behavior (it shares `EF50Archive2D` based on the import structure, so likely identical, but not verified).
- Concrete out-of-bounds rates from any actual run — none of the result directories under `results/models/ef50_zernike_*` were inspected.
