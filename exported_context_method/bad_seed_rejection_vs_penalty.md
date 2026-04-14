# Context: Bad-Seed Handling in `ef50_zernike.py`

## Question
Is the current code still assigning highly negative fitness (`-1e9`) to bad seeds instead of outright rejecting them?

## Short answer
**Both, by design.** The fitness function still tags invalid designs with a `-1e9` sentinel and a `"rejected": 1.0` metric, **but the runner now checks that flag and skips insertion into the MAP-Elites archive entirely** at every stage (calibration, seeding, and CMA-MEGA iteration). The `-1e9` value never reaches the archive — it only survives as feedback to the CMA emitter so the search distribution learns to avoid that region.

## Config relevant to this question
From [optical_rl/configs/ef50_zernike.yaml](optical_rl/configs/ef50_zernike.yaml):

```yaml
invalid_tol: 0.05
invalid_penalty_weight: 5.0
strict_entrance_pupil: false   # currently OFF
```

Note: `strict_entrance_pupil` is **false** in the current config, so the entrance-pupil rejection branch is inactive — only the BFL ≤ 0 (diverging-lens) rejection fires.

## Code evidence

### 1. Fitness function still emits `-1e9` for invalid designs
[optical_rl/qd/ef50_zernike.py:940-983](optical_rl/qd/ef50_zernike.py#L940-L983)

Two rejection paths inside `_fitness_and_metrics`:

- **BFL ≤ 0** (diverging lens): assigns `bad = theta_t.sum()*0 - 1e9` and returns metrics with `"rejected": 1.0`, `"rejected_bfl": 1.0`. Graph stays connected so autograd still works.
- **Strict entrance-pupil mode** (only when `strict_entrance_pupil=true`): if DeepLens silently fell back to `surfaces[0].r` for the pupil, same `-1e9` sentinel with `"rejected_pupil": 1.0`.

The soft penalties (intersection loss, invalid-fraction penalty above `invalid_tol`) still apply to *valid* designs and are unchanged.

### 2. Runner skips rejected designs at every stage

**Calibration** — [optical_rl/qd/ef50_zernike.py:1085-1092](optical_rl/qd/ef50_zernike.py#L1085-L1092):
```python
for theta in samples:
    result = self.problem.evaluate(theta, with_grads=False)
    if result.metrics.get("rejected", 0.0) > 0.0:
        continue   # don't let (0,0) measures skew percentile bounds
    m0_vals.append(result.measures[0])
    m1_vals.append(result.measures[1])
```

**Seeding** — [optical_rl/qd/ef50_zernike.py:1129-1138](optical_rl/qd/ef50_zernike.py#L1129-L1138):
```python
flat = self.problem.initial_theta()
r_flat = self.problem.evaluate(flat, with_grads=False)
if r_flat.metrics.get("rejected", 0.0) <= 0.0:
    self._maybe_log_seed_accept(self.archive.insert(r_flat), r_flat)
for theta in self.problem.sample_theta(...):
    r = self.problem.evaluate(theta, with_grads=False)
    if r.metrics.get("rejected", 0.0) > 0.0:
        continue   # not inserted into archive
    self._maybe_log_seed_accept(self.archive.insert(r), r)
```

**CMA-MEGA iteration** — [optical_rl/qd/ef50_zernike.py:1207-1217](optical_rl/qd/ef50_zernike.py#L1207-L1217):
```python
candidate_eval = self.problem.evaluate(candidate_theta, with_grads=False)
# Skip rejected candidates — do not insert into the archive.
# Their fitness still feeds back to the CMA emitter via batch_scores
# so the distribution learns to avoid them.
if candidate_eval.metrics.get("rejected", 0.0) > 0.0:
    improved = False
else:
    improved = self.archive.insert(candidate_eval)
scores.append(candidate_eval.fitness)
batch_scores.append(candidate_eval.fitness if improved else -np.inf)
```

## Why the design is split this way
- **`-1e9` in the fitness tensor**: keeps the autograd graph intact (graph would break on early `return None`), and gives the CMA emitter a strong gradient/score signal pushing it away from invalid regions.
- **`rejected` metric flag**: a clean, side-channel signal the runner uses to gate archive inserts so the MAP-Elites archive never contains poisoned cells with `-1e9` "winners."

## Net effect for the paper
For the EF50 Zernike runs with the current config (`strict_entrance_pupil: false`):
- Only diverging-lens (BFL ≤ 0) designs are hard-rejected from the archive.
- Front-group degenerate designs whose entrance pupil falls back to `surfaces[0].r` **are still inserted** into the archive (with whatever fitness they get from the fallback pupil), because the strict guard is off.
- Soft penalties (intersection, invalid-ray fraction) continue to shape the fitness of accepted designs.
