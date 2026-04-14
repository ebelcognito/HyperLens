# EF50 Zernike — Render & Config Bugfix Session

Context export from a debugging chat. Two related bugs in the EF50 Zernike QD (Quality-Diversity) optimization pipeline were diagnosed and fixed. Useful background for the "Methods / Implementation" portion of the paper, particularly around the fixed-EFL parameterization and the rendering / archive replay path.

## Project background (inferred from code)

- **Pipeline:** Quality-Diversity (CMA-ME style) search over Zernike-parameterized lens designs for the Canon EF50mm f/1.8 (5-element, cemented doublet rear group).
- **Lens template:** `DeepLens/datasets/lenses/camera/ef50mm_f1.8.json`
- **Design vector (theta):** curvatures of the optical surfaces.
  - **Free 11-D mode:** `theta = [c0..c5, c7..c11]` (11 free curvatures, c11 explicitly searched, `d_sensor` from BFL solve).
  - **Fixed-EFL 10-D mode:** `theta = [c0..c5, c7..c10]`. The 11th curvature `c11` is **derived** from the paraxial ABCD solve so EFL = `focal_length_target_mm` (50.52 mm). `d_sensor` follows from the paraxial BFL solve. This is the mode actually used for the run in question.
- **Behavioral descriptors:** two 1-based Noll Zernike coefficients (config: `descriptor_zernike_indices: [4, 11]`), giving a 2D archive grid (`64 × 64`).
- **Fitness:** polychromatic RMS spot (Fraunhofer C/d/F = 0.656 / 0.587 / 0.486 µm), `stage2` = mean over 3 meridional field points (0, 0.5, 1.0 × HFOV).
- **Config file:** `optical_rl/configs/ef50_zernike.yaml` — declares `focal_length_target_mm: 50.52`, `iterations: 5000`, the per-curvature `theta_bounds`, intersect/air/thickness penalties, etc.
- **Key files touched:**
  - `optical_rl/qd/ef50_zernike.py` — training entry point, `EF50ZernikeConfig` dataclass, `_load_config`, `main()`.
  - `optical_rl/qd/render_ef50_zernike.py` — archive replay / rendering, `_apply_theta_to_lens_ef50`, `_load_qd_config`, `render_design_ef50`, `render_archive_grid_ef50`.
  - `optical_rl/utils/config.py` — trivial YAML loader (`yaml.safe_load`).

## Bug 1 — Renderer crashed on a fixed-EFL archive

### Symptom

```
$ python optical_rl/qd/render_ef50_zernike.py
[render-ef50-zernike] archive:       results/models/ef50_zernike_20260409_115833/archive_cells.json
[render-ef50-zernike] EFL constraint: none (11-D archive)
[render-ef50-zernike] loaded 179 designs
[render-ef50-zernike] selected 9 diverse designs
[render-ef50-zernike] rendering #1/9 fitness=-0.00390
...
  File ".../render_ef50_zernike.py", line 231, in _apply_theta_to_lens_ef50
    raise ValueError(
ValueError: theta is 10-D (fixed-EFL archive) but focal_length_target_mm was not supplied.
Pass --config or --focal-length when rendering a fixed-EFL archive.
```

### Diagnosis

1. Inspected the archive JSON and confirmed each cell carries `theta` of length **10**, i.e. the archive was generated in fixed-EFL mode.
   ```python
   import json
   data = json.load(open(".../archive_cells.json"))
   first = data["cells"][0]
   len(first["theta"])  # → 10
   ```
2. `_apply_theta_to_lens_ef50` correctly raises when it sees `len(theta)==10` and no `focal_length_target_mm`. Lines ~228–248: 10-D branch calls `_derive_c11_from_efl(...)` and uses paraxial BFL for `d_sensor`; 11-D branch reads `c11 = arr[10]` and computes BFL from the full curvature set.
3. Walked the call chain `main → render_archive_grid_ef50 → render_design_ef50 → _apply_theta_to_lens_ef50` — all four layers thread `focal_length_target_mm` through correctly.
4. So the question reduced to: why is `main()` ending up with `focal_length_target_mm=None`?
5. Checked `_load_qd_config`. Resolution priority:
   1. explicit `--config`,
   2. any `*.yaml` sitting next to the archive in its run directory,
   3. fallback to `optical_rl/configs/ef50_zernike.yaml`.
6. Listed the run directory — **no YAML had been copied next to the archive**. So the loader should have fallen through to the default YAML.
7. Reproduced the loader manually:
   ```python
   from pathlib import Path
   from optical_rl.utils.config import load_config
   cfg = load_config("optical_rl/configs/ef50_zernike.yaml")
   cfg["qd"]["focal_length_target_mm"]  # → 50.52
   ```
   Loader works fine in isolation.
8. Looked at the timestamps:
   ```
   render_ef50_zernike.py     mtime 2026-04-09 12:06
   failed run                 ran   2026-04-09 12:04
   ```
   The script was edited **after** the failed run. The "EFL constraint: none (11-D archive)" log line in the user's traceback does not appear anywhere in the current source — confirming the failing run was against an older version of the file. The current source already prints `"EFL target: none loaded — required for 10-D archives ..."` instead, and threads the YAML fallback through `_load_qd_config` correctly.

### Resolution

No code change required for Bug 1 — it had already been fixed by the user's 12:06 edit, which (a) added the default-YAML fallback in `_load_qd_config` and (b) pulled `focal_length_target_mm` out of `qd_cfg` in `main()`. Verified by re-running the loader path end-to-end.

### Why this matters for the paper

The fixed-EFL parameterization is a deliberate methodological choice: it removes one degree of freedom from the search by constraining the back element via a paraxial solve, so every candidate has the same nominal focal length and the QD descriptors (Zernike coefficients) describe shape variation at fixed magnification rather than mixing focal-length drift with aberration content. This needs to be stated in Methods, and the rendering pipeline has to be aware of it because the archive is **not self-describing** — designs are stored as 10-D vectors and the renderer must reconstruct `c11` from the same paraxial constraint used during search. Mismatches between training-time and replay-time parameterization are a real failure mode and worth a sentence in the paper.

## Bug 2 — Training script crashed before the first iteration

### Symptom

```
$ python3 optical_rl/qd/ef50_zernike.py --config optical_rl/configs/ef50_zernike.yaml
Traceback (most recent call last):
  File ".../ef50_zernike.py", line 1300, in main
    cfg = _load_config(args.config)
  File ".../ef50_zernike.py", line 1179, in _load_config
    return EF50ZernikeConfig(
TypeError: EF50ZernikeConfig.__init__() got an unexpected keyword argument 'iterations'
```

### Diagnosis

1. `_load_config` (around lines 1155–1211) reads YAML keys via a `_get(key, default)` helper and constructs an `EF50ZernikeConfig(...)` with ~25 keyword arguments, including `iterations=int(_get("iterations", 1000))`.
2. Inspected the `@dataclass class EF50ZernikeConfig` (lines 140–208). Fields: `lens_path, object_depth_mm, wavelength_um, fitness_wavelengths_um, theta_bounds, stop_surface_index, focal_length_target_mm, descriptor_zernike_indices, archive_bins, archive_bounds, invalid_tol, invalid_penalty_weight, cma_eta, seed_count, calibration_count, num_rays, stage, zernike_field_index, intersect_penalty_weight, air_min_mm, thick_min_mm, device, initial_sigma, population_size, cma_seed, results_dir, run_name, log_every`. **No `iterations` field.**
3. Searched for `cfg.iterations` and `.iterations` in the file. Single hit:
   ```python
   # line 1304
   iterations = args.iterations if args.iterations is not None else 1000
   ```
   Two problems: (a) `EF50ZernikeConfig` rejects `iterations=` as an unknown kwarg, and (b) even if it didn't, `main()` was hardcoding the fallback to `1000`, silently ignoring the YAML's `iterations: 5000`.

### Fix

Two edits to `optical_rl/qd/ef50_zernike.py`:

1. Added the missing dataclass field (after `log_every`):
   ```python
   log_every: int = 50
   iterations: int = 1000
   ```

2. Made `main()` actually honor the loaded config:
   ```python
   # before
   iterations = args.iterations if args.iterations is not None else 1000
   # after
   iterations = args.iterations if args.iterations is not None else cfg.iterations
   ```

Now `--iterations` on the CLI still wins, but if it isn't passed the value comes from the YAML (`5000` for the EF50 run) instead of being silently clamped to `1000`.

### Why this matters for the paper

This is a small bug but it has a real implication for any results table: previous runs that did **not** pass `--iterations` on the command line were terminated at 1000 generations regardless of what the YAML claimed. If any reported numbers came from such runs, the actual budget was 1000 iterations of CMA-ME, not 5000. Worth checking against the run timestamps in `results/models/` before quoting iteration counts in the paper.

## Files modified in this session

- `optical_rl/qd/ef50_zernike.py`
  - Added `iterations: int = 1000` field to `EF50ZernikeConfig` (after `log_every`).
  - Replaced hardcoded `1000` fallback in `main()` with `cfg.iterations`.

## Files inspected (no edits)

- `optical_rl/qd/render_ef50_zernike.py` (`_apply_theta_to_lens_ef50`, `_load_qd_config`, `render_design_ef50`, `render_archive_grid_ef50`, `main`)
- `optical_rl/configs/ef50_zernike.yaml`
- `optical_rl/utils/config.py`
- `results/models/ef50_zernike_20260409_115833/archive_cells.json` (verified theta dim = 10, 179 designs)

## Open / unverified items worth flagging to the writing agent

- **Iteration-count audit.** Any historical EF50 run launched without `--iterations` was capped at 1000 generations because of the bug in `main()`, not 5000 as the YAML implies. If results numbers depend on iteration count, re-check which runs are affected before citing them.
- **Renderer/run-dir coupling.** The renderer falls back to the *repo* default YAML when no YAML sits next to the archive. That works as long as the default YAML still matches the parameterization the run was trained with. If `focal_length_target_mm` is ever changed in the default YAML between training and rendering, old archives will be replayed against the wrong constraint and produce subtly wrong `c11`/`d_sensor` values without crashing. Consider having the training script copy the YAML next to the archive (the renderer already prefers a co-located YAML if present). Worth one sentence in Methods or a footnote.
- **Archive descriptor choice.** The YAML uses `descriptor_zernike_indices: [4, 11]` while the comment in the same file says "(7, 11) = vertical coma + primary spherical". The active descriptors are `(4, 11)`, not `(7, 11)`. Make sure the paper text matches the actual indices in the YAML, not the stale comment.
