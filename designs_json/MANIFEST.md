# Design JSONs — 8 curated picks (4 septlets + 4 quadruplets)

Each JSON is a verbatim copy of the LensMEGA elite-export from its source run.
Filename prefix is the spotlight tag used in the rendered panels.

| Tag | RMS (refined) | Source run | Source file |
|---|---:|---|---|
| V1 | 2.72 µm  | `septlet_F3_20260425_224405`                | `elite_designs/elite_0020_34872k.json` |
| V2 | 4.00 µm  | `septlet_F3_20260425_224405`                | `elite_designs/posthoc_0068_r126_c129.json` |
| V3 | 9.56 µm  | `septlet_F3_20260425_224405`                | `elite_designs/posthoc_0101_r129_c168.json` |
| V4 | 13.32 µm | `septlet_F3_20260425_224405`                | `elite_designs/posthoc_0042_r137_c157.json` |
| Q1 | 16.83 µm | `quadruplet_s2s4_20260425_002055`           | `elite_designs/elite_0002_2610k.json` |
| Q4 | 20.29 µm | `quadruplet_discovery_materials_20260414_130002` | `elite_designs/elite_0016_8550k.json` |
| Q5 | 20.55 µm | `quadruplet_discovery_materials_20260414_130002` | `elite_designs/BEST_0027_13248k.json` |
| Q6 | 22.81 µm | `quadruplet_discovery_20260414_044915`      | `elite_designs/elite_0035_65916k.json` |

## Schema

Top-level keys (see `cooke_diverse_picks/README.md` for the canonical reference):
- `theta`, `fitness`, `rms_mm`, `efl_mm`, `sign_pattern`, `cosine_sim` — raw at discovery
- `measures` — 2-D Seidel descriptor `[S2_coma, S3_astigmatism]` (archive cell coordinate)
- `metrics` — full physics output (`rms_spot`, `efl_actual`, `fnum_actual`, `seidel_S1..S5`, etc.)
- `fingerprint_raw`, `fingerprint_refined` — per-element Seidel breakdown
- `refined` — L-BFGS-polished result; same nested schema as the raw block.
  **Use `refined.theta` to reproduce a render.**

## Reproducing a lens from one of these

```python
import json, numpy as np
d = json.load(open("V1_elite_0020_34872k_septlet.json"))
theta_refined = np.array(d["refined"]["theta"], dtype=np.float64)
# Feed theta_refined into render_spotlight_panels.build_lens(...) with the
# matching run's config.yaml — paths are listed in the table above.
```
