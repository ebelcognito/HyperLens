# RENDERING.md — how to pretty-render lens designs (authoritative)

Living instructions for producing the paper-style "pretty" lens cross-sections
(filled steel-blue glass + translucent R/G/B ray bundles). **Read this before
rendering any lens; keep it up to date when the pipeline changes.**

Reference of the target look:
`figures/octuplet_top17_eps030_lenses/design_01_rms_1.43.png`

---

## TL;DR

1. Use the Python in the **`hyperlens` miniforge env** — the repo's
   `activate_env.sh` / `activate_playpen_env.sh` are stale and the system
   anaconda `python` can't even import torch on this machine:
   ```
   PY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
   ```
2. **Render with the canonical recipe (`render_one`), NOT
   `GeoLens.draw_layout(style="filled")`.** `draw_layout`'s fill helper clips
   every element to the *smaller* of its two surface radii, which mangles
   meniscus and **cemented/fused** elements (white wedges, broken cement
   interfaces, thin empty slivers). The recipe traces each surface across its
   *own* semi-aperture, so fused doublets render as solid blocks with a thin
   internal cement curve. Full rationale:
   `figures/method_library_data/RENDERING_RECIPE.md`.
3. **Always rebuild the lens from `theta`** via the problem config — never
   round-trip through `prune_surf()` + `write_lens_json()` (corrupts d_sensor /
   materials; RMS explodes 2–40×). See RENDERING_RECIPE.md §10.
4. **Cemented templates need `cemented_interfaces`** passed to
   `load_lens_template`, or the per-glass element groups don't split and theta
   application throws `IndexError` (octuplet: `[3, 11]`).

---

## The one command for the f/4 designs

```bash
cd /nas/longleaf/home/hisham26/Desktop/hyperlens
/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python render_fnum4_designs.py
```
Reads `fnum4_designs/*.json`, writes `fnum4_designs_renders/<stem>_filled.png`
(+ a 2×3 contact sheet built separately). The script maps each design's
`system` field to the matching REFSWEEP config so the theta reconstructs the
exact optimized lens, then draws with the canonical `render_one`.

---

## How the correct renderer is wired

The cement-safe drawer lives in
`scripts_throwaway/render_spotlight_sixlet_designs.py` and is what produced the
reference figure (via `scripts_throwaway/render_octuplet_top17_eps030.py`):

```python
from scripts_throwaway.render_spotlight_sixlet_designs import render_one, BLUE_FILL
# render_one(ax, lens, fill_color) does, in order:
#   draw_filled_bundles(lens, ax)   # 3 fields (0, 0.5, 1.0 * HFOV) R/G/B envelopes
#   lens_fill_polygons(lens)        # per-surface span fill (cement/meniscus safe)
#   draw_lens_clean(lens, ax)       # thin outlines + edge connectors + dumbbell stop
```

Lens build (either of these is fine — both apply theta with cemented_interfaces):
- `render_fnum4_designs.py:build_lens(theta, cfg)` — uses the REFSWEEP config
  that matches each design's `source_run` (exact provenance).
- `scripts_throwaway.fig5_extract.make_geolens(...)` — uses the blessed
  `hl_{system}_seed42.yaml` configs; same templates, same DOF count.

Both call `_apply_theta_generic` from
`optical_rl/qd/render_lensmega_archive.py`.

### Style constants (recipe defaults — match the reference)
| knob | value |
| --- | --- |
| glass fill | `BLUE_FILL = "#7BAFD4"` (incumbent: `RED_FILL = "#E07B7B"`) |
| outline | `"#1F1F1F"`, `lw = 1.2`, `solid_capstyle="round"` |
| sensor | black, `lw = 1.6` |
| field colors | `["#1565C0", "#2E7D32", "#C62828"]` (on-axis → full-FOV) |
| filled bundle | `alpha = 0.32`, `num_rays = 9`, `wvln = 0.587 µm` |
| glass alpha under filled bundles | `0.78` |
| fill samples | `192` |
| figure | `figsize=(7.0, 3.4)`, `dpi=200`, `facecolor="white"` |

---

## DO NOT (traps that have bitten before)

- ❌ `lens.draw_layout(style="filled", ...)` for any system with cemented or
  strongly meniscus elements. It's the older
  `DeepLens/.../geolens_pkg/vis.py` drawer. `render_instructions.md` documents
  it — that doc is kept only for the marginal-ray debug notes; **do not use its
  filled-style call for fused surfaces.**
- ❌ `prune_surf()` → `write_lens_json()` → reload-and-render. Render from theta.
- ❌ omitting `cemented_interfaces` when loading a cemented template.
- ❌ the system anaconda `python` (broken sympy/mpmath); use the hyperlens env.

---

## Files

| path | role |
| --- | --- |
| `RENDERING.md` | this file — start here |
| `figures/method_library_data/RENDERING_RECIPE.md` | full recipe + rationale (§3 fill fix, §10 prune postmortem) |
| `render_fnum4_designs.py` | renders the 6 f/4 designs |
| `scripts_throwaway/render_spotlight_sixlet_designs.py` | `render_one` + cement-safe primitives |
| `scripts_throwaway/render_octuplet_top17_eps030.py` | produced the reference figure |
| `scripts_throwaway/fig5_extract.py` | `build_lens_template_and_solver`, `make_geolens`, `SYSTEMS` |
| `render_instructions.md` | OLD draw_layout pipeline — marginal-ray debug only; not for fused surfaces |
