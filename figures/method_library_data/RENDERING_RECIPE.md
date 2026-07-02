# Pretty lens-render recipe (with and without ray bundles)

Self-contained handoff doc for reproducing the figures
`figures/method_library_{system}_unpruned*.png` from scratch.  All paths
relative to `/nas/longleaf/home/hisham26/Desktop/hyperlens/`.  Drop this
file into another agent's prompt and they should be able to rebuild the
exact same look.

---

## TL;DR — the four ingredients that make this look good

1. **Rebuild the lens from theta on every render.  Do NOT round-trip
   through `prune_surf()` + `write_lens_json()`.**  That round-trip
   silently corrupts d_sensor and per-element materials/thicknesses
   (verified: EFLs drift 20–100 %, RMSs explode 2–40×).  See §10.
2. **Fill polygon respects each surface's own semi-aperture.**  The
   default DeepLens helper clips both surfaces to the *smaller* radius,
   leaving white triangles inside meniscus elements where the back
   surface is wider than the front.  We trace each surface from
   `-r_self` to `+r_self` and connect the endpoints with straight side
   walls.  See §3.
3. **Thin outline (`lw = 1.2`), not the chunky 1.9 that older scripts
   default to.**  Same with `solid_capstyle="round"` on every line so
   the corners don't look pixelated.
4. **Order the layers.**  Rays under the glass fill, glass fill under
   the dark surface outlines, sensor on top.  Concretely
   `zorder` 2 → 3 → (matplotlib default for outlines).  See §6.

---

## 1. Inputs

For each system (`sixlet` / `octuplet`) the renderer needs:

| input | path |
| --- | --- |
| Lens template JSON | `DeepLens/datasets/lenses/sixlet_50mm_f4_apsc_v2.json` (or `octuplet_50mm_f4_apsc_2cement.json`) |
| Problem config | `results/diversity_run_20260506_033447/configs/hl_{system}_seed42.yaml` |
| Per-design picks (anchors / parents / incumbent) | `figures/fig5_data/{system}_parents_and_incumbent.json` |

The picks JSON contains, for each design: its `theta` vector (full
free-parameter list, exactly the format the optimizer used), reported
`refined_RMS_um`, source-run provenance, and Seidel sums.  We use only
`theta` + `refined_RMS_um` for rendering.

---

## 2. Lens build (rebuild from theta, never reload from disk)

```python
from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)
import numpy as np

# Build the template + paraxial solver ONCE per system
template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
    SYSTEMS["octuplet"]["cfg"]
)

# Build a fresh GeoLens for each design's theta
lens = make_geolens(prob_cfg, p_problem, lens_path,
                    np.asarray(theta, dtype=np.float64),
                    template, solver)
```

`make_geolens` internally:

1. instantiates a fresh `GeoLens` from the template JSON,
2. calls `_apply_theta_generic` from `optical_rl/qd/render_lensmega_archive.py`
   which writes the curvatures, gaps, materials, and flange in place,
3. derives `d_sensor` from the solved EFL.

The resulting lens has the correct optical behavior — its RMS at the
runner's evaluation settings matches the `refined_RMS_um` from the
picks JSON.

---

## 3. The fill polygon (the bug + the fix)

### What's broken in the default helper

`optical_rl/qd/render_singlet_archive_designs.py:_lens_fill_polygons`
clips the polygon to the smaller of the two surface radii:

```python
r0 = float(s0.draw_r()) * 0.999
r1 = float(s1.draw_r()) * 0.999
r_eff = min(r0, r1)            # <-- the problem
rs = np.linspace(-r_eff, r_eff, num_samples)
z0 = s0.surface_with_offset(rs, 0)
z1 = s1.surface_with_offset(rs, 0)
```

That works for biconvex elements where both surfaces have similar `r`,
but for a meniscus / strongly asymmetric element where `r1 > r0` the
fill stops short of the back surface's true edge — visible white wedges
inside the glass.

### The fix

Trace each surface across **its own** range, then close the polygon
with implicit straight side walls between `(z0(±r0), ±r0)` and
`(z1(±r1), ±r1)`:

```python
def lens_fill_polygons(lens, num_samples=192):
    polys = []
    for i in range(len(lens.surfaces) - 1):
        s0 = lens.surfaces[i]
        s1 = lens.surfaces[i + 1]
        # only fill if mat2 is glass (n > 1.1)
        if not (hasattr(s0, "mat2") and float(s0.mat2.n) > 1.1):
            continue
        r0 = float(s0.draw_r()) * 0.999
        r1 = float(s1.draw_r()) * 0.999
        if r0 <= 1e-8 or r1 <= 1e-8:
            continue

        rs0 = np.linspace(-r0, r0, num_samples)   # front surface span
        rs1 = np.linspace(-r1, r1, num_samples)   # back  surface span
        rt0 = torch.tensor(rs0, dtype=torch.float32, device=lens.device)
        rt1 = torch.tensor(rs1, dtype=torch.float32, device=lens.device)
        z0 = s0.surface_with_offset(rt0, torch.zeros_like(rt0),
                                    valid_check=False).cpu().numpy()
        z1 = s1.surface_with_offset(rt1, torch.zeros_like(rt1),
                                    valid_check=False).cpu().numpy()

        # CCW polygon: front bottom-up, back top-down
        # (matplotlib closes the path implicitly with straight side walls)
        z_poly = np.concatenate([z0, z1[::-1]])
        x_poly = np.concatenate([rs0, rs1[::-1]])
        polys.append((z_poly, x_poly))
    return polys
```

Apply with:

```python
ax.fill(z_poly, x_poly, color=fill_color, alpha=fill_alpha,
        linewidth=0.0, zorder=3)
```

`linewidth=0.0` is important — we don't want a polygon outline; the
clean dark outline comes from the next stage and would otherwise read
double.

---

## 4. The clean outline (surfaces, edge connectors, aperture wedges, sensor)

```python
LINEWIDTH_LENS   = 1.2     # thinner than the 1.9 default in older scripts
LINEWIDTH_SENSOR = 1.6
OUTLINE_COLOR    = "#1F1F1F"
SENSOR_COLOR     = "#000000"

def draw_lens_clean(lens, ax, color=OUTLINE_COLOR, lw=LINEWIDTH_LENS):
    # 1. Surfaces and aperture stops
    for s in lens.surfaces:
        if s.__class__.__name__ == "Aperture":
            d = float(s.d.item()) if torch.is_tensor(s.d) else float(s.d)
            r = float(s.r)
            wedge_l = 0.05 * r            # short tick mark
            wedge_h = 0.18 * r            # vertical kicker
            for sign in (-1, +1):
                ax.plot([d - wedge_l, d + wedge_l], [sign * r, sign * r],
                        color="#666", lw=lw, solid_capstyle="round")
                ax.plot([d, d], [sign * r, sign * (r + wedge_h)],
                        color="#666", lw=lw, solid_capstyle="round")
        else:
            r_eff = float(s.draw_r())
            r = torch.linspace(-r_eff, r_eff, 256, device=s.device)
            z = s.surface_with_offset(
                r, torch.zeros(len(r), device=s.device), valid_check=False
            ).cpu().numpy()
            ax.plot(z, r.cpu().numpy(),
                    color=color, lw=lw, solid_capstyle="round")

    # 2. Edge connectors (straight line between consecutive glass surface
    #    rims) — only when mat2 is glass on the previous surface
    for i in range(len(lens.surfaces) - 1):
        s_prev = lens.surfaces[i]
        if not (hasattr(s_prev, "mat2") and float(s_prev.mat2.n) > 1.1):
            continue
        s_next = lens.surfaces[i + 1]
        r_prev = float(s_prev.draw_r())
        r_next = float(s_next.draw_r())
        sag_prev = s_prev.surface_with_offset(
            torch.tensor(r_prev, device=s_prev.device),
            torch.tensor(0.0, device=s_prev.device),
            valid_check=False).item()
        sag_next = s_next.surface_with_offset(
            torch.tensor(r_next, device=s_next.device),
            torch.tensor(0.0, device=s_next.device),
            valid_check=False).item()
        for sign in (-1, +1):
            ax.plot([sag_prev, sag_next], [sign * r_prev, sign * r_next],
                    color=color, lw=lw, solid_capstyle="round")

    # 3. Sensor plane (slightly thicker, full black)
    d_sensor = float(lens.d_sensor.item()) if torch.is_tensor(lens.d_sensor) else float(lens.d_sensor)
    r_sensor = float(lens.r_sensor)
    ax.plot([d_sensor, d_sensor], [-r_sensor, r_sensor],
            color=SENSOR_COLOR, lw=LINEWIDTH_SENSOR, solid_capstyle="round")
```

Everything uses `solid_capstyle="round"` so line endpoints don't look
square / pixelated when you zoom in.

The aperture-stop tick is two short dashes flanked by short verticals
on each side — small and unobtrusive but unambiguous as "this is the
stop".

---

## 5. Color palette

| element | hex | role |
| --- | --- | --- |
| `BLUE_FILL` | `#7BAFD4` | non-incumbent glass elements |
| `RED_FILL`  | `#E07B7B` | INCUMBENT glass elements (so the bad case stands out) |
| `OUTLINE`   | `#1F1F1F` | dark grey, softer than pure black |
| `SENSOR`    | `#000000` | sensor plane bar |
| Field colors (rays) | `#1565C0`, `#2E7D32`, `#C62828` | on-axis / mid-FOV / full-FOV |
| aperture-stop tick | `#666` | mid grey |

Both fill colors are the same tonal weight (Munsell value ~6) so blue
and red read as siblings rather than primary/garish.

For the per-cell title, the incumbent's text label also goes red (`color="C3"`)
so the visual + typographic emphasis match.

---

## 6. The three ray styles

### Style A — no rays
Just the fill + outlines.  Use `fill_alpha = 1.0` so the glass color
is fully saturated.

### Style B — line ray bundles (3 fields × 11 rays each)

Per-field: sample 11 point-source rays from infinity, trace to sensor
with `record=True`, redraw with restyled linewidth/alpha:

```python
FIELD_COLORS = ["#1565C0", "#2E7D32", "#C62828"]
RAY_LW = 0.55
RAY_ALPHA = 0.85
RAY_NUM_PER_FIELD = 11
RAY_WVLN_UM = 0.587
RAY_DEPTH_MM = -20000.0

for fov_frac, color in zip([0.0, 0.5, 1.0], FIELD_COLORS):
    ray = lens.sample_point_source_2D(
        fov=float(fov_frac * hfov_deg),
        depth=RAY_DEPTH_MM,
        num_rays=RAY_NUM_PER_FIELD,
        wvln=RAY_WVLN_UM,
    )
    ray.prop_to(-1.0)                              # back to z = -1 (entrance)
    _, rec = lens.trace2sensor(ray=ray, record=True)
    n_before = len(ax.lines)
    lens.draw_ray_2d(rec, ax=ax, fig=ax.figure, color=color)
    # draw_ray_2d uses lw=0.8 hardcoded — restyle after the fact
    for line in ax.lines[n_before:]:
        line.set_linewidth(RAY_LW)
        line.set_alpha(RAY_ALPHA)
```

When rays are drawn, set the glass fill alpha to **0.85** so rays read
through the glass.  Outlines still render fully opaque on top.

### Style C — filled ray bundles (envelope between extreme marginals)

Use the same sampling, but render with `draw_ray_2d_bundle` — it picks
the upper and lower marginal rays and fills the polygon between them:

```python
FILLED_BUNDLE_ALPHA = 0.32
FILLED_BUNDLE_NUM_RAYS = 9   # ≥2; more samples => smoother envelope

for fov_frac, color in zip([0.0, 0.5, 1.0], FIELD_COLORS):
    ray = lens.sample_point_source_2D(
        fov=float(fov_frac * hfov_deg),
        depth=RAY_DEPTH_MM,
        num_rays=FILLED_BUNDLE_NUM_RAYS,
        wvln=RAY_WVLN_UM,
    )
    ray.prop_to(-1.0)
    _, rec = lens.trace2sensor(ray=ray, record=True)
    lens.draw_ray_2d_bundle(rec, ax=ax, fig=ax.figure,
                            color=color, alpha=FILLED_BUNDLE_ALPHA)
```

When filled bundles are drawn, set the glass fill alpha to **0.78**
(slightly more transparent so the bundle colors show through inside
the glass).

### HFOV computation

`sample_point_source_2D(fov=...)` wants degrees.  Half-FOV from sensor
geometry:

```python
import math, numpy as np
def hfov_deg_for(lens):
    if hasattr(lens, "rfov") and lens.rfov is not None:
        try:
            return float(np.rad2deg(float(lens.rfov)))
        except Exception:
            pass
    half_sensor = (float(lens.sensor_size[0]) / 2.0
                   if hasattr(lens, "sensor_size") else float(lens.r_sensor))
    efl = float(getattr(lens, "foclen", 50.0))
    return math.degrees(math.atan(half_sensor / abs(efl)))
```

This matches the runner's stage-2 fields convention `(0, 0.5·HFOV, HFOV)`.

---

## 7. zorder / compositing order

Required for the layers to read correctly:

| zorder | element |
| --- | --- |
| matplotlib default (~2) | ray bundles (lines or filled polygons) |
| 3 | glass fill polygons |
| matplotlib default (~2.0 for `ax.plot`) | clean outlines (drawn LAST in code so they end up on top) |

In practice the call order matters more than the explicit zorder:

```python
def render_one(ax, lens, fill_color, ray_style):
    if ray_style == "lines":
        draw_line_rays(lens, ax)            # first: rays bottom layer
    elif ray_style == "filled":
        draw_filled_bundles(lens, ax)       # first: bundles bottom layer
    fill_alpha = (1.0 if ray_style == "none"
                  else 0.85 if ray_style == "lines"
                  else 0.78)
    for z_poly, x_poly in lens_fill_polygons(lens):
        ax.fill(z_poly, x_poly, color=fill_color,
                alpha=fill_alpha, linewidth=0.0, zorder=3)   # middle layer
    draw_lens_clean(lens, ax)               # last: outlines on top
    ax.set_aspect("equal")
    ax.set_axis_off()
```

---

## 8. Grid layout (2 × 4)

```
+----------------+----------------+----------------+----------------+
|   ANCHOR #1    |   ANCHOR #2    |   ANCHOR #3    |   INCUMBENT    |  <-- top row
+----------------+----------------+----------------+----------------+
| NEAR-PARENT #1 | NEAR-PARENT #2 | NEAR-PARENT #3 |    (blank)     |  <-- bottom row
+----------------+----------------+----------------+----------------+
```

```python
fig, axes = plt.subplots(2, 4, figsize=(15.5, 6), facecolor="white")
layout = [
    ("anchor",     0, 0, 0, "anchor #1"),
    ("anchor",     1, 0, 1, "anchor #2"),
    ("anchor",     2, 0, 2, "anchor #3"),
    ("incumbent",  0, 0, 3, "incumbent"),
    ("parent",     0, 1, 0, "near-parent #1"),
    ("parent",     1, 1, 1, "near-parent #2"),
    ("parent",     2, 1, 2, "near-parent #3"),
    (None,         0, 1, 3, ""),                # blank cell
]
for role, rank, row, col, label in layout: ...
```

Per-cell title: `"{label.upper()}\nRMS = {refined_RMS_um:.1f} um"` —
fontsize 10, color `"C3"` if incumbent else `"black"`.

Figure-level suptitle: `"{system} method-library — unpruned lens cross-sections ({style})"`
where style is `"no rays"` / `"with 3-field ray bundles"` / `"with 3-field filled bundles"`.

---

## 9. Save

```python
fig.tight_layout()
fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
```

`facecolor="white"` is set on both the Figure and the savefig because
some matplotlib versions otherwise pick up a transparent or grey
background, which makes the soft pastel fills look washed out.

---

## 10. Pruning postmortem (DO NOT DO THIS)

The first iteration of this workflow tried to call
`lens.prune_surf()` to trim each surface's semi-aperture to the
ray-traced extent, then `lens.write_lens_json(path)` to save the
pruned lens, then reload with `GeoLens(filename=path)` to render.

The reload corrupted the lens.  Verified table from
`figures/method_library_data/verify_pruned_rms.json`:

| design | reported RMS | pruned-JSON RMS | EFL after prune |
| --- | ---: | ---: | ---: |
| anchor #1 (octuplet) | 8.05 µm | 823 µm | 60.6 mm (target 50) |
| anchor #2 | 15.6 µm | 1428 µm | 78.9 mm |
| anchor #3 | 16.5 µm | 562 µm | 43.5 mm |
| incumbent | 736.6 µm | 1583 µm | 60.3 mm |
| parent #1 | 57.1 µm | 2429 µm | 103.0 mm |
| parent #2 | 64.6 µm | 115 µm | 57.5 mm |
| parent #3 | 57.3 µm | 1702 µm | 38.5 mm |

EFL drift of 20–100 % across every design, RMS off by 2–40×.  The
`prune_surf()` + `write_lens_json()` round-trip is dropping or
corrupting either the per-element thicknesses, the d_sensor, the
material indices, or some combination.

**The fix: skip pruning entirely.  Rebuild from theta on every render
(see §2).  The aperture polygons end up slightly larger than the
ray-traced extent — that's fine, the lenses still read as compact
optical stacks and the optical behavior is exact.**

If you ever need true pruned-aperture renders downstream, do the
prune *only* for the visual pass and keep theta as the authoritative
representation for any optical evaluation:

```python
# WRONG
lens = GeoLens(filename=pruned_json_path)
rms = evaluate(lens)            # gives nonsense

# RIGHT
lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
rms = evaluate(lens)            # matches the original refined RMS
lens.prune_surf()               # mutates only s.r values; safe for drawing
draw(lens)                      # apertures look tight, optics still correct
```

---

## 11. Full reference scripts

The throwaway implementations are at:

- `scripts_throwaway/method_library_render_unpruned_octuplet.py`
- `scripts_throwaway/method_library_render_unpruned_sixlet.py`
- `scripts_throwaway/fig5_octuplet_parents_incumbent.py` (picks the 7 designs)
- `scripts_throwaway/fig5_sixlet_parents_incumbent.py` (sixlet picker)

To regenerate everything:

```bash
cd /nas/longleaf/home/hisham26/Desktop/hyperlens
PYTHON=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python

# 1. Pick the 3 anchors + 3 near-parents + 1 incumbent (per system)
$PYTHON scripts_throwaway/fig5_octuplet_parents_incumbent.py
$PYTHON scripts_throwaway/fig5_sixlet_parents_incumbent.py

# 2. Render all three styles (per system)
$PYTHON scripts_throwaway/method_library_render_unpruned_octuplet.py
$PYTHON scripts_throwaway/method_library_render_unpruned_sixlet.py
```

Outputs land at `figures/method_library_{sixlet,octuplet}_unpruned*.png`.

---

## 12. Gotchas / things to double-check if you re-implement

- **Material check uses `mat2.n > 1.1`.**  In the codebase, "air" is
  represented as a Material with `n = 1.0`, so `> 1.1` correctly skips
  air gaps and stop-occluder gaps.  If you change this threshold,
  things like the cemented-pair side walls may stop drawing.
- **Aperture surfaces have no `c` attribute.**  Always check
  `s.__class__.__name__ == "Aperture"` before calling
  `s.surface_with_offset(...)`.  The render does this in
  `draw_lens_clean`.
- **`draw_ray_2d` hardcodes `lw=0.8`.**  We restyle the lines after
  the call (snippet in §6).  Don't try to pass `linewidth=...` to
  `draw_ray_2d` — it's ignored.
- **`prop_to(-1.0)` before `trace2sensor`.**  Without this, point-
  source rays at infinite depth start way upstream and the `record`
  array begins with z = `RAY_DEPTH_MM` ≈ -20000 mm, which makes the
  ray segments before the lens get clipped on a normal axis.  The
  prop pulls them back to z = -1 mm so the rays read as "arriving
  collimated from the left edge of the panel."
- **`r_sensor` is in mm.**  All length units in DeepLens are mm, so
  the `OBJ_DEPTH_MM = -20000.0` literal is "object 20 m in front of
  the lens", effectively infinity for these designs.
- **`refined_RMS_um` in the picks JSON is *not* recomputed by the
  renderer.**  It's the value the optimizer reported after L-BFGS.
  The renderer just labels the panel with it.  This is correct as
  long as you trust the source JSON; if you want a self-consistency
  check, evaluate the rebuilt lens with the runner's evaluation
  config and compare (the verification script
  `scripts_throwaway/verify_pruned_octuplet_rms.py` shows exactly
  how).

---

## 13. Where to tweak look-and-feel

| What you want | Where in the code |
| --- | --- |
| Different fill colors | `BLUE_FILL` / `RED_FILL` constants |
| Thinner / thicker outlines | `LINEWIDTH_LENS`, `LINEWIDTH_SENSOR` |
| Smoother fills | bump `FILL_NUM_SAMPLES` (default 192) |
| Different ray colors | `FIELD_COLORS` (3-tuple, on-axis → full-FOV) |
| More / fewer rays per field | `RAY_NUM_PER_FIELD` (lines), `FILLED_BUNDLE_NUM_RAYS` (filled) |
| Different ray wavelength | `RAY_WVLN_UM` (or loop over multiple wavelengths inside `draw_line_rays`) |
| Tighter / looser bundle alpha | `RAY_ALPHA`, `FILLED_BUNDLE_ALPHA` |
| Different grid layout | edit the `layout` list in `render_grid` |
| Bigger / smaller figure | `figsize=(...)` in `plt.subplots` |
| Higher DPI | `dpi=` in `fig.savefig` |

Everything else (axes off, equal aspect, white background) is locked
in and shouldn't need to change.
