> ⚠️ **OUTDATED for fused/meniscus elements.** The `draw_layout(style="filled")`
> pipeline described below clips each element to the *smaller* of its two
> surface radii, which breaks cemented (fused) doublets and strong meniscus
> elements (white wedges / broken cement interfaces). For pretty renders use
> **[RENDERING.md](RENDERING.md)** (the cement-safe `render_one` recipe). Keep
> this file only for the **marginal-ray debug mechanism** in §2.

# Rendering Lens Setups (Pretty / Filled Style) and the Marginal-Ray Debug Mechanism

This is a how-to for rendering the "pretty" lens diagrams used in the LensMEGA paper
figures (filled glass elements + translucent ray-bundle envelopes), plus the
already-built debugging mechanism that diagnoses why marginal rays sometimes
fail to make it through the system and leave gaps in the rendered bundle.

Everything described here already exists in the repo. Nothing needs to be
re-implemented — this is an orientation guide.

---

## 1. The pretty / filled rendering pipeline

### 1.1 Where the rendering primitives live

All filled-style drawing primitives are inside the patched DeepLens vis module:

- [DeepLens/deeplens/optics/geolens_pkg/vis.py](DeepLens/deeplens/optics/geolens_pkg/vis.py)

Three functions matter:

| Function | Lines | Purpose |
| --- | --- | --- |
| `GeoLens.draw_layout` | [vis.py:174-323](DeepLens/deeplens/optics/geolens_pkg/vis.py#L174-L323) | Top-level entry point. Picks `"line"` (default) or `"filled"` style, samples rays at 3 fields × RGB wavelengths, and dispatches to the lens-and-ray drawers. |
| `GeoLens.draw_lens_2d_filled` | [vis.py:447-527](DeepLens/deeplens/optics/geolens_pkg/vis.py#L447-L527) | Renders each glass element (front + back surface pair, detected via `surfaces[i].mat2.n > 1.1`) as a single closed `matplotlib.patches.Polygon` with a fill color and outline. Apertures and the sensor plane fall back to `draw_widget` / a plain line. |
| `GeoLens.draw_ray_2d_bundle` | [vis.py:529-579](DeepLens/deeplens/optics/geolens_pkg/vis.py#L529-L579) | Renders each (field, wavelength) bundle as a single translucent polygon between the top and bottom marginal rays. **This is where the survivor-filtering happens — see §2.** |

### 1.2 How to actually call it

The cleanest call site is `GeoLens.draw_layout(..., style="filled", ...)`. Use
this rather than poking the lower-level drawers directly.

```python
lens.draw_layout(
    filename=out_path,
    depth=float("inf"),     # parallel rays from infinity
    style="filled",         # <-- pretty style
    lens_title="",          # blank suptitle for paper figures
    glass_facecolor="#BFE3F2",   # pale cyan glass
    glass_edgecolor="black",
    glass_lw=1.2,
    bundle_alpha=0.22,      # see "tuning" below
)
```

The kwargs `glass_facecolor`, `glass_edgecolor`, `glass_lw`, and `bundle_alpha`
are the only knobs that change the look of the figure — there is no other
hidden state.

### 1.3 Reference call sites

These are the production scripts that already use this exact pipeline. Copy
from them rather than rebuilding:

- [render_filled_diagrams.py](render_filled_diagrams.py) — re-renders all paper picks (triplet/sixlet/septlet) in the filled style.
- [render_filled_collections.py](render_filled_collections.py) — current canonical version, also handles the decuplet and produces lens-only composites.
- [render_filled_paper.py](render_filled_paper.py) — earlier paper-figure pass.
- [render_filled_style_ab.py](render_filled_style_ab.py) — A/B style probe (tweak palette / alpha).

The shared `build_lens(theta, cfg, ...)` helper is duplicated across all of
them; it loads the lens template, applies a θ vector via `_apply_theta_generic`,
optionally rescales the stop to a target F/#, and optionally shrinks the
aperture. See [render_filled_collections.py:125-162](render_filled_collections.py#L125-L162) for the canonical copy.

### 1.4 Tuning notes baked into the existing code

These are the non-obvious choices, all already encoded in the call sites:

- **`bundle_alpha = 0.22`**, not the prototype default of 0.35.
  Rationale (per [render_filled_diagrams.py:84-86](render_filled_diagrams.py#L84-L86)): with three colored field bundles overlapping near the optical axis, 0.35 muddies the image. 0.22 keeps the bundles legible.
- **`num_rays = 9`** is hard-coded in `draw_layout` for filled style ([vis.py:205-207](DeepLens/deeplens/optics/geolens_pkg/vis.py#L205-L207)) — this is *intentional oversampling* so the survivor filter (§2) has rays to choose from. Line style uses 11 individually drawn rays.
- **`num_views = 3`** at `fov = linspace(0, rfov*0.99, 3)` — gives the on-axis / mid / corner-field bundles (R/G/B).
- **Aperture shrink (`aperture_shrink=0.92`)** for sixlet/septlet/decuplet ([render_filled_collections.py:157-161](render_filled_collections.py#L157-L161)). An 8% pupil shrink keeps the marginal ray a hair inside the true edge so it doesn't vignette against the rim of the next element. Triplets at F/8 are already stopped down enough that no shrink is needed (`aperture_shrink=1.0`).
- **`lens.post_computation()` is wrapped in try/except** after every aperture mutation — some surface configurations don't define it and we don't want that to break rendering.

### 1.5 Glass-element detection: the gotcha

`draw_lens_2d_filled` decides "this is a glass element" by checking
`surfaces[i].mat2.n > 1.1` and pairing with `surfaces[i+1]` ([vis.py:477-478](DeepLens/deeplens/optics/geolens_pkg/vis.py#L477-L478)).
This means:

- A **cemented doublet** (two glasses touching, no air gap) renders as **two
  abutting polygons**, not one — because each `(front, back)` pair with `mat2.n
  > 1.1` is a separate element. This is the correct behavior, just be aware.
- An **aperture** is detected by `cls == "Aperture"` and falls through to the
  outline drawer ([vis.py:471-474](DeepLens/deeplens/optics/geolens_pkg/vis.py#L471-L474)).
- An **air-bounded lone surface** (rare — e.g. a mirror in a hybrid system)
  falls back to `draw_widget` ([vis.py:514-517](DeepLens/deeplens/optics/geolens_pkg/vis.py#L514-L517)).

If a future render shows a glass element as just an outline instead of a fill,
that's the place to look.

---

## 2. The marginal-ray survivor mechanism (filled-bundle correctness)

### 2.1 The problem

Filled bundles are drawn as a polygon between the **top marginal** and
**bottom marginal** rays. If we sampled exactly 2 rays (extreme top + extreme
bottom from the pupil) and one of them vignettes, TIRs, or otherwise produces
NaN/inf in its path record, the polygon is built from a broken path → the
bundle either disappears or paints wildly outside the lens.

Symptom in past renders: ray bundle visibly stops short at one of the elements,
or the bundle envelope has a gap / spike.

### 2.2 The fix that's already in the code

Two-part:

**(a) Oversample.** [vis.py:207](DeepLens/deeplens/optics/geolens_pkg/vis.py#L207) sets `num_rays = 9 if style == "filled" else 11`. Nine evenly-spaced rays across the pupil for each field+wavelength.

**(b) Survivor filter inside `draw_ray_2d_bundle`** ([vis.py:546-562](DeepLens/deeplens/optics/geolens_pkg/vis.py#L546-L562)):

```python
finite_path = np.all(
    np.isfinite(rec[idx_view, :, :, 0])
    & np.isfinite(rec[idx_view, :, :, 2]),
    axis=1,
)
if finite_path.sum() < 2:
    continue
surv_idx = np.where(finite_path)[0]
entry_y = rec[idx_view, surv_idx, 0, 0]
i_lo = int(surv_idx[np.argmin(entry_y)])
i_hi = int(surv_idx[np.argmax(entry_y)])
```

Reading this:

1. Mark a ray as a "survivor" only if **every point along its traced path** is
   finite in both the y-axis and z-axis components.
2. If fewer than 2 survivors, skip the bundle (no polygon drawn — better than a
   broken polygon).
3. Among survivors, pick the **extremes by entry-y** (the y at the first
   surface) — these are the visible top/bottom marginals of whatever subset of
   the pupil actually got through.

The polygon is then closed as `top_path forward + bottom_path reversed` ([vis.py:563-577](DeepLens/deeplens/optics/geolens_pkg/vis.py#L563-L577)).

### 2.3 The diagnostic script

When a render still looks off — e.g. all 9 oversampled rays at some field
fail, so the bundle disappears entirely — use:

- [debug_filled_rays.py](debug_filled_rays.py)

What it does (per [debug_filled_rays.py:1-12](debug_filled_rays.py#L1-L12)):
For each of a curated set of "problem" lenses (currently 4: triplet T2,
sixlet S1, sixlet S3, septlet V3), it rebuilds the lens with the **exact same
pipeline** as `render_filled_collections.py` (including the F/# rescale and
the 0.92 aperture shrink), then traces 2 / 9 / 15 rays at three fields × RGB
wavelengths and reports per ray:

- whether the entire path is finite (survived) or hit NaN/inf
- the entry-y at the first surface (so you can see whether it was a marginal)
- which entry-y values failed

Sample interpretation:

- **`surviving=9/9`** at `num_rays=9` → bundle should render fine. Any visible
  defect is cosmetic, not a tracing failure.
- **`surviving=2/2` at `num_rays=2` but `surviving=9/9` at `num_rays=9`** →
  past behavior; the original 2-ray code would have lost one of the marginals.
  Confirms the oversampling+survivor patch is doing its job.
- **`surviving=0/9`** at high field → bundle will silently disappear. The fix
  is upstream of rendering — usually too aggressive a pupil for that field, or
  a surface that's TIRing for that wavelength. Try a smaller `aperture_shrink`
  (e.g. 0.85) or check whether the design itself is valid.

Run it with:

```bash
cd /home/hisham26/Desktop/RL
python debug_filled_rays.py
```

Adding a new "problem" lens: append a dict to `PROBLEMS` at
[debug_filled_rays.py:28-69](debug_filled_rays.py#L28-L69) with the same five
keys (`name`, `cfg`, `elite`, `refined_key`, `target_fnum`/`target_efl`/
`aperture_shrink`).

### 2.4 If you find a lens where bundles still look wrong

The order of operations to try, in increasing cost:

1. Run `debug_filled_rays.py` (after adding the lens to `PROBLEMS`) and read
   the per-field survivor counts.
2. If survivors at the failing field is **≥ 2 but small** (say 2 or 3 out of 9):
   the polygon is probably correct but visually thin. Increase `num_rays` for
   filled style at [vis.py:207](DeepLens/deeplens/optics/geolens_pkg/vis.py#L207) — try 15 — to give more candidates for the entry-y extremes.
3. If survivors is **0**: it's not a rendering problem. The lens really is
   blocking that field at the chosen pupil. Reduce `aperture_shrink` (e.g.
   0.92 → 0.85) at the call site, or for diagnosis only, drop `fov_ls` to
   `linspace(0, rfov*0.7, 3)` to confirm the corner field is the culprit.
4. If survivors is **9 but the bundle still looks wrong**: the polygon vertices
   are fine — the issue is that top/bottom of *entry-y* doesn't correspond to
   top/bottom *along the whole path* (rays cross). This is rare with a
   reasonable pupil, but possible. Inspect by temporarily changing
   `draw_ray_2d_bundle` to also stroke each survivor as a faint line so you can
   see which ray is where.

---

## 3. TL;DR for an agent inheriting this work

1. Render via `lens.draw_layout(filename=..., style="filled", glass_facecolor="#BFE3F2", glass_edgecolor="black", glass_lw=1.2, bundle_alpha=0.22)`. Don't reinvent the lens-building scaffolding — copy `build_lens` from [render_filled_collections.py](render_filled_collections.py).
2. For sixlet/septlet/decuplet, pass `aperture_shrink=0.92` when building the lens. For F/8 triplets, leave it at 1.0.
3. The marginal-ray survivor filter is already in place — `draw_ray_2d_bundle` oversamples to 9 rays and discards ones with NaN/inf in their path.
4. If bundles still look broken on a new lens, add it to `PROBLEMS` in [debug_filled_rays.py](debug_filled_rays.py) and run that script. The output tells you whether the issue is sample count, aperture, or the lens itself.
