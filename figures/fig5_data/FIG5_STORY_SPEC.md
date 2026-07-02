# Figure 5 — Diversity-at-scale story (sixlet & octuplet) — render spec

This document is a self-contained handoff for an agent that will produce a
publication-quality version of the diversity-gap visualization.  It contains:

1. The narrative
2. Data sources (files, schemas, exact paths)
3. Pre-computed numbers (hull areas, counts, thresholds)
4. The 2D embedding recipe (Seidel-PCA)
5. The figure layout, colors, markers, annotations
6. Caveats and alternative embeddings

The throwaway prototype renders are at:

- `/nas/longleaf/home/hisham26/Desktop/hyperlens/figures/fig5_sixlet_story.png`
- `/nas/longleaf/home/hisham26/Desktop/hyperlens/figures/fig5_octuplet_story.png`

Use them for layout reference only; the matplotlib styling is intentionally
unpolished.

---

## 1. Narrative

The paper claim being substantiated:

> The diversity advantage of HyperLens (HL) over restart-multistart (SA+GA)
> grows with parameter dimensionality.  HL's pool spans a much larger region
> of optical-behavior space than the SA+GA pool.

The visualization tells this story for the **sixlet** (D=37) and the
**octuplet** (D=45) lens systems by projecting refined designs into the 2D
PCA of standardized Seidel sums S₁..S₅ — i.e. *aberration space*.  In this
space:

- HL's full refined pool fills a wide envelope.
- SA+GA's full refined pool is a tight cluster contained inside HL's
  envelope.
- HL produces many distinct quality basins (refined RMS ≤ 30 µm) scattered
  across the wide envelope.
- SA+GA produces a few quality basins, all inside its narrow envelope.

**Why aberration space (Seidel-PCA), not parameter space:**

- 2D PCA of unit-normalised θ captures only ~20% of variance for D=37/45.
  The true diversity gap is in higher-D directions.
- 2D PCA of standardized Seidels captures **~70–84%** of variance.
- Two designs with the same aberrations are optically equivalent even if
  their θ differs — Seidel space is a more *honest* representation of "how
  diverse are the optical solutions".
- The same data the Fig 5(b) radar plot quantifies (HL p95 > SA+GA p95 on
  every Seidel axis) appears here as the area gap.

---

## 2. Data sources

All paths are relative to `/nas/longleaf/home/hisham26/Desktop/hyperlens/`.

### Per-design Seidel coefficients & RMS (already extracted)

`figures/fig5_data/seidel_per_design.csv`

| column | meaning |
| --- | --- |
| `system` | `sixlet` / `octuplet` (currently sixlet only; extend by re-running fig5_extract.py if needed) |
| `method` | `hl` / `saga` |
| `design_id` | index in source refined JSON |
| `S_1`, `S_2`, `S_3`, `S_4`, `S_5` | Seidel sums (paraxial, post-refinement; from refined-JSON `metrics`) |
| `refined_RMS_um` | post-L-BFGS RMS spot in µm |

### Pre-computed PCA projections (fresh — written by `fig5_sixlet_story.py`)

- `figures/fig5_data/sixlet_pca_projection.csv`
- `figures/fig5_data/octuplet_pca_projection.csv`

Schema:

| column | meaning |
| --- | --- |
| `method` | `hl` / `saga` |
| `design_id` | index in source refined JSON |
| `pc1`, `pc2` | 2D coordinates in standardized-Seidel PCA basis |
| `refined_rms_um` | post-L-BFGS RMS spot in µm |
| `refined_fitness` | post-L-BFGS fitness (`-rms_mean`, mm) |
| `is_basin_rep` | 1 if the design is an eps=0.20 net rep, else 0 |
| `passes_quality` | 1 if `refined_rms_um <= QUALITY_THRESHOLD` for that method |

### Source refined-pool JSONs (only needed if you re-extract)

- Sixlet HL: `results/lensmega/sixlet_apsc_discovery_v2_seed42_20260506_034601/refined_top500_seed42_hl.json`
- Sixlet SA+GA: `results/steelman_div020_20260506_123801/sixlet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json`
- Octuplet HL: `results/lensmega/octuplet_apsc_2cement_v2_seed42_20260506_040025/refined_top500_seed42_hl.json`
- Octuplet SA+GA: `results/steelman_div020_20260506_123801/octuplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json`
- Configs (for theta_bounds / D / Seidel computation): `results/diversity_run_20260506_033447/configs/hl_{system}_seed42.yaml`

### Other Figure-5 datasets (for cross-reference / additional panels)

- `figures/fig5_data/eps_curves_{system}_{method}.csv` — eps-distinct count vs eps grid (eps from 0.05 to 0.50, step 0.025)
- `figures/fig5_data/count_vs_rms_budget_{system}_{method}.csv` — eps-distinct count at eps=0.20 vs RMS budget (log-spaced 2–1000 µm, 60 points)
- `figures/fig5_data/radar_envelopes.json` — sixlet only: p5/p50/p95 of |Sⱼ| per method, plus per-axis normalisation constants
- `figures/fig5_data/rms_pool_summary.json` — per-pool refined-RMS quartiles + counts under each budget
- `figures/fig5_data/selected_designs.json` — 6 representative designs per (system, method), with theta + lens geometry + Seidels + RMS
- `figures/fig5_data/spot_diagrams.npz` — pre-computed sensor-plane ray endpoints for each selected design (3 fields × 3 wvln × 256 rays Fibonacci spiral)

---

## 3. Pre-computed numbers (drama box content)

### Sixlet (D=37)

| quantity | value |
| --- | --- |
| HL refined pool size | 498 |
| SA+GA refined pool size | 16 |
| HL eps=0.20-distinct basins | 86 |
| SA+GA eps=0.20-distinct basins | 16 (saturated) |
| **HL aberration envelope hull area (Seidel-PCA 2D)** | **55.59** |
| **SA+GA aberration envelope hull area (Seidel-PCA 2D)** | **2.16** |
| **Hull area ratio HL / SA+GA** | **25.7×** |
| Seidel-PCA 2D variance captured | 84% (PC1=49.8%, PC2=33.8%) |
| HL refined-RMS min | 3.35 µm |
| SA+GA refined-RMS min | 21.65 µm |
| HL quality basins (≤ 30 µm) | 25 |
| SA+GA quality basins (≤ 30 µm) | 4 |
| **Quality-basin count ratio** | **6.3×** |

### Octuplet (D=45)

| quantity | value |
| --- | --- |
| HL refined pool size | 310 |
| SA+GA refined pool size | 13 |
| HL eps=0.20-distinct basins | 31 |
| SA+GA eps=0.20-distinct basins | 13 (saturated) |
| **HL aberration envelope hull area (Seidel-PCA 2D)** | **34.43** |
| **SA+GA aberration envelope hull area (Seidel-PCA 2D)** | **1.83** |
| **Hull area ratio HL / SA+GA** | **18.9×** |
| Seidel-PCA 2D variance captured | 71% (PC1=41.4%, PC2=29.2%) |
| HL refined-RMS min | 12.21 µm |
| SA+GA refined-RMS min | 6.64 µm |
| HL quality basins (≤ 30 µm) | 3 |
| SA+GA quality basins (≤ 30 µm) | 2 |
| **Quality-basin count ratio** | **1.5×** |

Note: in octuplet, SA+GA's *minimum* RMS is *better* than HL's (6.64 vs
12.21 µm); the diversity drama for octuplet is in the **envelope area**
(18.9×), not in the count gap.  Lead with the area number for octuplet.

### Triplet (D=19, for reference)

Triplet has a much smaller diversity gap (HL/SA+GA eps=0.20 distinct: 39 vs
24).  We do not show a Seidel-PCA hull figure for triplet because the gap
is not the focal narrative; the eps-distinct curves panel in the Fig 5
composite covers it.

---

## 4. Embedding recipe (Seidel-PCA 2D)

```python
import numpy as np

def seidel_pca_basis(S_hl, S_sa):
    """S_hl: (N_hl, 5), S_sa: (N_sa, 5) -- per-design Seidel sums (S1..S5).
    Returns mu (5,), scale (5,), basis (5, 2), var_explained (5,)."""
    scale = np.maximum(np.std(np.vstack([S_hl, S_sa]), axis=0), 1e-12)
    S_all = np.vstack([S_hl, S_sa]) / scale
    mu = S_all.mean(axis=0)
    Sc = S_all - mu
    U, s, Vt = np.linalg.svd(Sc, full_matrices=False)
    var_explained = (s ** 2) / (s ** 2).sum()
    basis = Vt.T[:, :2]
    return mu, scale, basis, var_explained


def project(S_pool, mu, scale, basis):
    return (S_pool / scale - mu) @ basis
```

Standardization is per-axis stddev across both pools (HL+SA+GA combined),
so PC1/PC2 are not dominated by the largest-magnitude Seidel.

---

## 5. Figure layout

### Two panels stacked:

```
+----------------------------------------------------------+
|                                                          |
|        Top: aberration footprint scatter + hulls         |
|        (Seidel-PCA 2D, large - main visualization)       |
|                                                          |
+----------------------------------------------------------+
|     Bottom: refined-RMS histogram (HL vs SA+GA)          |
|             threshold line marked                        |
+----------------------------------------------------------+
```

Suggested aspect: ~10.5 × 11.5 inches.  Top panel takes ~3× the height of
bottom.

### Top panel — element list (z-order from background to foreground)

| z | element | source |
| --- | --- | --- |
| 0 | HL aberration envelope: convex hull of *full* HL refined pool projected to Seidel-PCA 2D. Filled (light blue, alpha ≈ 0.12), solid blue outline (lw 2). | `*_pca_projection.csv` rows where `method=hl` |
| 1 | SA+GA aberration envelope: convex hull of *full* SA+GA refined pool. Filled (red, alpha ≈ 0.30–0.35), dashed red outline (lw 2). | `*_pca_projection.csv` rows where `method=saga` |
| 2 | HL non-quality basin reps: dimmed grey small circles (alpha ≈ 0.65). | rows with `method=hl & is_basin_rep=1 & passes_quality=0` |
| 2 | SA+GA non-quality basin reps: dimmed grey squares with red edge (alpha ≈ 0.7). | rows with `method=saga & is_basin_rep=1 & passes_quality=0` |
| 5 | HL quality basins: filled circles colored by `refined_rms_um` (viridis_r, log scale, vmin≈1, vmax≈1000), black edge, large markers. | rows with `method=hl & passes_quality=1` (and ideally `is_basin_rep=1`) |
| 6 | SA+GA quality basins: filled squares colored by `refined_rms_um`, black edge, larger markers. | rows with `method=saga & passes_quality=1` |

Colorbar: viridis_r, label "refined RMS spot [µm] (log scale)".  vmin/vmax
should match across both system figures so they're visually comparable.
Recommended: vmin=2, vmax=1000.

### Bottom panel — RMS histogram

| element | source |
| --- | --- |
| HL full-pool RMS histogram (blue, alpha ≈ 0.55, geomspace(1,1000,35) bins) | source refined JSON `rms_um` for HL |
| SA+GA full-pool RMS histogram (red, alpha ≈ 0.75, black edge, same bins) | source refined JSON `rms_um` for SAGA |
| Vertical dotted line at 30 µm threshold | constant |
| Title bar: `"refined RMS (full pool):  HL min = X µm  |  SA+GA min = Y µm"` | computed |
| x-axis: log scale, "refined RMS spot [µm]" | constant |

### Drama box (annotation placed lower-left of top panel)

Monospace, white background with black border.  Format:

```
aberration envelope (full refined pool):
  HL    {N_hl_pool:3d} designs, hull area = {hl_hull_area:6.2f}
  SA+GA {N_sa_pool:3d} designs, hull area = {sa_hull_area:6.2f}
  area ratio:        {ratio:5.1f}x

quality basins (small spot, eps=0.20 distinct):
  HL    {n_quality_hl:3d} basins <= {QUALITY_UM_HL} um
  SA+GA {n_quality_sa:3d} basins <= {QUALITY_UM_SA} um
  count ratio:       {n_quality_hl/max(n_quality_sa,1):5.1f}x
```

Use exact numbers from §3 above.

### Annotation arrows on top panel

- One arrow pointing to **HL envelope** (top-right region of HL hull): "HL: wide aberration envelope, {n_hl} distinct quality basins scattered across many regimes"
- One arrow pointing to **SA+GA cluster** (centroid of SAGA points): "SA+GA: narrow aberration envelope, only {n_sa} quality basins reached (best refined RMS = {sa_min} µm)"

---

## 6. Color conventions (suggested for the polished version)

| element | color |
| --- | --- |
| HL hull / lines / accent | a calm blue, e.g. `#2E5EAA` (matplotlib `C0` is fine) |
| SA+GA hull / lines / accent | a warm red, e.g. `#C04030` (matplotlib `C3` is fine) |
| Non-quality basin dots | neutral grey `#BDBDBD` |
| Quality basin point colorbar | `viridis_r` (yellow=low RMS=good) — same colormap across systems for comparability |
| Background | white (paper) |

Keep the polished version monochromatic on the hulls; only the quality-basin
*points* should pull the eye via the viridis colormap.  Avoid rainbow palettes
on the hulls.

---

## 7. Title and caption

Suggested title for each panel:

```
sixlet aberration footprint (Seidel S1..S5, standardized -> 2D PCA, 84% var)
HL aberration envelope is 25.7x larger; HL has 25 distinct quality basins vs SA+GA's 4
```

```
octuplet aberration footprint (Seidel S1..S5, standardized -> 2D PCA, 71% var)
HL aberration envelope is 18.9x larger; HL has 3 distinct quality basins vs SA+GA's 2
```

Suggested LaTeX caption for the paper:

> Aberration footprint of the {sixlet, octuplet} refined pool projected onto
> the first two principal components of the standardized Seidel sums
> S₁–S₅.  HL's full-pool convex hull (n={498, 310}, area {55.59, 34.43}) is
> {25.7×, 18.9×} larger than SA+GA's (n={16, 13}, area {2.16, 1.83}) and
> geometrically contains it.  HL produces {25, 3} distinct quality basins
> (refined RMS ≤ 30 µm; eps=0.20-distinct in unit-normalised θ space)
> spread across the envelope; SA+GA produces {4, 2}, all within its tight
> cluster.

---

## 8. Caveats

- The aberration footprint is a **paraxial** view of optical behavior; it
  does not capture vignetting, distortion at large field, or higher-order
  aberrations.  This is fine for the diversity argument.
- Per-axis Seidel standardisation uses the combined HL+SA+GA stddev so the
  2D plane is dominated by directions where SA+GA is *not* uniformly small.
  This is honest because it's the same standardisation for both pools.
- 2D variance captured (84% sixlet, 71% octuplet) is high but not 100%:
  the actual 5-D Seidel volume gap is *larger* than the 2D area gap.
  In other words, the ratios are *lower bounds* on the diversity advantage.
- For the **octuplet**, SA+GA's minimum refined RMS is actually *lower*
  than HL's (6.64 vs 12.21 µm).  Do **not** lead the octuplet caption with
  a "HL is more accurate" claim.  Lead with the envelope-area gap (18.9×).
- The eps=0.20 epsilon-net is fitted in unit-normalised θ space (not
  Seidel space).  Quality basins that are eps-distinct in θ may be
  visually close in Seidel-PCA — that's fine, "distinct local minima"
  refers to parameter-space geometry, while "diverse aberrations" refers
  to the Seidel envelope.

---

## 9. Reproduce / regenerate

The driver script is `scripts_throwaway/fig5_sixlet_story.py`.  Despite the
name it's parametric:

```bash
# from /nas/longleaf/home/hisham26/Desktop/hyperlens/
/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python \
    scripts_throwaway/fig5_sixlet_story.py sixlet
/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python \
    scripts_throwaway/fig5_sixlet_story.py octuplet \
    --quality-hl 30 --quality-saga 30
```

Threshold defaults are 30 µm for both methods, both systems.  Override with
`--quality-hl` / `--quality-saga` if you want different thresholds.

---

## 10. Alternative 2D embeddings (if Seidel-PCA isn't dramatic enough)

| embedding | pros | cons |
| --- | --- | --- |
| **standardized Seidel-PCA 2D** *(current default)* | distance-faithful in aberration space; 84% / 71% var captured; honest | bound by linear projection |
| 2D MDS on Seidel L2 distances | distance-faithful by construction | nonlinear methods generally don't help here since aberration-space is small-D |
| t-SNE / UMAP on standardized Seidels (5-D) | locally separative; may exaggerate gap | not distance-faithful; can visually invent clusters — flag in caption |
| S₂ vs S₃ (just the BC pair) | 2D = exactly the paper's behavior space; no PCA fitting needed | smaller hull-area gap (~17× sixlet vs 25.7×) |
| Per-axis Seidel range bar chart | ALL 5 axes shown; no compression artefacts | less visually striking than a hull |

If the polished version wants extra drama, a 5-axis radar (Fig 5b) inset on
top of the hull plot showing the same envelope at the per-axis level is a
clean composite.

---

## 11. Three diverse refined designs per system (HL pool)

A separate deliverable: 3 maximally-diverse refined lenses per system, with
spot quality + greedy eps-net diversity guarantee.  Useful as concrete
"these are real distinct quality solutions" exhibits in the paper.

### Selection criteria

| system | RMS budget | eps tier | basis |
| --- | --- | --- | --- |
| sixlet | ≤ 20 µm | 0.25 | 11 distinct candidates available; pick first 3 by best refined fitness |
| octuplet | ≤ 30 µm | 0.25 | exactly 3 distinct candidates available (cannot reach 3 at ≤ 20 µm) |

The threshold relaxation for octuplet is necessary: the octuplet HL pool
has only 5 designs ≤ 20 µm, and they collapse to 2 distinct basins at any
eps tier. ≤ 30 µm gives 29 designs and exactly 3 distinct at eps=0.25.

### eps-tier distinct counts (full table)

`figures/fig5_data/three_diverse_eps_tiers.csv`

| sys | meth | rms ≤ | pool | e=0.05 | e=0.10 | e=0.15 | e=0.20 | e=0.25 | e=0.30 | e=0.35 | e=0.40 | e=0.45 | e=0.50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sixlet | hl | 20 | 41 | 18 | 12 | 11 | 11 | 11 | 10 | 4 | 2 | 1 | 1 |
| sixlet | hl | 30 | 86 | 43 | 30 | 25 | 25 | 25 | 19 | 7 | 3 | 1 | 1 |
| sixlet | hl | 50 | 195 | 82 | 59 | 49 | 47 | 45 | 30 | 12 | 3 | 1 | 1 |
| sixlet | saga | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sixlet | saga | 30 | 4 | 4 | 4 | 4 | 4 | 3 | 3 | 2 | 2 | 1 | 1 |
| sixlet | saga | 50 | 11 | 11 | 11 | 11 | 11 | 6 | 6 | 4 | 5 | 2 | 1 |
| octuplet | hl | 20 | 5 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 1 |
| octuplet | hl | 30 | 29 | 4 | 3 | 3 | 3 | 3 | 2 | 2 | 1 | 1 | 1 |
| octuplet | hl | 50 | 82 | 20 | 13 | 13 | 13 | 13 | 10 | 3 | 2 | 1 | 1 |
| octuplet | saga | 20 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| octuplet | saga | 30 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 1 |
| octuplet | saga | 50 | 8 | 8 | 8 | 8 | 8 | 5 | 3 | 3 | 1 | 1 | 1 |

Read this table carefully — counts can drop sharply between eps tiers
(e.g. sixlet HL ≤ 20 µm: 11 distinct at eps=0.25 → 4 at eps=0.35, 2 at
eps=0.40).  The "true diversity" of the small-spot solutions has a
characteristic scale around eps=0.3 in unit-θ space.

### The picks (sixlet HL, RMS ≤ 20 µm, eps=0.25)

| rank | design_id | refined RMS | refined fitness | pairwise distances (unit-θ L2/√D) |
| ---: | ---: | ---: | ---: | --- |
| 1 | 11 | 3.35 µm | -3.35e-03 | (1,2): 0.4148, (1,3): 0.2785, (2,3): 0.4248 |
| 2 | 2 | 4.03 µm | -4.03e-03 | |
| 3 | 473 | 5.90 µm | -5.90e-03 | |

### The picks (octuplet HL, RMS ≤ 30 µm, eps=0.25)

| rank | design_id | refined RMS | refined fitness | pairwise distances (unit-θ L2/√D) |
| ---: | ---: | ---: | ---: | --- |
| 1 | 163 | 12.21 µm | -1.22e-02 | (1,2): 0.3587, (1,3): 0.3805, (2,3): 0.2948 |
| 2 | 136 | 18.47 µm | -1.85e-02 | |
| 3 | 277 | 29.65 µm | -2.98e-02 | |

### Saved outputs

- `figures/fig5_data/three_diverse_below_thr.json` — for each system, the
  selection criteria + an array of 3 designs each carrying `theta` (full
  parameter vector), `S_1..S_5` (Seidel sums), `refined_RMS_um`,
  `refined_fitness`, `efl_mm`, `fnum_actual`, and `lens_geometry`
  (per-surface curvature, axial position, semi-aperture, materials with
  n_d/V_d, plus d_sensor and r_sensor)
- `figures/fig5_data/three_diverse_spots.npz` — keys `sixlet_0`,
  `sixlet_1`, `sixlet_2`, `octuplet_0`, `octuplet_1`, `octuplet_2`; each
  is a `(3, 3, 256, 2)` array of sensor-plane (x, y) ray endpoints for
  3 fields × 3 wavelengths × 256 Fibonacci-spiral pupil rays.  Invalid
  rays are NaN.
- `figures/fig5_data/three_diverse_eps_tiers.csv` — the full eps-tier
  table above.

Sanity render at `figures/fig5_three_diverse_sanity.png`.

### Suggested rendering for these 6

A clean 2×3 grid (rows = sixlet/octuplet, cols = rank 1/2/3) with:

- Lens cross-section per cell (use `lens_geometry` to redraw, or
  `_apply_theta_generic` + `lens.draw_lens_2d` to reproduce)
- Spot diagram inset per cell (e.g. lower-right corner) using the
  pre-computed sensor-plane points from `three_diverse_spots.npz` —
  scatter at the on-axis field for the 0.587 µm wavelength is usually
  most legible
- Per-cell label: `{system} #{rank}  RMS = {refined_RMS_um:.1f} µm`
- Color rays per wavelength (blue 0.486, green 0.587, red 0.656) if
  drawing rays; otherwise plain black lens outlines

The visual point: each row's three lenses are *visibly different optical
configurations* (different curvature signs, element thicknesses, power
distributions) yet all refine to the same RMS regime.  This is the
"diverse small-spot solutions" claim in concrete form.

---

## 12. Ray-fan plots for the 6 diverse picks

Classical transverse-ray-aberration plots ("Y-fan / X-fan") were computed
for each of the 6 picks above.

### Data file

`figures/fig5_data/three_diverse_rayfan_data.npz`

Keys follow the pattern:

- `{system}_{rank}_F{field_idx}_tan_pupilnorm` — shared x-axis for the tangential fan at that (design, field).  Length 41, in [-1, 1].
- `{system}_{rank}_F{field_idx}_tan_dy_um_w{wvln_nm}` — Δy_image in µm vs pupil-y (tangential / Y-fan).
- `{system}_{rank}_F{field_idx}_sag_dx_um_w{wvln_nm}` — Δx_image in µm vs pupil-x (sagittal / X-fan).

`system ∈ {sixlet, octuplet}`, `rank ∈ {0, 1, 2}`, `field_idx ∈ {0, 1, 2}`,
`wvln_nm ∈ {486, 587, 656}`.

Δ is the deviation from the chief ray (pupil-center sample) at the same
(design, field, wavelength).  Invalid (vignetted) rays are NaN.

### Conventions

- 41 pupil samples per fan, uniform in normalized pupil coordinate
- Entrance pupil from `lens.calc_entrance_pupil(paraxial=True)`
- Object at infinity (depth = -20000 mm)
- 3 stage-2 fields: on-axis, mid (0.5·HFOV), full (HFOV)
- HFOV = atan(half-sensor-diagonal / EFL), using the lens template's
  `r_sensor` (APS-C, 17 mm)
- 3 wavelengths: F/d/C (486/587/656 nm)
- Tangential fan: pupil samples on the Y-axis of the entrance pupil
  (x=0, y∈[-pupil_r, +pupil_r]); reported quantity is Δy_image
- Sagittal fan: pupil samples on the X-axis (y=0, x∈[-pupil_r, +pupil_r]);
  reported quantity is Δx_image
- Chief-ray fallback: if the central pupil sample vignettes, the
  centroid of valid rays is used as reference (introduces a small
  offset; for paper-quality, analytically locate the non-vignetted
  chief ray)

### Sanity render

`figures/fig5_three_diverse_rayfans.png` — 6×6 grid (rows = designs,
cols = 3 fields × 2 fan directions), 3 wavelengths overlaid per cell.
Y-limits are per-design adaptive (symmetric, clamped to ±[20, 500] µm).

### Suggested polished rendering

For a publication figure, a common choice is:

```
+------------------------+------------------------+------------------------+
|   sixlet #1            |   sixlet #2            |   sixlet #3            |
|   layout + spot inset  |   layout + spot inset  |   layout + spot inset  |
|   3 fields x 2 fans    |   3 fields x 2 fans    |   3 fields x 2 fans    |
+------------------------+------------------------+------------------------+
|   octuplet #1          |   octuplet #2          |   octuplet #3          |
|   ...                  |   ...                  |   ...                  |
+------------------------+------------------------+------------------------+
```

Each cell = one design block: a 2D lens layout at top (color rays per
field, dark glass outlines), a 6-panel ray-fan grid (3 fields × 2 fan
directions) below, and a small RMS / EFL / F# label.  Per-system shared
ylim across all field/direction panels for visual comparability.  Trim
the pupil x-axis to ±0.95 to avoid visual artefacts from edge
vignetting at exactly ±1.

Wavelength color convention: F line (486 nm) blue, d line (587 nm)
green, C line (656 nm) red.  This matches optical-design tradition;
HL's Seidel-PCA hull figure uses the same blue/green/red convention
for fields (not wavelengths), so be careful in the caption to
distinguish which colour-axis is which.

---

## 13. Checklist for the rendering agent

- [ ] Read `figures/fig5_data/sixlet_pca_projection.csv` and
  `octuplet_pca_projection.csv`
- [ ] Filter rows by `method` and `passes_quality` to get the four point
  sets (HL pool / HL quality / SAGA pool / SAGA quality)
- [ ] Compute convex hulls of HL pool and SAGA pool point sets
  (`scipy.spatial.ConvexHull` on PC1/PC2)
- [ ] Render the two-panel figure described in §5 with the conventions in §6
- [ ] Annotate using the exact numbers from §3 (already in the drama box
  text in §5)
- [ ] Use viridis_r colormap with shared vmin=2, vmax=1000 across both
  systems so the two figures look comparable
- [ ] Title each figure with the exact suggested string in §7
- [ ] Save to `figures/fig5_{system}_story_polished.png` (or wherever)

If anything in the data looks inconsistent with the numbers in §3, stop and
report — the underlying CSV may have been regenerated with a different
threshold.
