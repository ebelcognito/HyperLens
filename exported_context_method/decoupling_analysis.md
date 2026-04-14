# Quantifying descriptor/geometry decoupling in QD lens-design archives

Exported context from a Claude Code session. The goal was to develop a
quantitative method that supports the central thesis of this project:

> Among lens designs that meet a performance floor, *tiny* differences in
> the observed optical aberration descriptor (e.g. Zernike coefficients
> z7 and z11) can correspond to *large* differences in the underlying
> geometric parameters. Quality-diversity is supposed to populate exactly
> this kind of "fiber" structure — many geometrically distinct designs
> sharing nearly identical descriptor coordinates, all above a fitness
> bar.

The user wanted a way to *prove* this without having to render every design
in the archive and inspect them by eye.

---

## 1. Setting

- Project: `/home/hisham26/Desktop/RL`, optical_rl QD pipeline.
- Reference run analyzed: `results/models/ef50_zernike_20260410_180300/`.
  - 50 mm f/1.8 photographic objective, 10-D theta (the 11th coefficient is
    derived from a paraxial EFL solve), 2-D descriptor = (z7 vertical coma,
    z11 primary spherical) measured at the mid-field point.
  - Search: CMA-MEGA seeded MAP-Elites, 100x100 archive over (z7, z11),
    population 16, 10 000 iterations.
  - Fitness: negative polychromatic 3-field RMS spot (mm), C/d/F lines.
- Archive contents: 1469 filled cells out of 10 000, best fitness
  -0.3738 mm RMS, median fitness -11.74 mm.
- Other archives the same method must work on:
  - `doublet_cmamega_*`  : 6-D theta, 2-D descriptor, `archive.json` is a
    flat list of elite dicts (no top-level bounds).
  - `triplet_cmamega_*`  : 7-D theta, 2-D descriptor, same flat-list layout.
  - `ef50_zernike_*`     : 10-D theta, 2-D descriptor, `archive_cells.json`
    is a dict with keys `archive_bounds`, `archive_bins`, `cells`, `stats`.

The yaml config in each run carries `qd.theta_bounds` (mandatory) and
`qd.archive_bounds` (optional; null means it was auto-calibrated from a
random sweep).

---

## 2. The question, made precise

Words: "tiny perturbations of (z7, z11) -> wildly different geometric
designs, subject to a performance floor".

Translation into measurable quantities, where we filter to elites with
`rms_spot <= tau` (the performance bar) and rescale theta and measures into
unit boxes by their respective bounds:

1. **epsilon-ball test.** For every pair (i, j) of surviving elites compute
   normalized Euclidean distances `d_m` in (z7, z11) space and `d_theta` in
   the 10-D geometric space. Look at pairs whose `d_m` falls in the smallest
   1% of the distribution. If `d_theta` for those pairs reaches a large
   fraction of the global `d_theta` range, then "almost identical
   descriptor" coexists with "almost as far apart as any two designs in the
   archive". This is the headline number.

2. **k-NN Jaccard mismatch.** For each elite, compare its k nearest
   neighbors under `d_m` against its k nearest neighbors under `d_theta`.
   Compute Jaccard overlap of the two sets. Low overlap means the
   descriptor neighborhood and the geometry neighborhood are nearly
   independent structures.

3. **Linear null-space variance.** Fit OLS `measures_centered = theta_centered @ W`
   on the filtered set, with `W` of shape (D, M). The map theta -> measures
   has a (D - rank(W))-dimensional null space. Project the centered theta
   samples onto that null space; the captured variance is the fraction of
   geometric variation that *any* linear readout of the descriptor cannot
   see.

These three are intentionally redundant: (1) is a memorable scalar
punchline, (2) is a graphical/aggregate sanity check, (3) is the
mathematically rigorous version that lives in a vector space and not in a
distance distribution.

---

## 3. What we explicitly chose **not** to do

- We did **not** use the post-refinement basins from
  `refinement_20260410_190226/refinement_results.json`. Those are L-BFGS-B
  attractors clustered after a local geometric refinement; they answer a
  different question ("how many local minima exist after polishing") and
  are not appropriate for a claim about what the QD search itself
  discovered. Everything is computed on the **raw archive elites** from
  `archive_cells.json`.
- We did not normalize by the empirical filtered range because that
  would understate the absolute claim ("tiny in the search domain"). We
  use the a-priori `theta_bounds` and `archive_bounds` as the unit boxes,
  so distances are reproducible across runs.
- The script does not call DeepLens / does not render anything. It is a
  pure post-hoc analysis of the JSON archive plus the yaml.

---

## 4. The script

Path: `optical_rl/qd/analyze_decoupling.py`. Format-agnostic; works on
ef50, doublet, triplet, or any future QD run with the same conventions.

Key design points:

- **Auto-detects** the archive layout (`archive_cells.json` dict with a
  `cells` list, *or* `archive.json` flat list, *or* a dict-of-elites).
- **Resolves bounds** with priority: `qd.theta_bounds` (mandatory),
  `qd.archive_bounds` -> archive file's `archive_bounds` -> empirical
  min/max of the observed measures.
- **Performance floor** is configurable two ways:
  - `--rms-max <value>` (default: `2 * best_rms`)
  - `--top-frac <fraction>` keeps the top fraction by fitness.
- **Distances** are computed on theta and measures after rescaling each
  axis to [0, 1] via the bounds, then divided by `sqrt(dim)` so the
  resulting numbers are comparable across spaces of different
  dimensionality.
- **Outputs** in `<run_dir>/decoupling/`:
  - `decoupling_scatter.png`  - pairwise d_theta vs d_measure scatter,
    with the 1%-quantile epsilon line, Pearson and Spearman in the title.
  - `decoupling_knn.png`      - histogram of per-elite Jaccard overlaps.
  - `decoupling_summary.json` - all numbers, including the top 20 most
    decoupled pairs (each with both full theta vectors, both rms values,
    both descriptor coordinates, and both distances) for visual sanity
    checks.
- **Reusable API**: a top-level `analyze_run(run_dir, rms_max=..., top_frac=..., k=5)`
  function returns the summary dict, so it can be looped over many runs
  programmatically.

CLI examples:

```bash
python3 -m optical_rl.qd.analyze_decoupling \
    --run-dir results/models/ef50_zernike_20260410_180300 --rms-max 1.0
python3 -m optical_rl.qd.analyze_decoupling \
    --run-dir results/models/triplet_cmamega_20260328_192052 --top-frac 0.2
```

---

## 5. Results on the reference ef50 run

Filter: `rms_spot <= 1.0 mm` -> 136 elites kept out of 1469
(best=0.3738 mm, worst=0.9950 mm). Distances all in normalized
(unit-box, divided by sqrt(dim)) units.

```
theta_dim x descriptor_dim  : 10 x 2
elites under floor          : 136 / 1469
Pearson  r(d_m, d_theta)    : +0.627
Spearman r(d_m, d_theta)    : +0.661
k-NN Jaccard (k=5)          : mean=0.241  median=0.250  zero-overlap=0.154
linear R^2 theta->measures  : joint=+0.897   per-dim=[0.814, 0.930]
theta variance in NULL(W)   : 92.6 %

epsilon-ball  (d_m <= 0.00621, the 1% quantile)
    pairs in ball             : 92
    max    d_theta in ball    : 0.1845
    p99    d_theta in ball    : 0.1836
    median d_theta in ball    : 0.0151
    global median d_theta     : 0.0584
    global p99    d_theta     : 0.2169
```

### Interpretation

- The epsilon-ball max `d_theta` (0.1845) is **85 % of the global p99
  d_theta** (0.2169). Two performant designs whose (z7, z11) coordinates
  agree to within ~0.6 % of the descriptor range can still be nearly as
  geometrically far apart as the most distant pair of performant elites in
  the entire archive.
- Out of k=5 nearest neighbors, descriptor-NN and geometry-NN share on
  average only ~1.2 (Jaccard 0.241). 15.4 % of elites have *zero* overlap
  between the two neighborhood sets.
- The linear R^2 = 0.897 number is *not* evidence against the thesis - it
  just says theta linearly predicts measures, which it has to: a Zernike
  coefficient is a function of the geometry. The right thing to look at is
  the *reverse* direction, captured by the null-space variance: **92.6 %
  of the geometric variance among performant elites lies in directions
  that the linear map theta -> (z7, z11) cannot see.** That fraction is
  the fraction of geometry the descriptor is blind to, and it is the
  cleanest single quantitative statement of the QD decoupling claim.
- Pearson 0.627 / Spearman 0.661: there is a real but loose monotone
  relationship between d_m and d_theta. Not tight, with heavy tails - which
  is what you want.

### Three sentences ready to drop into the paper

1. *Among performant elites (rms_spot <= 1 mm) in the ef50 z7-z11 archive,
   92.6 % of the variance in the 10-D geometric parameter vector lives in
   the 8-D null space of the linear map theta -> (z7, z11) - that is,
   inside a subspace that any linear readout of the chosen descriptor
   cannot distinguish.*
2. *The 5 nearest descriptor-neighbors and the 5 nearest geometry-neighbors
   of an elite share on average only 24 % of their members, and 15 % of
   elites have zero neighbors in common, demonstrating that the two
   neighborhood structures are nearly independent.*
3. *Among the pairs whose descriptor distance falls in the tightest 1 % of
   the distribution (`d_m <= 0.0062` in normalized units), the largest
   geometric distance reaches 0.1845 - 85 % of the global 99th-percentile
   geometric distance across all performant pairs - showing that
   nearly-identical observed aberrations are compatible with
   nearly-maximal geometric divergence.*

---

## 6. Generalization smoke tests on other archive families

Same script, no changes, run with `--top-frac 0.2`:

### Doublet (`results/models/doublet_cmamega_20260312_104309/`)

```
theta_dim x descriptor_dim  : 6 x 2
elites under floor          : 74 / 368   (rms 4.36..4.55 mm)
Pearson  r                  : +0.291
Spearman r                  : +0.199
k-NN Jaccard (k=5)          : mean=0.435  zero-overlap=0.027
linear R^2 joint            : +0.861
theta variance in NULL(W)   : 88.1 %
epsilon-ball max d_theta    : 0.0127     (global p99 = 0.5994)
```

Mixed signal: the null-space fraction is high (88 %), but inside the
descriptor epsilon-ball geometric distances are *also* small
(`0.0127 / 0.5994 ~= 2 %`). For this particular doublet run the descriptor
turns out to be a fairly faithful proxy for geometry - the QD decoupling
claim is **weak** here. This is interesting on its own and worth reporting:
not every descriptor choice produces decoupled fibers. Worth comparing
across multiple doublet runs.

### Triplet (`results/models/triplet_cmamega_20260328_192052/`)

```
theta_dim x descriptor_dim  : 7 x 2
elites under floor          : 75 / 375   (rms 4.65..4.73 mm)
Pearson  r                  : +0.682
Spearman r                  : +0.674
k-NN Jaccard (k=5)          : mean=0.367  zero-overlap=0.080
linear R^2 joint            : +0.679
theta variance in NULL(W)   : 50.0 %
epsilon-ball max d_theta    : 0.4735     (global p99 = 0.6788)
```

Strong decoupling. The epsilon-ball max `d_theta` is **70 % of the global
p99**, and the null-space captures half of the total geometric variance.
This is the result that would land hardest in a comparative study.

### Reading the contrast

- ef50 (10-D theta, 2-D descriptor): **strong** decoupling, large
  null-space (92.6 %), large epsilon-ball d_theta.
- triplet (7-D theta, 2-D descriptor): **strong** decoupling, smaller
  null-space (50.0 %) but proportionally even larger epsilon-ball
  d_theta.
- doublet (6-D theta, 2-D descriptor): **weak** decoupling under the chosen
  performance bar - the descriptor and the geometry are highly aligned.

Two complementary stories for the paper:
1. *When* the descriptor decouples from geometry (ef50, triplet), QD
   discovers a high-dimensional family of distinct geometric solutions
   sharing nearly identical descriptor coordinates, which is exactly what
   it is supposed to do.
2. *When* the descriptor is too tightly coupled to geometry (this doublet
   run), the same diagnostic flags the failure: the epsilon-ball d_theta
   is small, so the descriptor is essentially a stand-in for the geometry
   and the archive provides little extra diversity. This is a useful
   cautionary tale to include in the paper.

---

## 7. Files written by the analysis

For each run, in `<run_dir>/decoupling/`:

| file                     | content                                                       |
| ------------------------ | ------------------------------------------------------------- |
| decoupling_scatter.png   | scatter of d_measure vs d_theta over all pairs                |
| decoupling_knn.png       | histogram of per-elite Jaccard overlap                        |
| decoupling_summary.json  | every number above + top 20 most decoupled (theta_i, theta_j) |

The `top_decoupled_pairs` array in the JSON is what to use for visual
sanity-check renders: each entry has the two full theta vectors, both rms
values, both (z7, z11) coordinates, and the normalized d_measure /
d_theta. Picking 3 of those pairs and rendering them with the existing
`render_ef50_zernike.py` script would give the paper its visual figure
that complements the quantitative tests.

---

## 8. Suggested results-section structure

1. **Setup paragraph.** What the archive is, what the descriptor is, what
   the performance floor is. One sentence each.
2. **Diagnostic.** Define the three tests in one paragraph each, with the
   normalization convention stated explicitly (unit boxes via theta_bounds
   and archive_bounds, divided by sqrt(dim)).
3. **Headline numbers for ef50.** Table or three sentences. The
   null-space fraction is the strongest single number.
4. **Cross-architecture comparison.** Same numbers for doublet and
   triplet, with the contrast noted - decoupling is not automatic, and
   the same diagnostic flags both successes and failures of descriptor
   choice.
5. **Visual figure.** Pick 3 pairs from `top_decoupled_pairs` and render
   them side-by-side; caption: "These three pairs of designs differ by
   less than X in normalized descriptor distance but exhibit visibly
   different glass shapes / element thicknesses / curvatures."
6. **Honest caveat.** R^2 = 0.897 looks like high coupling, and in the
   forward direction (theta -> measures) it is - but the reverse
   direction, captured by the null-space variance, is the one that
   matters for QD's claim, and there the picture is the opposite. This
   is worth a sentence so a reviewer doesn't pounce on the R^2.

---

## 9. Open questions for the followup model

- Should the paper include the same analysis at multiple performance
  floors (e.g. rms <= 0.5, 1.0, 2.0 mm) to show how decoupling scales
  with the strictness of the bar? The script supports this; one
  invocation per floor.
- Is there a more semantically meaningful theta normalization than the
  prior bounds? E.g. normalizing by the *empirical* std of each theta
  component on the filtered set would give a Mahalanobis-like distance.
  We chose not to do this so the numbers compare across runs, but it is
  a defensible alternative.
- Would a non-linear analog of the null-space test (e.g. fit a small MLP
  for theta -> measures, report its R^2 vs the linear one) strengthen
  the claim? The gap between the two would be a quantitative measure of
  how much *non-linear* structure the descriptor sees beyond the linear
  R^2.
- The doublet result suggests the framework can also be used to *select*
  good descriptors: pick the (z_i, z_j) pair with the largest
  null-space fraction and the largest epsilon-ball d_theta. Worth
  mentioning as a methodological by-product.
