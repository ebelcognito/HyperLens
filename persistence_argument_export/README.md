# Persistence-then-Improvement: export folder

This folder packages everything needed to write the "persistence-driven
basin discovery" subsection of the LensMEGA paper, using the seventuplet
`-+--+-` family as the demonstrative example.

## Reading order for an inheriting agent

1. **`DISCUSSION.md`** — the full discussion arc with the user's own
   framing of the persistence argument, in their words. Start here to
   understand *why* this example was chosen and *what* claims it is
   meant to support.
2. **`PERSISTENCE_DEMO.md`** — the original deeper analysis writeup of
   the lineage chain (one terminus → 7-link parent chain → seed).
3. **`PAPER_DEMO_TEXT.md`** — paper-ready prose for the results
   section, suggested figure captions, and a "how this connects to
   the rest of the paper" table.
4. **`lineage_chain_full.json`** — the 7-link parent chain extracted
   directly from the run's `lineage.jsonl`. Ancient→recent ordering;
   each entry has `gen`, `theta_hash`, `parent_theta_hash`, `cell`,
   `measures (S2, S3)`, `fitness`, `rms_um`, `seidel_all`,
   `invalid_fraction`, `efl_actual`, `source`, `insert_status`.

## State JSONs (the three rendered stages of the demo figure)

| file | role in the figure |
|---|---|
| `state_A_gen824_parent.json` | Stage A — bad-but-stored ancestor (gen 824) |
| `state_B_elite_0020_raw.json` | Stage B — direct CMA-MEGA child of A (raw, no L-BFGS) |
| `state_C_elite_0020_refined.json` | Stage C — same individual after L-BFGS polish |

Plus the two siblings of `elite_0020` that share the same parent chain
(both branched from gen 824 in the same generation):

| file | refined RMS |
|---|---:|
| `state_sibling_elite_0006.json` | 7.5 µm (cell (128, 137)) |
| `state_sibling_elite_0019.json` | 2.8 µm (cell (137, 139)) |

All five state JSONs include the full 43-D `theta`, the (S₂, S₃)
descriptor cell, the per-element Seidel fingerprint, the EFL, and the
on-the-fly L-BFGS refinement output (where applicable).

## Numerical-results JSONs

| file | content |
|---|---|
| `gradient_alignment_gen824.json` | per-Seidel cosines at gen 824, 43-D random-vector null, subspace-projection result (1.000) |
| `alignment_summary.json` | local-landscape regression and 3-D plane-fit values |

## Figures (`figures/`)

| file | what it shows |
|---|---|
| `composite_persistence_demo.png` | 3-stage composite: (lens diagram + spot map) for each of A / B / C, common ±2.6 mm spot scale |
| `arrows_2d.png` | (S₂, L) and (S₃, L) panels — coloured arrows for `−∇L` and the descent-aligned descriptor gradients, plus a dashed grey "mean LC" arrow |
| `arrows_3d.png` | the same arrows in (S₂, S₃, L) space |
| `gradient_alignment_gen824.png` | per-Seidel `|cos|` bars + subspace-projection result |
| `walks_2d.png` / `walks_3d.png` | alternate visualisation: actual walks along ±gradients instead of arrows |

## Reproducibility one-liners

```bash
# Walk backward through the parent chain of any deep elite:
python3 family_a_analysis/lineage_three.py

# Re-render the 3-stage composite:
python3 render_persistence_demo.py

# Re-compute gradient alignment + subspace projection at gen 824:
python3 family_a_analysis/compute_gen824_gradients.py

# Re-render the arrow-based gradient-alignment figure:
python3 family_a_analysis/visualize_gradient_arrows.py
```

All scripts read from `results/lensmega/septlet_F3_20260425_224405/`
and the elite-design JSONs. **No re-running of the QD search itself
is needed.**

## Headline numbers to keep at the top of mind

- **6 of 7 ancestors had raw RMS > 12 mm.** Three entered the archive
  as first-occupants of empty (S₂, S₃) cells. None would have survived
  a fitness-ranked priority queue.
- **Total chain fitness improvement:** −23.29 → −0.82, a 28× drop —
  achieved through CMA-MEGA emissions only, no L-BFGS in the chain.
- **Per-Seidel alignment at gen 824:** `|cos(−∇L, ∇M_S₂)| = 0.38`,
  `|cos(−∇L, ∇M_S₃)| = 0.43`. 43-D random null = 0.13.
- **Subspace projection of Δθ from gen 824 → elite_0020:** 1.000 onto
  the 3-D framework subspace `span{∇L, ∇M_S₂, ∇M_S₃}` (random null
  0.249).
- **Mean LC arrow:** the equal-weighted mean of the three descent-
  aligned gradient directions has cos = **0.99** with `−∇L` —
  combining the descriptor gradients essentially recovers the descent
  direction itself.
- **Three deep siblings from the same chain:** `elite_0006` (7.5 µm),
  `elite_0019` (2.8 µm), `elite_0020` (2.7 µm) — three elites,
  one chain, one parent at the branch point.

## Caveat to flag

There is an autograd vs finite-differences sign disagreement on
`∇M_S₃` at this point (cos = 0.57; magnitudes match). The
gradient-alignment study's `|cos|` numbers are fine (it reported
magnitudes), but the *signed* relations involving S₃ should be
recomputed via FD if the paper text relies on them. The
visualisations in this folder use FD-computed gradients to be
empirically honest.
