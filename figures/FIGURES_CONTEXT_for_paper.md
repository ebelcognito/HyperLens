# Figures Context Dump — octuplet CHAMPION paper

Living index of the PNGs that are paper-relevant + what to do with them.

---

## 1. Archive coverage figures (Mathematica v16)

Two-axis convention (now correct after FrameLabel fix): **X = Coma S₂ (×10⁻³)**,
**Y = Astigmatism S₃ (×10⁻³)**.

| file | purpose | spotlight set | hull threshold | button size |
|---|---|---|---|---|
| `results/figure_archive_demo_octuplet_CHAMPION/v16oct_main_hull-fit_gt_m10_sol-150_m4_SPREAD.png` | The "preferred" baseline archive figure, 4 spread-out picks (right / bottom / super / common). | spread (visual-position based) | fit ≥ −10 | full |
| `results/figure_archive_demo_octuplet_CHAMPION/v16oct_main_hull-fit_gt_m3p7_sol-150_m4_DIVERSE_smallbtn.png` | Tighter hull, smaller buttons, d1/d2/d3/common ε-distinct picks. | d1/d2/d3/common (ε-distinct in 45-D θ) | fit ≥ −3.7 | half (SPOT_SCALE=0.5) |
| `results/figure_archive_demo_octuplet_CHAMPION/v16oct_main_hull-fit_gt_m10_sol-150_m4_TOP10under20.png` | Top-10 ε-distinct sub-20 µm designs, pooled across **13 octuplet runs**. | top-10 under 20 µm | fit ≥ −10 | half |
| `results/figure_archive_demo_octuplet_CHAMPION/v16oct_main_hull-fit_gt_m10_sol-150_m4_UNDER20vsSAGA.png` | **All 70 sub-20 µm HL designs colored by SA+GA-hull membership**: 38 blue (outside SA+GA), 32 red (inside). The "where SA+GA misses" visual. | 70 sub-20 picks | fit ≥ −10 | small (SPOT_SCALE=0.3) |
| `results/figure_archive_demo_octuplet_CHAMPION/v16oct_main_hull-fit_gt_m10_sol-150_m4_STORYBOARD4.png` | The **4-design storyboard chart**: 2 rightmost + 1 bottom-most + 1 shared (20–25 µm). | 4 storyboard | fit ≥ −10 | full |

### What to do with these

* Pick **`STORYBOARD4.png`** as the main results-section archive figure. It
  emphasizes the 4 narrative designs.
* Use `UNDER20vsSAGA.png` as a discussion-section / appendix figure to
  defend the "HL covers BD regions SA+GA misses" claim quantitatively.
* `SPREAD.png` is a back-up — same hull, more generic spotlights.
* The `_DIVERSE_smallbtn.png` is the "tighter-quality" view; consider as a
  side-by-side companion if a reviewer asks about the fit ≥ −10 hull being
  too generous.

---

## 2. Lens-design library grids

| file | content |
|---|---|
| `figures/octuplet_top10_under20um.png` | Clean 2×5 grid, top 10 ε-distinct sub-20 µm lens cross-sections, NO per-card text. |
| `figures/octuplet_top10_under20um_LABELED.png` | Same 10 lenses, WITH source-run + RMS labels under each card (kept for reference). |
| `figures/octuplet_storyboard4_lenses.png` | The 4 storyboard designs as a 2×2 lens-card grid with role titles + RMS/S₂/S₃ labels. |

### What to do with these

* `octuplet_top10_under20um.png` is the "look at the variety" punchline. Put
  it next to the archive chart in the paper.
* `octuplet_storyboard4_lenses.png` pairs with the Seidel + robust panels
  in §4 to tell a per-design story.
* All lens cards use the soft-blue / soft-red glass convention (`#7BAFD4` /
  `#E07B7B`); shared/common designs are red, HL-exclusive are blue.
* **Note on vertical whitespace**: the lens aspect is ~2.2:1 (75 mm long ×
  34 mm tall), so panels have lots of empty space above/below the lens.
  Fix is to drop figsize height (e.g. 20×9 → 20×6) and tighten hspace
  — see `scripts_throwaway/render_octuplet_top10_under20um.py`.

---

## 3. Per-design analytical panels (Seidel + stability ribbon)

**Paper-grade simplified version**, 4 PNGs (one per storyboard design):

| file | role | RMS µm | S₂ ×10⁻³ | S₃ ×10⁻³ | source run |
|---|---|---|---|---|---|
| `figures/storyboard4_combined_panels/storyboard_right1_seidel_robust.png` | Rightmost #1 (high coma) | 19.84 | +0.56 | +0.08 | CHAMPION |
| `figures/storyboard4_combined_panels/storyboard_right2_seidel_robust.png` | Rightmost #2 (high coma) | 15.97 | +0.51 | −0.35 | CHAMPION |
| `figures/storyboard4_combined_panels/storyboard_bottom_seidel_robust.png` | Bottom-most (low astig) | 17.25 | +0.09 | −0.71 | F_patience150 |
| `figures/storyboard4_combined_panels/storyboard_common_seidel_robust.png` | Shared basin (in SA+GA hull, 20–25 µm) | 20.03 | −0.03 | +0.08 | fig4 seed-43 |

Each PNG: **Seidel bar chart (|S₂|, |S₃|) on the left + RMS-vs-field stability
ribbon on the right**, side-by-side. Style: endpoint-only y-ticks, large
bold sans-serif, **no axis labels / no headers / no titles** — minimum chrome.

### What to do with these

* Pair each PNG with its matching lens card from the `_lenses.png` 2×2
  grid (or extract from `octuplet_storyboard4_lenses.png` if needed).
* Layout suggestion: a 4-row table where each row is **lens crossection |
  combined Seidel+robust panel**. Each row is one of the 4 storyboard
  designs.
* The ribbon thickness = wavelength chromatic spread; the line is the
  per-field median across {486, 587, 656 nm}. Lower y-extent = sharper.

---

## 4. Older 12-PNG triplet set (probably can be retired)

`figures/champion_spotlight_panels/` contains 12 PNGs from the earlier
d1/d2/d3/common diverse picks — three panels per design (lens, seidel,
robust). The **combined seidel-robust style in §3 supersedes the separate
seidel + robust panels here.** Keep for reference; don't use in the paper.

---

## 5. Compute and provenance summary

| | run | refined-pool source |
|---|---|---|
| primary CHAMPION (octuplet) | `results/lensmega/octuplet_REFSWEEP_CHAMPION_seed42_20260509_145427/` | full 100 k budget, n=446 |
| BLIND_COMBO (octuplet, partial budget) | `results/lensmega/octuplet_REFSWEEP_BLIND_COMBO_seed42_20260509_074619/` | 54 % budget, n=103 |
| sub-20 µm pool (across all runs) | `/tmp/octuplet_under20_resolved.json` (regenerated) | 70 designs |
| storyboard 4 picks | `results/figure_archive_demo_octuplet_CHAMPION/spotlights_storyboard4.json` | — |

---

## 6. Companion paper docs (already written)

* `figures/CHAMPION_results_for_paper.md` — full octuplet CHAMPION results dump
* `figures/CHAMPION_sixlet_results_for_paper.md` — sixlet CHAMPION companion
* `figures/RESULTS_summary_for_paper.md` — cross-topology summary
* `figures/REPORT_octuplet_hl_ablation.md` — full ablation context (octuplet)
* `figures/REPORT_hl_hyperparam_ablation.md` — full ablation context (sixlet)

---

## 7. Open items / TODO

* [ ] Refine the remaining 394 sixlet CHAMPION designs (lift the top-500
      cap) — script staged at
      `scripts_throwaway/refine_sixlet_champion_uncapped.sh`, needs compute.
* [ ] (Optional) Reduce vertical whitespace in
      `octuplet_top10_under20um.png` by re-rendering at figsize 20×6.
* [ ] (Optional) Re-render `STORYBOARD4.png` archive with per-spot text
      labels (e.g. "R1", "R2", "B", "C") next to each circle so the chart
      cross-references the 4 lens cards visually.
* [ ] (Optional) 4-row paper-section composite: lens card | seidel+ribbon
      panel, stacked for all 4 storyboard designs.

---

*Generated 2026-05-11. Companion to the figures themselves; refresh when
new figures are added or the paper outline changes.*
