#!/usr/bin/env python
"""Composite paper figure: HL CHAMPION vs SA+GA archive coverage.

Left  panel: wide view at HL fit ≥ -10  (HL hull contains SAGA hull, 17.3x larger)
Right panel: tight zoom at HL fit ≥ -2   (high-quality core)
Bottom:     color legend strip

All input PNGs already rendered by render_archive_v16_oct.wl with matched-
threshold SAGA hulls (BD-clipped). This script just composes them into a
single paper-ready figure.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
DD = ROOT / "results/figure_archive_demo_octuplet_CHAMPION"
OUT = ROOT / "figures/octuplet_CHAMPION_vs_SAGA_archive.png"

WIDE = DD / "v16oct_main_hull-fit_gt_m10_sol-150.png"
TIGHT = DD / "v16oct_main_hull-fit_gt_m2_sol-150_tight.png"
LEGEND = DD / "v16oct_legend.png"

assert WIDE.exists() and TIGHT.exists() and LEGEND.exists()

img_wide = mpimg.imread(WIDE)
img_tight = mpimg.imread(TIGHT)
img_legend = mpimg.imread(LEGEND)

fig = plt.figure(figsize=(20, 12), facecolor="white")
gs = fig.add_gridspec(
    2, 2,
    left=0.03, right=0.97,
    top=0.85, bottom=0.04,
    wspace=0.06, hspace=0.04,
    height_ratios=[1.0, 0.18],
    width_ratios=[1.0, 1.0],
)

ax_w = fig.add_subplot(gs[0, 0])
ax_w.imshow(img_wide)
ax_w.set_axis_off()
ax_w.set_title(
    "Wide view  ·  HL hull at fitness ≥ −10  "
    "(coverage of stable, reasonable designs)",
    fontsize=15, family="serif", style="italic", color="#1A1A1A", pad=10,
)

ax_t = fig.add_subplot(gs[0, 1])
ax_t.imshow(img_tight)
ax_t.set_axis_off()
ax_t.set_title(
    "Tight core  ·  HL hull at fitness ≥ −2  "
    "(high-quality designs, refined-pool overlay)",
    fontsize=15, family="serif", style="italic", color="#1A1A1A", pad=10,
)

ax_l = fig.add_subplot(gs[1, :])
ax_l.imshow(img_legend)
ax_l.set_axis_off()

# Top header (more vertical room)
fig.text(
    0.5, 0.945,
    "Octuplet CHAMPION vs SA+GA — archive coverage at matched fitness thresholds",
    ha="center", va="center",
    fontsize=22, family="serif", style="italic", color="#1A1A1A",
)
fig.text(
    0.5, 0.905,
    "HL fit ≥ −10 hull is 17.3× SAGA's matched hull and contains every SAGA point "
    "(1 SAGA outlier outside HL's archive BD bounds excluded).",
    ha="center", va="center",
    fontsize=13, family="serif", style="italic", color="#3A3A3A",
)

# Outer frame
box = FancyBboxPatch(
    (0.008, 0.008), 0.984, 0.984,
    boxstyle="round,pad=0.0,rounding_size=0.008",
    transform=fig.transFigure,
    edgecolor="#1A1A1A", facecolor="none", linewidth=1.4,
    zorder=10, clip_on=False,
)
fig.patches.append(box)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, facecolor="white")
plt.close(fig)
print(f"-> {OUT}")
