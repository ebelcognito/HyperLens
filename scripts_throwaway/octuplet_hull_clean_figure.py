#!/usr/bin/env python
"""ONE clean figure — HL CHAMPION vs SA+GA on octuplet.

Hull threshold:   fit ≥ -10 (= SA+GA's own `tracker_threshold` from hyperparameters.json)
Points threshold: refined RMS < 50 µm

Layout: single panel, square, clean. Warm-band contour core left in; HL outer hull
in blue contains SAGA outer hull in red. Sub-30-µm points overlaid (HL blue, SAGA
rose). Top-right corner has a compact stats card. ε-distinct annotation makes the
diversity argument visually direct.

Inputs (all already exist on disk):
  results/figure_archive_demo_octuplet_CHAMPION/{archive_grid_v9_dual.csv,
       contours_v9_dual.json, hl_hull_fit_gt_m4.csv, saga_hull_fit_gt_m4.csv,
       hl_solutions_lt50um.csv, saga_solutions_lt50um.csv}

Output: figures/octuplet_hull_clean_HL_vs_SAGA.png
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
DD = ROOT / "results/figure_archive_demo_octuplet_CHAMPION"
CFG = ROOT / "optical_rl/qd/lensmega/config/examples/octuplet_apsc_2cement_v2.yaml"
OUT = ROOT / "figures/octuplet_hull_clean_HL_vs_SAGA.png"

# Refined-pool sources for ε-distinct (need θ vectors).
HL_CHAMP = ROOT / "results/lensmega/octuplet_REFSWEEP_CHAMPION_seed42_20260509_145427/refined_top500_seed42_CHAMPION.json"
SAGA_A   = ROOT / "results/diversity_run_20260506_033447/octuplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json"

# --- Style constants (matches the v16 Mathematica palette where possible) ---
HL_HULL_COLOR   = "#1A4A7A"   # deeper blue
HL_HULL_FILL    = "#7BAFD4"
HL_DOT_COLOR    = "#0A3A5C"
SAGA_HULL_COLOR = "#9A2A3A"
SAGA_HULL_FILL  = "#D49099"
SAGA_DOT_COLOR  = "#7A0C1C"
WARM_CMAP = [
    "#322036",   # deep purple-grey
    "#562842",   # plum
    "#834730",   # warm brown
    "#B07043",   # ochre
    "#D89953",   # gold
    "#F0C683",   # pale gold
    "#FAE7B3",   # cream
]

# --- Inputs ---
mat = np.loadtxt(DD/"archive_grid_v9_dual.csv", delimiter=",")
contours = json.load(open(DD/"contours_v9_dual.json"))
hl_hull = np.loadtxt(DD/"hl_hull_fit_gt_m4.csv", delimiter=",", skiprows=1)
sg_hull = np.loadtxt(DD/"saga_hull_fit_gt_m4.csv", delimiter=",", skiprows=1)
hl_sols = np.loadtxt(DD/"hl_solutions_lt50um.csv", delimiter=",", skiprows=1)
sg_sols = np.loadtxt(DD/"saga_solutions_lt50um.csv", delimiter=",", skiprows=1)

# Archive bounds (octuplet) — for the warm-band image data range
S2_LO, S2_HI = -0.005, 0.005
S3_LO, S3_HI = -0.008, 0.008

# Zoom viewport bounds — cover both hulls (HL fit≥−4 + SA+GA matched) with
# ~20% padding. SAGA's top vertex at S3≈+4.3e-3 needs to be visible.
S2_VIEW_LO, S2_VIEW_HI = -2.5e-3, 6.2e-3
S3_VIEW_LO, S3_VIEW_HI = -4.0e-3, 5.7e-3

# --- ε-distinct calc (matches earlier audit) ---
cfg = yaml.safe_load(open(CFG))
b = np.array(cfg['problem']['theta_bounds'], float)
lo, hi = b[:, 0], b[:, 1]
D = len(lo); rng = hi - lo; sqrtD = np.sqrt(D)


def load_designs_with_theta(json_path, gate_um=30.0):
    data = json.load(open(json_path))
    designs = data["designs"] if isinstance(data, dict) else data
    th, rms, fit = [], [], []
    for d in designs:
        if d.get("rms_um", 1e9) >= gate_um: continue
        if "theta" not in d: continue
        t = np.asarray(d["theta"], float)
        if t.size != D: continue
        th.append(t); rms.append(float(d["rms_um"]))
        fit.append(float(d.get("refined_fitness", -d["rms_um"])))
    if not th: return np.empty((0, D)), np.array([]), np.array([])
    return (np.array(th) - lo) / rng, np.array(rms), np.array(fit)


def eps_distinct(theta_norm, fitness, eps):
    if len(theta_norm) == 0: return 0
    order = np.argsort(-fitness); accepted = []
    for k in order:
        u = theta_norm[k]
        if all(np.linalg.norm(u - a)/sqrtD > eps for a in accepted):
            accepted.append(u)
    return len(accepted)


RMS_GATE_UM = 50.0
th_hl, rms_hl, fit_hl = load_designs_with_theta(HL_CHAMP, gate_um=RMS_GATE_UM)
th_sa, rms_sa, fit_sa = load_designs_with_theta(SAGA_A,   gate_um=RMS_GATE_UM)
eps_hl_20 = eps_distinct(th_hl, fit_hl, 0.20)
eps_hl_30 = eps_distinct(th_hl, fit_hl, 0.30)
eps_sa_20 = eps_distinct(th_sa, fit_sa, 0.20)
eps_sa_30 = eps_distinct(th_sa, fit_sa, 0.30)
print(f"HL CHAMPION  <{RMS_GATE_UM:.0f}µm: n={len(rms_hl)}  min={rms_hl.min():.2f}  ε=.20→{eps_hl_20}  ε=.30→{eps_hl_30}")
print(f"SAGA Pool A  <{RMS_GATE_UM:.0f}µm: n={len(rms_sa)}  min={rms_sa.min():.2f}  ε=.20→{eps_sa_20}  ε=.30→{eps_sa_30}")

# Hull areas + raw cell counts (for stats card / legend)
hl_area = ConvexHull(hl_hull).volume if len(hl_hull) >= 3 else 0.0
sg_area = ConvexHull(sg_hull).volume if len(sg_hull) >= 3 else 0.0
ratio = hl_area / sg_area if sg_area > 0 else float("inf")
# Re-derive n_cells (designs above the matched fit threshold) so the legend is honest.
HL_FIT_THR = -4.0
hl_archive = np.loadtxt(DD/"hl_archive.csv", delimiter=",", skiprows=1)
hl_n_cells = int((hl_archive[:, 2] >= HL_FIT_THR).sum())
sg_recomp = json.load(open(DD/"saga_descriptor_recompute.json"))
sg_arr = np.array([(d["S2"], d["S3"], d["fit"]) for d in sg_recomp["sg_unrefined"]])
sg_in_bd = (sg_arr[:,0]>=S2_LO)&(sg_arr[:,0]<=S2_HI)&(sg_arr[:,1]>=S3_LO)&(sg_arr[:,1]<=S3_HI)
sg_n_cells = int(((sg_arr[:, 2] >= HL_FIT_THR) & sg_in_bd).sum())

# --- Containment audit ---
hl_path = MplPath(hl_hull)
sg_inside_hl = hl_path.contains_points(sg_hull)
print(f"SAGA hull vertices inside HL hull: {sg_inside_hl.sum()}/{len(sg_hull)}")
sg_pts_inside = hl_path.contains_points(sg_sols[:, :2]) if len(sg_sols) else np.array([], dtype=bool)
print(f"SAGA sub-30 points inside HL hull: {sg_pts_inside.sum()}/{len(sg_sols)}")

# --- Render ---
# Wider aspect than tall since the zoom-bbox is wider (S2 span 7.4e-3) than
# tall (S3 span 6.1e-3). Pick 13:8.5 to give equal-aspect plot + headroom for
# title + side room for stats card.
fig = plt.figure(figsize=(13, 9.0), facecolor="white")
ax = fig.add_axes([0.085, 0.085, 0.62, 0.78], facecolor="#F8F6F1")

# Background warm-band image — pre-quantized 7-inside / 5-outside band field
# from build_inputs.py (percentile-equal bands, looks great).
img = np.flipud(mat)
rgba = np.full(img.shape + (4,), [0.965, 0.955, 0.935, 1.0])
inside = img > 0
out_cov = (img > -1.5) & (img < 0)
nbands = len(WARM_CMAP)
warm_rgb = np.array([
    [int(c[1:3], 16)/255, int(c[3:5], 16)/255, int(c[5:7], 16)/255]
    for c in WARM_CMAP
])
cool_rgb = np.array([
    [0.965, 0.955, 0.935],
    [0.90, 0.89, 0.86],
    [0.82, 0.83, 0.81],
    [0.72, 0.76, 0.76],
    [0.55, 0.66, 0.70],
])
if inside.any():
    bins = np.clip((img[inside] * (nbands - 1)).astype(int), 0, nbands - 1)
    rgba[inside, :3] = warm_rgb[bins]
if out_cov.any():
    v = -img[out_cov]
    bins = np.clip((v * (len(cool_rgb) - 1)).astype(int), 0, len(cool_rgb) - 1)
    rgba[out_cov, :3] = cool_rgb[bins]

# Note: we plot S2 on x-axis (Coma), S3 on y-axis (Astigmatism) — matches the
# v16 Mathematica DataRange order. Image data is (rows, cols) = (S3_bin, S2_bin).
ax.imshow(rgba, origin="lower", extent=[S2_LO, S2_HI, S3_LO, S3_HI],
          aspect="auto", interpolation="nearest", zorder=1)

# HL fit≥-10 hull (outer, contains SAGA)
v = hl_hull
ax.fill(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]),
        color=HL_HULL_FILL, alpha=0.16, zorder=2)
ax.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]),
        color=HL_HULL_COLOR, linewidth=2.6, zorder=4,
        label=f"HL  fit ≥ −4  hull (n={hl_n_cells}, area={hl_area:.2e})")

# SAGA fit≥-10 hull (BD-clipped, inner)
v = sg_hull
ax.fill(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]),
        color=SAGA_HULL_FILL, alpha=0.30, zorder=3)
ax.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]),
        color=SAGA_HULL_COLOR, linewidth=2.4, zorder=5,
        label=f"SA+GA  fit ≥ −4  hull (n={sg_n_cells}, area={sg_area:.2e})")

# Sub-30-µm refined points — HL CHAMPION
if len(hl_sols):
    ax.scatter(hl_sols[:, 0], hl_sols[:, 1],
               s=44, marker="o", c=HL_DOT_COLOR, edgecolors="white", linewidths=0.7,
               zorder=8,
               label=f"HL refined RMS < 50 µm  (n={len(hl_sols)}, ε=0.30 → {eps_hl_30} distinct)")

# Sub-30-µm refined points — SAGA Pool A
if len(sg_sols):
    ax.scatter(sg_sols[:, 0], sg_sols[:, 1],
               s=44, marker="o", c=SAGA_DOT_COLOR, edgecolors="white", linewidths=0.7,
               zorder=9,
               label=f"SA+GA refined RMS < 50 µm  (n={len(sg_sols)}, ε=0.30 → {eps_sa_30} distinct)")

# Cosmetic
ax.set_xlim(S2_VIEW_LO, S2_VIEW_HI)
ax.set_ylim(S3_VIEW_LO, S3_VIEW_HI)
ax.set_aspect("equal")   # geometrically faithful (same scale on both axes)
ax.set_xlabel("Coma  S₂  (×10⁻³)", fontsize=13, family="serif", labelpad=8)
ax.set_ylabel("Astigmatism  S₃  (×10⁻³)", fontsize=13, family="serif", labelpad=8)
# Format ticks as ×10⁻³
def tick_fmt(x, pos): return f"{x*1e3:+.1f}".rstrip("0").rstrip(".") if x != 0 else "0"
ax.xaxis.set_major_formatter(plt.FuncFormatter(tick_fmt))
ax.yaxis.set_major_formatter(plt.FuncFormatter(tick_fmt))
ax.grid(alpha=0.20, linewidth=0.4)

leg = ax.legend(loc="lower left", fontsize=9.5, frameon=True, fancybox=True,
                framealpha=0.96, edgecolor="#444", facecolor="white")
leg.set_zorder(20)

# Stats card (right-side panel, separate from the plot — no overlap with axes)
card_x, card_y, card_w, card_h = 0.730, 0.40, 0.250, 0.46
fig.add_axes([card_x, card_y, card_w, card_h], facecolor="#FFFFFFBF")
ax_card = fig.axes[-1]
ax_card.set_xticks([]); ax_card.set_yticks([])
for sp in ax_card.spines.values():
    sp.set_color("#444"); sp.set_linewidth(1.0)
ax_card.text(0.5, 0.96, "Hull comparison",
             ha="center", va="top", fontsize=13, family="serif",
             style="italic", color="#1A1A1A", transform=ax_card.transAxes)
ax_card.text(0.5, 0.89, "octuplet, 100 k iter, seed 42",
             ha="center", va="top", fontsize=10, family="serif",
             color="#3A3A3A", transform=ax_card.transAxes)
lines = [
    ("Hull threshold:",      "fit ≥ −4   (high-quality design gate, raw fitness)"),
    ("Points threshold:",    "refined RMS < 50 µm"),
    ("HL hull / SAGA hull:", f"{ratio:.1f}×  larger  ·  contains {sg_inside_hl.sum()}/{len(sg_hull)} SAGA hull verts"),
    (f"Sub-{int(RMS_GATE_UM)} µm pool:",     f"HL  n={len(hl_sols)}   SA+GA  n={len(sg_sols)}"),
    ("ε-distinct  @ ε=0.30:", f"HL  {eps_hl_30}   SA+GA  {eps_sa_30}     ←  HL wins {eps_hl_30/max(eps_sa_30,1):.1f}×"),
    ("Min refined RMS:",     f"HL  {rms_hl.min():.2f} µm    SAGA  {rms_sa.min():.2f} µm"),
]
y = 0.78
for lbl, val in lines:
    ax_card.text(0.04, y, lbl, ha="left", va="top",
                 fontsize=10, family="serif", weight="semibold",
                 color="#222", transform=ax_card.transAxes)
    ax_card.text(0.04, y - 0.06, val, ha="left", va="top",
                 fontsize=10, family="serif", color="#000",
                 transform=ax_card.transAxes)
    y -= 0.14

# Top title
fig.text(0.5, 0.955,
         f"Octuplet  ·  HL CHAMPION vs SA+GA  ·  fit ≥ −4 hulls + sub-{int(RMS_GATE_UM)} µm pool overlay",
         ha="center", va="center", fontsize=17, family="serif",
         style="italic", color="#1A1A1A")
fig.text(0.5, 0.920,
         f"HL's high-quality hull (fit ≥ −4) is {ratio:.1f}× SA+GA's; "
         f"at sub-{int(RMS_GATE_UM)} µm quality HL produces {eps_hl_30} ε-distinct lens forms ({len(hl_sols)} designs) vs SA+GA's {eps_sa_30} ({len(sg_sols)} designs).",
         ha="center", va="center", fontsize=10.5, family="serif",
         style="italic", color="#3A3A3A")

# Outer frame
box = FancyBboxPatch(
    (0.008, 0.008), 0.984, 0.984,
    boxstyle="round,pad=0.0,rounding_size=0.008",
    transform=fig.transFigure,
    edgecolor="#1A1A1A", facecolor="none", linewidth=1.3,
    zorder=10, clip_on=False,
)
fig.patches.append(box)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, facecolor="white")
plt.close(fig)
print(f"\n-> {OUT}")
