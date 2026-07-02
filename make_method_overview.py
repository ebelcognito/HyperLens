"""Three-panel method overview: seeding -> CMA mutation -> filled archive.

Saves:
  method_overview_archive.png / .pdf  (combined figure with labels)
  panel_a_seeding.png                 (standalone, no labels, thick border)
  panel_b_cma.png                     (standalone, no labels, thick border)
  panel_c_filled.png                  (standalone, no labels, thick border)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

GRID = 6
CENTER = np.array([GRID / 2 - 0.5, GRID / 2 - 0.5])

# Blue -> dark-blue ramp matching the paper's lens / chief-ray palette.
# Anchors: pale paper-glass blue  ->  mid blue  ->  deep navy.
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "paper_blue", ["#D6EAF4", "#7BAFD4", "#1B4965"], N=256,
)
FRAME = "#1a1a1a"
SELECT_RING = "#e85d04"
BG_FILL = "#e6e8ee"  # faint grey-blue, like the reference archive background


def tw(t):
    """Map intensity t in [0, 1] onto the paper blue ramp (pale -> deep navy)."""
    t = float(np.clip(t, 0.0, 1.0))
    return BLUE_RAMP(0.10 + 0.85 * t)


GREENS = colormaps["Greens"]


def gn(t):
    """Map intensity t in [0, 1] onto the Greens cmap (light -> dark)."""
    t = float(np.clip(t, 0.0, 1.0))
    return GREENS(0.35 + 0.55 * t)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.linewidth": 0.6,
})


# ---------- precompute all randomness deterministically ----------
def precompute(seed=7):
    rng = np.random.default_rng(seed)

    # Panel A: seed cells, more spread out
    n = 5
    n_center = int(round(0.5 * n))
    n_far = n - n_center
    seed_cells = []
    while len(seed_cells) < n_center:
        xy = rng.normal(loc=CENTER, scale=GRID * 0.22)
        ix, iy = int(round(xy[0])), int(round(xy[1]))
        if 0 <= ix < GRID and 0 <= iy < GRID and (ix, iy) not in seed_cells:
            seed_cells.append((ix, iy))
    while len(seed_cells) < n_center + n_far:
        ix = int(rng.integers(0, GRID))
        iy = int(rng.integers(0, GRID))
        if (ix, iy) not in seed_cells:
            seed_cells.append((ix, iy))

    seed_a_t = []
    for (ix, iy) in seed_cells:
        d = np.linalg.norm(np.array([ix, iy]) - CENTER) / (GRID / 2)
        t = float(np.clip(1.0 - d * 0.7 + rng.normal(0, 0.25), 0.25, 1.0))
        seed_a_t.append(t)

    # Panel B: pick selected, sample CMA offspring far out
    seed_arr = np.array(seed_cells)
    dists = np.linalg.norm(seed_arr - CENTER, axis=1)
    selected = tuple(int(v) for v in seed_arr[np.argmin(dists)])
    sx, sy = selected

    cov = np.array([[2.4, 0.6], [0.6, 1.7]])
    min_radius = GRID * 0.25
    offspring = []
    tries = 0
    while len(offspring) < 3 and tries < 2000:
        tries += 1
        dxy = rng.multivariate_normal([0, 0], cov)
        if np.linalg.norm(dxy) < min_radius:
            continue
        ix, iy = int(round(sx + dxy[0])), int(round(sy + dxy[1]))
        if 0 <= ix < GRID and 0 <= iy < GRID and (ix, iy) != selected:
            if (ix, iy) not in offspring and (ix, iy) not in seed_cells:
                offspring.append((ix, iy))
    offspring_t = [float(rng.uniform(0.8, 1.0)) for _ in offspring]

    # Panel C: density field + outliers
    yy, xx = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    dgrid = np.sqrt((xx - CENTER[0]) ** 2 + (yy - CENTER[1]) ** 2)
    base = np.exp(-(dgrid / (GRID * 0.32)) ** 2)
    noise = rng.normal(0, 0.18, size=base.shape)
    field = np.clip(base + noise, 0, None)

    thresh = np.quantile(field, 0.35)
    filled = field > thresh

    n_outliers = max(1, int(0.029 * GRID * GRID))  # ~3% density
    outlier_positions = []
    while len(outlier_positions) < n_outliers:
        ix = int(rng.integers(0, GRID))
        iy = int(rng.integers(0, GRID))
        if dgrid[iy, ix] > GRID * 0.18 and (ix, iy) not in outlier_positions:
            outlier_positions.append((ix, iy))

    outlier_mask = np.zeros_like(filled, dtype=bool)
    for (ix, iy) in outlier_positions:
        outlier_mask[iy, ix] = True
        filled[iy, ix] = True

    cell_noise = rng.normal(0, 0.05, size=field.shape)
    outlier_t = {pos: float(rng.uniform(0.0, 0.05)) for pos in outlier_positions}

    # Local-step offspring: 5 cells *close* to the selected elite (small LC of
    # behavior gradients), used by the green-near-elite archive panel.
    # 3 offspring spread roughly uniformly around the elite at moderate radius.
    # Sample one angle per "wedge" so they end up well distributed.
    n_local = 3
    phase = float(rng.uniform(0, 2 * np.pi))
    base_angles = 2 * np.pi * np.arange(n_local) / n_local + phase
    local_r_lo, local_r_hi = 1.5, 2.4
    local_offspring = []
    for k, base_a in enumerate(base_angles):
        placed = False
        for _ in range(40):
            a = base_a + rng.uniform(-0.45, 0.45)
            r = rng.uniform(local_r_lo, local_r_hi)
            ix = int(round(sx + r * np.cos(a)))
            iy = int(round(sy + r * np.sin(a)))
            if (0 <= ix < GRID and 0 <= iy < GRID
                    and (ix, iy) != selected
                    and (ix, iy) not in local_offspring
                    and (ix, iy) not in seed_cells):
                local_offspring.append((ix, iy))
                placed = True
                break
        if not placed:
            for r in (1.8, 2.2, 1.4):
                a = base_a
                ix = int(round(sx + r * np.cos(a)))
                iy = int(round(sy + r * np.sin(a)))
                if (0 <= ix < GRID and 0 <= iy < GRID
                        and (ix, iy) not in local_offspring
                        and (ix, iy) != selected):
                    local_offspring.append((ix, iy))
                    break
    local_offspring_t = [float(rng.uniform(0.55, 1.0)) for _ in local_offspring]

    # Per-arrow green shade for the standalone arrows diagram (matches CMA offspring).
    arrow_green_t = [float(rng.uniform(0.6, 1.0)) for _ in offspring]

    return {
        "seed_cells": seed_cells,
        "seed_a_t": seed_a_t,
        "selected": selected,
        "offspring": offspring,
        "offspring_t": offspring_t,
        "field": field,
        "filled": filled,
        "outlier_mask": outlier_mask,
        "cell_noise": cell_noise,
        "outlier_t": outlier_t,
        "local_offspring": local_offspring,
        "local_offspring_t": local_offspring_t,
        "arrow_green_t": arrow_green_t,
    }


# ---------- drawing helpers ----------
def draw_grid_frame(ax, clean=False):
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(-0.5, GRID - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(BG_FILL)
    for s in ax.spines.values():
        s.set_color(FRAME)
        s.set_linewidth(2.0 if clean else 0.8)
    if not clean:
        step = 5 if GRID >= 10 else max(1, GRID // 3)
        for k in range(step, GRID, step):
            ax.axvline(k - 0.5, color="#d6d8dd", lw=0.4, zorder=0)
            ax.axhline(k - 0.5, color="#d6d8dd", lw=0.4, zorder=0)


def paint_cell(ax, x, y, color, edge=None, lw=0.0, zorder=2):
    rect = mpatches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                              facecolor=color, edgecolor=edge,
                              linewidth=lw, zorder=zorder)
    ax.add_patch(rect)


# ---------- panels ----------
def panel_seeding(ax, data, clean=False):
    draw_grid_frame(ax, clean=clean)
    for (ix, iy), t in zip(data["seed_cells"], data["seed_a_t"]):
        paint_cell(ax, ix, iy, tw(t))


def panel_cma(ax, data, clean=False):
    draw_grid_frame(ax, clean=clean)
    selected = data["selected"]
    sx, sy = selected
    for (ix, iy) in data["seed_cells"]:
        if (ix, iy) == selected:
            paint_cell(ax, ix, iy, tw(1.0), zorder=4)
        else:
            paint_cell(ax, ix, iy, tw(0.35))

    ring = mpatches.Circle((sx, sy), radius=0.55, fill=False,
                           edgecolor=SELECT_RING, linewidth=1.6, zorder=5)
    ax.add_patch(ring)

    for (ix, iy), t in zip(data["offspring"], data["offspring_t"]):
        paint_cell(ax, ix, iy, tw(t), zorder=3)

    for (ix, iy) in data["offspring"]:
        arr = FancyArrowPatch((sx, sy), (ix, iy),
                              arrowstyle="-|>,head_length=8,head_width=6",
                              mutation_scale=2.2,
                              color=SELECT_RING, lw=2.6, alpha=0.95,
                              zorder=6, shrinkA=6, shrinkB=4)
        ax.add_patch(arr)


def panel_local_green(ax, data, clean=False):
    """Archive view with 5 GREEN offspring close to the selected elite, no arrows."""
    draw_grid_frame(ax, clean=clean)
    selected = data["selected"]
    sx, sy = selected

    for (ix, iy) in data["seed_cells"]:
        if (ix, iy) == selected:
            paint_cell(ax, ix, iy, tw(1.0), zorder=4)
        else:
            paint_cell(ax, ix, iy, tw(0.35))

    ring = mpatches.Circle((sx, sy), radius=0.55, fill=False,
                           edgecolor=SELECT_RING, linewidth=1.6, zorder=5)
    ax.add_patch(ring)

    for (ix, iy), t in zip(data["local_offspring"], data["local_offspring_t"]):
        paint_cell(ax, ix, iy, gn(t), zorder=3)


def figure_arrows_only(data, out_path):
    """Standalone abstract diagram: source dot + 5 thick arrows + 5 green endpoints.
    No archive grid, no axes, white background."""
    selected = data["selected"]
    offspring = data["offspring"]
    sx, sy = selected

    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    xs = [sx] + [p[0] for p in offspring]
    ys = [sy] + [p[1] for p in offspring]
    pad = 1.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ARROW_BLUE = "#1B4965"
    SRC_EDGE   = "#0E2B40"
    for (ix, iy) in offspring:
        arr = FancyArrowPatch((sx, sy), (ix, iy),
                              arrowstyle="-|>,head_length=10,head_width=7",
                              mutation_scale=2.4,
                              color=ARROW_BLUE, lw=3.4, alpha=1.0,
                              zorder=2, shrinkA=10, shrinkB=8)
        ax.add_patch(arr)

    src = mpatches.Circle((sx, sy), radius=0.32,
                          facecolor=ARROW_BLUE, edgecolor=SRC_EDGE,
                          linewidth=1.4, zorder=4)
    ax.add_patch(src)

    for (ix, iy), t in zip(offspring, data["arrow_green_t"]):
        dot = mpatches.Circle((ix, iy), radius=0.26,
                              facecolor=gn(t), edgecolor="#0e3d18",
                              linewidth=1.0, zorder=3)
        ax.add_patch(dot)

    fig.savefig(out_path, dpi=600, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)


def panel_filled(ax, data, clean=False):
    draw_grid_frame(ax, clean=clean)
    field = data["field"]
    filled = data["filled"]
    outlier_mask = data["outlier_mask"]
    cell_noise = data["cell_noise"]
    outlier_t = data["outlier_t"]

    for iy in range(GRID):
        for ix in range(GRID):
            if not filled[iy, ix]:
                continue
            if outlier_mask[iy, ix]:
                t = outlier_t[(ix, iy)]
            else:
                v = field[iy, ix]
                t = float(np.clip(0.25 + 0.85 * v + cell_noise[iy, ix], 0.15, 1.0))
            paint_cell(ax, ix, iy, tw(t))


# ---------- compose ----------
import os
data = precompute(seed=7)
base = "/nas/longleaf/home/hisham26/Desktop/hyperlens/figure 2"
os.makedirs(base, exist_ok=True)

# combined figure (with titles + connector arrows)
fig = plt.figure(figsize=(15.0, 5.6))
gs = fig.add_gridspec(1, 3, wspace=0.18, left=0.03, right=0.985,
                      top=0.84, bottom=0.06)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
panel_seeding(ax1, data)
panel_cma(ax2, data)
panel_filled(ax3, data)

titles = [
    ("(a)  Seeding",        "random initial samples populate the archive"),
    ("(b)  CMA-MEGA step",  "offspring sampled around a selected elite"),
    ("(c)  Filled archive", "elites accumulated across generations"),
]
for ax, (t, sub) in zip([ax1, ax2, ax3], titles):
    ax.set_title(t, fontsize=12, loc="left", pad=18, color=FRAME, weight="bold")
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5,
            color="#555555", ha="left", va="bottom")

for x_from, x_to in [(0.345, 0.365), (0.665, 0.685)]:
    arr = FancyArrowPatch((x_from, 0.50), (x_to, 0.50),
                          transform=fig.transFigure,
                          arrowstyle="-|>", mutation_scale=14,
                          color="#888888", lw=1.0)
    fig.patches.append(arr)

fig.savefig(f"{base}/method_overview_archive.png", dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig(f"{base}/method_overview_archive.pdf",
            bbox_inches="tight", facecolor="white")
print(f"wrote {base}/method_overview_archive.png")
print(f"wrote {base}/method_overview_archive.pdf")
plt.close(fig)


# ---------- standalone panels for Illustrator ----------
panel_specs = [
    ("panel_a_seeding",      panel_seeding),
    ("panel_b_cma",          panel_cma),
    ("panel_b_local_green",  panel_local_green),
    ("panel_c_filled",       panel_filled),
]
for name, panel_fn in panel_specs:
    fig_s = plt.figure(figsize=(6.0, 6.0))
    ax_s = fig_s.add_axes([0.02, 0.02, 0.96, 0.96])
    panel_fn(ax_s, data, clean=True)
    out = f"{base}/{name}.png"
    fig_s.savefig(out, dpi=600, bbox_inches="tight",
                  facecolor="white", pad_inches=0.04)
    print(f"wrote {out}")
    plt.close(fig_s)

# Standalone abstract arrows diagram (no archive context)
arrows_out = f"{base}/arrows_only.png"
figure_arrows_only(data, arrows_out)
print(f"wrote {arrows_out}")
