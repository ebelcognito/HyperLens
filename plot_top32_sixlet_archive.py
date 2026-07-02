"""Combined archive map of the 32 mono-λ sixlet picks.

For each design we:
  - Look up its parent-run refined_top500 entry to get (m0_S2_coma, m1_S3_astigmatism)
  - Project onto the shared 512x512 descriptor grid
     (bounds [[-0.008, 0.008], [-0.012, 0.012]] from sixlet_apsc_discovery_v2)
  - Aggregate: cell -> best mono_fitness if multiple designs collide

Render:
  - Square view centered on archive center (cell 256, 256)
  - Side = 2 * R_max where R_max is the furthest occupied cell's radial dist
  - Empty cells white; occupied cells colored by symlog(fitness) clipped to [-1, 0]
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
TOP32 = ROOT / "scripts_throwaway/mono587_top40_sixlet_eps030_sub10.json"
OUT_PNG = ROOT / "mono587_top32_sixlet_renders/archive_map_top32.png"

BINS = (32768, 32768)  # finer than the archive's 512 so top-fitness designs separate
BOUNDS = np.array([[-0.008, 0.008], [-0.012, 0.012]])  # S2, S3 (archive bounds)
CENTER = (BINS[0] // 2, BINS[1] // 2)


def fetch_descriptor(run, idx):
    fps = sorted(glob.glob(str(ROOT / f"results/lensmega/{run}/refined_top*.json")))
    if not fps:
        return None
    d = json.load(open(fps[0]))
    if idx >= len(d["designs"]):
        return None
    m = d["designs"][idx].get("metrics", {})
    s2 = m.get("m0_S2_coma", m.get("seidel_S2"))
    s3 = m.get("m1_S3_astigmatism", m.get("seidel_S3"))
    return None if s2 is None or s3 is None else (float(s2), float(s3))


def bin_index(val, lo, hi, n):
    """Map continuous val in [lo, hi] -> integer bin [0, n-1]."""
    f = (val - lo) / (hi - lo)
    return int(np.clip(np.floor(f * n), 0, n - 1))


def main():
    bundle = json.load(open(TOP32))
    designs = bundle["designs"][:32]

    cell_fit = {}     # (i, j) -> best mono_fitness
    cell_ranks = {}   # (i, j) -> list of ranks
    skipped = []
    for d in designs:
        rank = int(d["rank"]); fit = float(d["mono_fitness"])
        coords = fetch_descriptor(d["run"], int(d["idx"]))
        if coords is None:
            skipped.append(rank); continue
        s2, s3 = coords
        i = bin_index(s2, *BOUNDS[0], BINS[0])
        j = bin_index(s3, *BOUNDS[1], BINS[1])
        if (i, j) not in cell_fit or fit > cell_fit[(i, j)]:
            cell_fit[(i, j)] = fit
        cell_ranks.setdefault((i, j), []).append(rank)

    n_cells = len(cell_fit)
    n_collisions = sum(1 for v in cell_ranks.values() if len(v) > 1)
    print(f"Designs placed: {sum(len(v) for v in cell_ranks.values())} "
          f"in {n_cells} cells; {n_collisions} cell(s) with >1 design.")
    if skipped:
        print(f"Skipped (no descriptor in source JSON): {skipped}")
    for cell, ranks in sorted(cell_ranks.items()):
        if len(ranks) > 1:
            print(f"  cell {cell}: ranks {ranks}")

    # Radial extent from center
    R = 0.0
    for (i, j) in cell_fit:
        r = np.hypot(i - CENTER[0], j - CENTER[1])
        if r > R: R = r
    R_ceil = int(np.ceil(R)) + 1
    print(f"R_max = {R:.2f} cells; using window radius {R_ceil}")

    # Build the cropped image
    W = 2 * R_ceil
    img = np.full((W, W), np.nan, dtype=np.float64)
    i0 = CENTER[0] - R_ceil
    j0 = CENTER[1] - R_ceil
    for (i, j), f in cell_fit.items():
        img[i - i0, j - j0] = f

    # Render
    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="white")
    norm = SymLogNorm(linthresh=1e-3, linscale=1.0, vmin=-1.0, vmax=0.0, base=10)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")
    masked = np.ma.masked_invalid(img)
    # Note: img is indexed [i, j] = (S2-bin, S3-bin). Display with S2 on x-axis,
    # S3 on y-axis. Transpose so columns become S2 and rows become S3.
    im = ax.imshow(
        masked.T,
        origin="lower",
        cmap=cmap, norm=norm,
        extent=[i0 - CENTER[0], i0 + W - CENTER[0],
                j0 - CENTER[1], j0 + W - CENTER[1]],
        interpolation="nearest",
    )
    ax.axhline(0, color="0.7", lw=0.6, zorder=1)
    ax.axvline(0, color="0.7", lw=0.6, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlabel(f"S2 (coma) bin offset from center  (bin width = {(BOUNDS[0,1]-BOUNDS[0,0])/BINS[0]:.2e})")
    ax.set_ylabel(f"S3 (astigmatism) bin offset from center  (bin width = {(BOUNDS[1,1]-BOUNDS[1,0])/BINS[1]:.2e})")
    ax.set_title(f"Sixlet top-32 ε-net picks — combined archive map\n"
                 f"grid {BINS[0]}×{BINS[1]}; {n_cells} cells filled, "
                 f"{n_collisions} collisions; window = 2·R_max = {W} cells",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("mono fitness (symlog, clipped to [-1, 0])")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
