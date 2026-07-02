"""Combined raw-state archive map of the 32 mono-λ sixlet picks.

For each pick, resolve pool_idx → parent run's archive_cells.cells[i], pull the
raw (S2, S3) measures and raw fitness, project onto a shared 2048x2048 grid
(matches the finest parent archive's resolution), and render as in the post-
refinement plot but using raw values.
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
OUT_PNG = ROOT / "mono587_top32_sixlet_renders/archive_map_top32_raw.png"

BINS = (2048, 2048)
BOUNDS = np.array([[-0.008, 0.008], [-0.012, 0.012]])
CENTER = (BINS[0] // 2, BINS[1] // 2)

_cache = {}
def archive_cells(run):
    if run not in _cache:
        p = ROOT / f"results/lensmega/{run}/archive_cells.json"
        _cache[run] = json.load(open(p)) if p.exists() else None
    return _cache[run]


def fetch_raw(run, idx):
    """(S2, S3, raw_fitness)."""
    fps = sorted(glob.glob(str(ROOT / f"results/lensmega/{run}/refined_top*.json")))
    if not fps: return None
    d = json.load(open(fps[0]))
    if idx >= len(d["designs"]): return None
    des = d["designs"][idx]
    pool_idx = des.get("pool_idx")
    ac = archive_cells(run)
    if ac is None or pool_idx is None or pool_idx >= len(ac["cells"]):
        return None
    cell = ac["cells"][pool_idx]
    return cell["measures"][0], cell["measures"][1], float(cell["fitness"])


def bin_index(val, lo, hi, n):
    return int(np.clip(np.floor((val - lo) / (hi - lo) * n), 0, n - 1))


def main():
    bundle = json.load(open(TOP32))
    designs = bundle["designs"][:32]

    cell_fit = {}
    cell_ranks = {}
    out_of_bounds = []
    skipped = []
    for d in designs:
        rank = int(d["rank"])
        r = fetch_raw(d["run"], int(d["idx"]))
        if r is None:
            skipped.append(rank); continue
        s2, s3, fit = r
        oob = (s2 < BOUNDS[0,0] or s2 > BOUNDS[0,1]
               or s3 < BOUNDS[1,0] or s3 > BOUNDS[1,1])
        if oob:
            out_of_bounds.append((rank, s2, s3, fit))
        i = bin_index(s2, *BOUNDS[0], BINS[0])
        j = bin_index(s3, *BOUNDS[1], BINS[1])
        if (i, j) not in cell_fit or fit > cell_fit[(i, j)]:
            cell_fit[(i, j)] = fit
        cell_ranks.setdefault((i, j), []).append(rank)

    n_cells = len(cell_fit)
    n_coll = sum(1 for v in cell_ranks.values() if len(v) > 1)
    print(f"Placed {sum(len(v) for v in cell_ranks.values())} designs in {n_cells} cells; "
          f"{n_coll} collision(s).")
    if skipped:
        print(f"Skipped (no pool_idx → archive cell): {skipped}")
    if out_of_bounds:
        print(f"Out-of-bounds (clamped to edge):")
        for r, s2, s3, fit in out_of_bounds:
            print(f"  rank {r}: S2={s2:+.3e}  S3={s3:+.3e}  fit={fit:+.4f}")
    for cell, ranks in sorted(cell_ranks.items()):
        if len(ranks) > 1:
            print(f"  cell {cell}: ranks {ranks}")

    # Radial extent from center
    R = 0.0
    for (i, j) in cell_fit:
        r = np.hypot(i - CENTER[0], j - CENTER[1])
        if r > R: R = r
    R_ceil = int(np.ceil(R)) + 1
    W = 2 * R_ceil
    print(f"R_max = {R:.2f} cells, window = {W} cells")

    img = np.full((W, W), np.nan)
    i0, j0 = CENTER[0] - R_ceil, CENTER[1] - R_ceil
    for (i, j), f in cell_fit.items():
        img[i - i0, j - j0] = f

    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="white")
    norm = SymLogNorm(linthresh=1e-3, linscale=1.0, vmin=-1.0, vmax=0.0, base=10)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")
    im = ax.imshow(
        np.ma.masked_invalid(img).T,
        origin="lower", cmap=cmap, norm=norm,
        extent=[i0 - CENTER[0], i0 + W - CENTER[0],
                j0 - CENTER[1], j0 + W - CENTER[1]],
        interpolation="nearest",
    )
    ax.axhline(0, color="0.7", lw=0.6); ax.axvline(0, color="0.7", lw=0.6)
    ax.set_aspect("equal")
    bw_s2 = (BOUNDS[0,1] - BOUNDS[0,0]) / BINS[0]
    bw_s3 = (BOUNDS[1,1] - BOUNDS[1,0]) / BINS[1]
    ax.set_xlabel(f"S2 (coma) bin offset (bin width = {bw_s2:.2e})")
    ax.set_ylabel(f"S3 (astigmatism) bin offset (bin width = {bw_s3:.2e})")
    ax.set_title(f"Sixlet top-32 raw archive map  ({BINS[0]}×{BINS[1]} grid)\n"
                 f"{n_cells} cells, {n_coll} collision(s); {len(out_of_bounds)} oob "
                 f"(clamped); window = 2·R_max = {W}", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("raw fitness (symlog, clipped to [-1, 0])")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
