"""Raw-archive map of the 32 mono-λ sixlet picks, rendered on a SMALL square
grid sized to fit the in-bounds designs.

Strategy:
  - Exclude designs whose raw (S2, S3) lies outside the archive bounds
    (clamping them onto a small grid would just pile them up at corners
    and dominate R_max). These are flagged in the console.
  - Compute R_max = max sqrt(S2^2 + S3^2) over surviving designs in continuous
    units. Pick bin width = 2*R_max / N so the N×N window exactly covers a
    centered square of side 2*R_max.
  - Aggregate cells by best fitness, count collisions, annotate counts on
    cells where multiple designs land.
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
OUT_PNG = ROOT / "mono587_top32_sixlet_renders/archive_map_top32_raw_small.png"

N = 32   # grid side
BOUNDS = np.array([[-0.008, 0.008], [-0.012, 0.012]])  # archive nominal bounds

_cache = {}
def archive_cells(run):
    if run not in _cache:
        p = ROOT / f"results/lensmega/{run}/archive_cells.json"
        _cache[run] = json.load(open(p)) if p.exists() else None
    return _cache[run]

def fetch_raw(run, idx):
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


def main():
    bundle = json.load(open(TOP32))
    designs = bundle["designs"][:32]

    surv, skipped, oob = [], [], []
    for d in designs:
        rank = int(d["rank"])
        r = fetch_raw(d["run"], int(d["idx"]))
        if r is None:
            skipped.append(rank); continue
        s2, s3, fit = r
        if (s2 < BOUNDS[0,0] or s2 > BOUNDS[0,1]
            or s3 < BOUNDS[1,0] or s3 > BOUNDS[1,1]):
            oob.append((rank, s2, s3, fit))
        else:
            surv.append((rank, s2, s3, fit))
    print(f"Resolved {len(surv) + len(oob)}/32. "
          f"In-bounds: {len(surv)}, OOB excluded: {len(oob)}, "
          f"unresolvable: {len(skipped)}")
    if skipped: print(f"  unresolvable ranks: {skipped}")
    if oob:
        print(f"  OOB excluded:")
        for r, s2, s3, f in oob:
            print(f"    rank {r}: S2={s2:+.3e} S3={s3:+.3e} fit={f:+.4f}")

    # Radial extent in continuous units
    R = max(float(np.hypot(s2, s3)) for _, s2, s3, _ in surv)
    bin_w = 2 * R / N
    print(f"R_max (in-bounds) = {R:.3e}  => bin width = {bin_w:.3e} for N={N}")

    # Bin: center cell of the N×N grid is at (N/2, N/2). cell index i = floor((s2 + R) / bin_w)
    def to_bin(s2, s3):
        i = int(np.clip(np.floor((s2 + R) / bin_w), 0, N - 1))
        j = int(np.clip(np.floor((s3 + R) / bin_w), 0, N - 1))
        return i, j

    cell_fit, cell_ranks = {}, {}
    for rank, s2, s3, fit in surv:
        c = to_bin(s2, s3)
        if c not in cell_fit or fit > cell_fit[c]:
            cell_fit[c] = fit
        cell_ranks.setdefault(c, []).append(rank)
    n_coll = sum(1 for v in cell_ranks.values() if len(v) > 1)
    print(f"Placed {len(surv)} designs in {len(cell_fit)} cells, {n_coll} collisions.")
    for c, ranks in sorted(cell_ranks.items()):
        if len(ranks) > 1:
            print(f"  cell {c}: ranks {ranks}")

    # Image
    img = np.full((N, N), np.nan)
    for (i, j), f in cell_fit.items():
        img[i, j] = f

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7.0, 7.0), facecolor="white")
    norm = SymLogNorm(linthresh=0.1, linscale=1.0, vmin=-100.0, vmax=0.0, base=10)
    cmap = plt.cm.magma.copy()
    cmap.set_bad("#F5F5F7")    # soft off-white for empty cells
    im = ax.imshow(
        np.ma.masked_invalid(img).T,
        origin="lower", cmap=cmap, norm=norm,
        extent=[-R, R, -R, R], interpolation="nearest",
    )
    # Soft origin lines
    ax.axhline(0, color="#999", lw=0.7, alpha=0.5)
    ax.axvline(0, color="#999", lw=0.7, alpha=0.5)
    # Strip the box; keep only bottom & left spines as faint ticks
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color("#666"); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors="#444", labelsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("S2  (coma)", fontsize=11, color="#222", labelpad=8)
    ax.set_ylabel("S3  (astigmatism)", fontsize=11, color="#222", labelpad=8)
    ax.set_title(f"Sixlet top-32 raw archive  ·  {N}×{N} grid",
                 fontsize=12, color="#222", pad=14)
    cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.025)
    cbar.set_label("raw fitness", fontsize=10, color="#222", labelpad=8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors="#444", labelsize=9)
    fig.savefig(OUT_PNG, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
