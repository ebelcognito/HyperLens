"""Rebuild the 8x4 sixlet grid with a numbered circle (the design rank) on
the top-left corner of each lens render. Reads the already-cropped per-design
PNGs from mono587_top32_sixlet_renders/.
"""
import os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DIR = "/nas/longleaf/home/hisham26/Desktop/hyperlens/mono587_top32_sixlet_renders"


def main():
    pngs = sorted(p for p in glob.glob(os.path.join(DIR, "rank_*_filled.png")))
    assert len(pngs) >= 32, f"need 32 picks, got {len(pngs)}"
    pngs = pngs[:32]

    # Each PNG is named rank_NN_mono_*. Pull rank from filename for the circle.
    def rank_of(p):
        name = os.path.basename(p)
        # rank_NN_...
        return int(name.split("_")[1])

    sample = Image.open(pngs[0])
    W, H = sample.size  # ~1700×850 for sixlet
    aspect = H / W

    nrows, ncols = 8, 4
    tile_w_in = 3.6
    tile_h_in = tile_w_in * aspect

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * tile_w_in, nrows * tile_h_in),
        gridspec_kw=dict(hspace=0.05, wspace=0.04),
    )

    for ax, p in zip(axes.flat, pngs):
        ax.imshow(Image.open(p))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        rk = rank_of(p)
        # Small italic-serif numeral in a soft-bordered circle (Mathematica style).
        ax.annotate(
            str(rk),
            xy=(0.040, 0.905), xycoords="axes fraction",
            ha="center", va="center",
            fontsize=9, color="#0B2545",
            family="serif", style="italic",
            bbox=dict(
                boxstyle="circle,pad=0.32",
                facecolor="white",
                edgecolor="#0B2545", linewidth=0.7,
                alpha=0.92,
            ),
            zorder=10,
        )

    out = os.path.join(DIR, "grid_8x4_indexed.png")
    fig.savefig(out, dpi=200, bbox_inches="tight",
                pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
