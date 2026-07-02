"""Crop triplet pretty-render PNGs (1700x850) to central 1400x850 in place,
then rebuild the 8x4 grid.
"""
import os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DIR = "/nas/longleaf/home/hisham26/Desktop/hyperlens/mono587_top32_triplet_renders"
W_IN, H_IN = 1700, 850
W_OUT, H_OUT = 1400, 850
BOX = ((W_IN - W_OUT)//2, (H_IN - H_OUT)//2,
       (W_IN - W_OUT)//2 + W_OUT,
       (H_IN - H_OUT)//2 + H_OUT)


def main():
    pngs = sorted(glob.glob(os.path.join(DIR, "*_filled.png")))
    for p in pngs:
        im = Image.open(p)
        if im.size != (W_IN, H_IN):
            print(f"  skip (size {im.size}): {os.path.basename(p)}")
            continue
        im.crop(BOX).save(p)
        print(f"  cropped {os.path.basename(p)}")

    nrows, ncols = 8, 4
    tile_w = 3.6
    tile_h = tile_w * H_OUT / W_OUT
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * tile_w, nrows * tile_h),
        gridspec_kw=dict(hspace=0.05, wspace=0.04),
    )
    for ax, p in zip(axes.flat, pngs[:32]):
        ax.imshow(Image.open(p))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    out = os.path.join(DIR, "grid_8x4.png")
    fig.savefig(out, dpi=200, bbox_inches="tight",
                pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"grid -> {out}")


if __name__ == "__main__":
    main()
