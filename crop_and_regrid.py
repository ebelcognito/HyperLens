"""Crop all 1920x1440 filled lens renders to a central 1700x850 window and
rebuild the 6x4 grid composites. Operates in place on the two render dirs.
"""
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

W_FULL, H_FULL = 1920, 1440
W_CROP, H_CROP = 1700, 850
LEFT = (W_FULL - W_CROP) // 2          # 110
TOP = (H_FULL - H_CROP) // 2           # 295
BOX = (LEFT, TOP, LEFT + W_CROP, TOP + H_CROP)  # (110, 295, 1810, 1145)

DIRS = [
    "/nas/longleaf/home/hisham26/Desktop/hyperlens/design_jsons_renders",
    "/nas/longleaf/home/hisham26/Desktop/hyperlens/mono587_top24_renders",
]


def crop_dir(d):
    pngs = sorted(p for p in glob.glob(os.path.join(d, "*_filled.png")))
    print(f"\n{d}: {len(pngs)} pngs")
    for p in pngs:
        im = Image.open(p)
        if im.size != (W_FULL, H_FULL):
            print(f"  skip (size {im.size}): {os.path.basename(p)}")
            continue
        im.crop(BOX).save(p)
        print(f"  cropped {os.path.basename(p)}")
    return pngs


def regrid(pngs, out_path, nrows=6, ncols=4):
    # tile aspect ratio = W_CROP / H_CROP = 2.0
    tile_w = 3.6
    tile_h = tile_w * H_CROP / W_CROP
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * tile_w, nrows * tile_h),
        gridspec_kw=dict(hspace=0.05, wspace=0.04),
    )
    for ax, p in zip(axes.flat, pngs):
        ax.imshow(Image.open(p))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"grid -> {out_path}")


def main():
    for d in DIRS:
        pngs = crop_dir(d)
        if len(pngs) >= 24:
            regrid(pngs[:24], os.path.join(d, "grid_6x4.png"))


if __name__ == "__main__":
    main()
