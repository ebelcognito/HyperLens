"""Re-render the 32 triplet pretty PNGs at 1700x1000, crop to central
1400x1000, and rebuild the 8x4 grid.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = "/nas/longleaf/home/hisham26/Desktop/hyperlens"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "DeepLens"))

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)
from scripts_throwaway.render_spotlight_sixlet_designs import (
    render_one, BLUE_FILL,
)

SRC_JSON = os.path.join(
    REPO, "scripts_throwaway/mono587_top32_triplet_eps030_sub10.json"
)
OUT_DIR = os.path.join(REPO, "mono587_top32_triplet_renders")
os.makedirs(OUT_DIR, exist_ok=True)

# Raw render: 1700x1000 (aspect 1.7) — gives more vertical breathing room
# than the prior 1700x850, then crop horizontally to 1400 wide.
FIG_W_IN, FIG_H_IN, DPI = 8.5, 5.0, 200
W_RAW, H_RAW = 1700, 1000
W_OUT, H_OUT = 1400, 1000
BOX = ((W_RAW - W_OUT)//2, 0, (W_RAW - W_OUT)//2 + W_OUT, H_OUT)


def render_design(theta, out_path, prob_cfg, p_problem, lens_path, template, solver):
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")
    ax.set_facecolor("white")
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    lens = make_geolens(prob_cfg, p_problem, lens_path,
                        np.asarray(theta, dtype=np.float64), template, solver)
    render_one(ax, lens, fill_color=BLUE_FILL)
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    im = Image.open(out_path)
    if im.size == (W_RAW, H_RAW):
        im.crop(BOX).save(out_path)


def main():
    bundle = json.load(open(SRC_JSON))
    designs = bundle["designs"][:32]
    assert len(designs) == 32

    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["triplet"]["cfg"])

    renders = []
    for d in designs:
        rank = int(d["rank"])
        mono_rms = float(d["mono_rms_um"])
        stem = f"rank_{rank:02d}_mono_{mono_rms:06.2f}um"
        out_path = os.path.join(OUT_DIR, f"{stem}_filled.png")
        print(f"  {stem}", flush=True)
        try:
            render_design(d["theta"], out_path,
                          prob_cfg, p_problem, lens_path, template, solver)
        except Exception as e:
            print(f"    FAIL: {e}"); continue
        renders.append({"png": out_path, "rank": rank})

    nrows, ncols = 8, 4
    tile_w = 3.6
    tile_h = tile_w * H_OUT / W_OUT
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * tile_w, nrows * tile_h),
        gridspec_kw=dict(hspace=0.05, wspace=0.04),
    )
    for ax, it in zip(axes.flat, renders):
        ax.imshow(Image.open(it["png"]))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    out = os.path.join(OUT_DIR, "grid_8x4.png")
    fig.savefig(out, dpi=200, bbox_inches="tight",
                pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"\ngrid -> {out}")


if __name__ == "__main__":
    main()
