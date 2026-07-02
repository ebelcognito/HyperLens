#!/usr/bin/env python
"""Compose per-design 'card' PNGs: layout (top) + elegant spot (bottom),
matched widths, white pad between. One card per design, 8 cards total
(4 sixlet + 4 octuplet). Drop the folder into Overleaf and \\includegraphics
each card.
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
SRC  = ROOT / "figures" / "zemax_top8_diverse_fnum5"
OUT  = ROOT / "figures" / "topology_cards"
PAD_PX = 18                       # white gap between layout and spot

OUT.mkdir(parents=True, exist_ok=True)

def compose(layout_path: Path, spot_path: Path, out_path: Path):
    layout = Image.open(layout_path).convert("RGB")
    spot   = Image.open(spot_path).convert("RGB")
    target_w = max(layout.width, spot.width)
    def fit(im):
        if im.width == target_w:
            return im
        h = round(im.height * target_w / im.width)
        return im.resize((target_w, h), Image.LANCZOS)
    layout = fit(layout)
    spot   = fit(spot)
    H = layout.height + PAD_PX + spot.height
    canvas = Image.new("RGB", (target_w, H), "white")
    canvas.paste(layout, (0, 0))
    canvas.paste(spot,   (0, layout.height + PAD_PX))
    canvas.save(out_path, optimize=True)
    return out_path

def main():
    for sys in ("sixlet", "octuplet"):
        for rank in (1, 2, 3, 4):
            lp = SRC / f"{sys}_{rank}_layout.png"
            sp = SRC / f"{sys}_{rank}_spot.png"
            op = OUT / f"{sys}_{rank}.png"
            compose(lp, sp, op)
            print(f"  -> {op.relative_to(ROOT)}  ({op.stat().st_size//1024} KB)",
                  flush=True)


if __name__ == "__main__":
    main()
