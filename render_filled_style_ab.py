"""A/B prototype for the new filled-glass + ray-bundle render style.

Loads the same elite as render_extra_elite.py (cooke_triplet_materials,
elite_0012_11988k, F/8 refined) and produces two PNGs side by side:

  paper_figures/method_overview_renders/style_ab_line.png    (current style)
  paper_figures/method_overview_renders/style_ab_filled.png  (new filled style)

Uses GeoLens.draw_layout(style="line"|"filled") from the patched
DeepLens vis.py — no lensmega-side reimplementation.
"""
import json
import os
import sys

sys.path.insert(0, "/home/hisham26/Desktop/RL")
sys.path.insert(0, "/home/hisham26/Desktop/RL/DeepLens")

from optical_rl.qd.lensmega.config.loader import load_config
from render_method_overview import (
    CFG, OUT, FNUM_TARGET, EFL_TARGET, build_lens_from_theta,
)

ELITE_JSON = (
    "/home/hisham26/Desktop/RL/results/lensmega/"
    "cooke_triplet_materials_20260414_173935/"
    "elite_designs_F8p0/elite_0012_11988k.json"
)


def main():
    cfg = load_config(CFG)
    with open(ELITE_JSON) as f:
        d = json.load(f)
    ref = d.get("refined_F8") or d.get("refined")
    theta = ref["theta"]
    rms_um = float(ref["rms_mm"]) * 1000.0
    print(f"Source: {os.path.basename(ELITE_JSON)}  refined RMS: {rms_um:.2f} um")

    line_path = os.path.join(OUT, "style_ab_line.png")
    filled_path = os.path.join(OUT, "style_ab_filled.png")

    lens_a = build_lens_from_theta(
        theta, cfg, device="cpu",
        target_fnum=FNUM_TARGET, target_efl=EFL_TARGET,
    )
    lens_a.draw_layout(
        filename=line_path,
        depth=float("inf"),
        style="line",
        lens_title="line style (current)",
    )
    print(f"OK -> {line_path}")

    lens_b = build_lens_from_theta(
        theta, cfg, device="cpu",
        target_fnum=FNUM_TARGET, target_efl=EFL_TARGET,
    )
    lens_b.draw_layout(
        filename=filled_path,
        depth=float("inf"),
        style="filled",
        lens_title="filled style (new)",
        glass_facecolor="#BFE3F2",
        glass_edgecolor="black",
        glass_lw=1.2,
        bundle_alpha=0.35,
    )
    print(f"OK -> {filled_path}")


if __name__ == "__main__":
    main()
