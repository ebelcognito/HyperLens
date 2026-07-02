#!/usr/bin/env python
"""Quick sanity render of the 3 diverse picks per system."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
DATA = ROOT / "figures" / "fig5_data"
OUT_PNG = ROOT / "figures" / "fig5_three_diverse_sanity.png"

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)


def main():
    picks = json.loads((DATA / "three_diverse_below_thr.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for row, sys_name in enumerate(("sixlet", "octuplet")):
        template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(SYSTEMS[sys_name]["cfg"])
        for col, design in enumerate(picks[sys_name]["designs"]):
            ax = axes[row, col]
            theta = np.asarray(design["theta"], dtype=np.float64)
            try:
                lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
                lens.draw_lens_2d(ax=ax, fig=ax.figure, color="black")
            except Exception as e:
                ax.text(0.5, 0.5, f"build err\n{type(e).__name__}",
                        ha="center", va="center", transform=ax.transAxes)
            ax.set_title(
                f"{sys_name} #{col+1}  id={design['design_id']}  "
                f"RMS={design['refined_RMS_um']:.1f} um",
                fontsize=8,
            )
            ax.set_aspect("equal")
            ax.set_axis_off()

    fig.suptitle(
        "Three diverse refined designs per system (HL pool)\n"
        "sixlet: RMS<=20um eps=0.25  |  octuplet: RMS<=30um eps=0.25",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
