#!/usr/bin/env python
"""Sanity render of the 3 (anchor, parent) pairs + 1 incumbent for octuplet.

Layout: 2 rows x 4 cols.
  Row 1: anchor1, anchor2, anchor3, blank (or title)
  Row 2: parent1, parent2, parent3, incumbent

The user should see that parents visually resemble their anchors (same
basic glass arrangement) but the parent's spot is larger; the incumbent
should look obviously different/bad.
"""
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
OUT_PNG = ROOT / "figures" / "fig5_octuplet_parents_sanity.png"

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)


def draw_design(ax, lens, title, *, color="black"):
    try:
        lens.draw_lens_2d(ax=ax, fig=ax.figure, color=color)
    except Exception as e:
        ax.text(0.5, 0.5, f"draw err\n{type(e).__name__}",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.set_axis_off()


def main():
    payload = json.loads((DATA / "octuplet_parents_and_incumbent.json").read_text())
    anchors = payload["anchors_used"]
    designs = payload["designs"]
    parents = [d for d in designs if d["role"] == "parent"]
    parents.sort(key=lambda d: d["rank"])
    incumbent = next((d for d in designs if d["role"] == "incumbent"), None)

    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["octuplet"]["cfg"]
    )

    fig, axes = plt.subplots(2, 4, figsize=(15, 5.5))
    for col in range(4):
        ax_a = axes[0, col]
        ax_p = axes[1, col]
        if col < 3:
            # Top: anchor
            a = anchors[col]
            try:
                lens_a = make_geolens(prob_cfg, p_problem, lens_path,
                                       np.asarray(a["theta"]), template, solver)
                draw_design(
                    ax_a, lens_a,
                    f"ANCHOR #{col+1}\nrefined RMS = {a['refined_RMS_um']:.1f} um\nsource: {a['source_run']} id={a['design_id']}",
                    color="C0",
                )
            except Exception as e:
                ax_a.text(0.5, 0.5, f"anchor err: {e}", ha="center", va="center", transform=ax_a.transAxes)
                ax_a.set_axis_off()

            # Bottom: parent
            p = parents[col]
            try:
                lens_p = make_geolens(prob_cfg, p_problem, lens_path,
                                       np.asarray(p["theta"]), template, solver)
                d_anchor = p["anchor_info"]["unit_theta_distance"]
                draw_design(
                    ax_p, lens_p,
                    f"NEAR-PARENT #{col+1}\nrefined RMS = {p['refined_RMS_um']:.1f} um\nd to anchor #{col+1} = {d_anchor:.3f}\nsource: {p['source_run']} id={p['design_id']}",
                    color="#666",
                )
            except Exception as e:
                ax_p.text(0.5, 0.5, f"parent err: {e}", ha="center", va="center", transform=ax_p.transAxes)
                ax_p.set_axis_off()
        else:
            # Top-right: blank with title
            ax_a.text(0.5, 0.5, "INCUMBENT\n(obviously bad)",
                      ha="center", va="center", transform=ax_a.transAxes,
                      fontsize=11, weight="bold", color="C3")
            ax_a.set_axis_off()
            # Bottom-right: incumbent
            if incumbent is not None:
                try:
                    lens_i = make_geolens(prob_cfg, p_problem, lens_path,
                                          np.asarray(incumbent["theta"]), template, solver)
                    draw_design(
                        ax_p, lens_i,
                        f"INCUMBENT\nrefined RMS = {incumbent['refined_RMS_um']:.1f} um\nsource: {incumbent['source_run']} id={incumbent['design_id']}",
                        color="C3",
                    )
                except Exception as e:
                    ax_p.text(0.5, 0.5, f"incumbent err: {e}", ha="center", va="center", transform=ax_p.transAxes)
                    ax_p.set_axis_off()

    fig.suptitle(
        "Octuplet: 3 anchor refined designs (top) and their nearest 'near-parent' designs (bottom)\n"
        "+ one obviously-bad incumbent. Anchors RMS << Parents RMS, but they share a parameter neighborhood.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
