#!/usr/bin/env python
"""Hull comparison at refined RMS < 30 µm — HL CHAMPION & BLIND_COMBO vs SAGA.

Three panels:
  - HL CHAMPION (patience=20)  vs SAGA at RMS<30 hulls
  - HL BLIND_COMBO (patience=30) vs SAGA at RMS<30 hulls (BLIND_COMBO's "deep basin" alpha)
  - both HL pools combined vs SAGA at RMS<30 hulls (showcase of the alpha trade)
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
from matplotlib.patches import FancyBboxPatch
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
OUT = ROOT / "figures/octuplet_hull_rms_lt30_HL_vs_SAGA.png"

CHAMP_RUN = ROOT / "results/lensmega/octuplet_REFSWEEP_CHAMPION_seed42_20260509_145427"
BLIND_RUN = ROOT / "results/lensmega/octuplet_REFSWEEP_BLIND_COMBO_seed42_20260509_074619"
SAGA_RECOMP = ROOT / "results/figure_archive_demo_octuplet_CHAMPION/saga_descriptor_recompute.json"

RMS_GATE_UM = 30.0


def load_hl_pool(run_dir, json_name):
    refined = json.load(open(run_dir/json_name))["designs"]
    fit_to_cell = {}
    ac_json = run_dir/"archive_cells.json"
    if ac_json.exists():
        ac = json.load(open(ac_json))
        fit_to_cell = {round(c["fitness"], 8): c["measures"] for c in ac["cells"]}
    else:
        # Fallback: load archive.pkl and walk cells.
        import pickle
        sys.path.insert(0, str(ROOT))
        ar = pickle.load(open(run_dir/"archive.pkl", "rb"))
        # Archive object has _cells: dict[bin_idx -> Cell with .measures and .fitness]
        cells = getattr(ar, "_cells", None) or getattr(ar, "cells", None)
        if isinstance(cells, dict):
            iterable = cells.values()
        else:
            iterable = cells
        for c in iterable:
            m = getattr(c, "measures", None)
            f = getattr(c, "fitness", None)
            if m is None or f is None: continue
            fit_to_cell[round(float(f), 8)] = [float(m[0]), float(m[1])]
    pts = []
    for d in refined:
        if d.get("rms_um", 1e9) >= RMS_GATE_UM: continue
        m = fit_to_cell.get(round(d["raw_fitness"], 8))
        if m is None: continue
        pts.append([m[0], m[1], d["rms_um"]])
    return np.array(pts) if pts else np.empty((0, 3))


def load_saga_pool():
    recomp = json.load(open(SAGA_RECOMP))
    pts = [[d["S2"], d["S3"], d["rms_um"]] for d in recomp["sg_refined"] if d["rms_um"] < RMS_GATE_UM]
    return np.array(pts) if pts else np.empty((0, 3))


def hull_or_none(pts):
    if len(pts) < 3:
        return None
    return ConvexHull(pts[:, :2])


def plot_panel(ax, hl_pts, sg_pts, hl_label, sg_label, title):
    """Draw a single hull comparison panel."""
    # Hulls
    hl_hull = hull_or_none(hl_pts)
    sg_hull = hull_or_none(sg_pts)

    # HL filled hull
    if hl_hull:
        v = hl_pts[hl_hull.vertices, :2]
        ax.fill(v[:, 0]*1e3, v[:, 1]*1e3, color="#4A87B8", alpha=0.18, zorder=1)
        ax.plot(np.append(v[:, 0], v[0, 0])*1e3, np.append(v[:, 1], v[0, 1])*1e3,
                color="#1A4A7A", linewidth=2.4, zorder=3)

    # SAGA filled hull
    if sg_hull:
        v = sg_pts[sg_hull.vertices, :2]
        ax.fill(v[:, 0]*1e3, v[:, 1]*1e3, color="#C8606E", alpha=0.18, zorder=2)
        ax.plot(np.append(v[:, 0], v[0, 0])*1e3, np.append(v[:, 1], v[0, 1])*1e3,
                color="#9A2A3A", linewidth=2.4, zorder=4)

    # HL points
    if len(hl_pts):
        ax.scatter(hl_pts[:, 0]*1e3, hl_pts[:, 1]*1e3,
                   s=22, c="#1A4A7A", edgecolors="white", linewidth=0.5,
                   alpha=0.9, zorder=5, label=f"{hl_label}  (n={len(hl_pts)}, min RMS={hl_pts[:,2].min():.2f} µm)")

    # SAGA points — distinguish inside-vs-outside HL hull
    if len(sg_pts):
        if hl_hull:
            hl_path = MplPath(hl_pts[hl_hull.vertices, :2])
            inside = hl_path.contains_points(sg_pts[:, :2])
        else:
            inside = np.zeros(len(sg_pts), dtype=bool)
        ax.scatter(sg_pts[inside, 0]*1e3, sg_pts[inside, 1]*1e3,
                   s=22, c="#9A2A3A", edgecolors="white", linewidth=0.5,
                   alpha=0.9, zorder=5,
                   label=f"{sg_label} ∈ HL hull  (n={inside.sum()}, min RMS={sg_pts[inside, 2].min():.2f} µm)" if inside.any() else None)
        if (~inside).any():
            ax.scatter(sg_pts[~inside, 0]*1e3, sg_pts[~inside, 1]*1e3,
                       s=44, marker="X", c="#9A2A3A", edgecolors="black", linewidth=0.8,
                       zorder=6,
                       label=f"{sg_label} ∉ HL hull  (n={(~inside).sum()}, min RMS={sg_pts[~inside, 2].min():.2f} µm)")

    ax.set_xlabel("Coma  S₂  (×10⁻³)", fontsize=11, family="serif")
    ax.set_ylabel("Astigmatism  S₃  (×10⁻³)", fontsize=11, family="serif")
    ax.set_title(title, fontsize=13, family="serif", style="italic", pad=8)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper right", frameon=True, fancybox=True, framealpha=0.95)
    # caption with hull stats
    if hl_hull and sg_hull:
        ratio = hl_hull.volume / sg_hull.volume
        n_inside = MplPath(hl_pts[hl_hull.vertices, :2]).contains_points(sg_pts[:, :2]).sum()
        msg = f"HL hull / SAGA hull area = {ratio:.2f}×   ·   SAGA inside HL hull: {n_inside}/{len(sg_pts)}"
        ax.text(0.5, -0.16, msg, transform=ax.transAxes,
                ha="center", va="top", fontsize=10, family="serif",
                style="italic", color="#3A3A3A")


def main():
    sg = load_saga_pool()
    print(f"SAGA refined <{RMS_GATE_UM} µm: n={len(sg)}, min RMS = {sg[:, 2].min():.2f}")
    champ = load_hl_pool(CHAMP_RUN, "refined_top500_seed42_CHAMPION.json")
    print(f"HL CHAMPION refined <{RMS_GATE_UM} µm: n={len(champ)}, min RMS = {champ[:, 2].min():.2f}")
    blind = load_hl_pool(BLIND_RUN, "refined_top500_seed42_BLIND_COMBO.json")
    print(f"HL BLIND_COMBO refined <{RMS_GATE_UM} µm: n={len(blind)}, min RMS = {blind[:, 2].min():.2f}")

    combined = np.vstack([champ, blind]) if len(blind) else champ
    print(f"HL CHAMPION ∪ BLIND_COMBO refined <{RMS_GATE_UM} µm: n={len(combined)}, min RMS = {combined[:, 2].min():.2f}")

    fig = plt.figure(figsize=(20, 8.0), facecolor="white")
    gs = fig.add_gridspec(1, 3, left=0.04, right=0.985, top=0.83, bottom=0.13, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_panel(ax1, champ, sg, "HL CHAMPION (patience=20)", "SAGA", "HL CHAMPION  vs  SA+GA")

    ax2 = fig.add_subplot(gs[0, 1])
    plot_panel(ax2, blind, sg, "HL BLIND_COMBO (patience=30)", "SAGA", "HL BLIND_COMBO  vs  SA+GA")

    ax3 = fig.add_subplot(gs[0, 2])
    plot_panel(ax3, combined, sg,  "HL  (CHAMPION ∪ BLIND_COMBO)", "SAGA", "HL combined  vs  SA+GA")

    fig.text(0.5, 0.95,
             "Convex-hull comparison — designs with refined RMS < 30 µm",
             ha="center", va="center", fontsize=20, family="serif",
             style="italic", color="#1A1A1A")
    fig.text(0.5, 0.91,
             "X = SA+GA design outside HL hull  (where SA+GA's GA-driven local search beat HL's CMA-ES at this quality threshold)",
             ha="center", va="center", fontsize=12, family="serif",
             style="italic", color="#3A3A3A")

    box = FancyBboxPatch(
        (0.008, 0.008), 0.984, 0.984,
        boxstyle="round,pad=0.0,rounding_size=0.008",
        transform=fig.transFigure,
        edgecolor="#1A1A1A", facecolor="none", linewidth=1.4,
        zorder=10, clip_on=False,
    )
    fig.patches.append(box)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
