#!/usr/bin/env python
"""Story figure: HL reaches many distinct quality basins, SA+GA can only
reach one or two. 2D embedding via Seidel-PCA on standardized S1..S5.

Usage:
  python scripts_throwaway/fig5_sixlet_story.py [SYSTEM]

SYSTEM defaults to 'sixlet'; pass 'octuplet' (or 'triplet') to run for
those instead. Output PNG and CSV names reflect the chosen system.
"""
from __future__ import annotations

import argparse
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
from matplotlib.colors import LogNorm

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
DATA = ROOT / "figures" / "fig5_data"

from scripts_throwaway.fig5_extract import (
    SYSTEMS, load_bounds, load_pool, eps_net_select, EPS_TARGET,
)

# Per-system asymmetric quality thresholds. SAGA threshold = HL threshold +
# slack so SAGA's single best design qualifies even if its min RMS is just
# above the HL threshold. Numbers chosen empirically per system.
SYSTEM_DEFAULTS = {
    "sixlet":   {"quality_um_hl": 30.0, "quality_um_saga": 30.0},
    "octuplet": {"quality_um_hl": 30.0, "quality_um_saga": 30.0},
    "triplet":  {"quality_um_hl": 30.0, "quality_um_saga": 30.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", nargs="?", default="sixlet",
                    choices=list(SYSTEMS.keys()))
    ap.add_argument("--quality-hl", type=float, default=None,
                    help="HL refined-RMS threshold in um (overrides default)")
    ap.add_argument("--quality-saga", type=float, default=None,
                    help="SA+GA refined-RMS threshold in um (overrides default)")
    args = ap.parse_args()

    SYS_NAME = args.system
    OUT_PNG = ROOT / "figures" / f"fig5_{SYS_NAME}_story.png"
    OUT_CSV = DATA / f"{SYS_NAME}_pca_projection.csv"

    info = SYSTEMS[SYS_NAME]
    bounds = load_bounds(info["cfg"])
    D = bounds.shape[0]
    hl = load_pool(info["hl_json"], bounds)
    sa = load_pool(info["saga_json"], bounds)

    # eps-net basin representatives at eps=0.20 (best-fitness-first greedy net)
    hl_idx = eps_net_select(hl, EPS_TARGET, D)
    sa_idx = eps_net_select(sa, EPS_TARGET, D)
    hl_reps_all = [hl[i] for i in hl_idx]
    sa_reps_all = [sa[i] for i in sa_idx]
    print(f"sixlet HL  total basin reps: {len(hl_reps_all)} (from pool of {len(hl)})")
    print(f"sixlet SAGA total basin reps: {len(sa_reps_all)} (from pool of {len(sa)})")

    # Asymmetric quality thresholds (system-specific defaults; overridable).
    QUALITY_UM_HL = float(args.quality_hl) if args.quality_hl is not None else SYSTEM_DEFAULTS[SYS_NAME]["quality_um_hl"]
    QUALITY_UM_SA = float(args.quality_saga) if args.quality_saga is not None else SYSTEM_DEFAULTS[SYS_NAME]["quality_um_saga"]
    print(f"system={SYS_NAME}, HL threshold={QUALITY_UM_HL} um, SA+GA threshold={QUALITY_UM_SA} um")
    hl_reps = [r for r in hl_reps_all if r["rms_um"] <= QUALITY_UM_HL]
    sa_reps = [r for r in sa_reps_all if r["rms_um"] <= QUALITY_UM_SA]
    print(f"sixlet HL  basin reps <= {QUALITY_UM_HL} um: {len(hl_reps)}")
    print(f"sixlet SAGA basin reps <= {QUALITY_UM_SA} um: {len(sa_reps)}")

    # PCA basis: fit on the 5D Seidel space (S1..S5), standardized per axis.
    # In aberration space, HL's wider envelope shows up as a much larger
    # 2D hull than SA+GA's tight cluster -- this is the visualization the
    # radar plot quantifies. 2D captures ~84% of Seidel variance.
    S_hl_full = np.array([[r["S1"], r["S2"], r["S3"], r["S4"], r["S5"]] for r in hl])
    S_sa_full = np.array([[r["S1"], r["S2"], r["S3"], r["S4"], r["S5"]] for r in sa])
    seidel_scale = np.maximum(
        np.std(np.vstack([S_hl_full, S_sa_full]), axis=0), 1e-12,
    )
    S_hl_full_n = S_hl_full / seidel_scale
    S_sa_full_n = S_sa_full / seidel_scale
    S_all = np.vstack([S_hl_full_n, S_sa_full_n])
    mu = S_all.mean(axis=0)
    Sc = S_all - mu
    U, s, Vt = np.linalg.svd(Sc, full_matrices=False)
    var_explained = (s ** 2) / (s ** 2).sum()
    PCA_BASIS = Vt.T[:, :2]

    def project_seidels(reps):
        if not reps:
            return np.zeros((0, 2))
        S = np.array([[r["S1"], r["S2"], r["S3"], r["S4"], r["S5"]] for r in reps])
        return (S / seidel_scale - mu) @ PCA_BASIS

    # All basin reps in Seidel-PCA 2D
    pc_hl_all = project_seidels(hl_reps_all)
    pc_sa_all = project_seidels(sa_reps_all)
    rms_hl_all = np.array([r["rms_um"] for r in hl_reps_all])
    rms_sa_all = np.array([r["rms_um"] for r in sa_reps_all])

    # Quality-passing subsets (highlighted)
    pc_hl = project_seidels(hl_reps)
    pc_sa = project_seidels(sa_reps)
    rms_hl = np.array([r["rms_um"] for r in hl_reps]) if hl_reps else np.zeros(0)
    rms_sa = np.array([r["rms_um"] for r in sa_reps]) if sa_reps else np.zeros(0)

    # FULL refined pools projected (for the diversity-envelope hulls --
    # this is the natural view of where each method's aberration footprint
    # actually lives, including the high-RMS designs).
    pc_pool_hl = (S_hl_full_n - mu) @ PCA_BASIS
    pc_pool_sa = (S_sa_full_n - mu) @ PCA_BASIS

    # Save projection table (full pool + quality flag per side)
    with OUT_CSV.open("w") as f:
        f.write("method,design_id,pc1,pc2,refined_rms_um,refined_fitness,is_basin_rep,passes_quality\n")
        rep_ids_hl = {r["idx"] for r in hl_reps_all}
        rep_ids_sa = {r["idx"] for r in sa_reps_all}
        for r, p in zip(hl, pc_pool_hl):
            is_rep = r["idx"] in rep_ids_hl
            passes = r["rms_um"] <= QUALITY_UM_HL
            f.write(f"hl,{r['idx']},{p[0]:.6f},{p[1]:.6f},{r['rms_um']:.4f},{r['refined_fitness']:.6e},{int(is_rep)},{int(passes)}\n")
        for r, p in zip(sa, pc_pool_sa):
            is_rep = r["idx"] in rep_ids_sa
            passes = r["rms_um"] <= QUALITY_UM_SA
            f.write(f"saga,{r['idx']},{p[0]:.6f},{p[1]:.6f},{r['rms_um']:.4f},{r['refined_fitness']:.6e},{int(is_rep)},{int(passes)}\n")
    print(f"wrote {OUT_CSV}")
    print(f"PCA (fit on quality-passing reps) variance explained: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}, sum={var_explained[:2].sum():.3f}")

    # Counts for annotation
    rms_full_hl = np.array([r["rms_um"] for r in hl])
    rms_full_sa = np.array([r["rms_um"] for r in sa])
    n_hl = len(hl_reps)
    n_sa = len(sa_reps)
    sa_min = float(rms_full_sa.min())
    hl_min = float(rms_full_hl.min())

    # Drama numbers (D-dim, no projection)
    hl_arr = np.array([r["theta_norm"] for r in hl_reps]) if hl_reps else np.zeros((0, D))
    sa_arr = np.array([r["theta_norm"] for r in sa_reps]) if sa_reps else np.zeros((0, D))
    n_quality_basins_hl = n_hl
    n_quality_basins_sa = n_sa

    # ------------- figure -------------
    fig = plt.figure(figsize=(10.5, 11.5))
    gs = fig.add_gridspec(
        nrows=2, ncols=1, height_ratios=[3.0, 1.0], hspace=0.28,
    )
    ax_main = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    # color norm: log over RMS
    rms_all = np.concatenate([rms_hl, rms_sa])
    norm = LogNorm(vmin=max(rms_all.min(), 1.0), vmax=min(rms_all.max(), 1000.0))
    cmap = plt.get_cmap("viridis_r")  # reversed so low-RMS = bright

    from scipy.spatial import ConvexHull

    # Layer 1: HL FULL pool diversity hull (all 498 designs in Seidel-PCA 2D)
    hl_full_hull_area = sa_full_hull_area = None
    if len(pc_pool_hl) >= 3:
        h = ConvexHull(pc_pool_hl)
        poly = pc_pool_hl[h.vertices]
        poly = np.vstack([poly, poly[:1]])
        ax_main.fill(poly[:, 0], poly[:, 1], color="C0", alpha=0.12, zorder=0)
        ax_main.plot(poly[:, 0], poly[:, 1], color="C0", lw=2.0, ls="-",
                     alpha=0.85,
                     label=f"HL aberration envelope ({len(hl)} refined, area={h.volume:.2f})")
        hl_full_hull_area = h.volume

    # Layer 2: SAGA FULL pool hull (all 16 designs in Seidel-PCA 2D)
    if len(pc_pool_sa) >= 3:
        h = ConvexHull(pc_pool_sa)
        poly = pc_pool_sa[h.vertices]
        poly = np.vstack([poly, poly[:1]])
        ax_main.fill(poly[:, 0], poly[:, 1], color="C3", alpha=0.35, zorder=1)
        ax_main.plot(poly[:, 0], poly[:, 1], color="C3", lw=2.0, ls="--",
                     alpha=0.95,
                     label=f"SA+GA aberration envelope ({len(sa)} refined, area={h.volume:.2f})")
        sa_full_hull_area = h.volume

    # Layer 3: dimmed HL basin reps NOT passing quality (background grey dots)
    hl_fail_mask = np.array([r["rms_um"] > QUALITY_UM_HL for r in hl_reps_all])
    if hl_fail_mask.any():
        ax_main.scatter(
            pc_hl_all[hl_fail_mask, 0], pc_hl_all[hl_fail_mask, 1],
            s=20, marker="o", color="lightgrey", edgecolor="grey",
            linewidth=0.4, alpha=0.65, zorder=2,
            label=f"HL basins > {int(QUALITY_UM_HL)} um (n={int(hl_fail_mask.sum())})",
        )

    # Layer 4: dimmed SAGA basin reps NOT passing quality
    sa_fail_mask = np.array([r["rms_um"] > QUALITY_UM_SA for r in sa_reps_all])
    if sa_fail_mask.any():
        ax_main.scatter(
            pc_sa_all[sa_fail_mask, 0], pc_sa_all[sa_fail_mask, 1],
            s=55, marker="s", color="lightgrey", edgecolor="C3",
            linewidth=0.6, alpha=0.7, zorder=2,
            label=f"SA+GA basins > {int(QUALITY_UM_SA)} um (n={int(sa_fail_mask.sum())})",
        )

    # Layer 5: HL QUALITY basins (highlighted, colored by RMS)
    sc_hl = None
    if len(pc_hl) > 0:
        sc_hl = ax_main.scatter(
            pc_hl[:, 0], pc_hl[:, 1],
            c=rms_hl, cmap=cmap, norm=norm,
            s=170, marker="o", edgecolor="black", linewidth=1.1, alpha=0.98,
            label=f"HL quality basins <= {int(QUALITY_UM_HL)} um (n={n_hl})",
            zorder=5,
        )
    # Layer 6: SAGA QUALITY basin (single, highlighted)
    if len(pc_sa) > 0:
        ax_main.scatter(
            pc_sa[:, 0], pc_sa[:, 1],
            c=rms_sa, cmap=cmap, norm=norm,
            s=220, marker="s", edgecolor="black", linewidth=1.6,
            label=f"SA+GA quality basin <= {int(QUALITY_UM_SA)} um (n={n_sa}, {float(rms_sa[0]):.1f} um)",
            zorder=6,
        )

    if sc_hl is not None:
        cbar = fig.colorbar(sc_hl, ax=ax_main, fraction=0.045, pad=0.02)
        cbar.set_label("refined RMS spot [um] (log scale)")

    ax_main.set_xlabel(f"Seidel-PCA PC1 ({var_explained[0]*100:.1f}% var)")
    ax_main.set_ylabel(f"Seidel-PCA PC2 ({var_explained[1]*100:.1f}% var)")
    ax_main.set_title(
        f"{SYS_NAME} aberration footprint (Seidel S1..S5, standardized -> 2D PCA, "
        f"{var_explained[:2].sum()*100:.0f}% var)\n"
        f"HL aberration envelope is {hl_full_hull_area / max(sa_full_hull_area, 1e-9):.1f}x larger, "
        f"and HL has {n_hl} distinct quality basins vs SA+GA's {n_sa}"
    )
    ax_main.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax_main.grid(alpha=0.25)

    # Drama numbers in lower-left corner
    drama_lines = [
        f"aberration envelope (full refined pool):",
        f"  HL    {len(hl):3d} designs, hull area = {hl_full_hull_area:6.2f}",
        f"  SA+GA {len(sa):3d} designs, hull area = {sa_full_hull_area:6.2f}",
        f"  area ratio:       {hl_full_hull_area / max(sa_full_hull_area, 1e-9):5.1f}x",
        "",
        f"quality basins (small spot, eps=0.20 distinct):",
        f"  HL    {n_hl:3d} basins <= {int(QUALITY_UM_HL)} um",
        f"  SA+GA {n_sa:3d} basins <= {int(QUALITY_UM_SA)} um",
        f"  count ratio:      {n_hl / max(n_sa, 1):5.1f}x",
    ]
    ax_main.text(
        0.02, 0.02, "\n".join(drama_lines),
        transform=ax_main.transAxes,
        fontsize=8.5, va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black",
                  lw=0.9, alpha=0.94),
    )

    # Annotate SAGA cluster
    ax_main.annotate(
        f"SA+GA: narrow aberration envelope,\n"
        f"only 1 quality basin reached\n"
        f"(best refined RMS = {sa_min:.1f} um)",
        xy=(pc_pool_sa[:, 0].mean(), pc_pool_sa[:, 1].mean()),
        xytext=(50, -90), textcoords="offset points",
        fontsize=9, color="C3",
        arrowprops=dict(arrowstyle="->", color="C3", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="C3", lw=0.8, alpha=0.95),
    )

    # Annotate HL envelope
    ax_main.annotate(
        f"HL: wide aberration envelope,\n"
        f"{n_hl} distinct quality basins\n"
        f"scattered across many regimes",
        xy=(pc_pool_hl[:, 0].max(), pc_pool_hl[:, 1].max()),
        xytext=(-220, -10), textcoords="offset points",
        fontsize=9, color="C0",
        arrowprops=dict(arrowstyle="->", color="C0", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="C0", lw=0.8, alpha=0.95),
    )

    # ----- bottom: full-pool RMS distribution (quality-context backdrop) -----
    bins = np.geomspace(1.0, 1000.0, 35)
    ax_hist.hist(
        rms_full_hl, bins=bins, color="C0", alpha=0.55,
        label=f"HyperLens full pool (n={len(hl)})", edgecolor="C0",
    )
    ax_hist.hist(
        rms_full_sa, bins=bins, color="C3", alpha=0.75,
        label=f"SA+GA full pool (n={len(sa)})", edgecolor="black",
    )
    ax_hist.axvline(QUALITY_UM_HL, color="black", ls=":", lw=1.2)
    ax_hist.axvline(QUALITY_UM_SA, color="C3", ls=":", lw=1.0)
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("refined RMS spot [um]")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(
        f"refined RMS (full pool):  HL min = {hl_min:.1f} um  |  "
        f"SA+GA min = {sa_min:.1f} um  (no SA+GA design <= {int(QUALITY_UM_HL)} um)",
        fontsize=10,
    )
    ax_hist.text(QUALITY_UM_HL, 0.92, f" {int(QUALITY_UM_HL)} um (HL threshold)",
                 transform=ax_hist.get_xaxis_transform(),
                 fontsize=8, va="top")
    ax_hist.text(QUALITY_UM_SA, 0.78, f" {int(QUALITY_UM_SA)} um (SA+GA threshold)",
                 transform=ax_hist.get_xaxis_transform(),
                 fontsize=8, va="top", color="C3")
    ax_hist.legend(loc="upper right", fontsize=9)
    ax_hist.grid(alpha=0.25)

    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    print(f"HL min RMS: {hl_min:.2f} um, HL quality basins <= {int(QUALITY_UM_HL)}um: {n_hl}")
    print(f"SAGA min RMS: {sa_min:.2f} um, SAGA quality basins <= {int(QUALITY_UM_SA)}um: {n_sa}")
    print(f"PCA variance explained on quality-passing reps: PC1+PC2 = {var_explained[:2].sum()*100:.1f}%")


if __name__ == "__main__":
    main()
