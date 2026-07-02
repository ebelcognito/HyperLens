#!/usr/bin/env python
"""Diverse-design library figure: 10 HL refined designs per system, picked by
eps-net in normalized theta space, rendered as filled-bundle lens cross-sections
in a 2x5 grid. All panels in the same soft blue. No insets, no Seidel chart, no
spot diagram -- just the lens cards. The aesthetic spread IS the argument.

Usage:
  python scripts_throwaway/render_diverse_library.py            # both systems
  python scripts_throwaway/render_diverse_library.py sixlet
  python scripts_throwaway/render_diverse_library.py octuplet

Outputs:
  results/figure_archive_demo/diverse_library_sixlet.png
  results/figure_archive_demo_octuplet/diverse_library_octuplet.png

See HANDOFF_DIVERSE_LIBRARY.md for the design rationale.
"""
from __future__ import annotations
import json, os, sys
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

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
    load_bounds, load_pool, eps_net_select,
)
from scripts_throwaway.render_spotlight_sixlet_designs import render_one

BLUE_FILL = "#7BAFD4"
N_TARGET = 6
N_ROWS, N_COLS = 3, 2
RMS_GATE_UM = 40.0
EPS_LADDER = [0.40, 0.35, 0.30, 0.25, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.05, 0.03]

OUT_DIRS = {
    "sixlet":   ROOT / "results" / "figure_archive_demo",
    "octuplet": ROOT / "results" / "figure_archive_demo_octuplet",
}


def drop_spotlights(pool, spotlights, system):
    """Remove designs that match a spotlight. Sixlet stores refined_theta in
    spotlights.json; octuplet doesn't, so we match on (S2, S3, rms_um)."""
    if system == "sixlet":
        spot_thetas = [np.asarray(s["refined_theta"], dtype=np.float64)
                       for s in spotlights if "refined_theta" in s]
        def is_spot(r):
            return any(np.allclose(r["theta"], st, atol=1e-9) for st in spot_thetas)
    else:
        # Octuplet spotlights store S2/S3 from a separate descriptor recompute,
        # so they don't match the pool's stored values. rms_um does match
        # exactly (verified empirically), so use that as the join key.
        spot_rms = {round(s["rms_um"], 4) for s in spotlights}
        def is_spot(r):
            return round(r["rms_um"], 4) in spot_rms
    n_before = len(pool)
    cleaned = [r for r in pool if not is_spot(r)]
    print(f"  dropped {n_before - len(cleaned)} spotlight(s) from pool")
    return cleaned


def pick_diverse(pool_clean, dim, n_target):
    """Walk eps ladder; take the loosest eps that still yields >= n_target picks."""
    for eps in EPS_LADDER:
        idx = eps_net_select(pool_clean, eps, dim)
        if len(idx) >= n_target:
            chosen = [pool_clean[i] for i in idx[:n_target]]
            print(f"  eps={eps:.2f}: {len(idx)} accepted, taking first {n_target}")
            return chosen, eps
    raise RuntimeError(f"could not fill library (n={n_target}) at any eps in ladder")


def report_pairwise(chosen, dim, eps):
    """Print pairwise normalized-theta distances and Seidel signatures."""
    sqrtD = np.sqrt(dim)
    n = len(chosen)
    pdists = []
    for a in range(n):
        for b in range(a + 1, n):
            d = np.linalg.norm(chosen[a]["theta_norm"] - chosen[b]["theta_norm"]) / sqrtD
            pdists.append(d)
    print(f"  pairwise normalized-theta dist: min={min(pdists):.3f} (eps={eps:.2f}) "
          f"max={max(pdists):.3f} mean={np.mean(pdists):.3f}")
    print(f"  Seidel signatures (S1..S5) per pick:")
    for k, r in enumerate(chosen):
        sig = " ".join(f"{r[k2]:+.2e}" for k2 in ("S1", "S2", "S3", "S4", "S5"))
        print(f"    #{k+1:2d}  rms={r['rms_um']:5.1f} um   {sig}")


def render_grid(chosen, system, out_png, n_more):
    """Render N_TARGET panels in N_ROWS x N_COLS, framed by a rounded box.
    A serif header on top reports the mean multi-wavelength RMS over the picks;
    a small italic footer hints that n_more diverse designs exist beyond the grid.
    """
    from matplotlib.patches import FancyBboxPatch

    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS[system]["cfg"]
    )

    avg_rms = float(np.mean([d["rms_um"] for d in chosen]))

    # Lens cards are landscape (~3-4:1). 2 per row x 3 rows -> roughly 14 x 10.
    fig = plt.figure(figsize=(14.0, 10.0), facecolor="white")
    gs = fig.add_gridspec(
        N_ROWS, N_COLS,
        left=0.05, right=0.95,
        top=0.88, bottom=0.07,
        wspace=0.08, hspace=0.10,
    )
    for k, design in enumerate(chosen):
        ax = fig.add_subplot(gs[k // N_COLS, k % N_COLS])
        ax.set_facecolor("white")
        try:
            lens = make_geolens(prob_cfg, p_problem, lens_path,
                                np.asarray(design["theta"], dtype=np.float64),
                                template, solver)
            render_one(ax, lens, fill_color=BLUE_FILL)
        except Exception as e:
            print(f"  WARN: design #{k+1} (idx={design['idx']}) render failed: {e}")
            ax.text(0.5, 0.5, "render failed", ha="center", va="center",
                    transform=ax.transAxes, color="red")
            ax.set_axis_off()

    # Mathematica-style serif header
    fig.text(
        0.5, 0.945,
        f"Average Multiwavelength RMS:  {avg_rms:.1f}  µm",
        ha="center", va="center",
        fontsize=22, family="serif", style="italic",
        color="#1A1A1A",
    )

    # "+ N more" footer
    fig.text(
        0.5, 0.035,
        f"… plus {n_more} more diverse designs with RMS < {RMS_GATE_UM:.0f} µm",
        ha="center", va="center",
        fontsize=14, family="serif", style="italic",
        color="#5A5A5A",
    )

    # Outer frame around everything
    box = FancyBboxPatch(
        (0.012, 0.012), 0.976, 0.976,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        transform=fig.transFigure,
        edgecolor="#1A1A1A", facecolor="none", linewidth=1.6,
        zorder=10, clip_on=False,
    )
    fig.patches.append(box)

    fig.savefig(out_png, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  -> {out_png}")


def run_system(system):
    print("=" * 70)
    print(f"DIVERSE LIBRARY: {system}")
    print("=" * 70)

    cfg_path = SYSTEMS[system]["cfg"]
    hl_json  = SYSTEMS[system]["hl_json"]
    out_dir  = OUT_DIRS[system]
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = load_bounds(cfg_path)
    pool = load_pool(hl_json, bounds)
    print(f"  loaded {len(pool)} designs (rms < 1000 um)")

    pool_lt40 = [r for r in pool if r["rms_um"] < RMS_GATE_UM]
    print(f"  {len(pool_lt40)} designs with rms < {RMS_GATE_UM} um")

    spotlights = json.loads((out_dir / "spotlights.json").read_text())
    pool_clean = drop_spotlights(pool_lt40, spotlights, system)

    if len(pool_clean) < N_TARGET:
        raise RuntimeError(f"only {len(pool_clean)} designs available after filtering")

    dim = pool_clean[0]["theta"].size
    chosen, eps_used = pick_diverse(pool_clean, dim, N_TARGET)

    report_pairwise(chosen, dim, eps_used)

    n_more = len(pool_clean) - N_TARGET  # other sub-RMS-gate designs not shown
    out_png = out_dir / f"diverse_library_{system}.png"
    render_grid(chosen, system, out_png, n_more=n_more)

    manifest = {
        "system": system,
        "n_target": N_TARGET,
        "rms_gate_um": RMS_GATE_UM,
        "eps_used": eps_used,
        "picks": [
            {"rank": k + 1, "design_idx": int(r["idx"]),
             "rms_um": r["rms_um"],
             "S1": r["S1"], "S2": r["S2"], "S3": r["S3"],
             "S4": r["S4"], "S5": r["S5"]}
            for k, r in enumerate(chosen)
        ],
    }
    manifest_path = out_dir / f"diverse_library_{system}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  -> {manifest_path}")


def main():
    if len(sys.argv) > 1:
        systems = [sys.argv[1]]
    else:
        systems = ["sixlet", "octuplet"]
    for s in systems:
        if s not in SYSTEMS:
            sys.exit(f"unknown system: {s}")
        run_system(s)


if __name__ == "__main__":
    main()
