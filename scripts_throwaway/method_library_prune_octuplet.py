#!/usr/bin/env python
"""Prune the 7 octuplet method-library designs (3 anchors, 3 near-parents, 1
incumbent) by ray-traced aperture trimming, then render each pruned lens
without rays.

For each design:
  1. Build a fresh GeoLens with theta applied via _apply_theta_generic
  2. Capture the original semi-apertures (r) per surface
  3. Call lens.prune_surf() (DeepLens primitive) to trim r to the ray-traced
     extent + automatic margin
  4. Save the pruned lens as JSON
  5. Render the pruned lens cross-section (no rays drawn)

Outputs:
  figures/method_library_data/pruned_lenses/{role}_{rank}.json
  figures/method_library_data/pruned_summary.json   (CAs before/after, RMS)
  figures/method_library_octuplet_pruned.png       (sanity grid render)
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
DATA_IN = ROOT / "figures" / "fig5_data"
OUT_DIR = ROOT / "figures" / "method_library_data"
OUT_LENSES = OUT_DIR / "pruned_lenses"
OUT_SUMMARY = OUT_DIR / "pruned_summary.json"
OUT_PNG = ROOT / "figures" / "method_library_octuplet_pruned.png"
OUT_LENSES.mkdir(parents=True, exist_ok=True)

from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)


def collect_designs():
    """Return list of (role, rank, design_dict) for the 7 octuplet designs."""
    payload = json.loads((DATA_IN / "octuplet_parents_and_incumbent.json").read_text())
    out = []
    for k, a in enumerate(payload["anchors_used"]):
        out.append(("anchor", k, a))
    for d in payload["designs"]:
        if d["role"] == "parent":
            out.append(("parent", int(d["rank"]), d))
        elif d["role"] == "incumbent":
            out.append(("incumbent", int(d.get("rank", 0)), d))
    return out


def get_cas(lens):
    return [float(s.r) for s in lens.surfaces]


def prune_design(template, solver, prob_cfg, lens_path, p_problem, theta):
    """Build, prune in place, return (lens, ca_before, ca_after, ok)."""
    lens = make_geolens(prob_cfg, p_problem, lens_path, np.asarray(theta), template, solver)
    ca_before = get_cas(lens)
    try:
        lens.prune_surf()  # ray-trace + auto margin (10% for r_sensor>=10mm)
        ca_after = get_cas(lens)
        ok = True
    except Exception as e:
        print(f"    prune_surf failed: {e}")
        ca_after = ca_before
        ok = False
    return lens, ca_before, ca_after, ok


def save_lens_json(lens, path):
    """Use DeepLens write_lens_json if available; else write a minimal record."""
    try:
        lens.write_lens_json(str(path))
        return True
    except Exception as e:
        print(f"    write_lens_json failed: {e}; writing minimal fallback")
        rec = {
            "foclen": float(getattr(lens, "foclen", float("nan"))),
            "fnum": float(getattr(lens, "fnum", float("nan"))) if hasattr(lens, "fnum") else None,
            "r_sensor": float(getattr(lens, "r_sensor", float("nan"))),
            "d_sensor": float(lens.d_sensor.item()) if torch.is_tensor(lens.d_sensor) else float(lens.d_sensor),
            "surfaces": [
                {
                    "type": type(s).__name__,
                    "r": float(s.r),
                    "d": float(s.d.item()) if torch.is_tensor(s.d) else float(s.d),
                    "c": float(s.c.item()) if (hasattr(s, "c") and torch.is_tensor(s.c)) else (float(s.c) if hasattr(s, "c") else 0.0),
                    "mat2": getattr(getattr(s, "mat2", None), "name", None),
                }
                for s in lens.surfaces
            ],
        }
        Path(path).write_text(json.dumps(rec, indent=2))
        return False


def main():
    designs = collect_designs()
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["octuplet"]["cfg"]
    )

    summary = {"system": "octuplet", "designs": []}

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6))
    layout = [
        ("anchor",     0, 0, 0, "C0"),
        ("anchor",     1, 0, 1, "C0"),
        ("anchor",     2, 0, 2, "C0"),
        ("incumbent",  0, 0, 3, "C3"),
        ("parent",     0, 1, 0, "#666"),
        ("parent",     1, 1, 1, "#666"),
        ("parent",     2, 1, 2, "#666"),
        (None,         0, 1, 3, None),  # blank cell
    ]
    by_key = {(role, rank): des for (role, rank, des) in designs}

    for role, rank, row, col, color in layout:
        ax = axes[row, col]
        if role is None:
            ax.set_axis_off()
            continue
        des = by_key.get((role, rank))
        if des is None:
            ax.text(0.5, 0.5, f"{role}#{rank+1}\nMISSING", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue

        theta = des["theta"]
        rms = float(des["refined_RMS_um"])
        src = des.get("source_run", "?")
        did = des.get("design_id", "?")

        print(f"  prune {role}#{rank+1}: source={src} id={did} RMS={rms:.2f} um")
        lens, ca_before, ca_after, ok = prune_design(
            template, solver, prob_cfg, lens_path, p_problem, theta,
        )
        # Save pruned lens JSON
        out_json = OUT_LENSES / f"{role}_{rank}.json"
        wrote = save_lens_json(lens, out_json)
        print(f"    wrote {out_json}  (write_lens_json ok={wrote})  prune_ok={ok}")

        # Per-surface CA shrink summary
        shrink_pct = [
            (1.0 - ca_after[i] / ca_before[i]) * 100.0 if ca_before[i] > 0 else 0.0
            for i in range(len(ca_before))
        ]
        max_shrink = max(shrink_pct) if shrink_pct else 0.0
        median_shrink = float(np.median(shrink_pct)) if shrink_pct else 0.0

        summary["designs"].append({
            "role": role,
            "rank": rank,
            "source_run": src,
            "design_id": did,
            "refined_RMS_um": rms,
            "ca_before_mm": ca_before,
            "ca_after_mm": ca_after,
            "ca_max_shrink_pct": max_shrink,
            "ca_median_shrink_pct": median_shrink,
            "prune_ok": ok,
            "lens_json": str(out_json),
        })

        # Render pruned lens, NO RAYS
        try:
            lens.draw_lens_2d(ax=ax, fig=ax.figure, color=color)
        except Exception as e:
            ax.text(0.5, 0.5, f"draw err: {e}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_title(
            f"{role.upper()} #{rank+1}\nRMS={rms:.1f} um  "
            f"(median CA shrink {median_shrink:.0f}%)",
            fontsize=8,
        )
        ax.set_aspect("equal")
        ax.set_axis_off()

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_SUMMARY}")

    fig.suptitle(
        "Octuplet method-library: pruned lens cross-sections (no rays)\n"
        "top: 3 anchors + 1 incumbent;  bottom: 3 near-parents",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
