"""
Paper intro figure — "Same spot, different aberration character"

Takes an iso-performance set of Cooke triplet elites (all ~12 um refined
RMS) from the clean 2-D archive and shows in one frame that:

  * generic behaviour descriptors (spot RMS, EFL error) CANNOT distinguish
    them — they collapse onto a single point;
  * Seidel descriptors (S2, S3, S4) spread them across the archive —
    each design has a distinct aberration profile despite having the
    same spot performance;
  * the lens layouts show they are structurally different designs.

All 5 Seidel values are recomputed directly from each elite's refined
theta using the same SeidelDescriptor the main run uses, so we're
reporting the *system-sum* Seidels.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

sys.path.insert(0, "/home/hisham26/Desktop/RL")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.config.schema import RunConfig
from optical_rl.qd.lensmega.descriptors.seidel import SeidelDescriptor
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic


ARCHIVE_DIR = (
    "/home/hisham26/Desktop/RL/results/lensmega/"
    "cooke_triplet_materials_20260414_173935"
)
OUT_DIR = "/home/hisham26/Desktop/RL/paper_figures/iso_spot_character"
os.makedirs(OUT_DIR, exist_ok=True)


def load_refined_elites():
    paths = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "elite_designs", "*.json")))
    elites = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if "refined" not in d or not d["refined"]:
            continue
        ref = d["refined"]
        if "theta" not in ref or "rms_mm" not in ref:
            continue
        name = os.path.basename(p)[:-5]
        efl = ref.get("efl_mm", 50.0)
        try:
            efl_f = float(efl)
        except Exception:
            efl_f = 50.0
        elites.append({
            "name": name,
            "json_path": p,
            "theta_refined": np.array(ref["theta"], dtype=float),
            "theta_raw": np.array(d["theta"], dtype=float),
            "rms_um": float(ref["rms_mm"]) * 1000.0,
            "efl_err_pct": (efl_f - 50.0) / 50.0 * 100.0,
            "s2_2d": float(d["measures"][0]),
            "s3_2d": float(d["measures"][1]),
            "lens_png": os.path.join(ARCHIVE_DIR, "elite_designs",
                                     f"{name}_refined.png"),
        })
    return elites


def pick_iso_quartet(elites, band=(11.5, 13.5)):
    pool = [e for e in elites if band[0] <= e["rms_um"] <= band[1]]
    if len(pool) < 4:
        return pool
    # Greedy farthest-point-in-2D-Seidel selection, to get
    # Seidel-diverse designs at fixed RMS
    coords = np.array([[e["s2_2d"], e["s3_2d"]] for e in pool])
    coords_n = coords / (np.abs(coords).max(axis=0) + 1e-12)
    idx0 = int(np.argmax(np.linalg.norm(coords_n, axis=1)))
    sel = [idx0]
    while len(sel) < 4:
        d_to = np.min(np.stack([
            np.linalg.norm(coords_n - coords_n[i], axis=1) for i in sel
        ]), axis=0)
        d_to[sel] = -1
        sel.append(int(np.argmax(d_to)))
    return [pool[i] for i in sel]


def setup_descriptor():
    cfg = load_config(os.path.join(ARCHIVE_DIR, "config.yaml"))
    p = cfg.problem
    template, lens = load_lens_template(
        p.lens_path,
        wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        device="cpu",
    )
    solver = GeneralizedParaxialSolver(template)
    desc = SeidelDescriptor(["S1", "S2", "S3", "S4", "S5"])
    desc.prepare(template, device="cpu")
    return cfg, template, lens, solver, desc


def compute_full_seidel(theta, cfg, template, lens, solver, desc):
    p = cfg.problem
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=False)
    _apply_theta_generic(
        lens, theta_t,
        template=template, solver=solver,
        focal_length_target_mm=p.focal_length_target_mm,
        device="cpu",
        parameterization=p.parameterization,
        free_curvature_indices=p.free_curvature_indices,
        free_gap_indices=p.free_gap_indices,
        variable_flange=p.variable_flange,
        variable_materials=p.variable_materials,
        free_material_indices=p.free_material_indices,
    )
    # Same field index the run used
    try:
        hfov_deg = float(np.rad2deg(float(lens.rfov)))
    except Exception:
        hfov_deg = 15.0
    field_y_deg = hfov_deg  # use the max field used during fitness eval
    out = desc.compute(
        lens,
        field_y_deg=field_y_deg,
        object_depth_mm=p.object_depth_mm,
        wavelength_um=p.wavelength_um,
        device="cpu",
    )
    return out.detach().numpy()


def main():
    elites = load_refined_elites()
    print(f"loaded {len(elites)} refined elites")

    cfg, template, lens, solver, desc = setup_descriptor()

    # compute full 5-Seidel for every refined elite (fast — 37 evals)
    for e in elites:
        e["seidel_full"] = compute_full_seidel(
            e["theta_refined"], cfg, template, lens, solver, desc
        )

    quartet = pick_iso_quartet(elites, band=(11.5, 13.5))
    # sort by S2 for consistent left-to-right ordering
    quartet = sorted(quartet, key=lambda e: e["seidel_full"][1])
    print("\niso-RMS quartet:")
    for e in quartet:
        s = e["seidel_full"]
        print(f"  {e['name']:<30} rms={e['rms_um']:>6.2f} um  "
              f"S1={s[0]:+.2e} S2={s[1]:+.2e} S3={s[2]:+.2e} "
              f"S4={s[3]:+.2e} S5={s[4]:+.2e}")

    # ----- build figure -----
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[0.9, 1.1, 1.1],
        hspace=0.35, wspace=0.30,
        left=0.05, right=0.98, top=0.93, bottom=0.06,
    )
    colors = ["#E45756", "#4C78A8", "#54A24B", "#F58518"]

    # Row 1: four lens layouts
    for i, e in enumerate(quartet):
        ax = fig.add_subplot(gs[0, i])
        if os.path.exists(e["lens_png"]):
            ax.imshow(imread(e["lens_png"]))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(colors[i]); sp.set_linewidth(3)
        ax.set_title(f"Lens {i+1}\nspot RMS = {e['rms_um']:.2f} μm",
                     fontsize=11, color=colors[i], fontweight="bold")

    # Row 2+3 left: generic behaviour space (rms, efl_err) for ALL elites
    ax_beh = fig.add_subplot(gs[1:, :2])
    all_rms = np.array([e["rms_um"] for e in elites])
    all_efl = np.array([abs(e["efl_err_pct"]) for e in elites])
    ax_beh.scatter(all_rms, all_efl, s=40, c="lightgray", edgecolor="#999",
                   linewidths=0.4, alpha=0.7, label="archive elites")
    for i, e in enumerate(quartet):
        ax_beh.scatter(e["rms_um"], abs(e["efl_err_pct"]),
                       s=260, c=colors[i], edgecolor="black",
                       linewidths=1.5, zorder=5, label=f"Lens {i+1}")
    ax_beh.axvspan(11.5, 13.5, color="gold", alpha=0.18, zorder=0,
                   label="iso-RMS band (11.5–13.5 μm)")
    ax_beh.set_xlabel("refined spot RMS (μm)", fontsize=12)
    ax_beh.set_ylabel("|EFL error| (%)", fontsize=12)
    ax_beh.set_title(
        "Generic behaviour descriptor: (spot RMS, |EFL error|)\n"
        "→ the four iso-performance lenses are indistinguishable",
        fontsize=12)
    ax_beh.grid(True, alpha=0.3)
    ax_beh.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Row 2+3 right: (S2, S3, S4) 3-D scatter for ALL elites
    ax_sei = fig.add_subplot(gs[1:, 2:], projection="3d")
    all_s = np.array([e["seidel_full"] for e in elites])
    ax_sei.scatter(
        all_s[:, 1] * 1e4, all_s[:, 2] * 1e4, all_s[:, 3] * 1e4,
        s=35, c="lightgray", edgecolor="#999", linewidths=0.3,
        alpha=0.6, label="archive elites",
    )
    for i, e in enumerate(quartet):
        s = e["seidel_full"]
        ax_sei.scatter(
            s[1] * 1e4, s[2] * 1e4, s[3] * 1e4,
            s=280, c=colors[i], edgecolor="black", linewidths=1.2,
            label=f"Lens {i+1}",
        )
    ax_sei.set_xlabel(r"$S_2$ (coma) [$\times 10^{-4}$]", fontsize=10)
    ax_sei.set_ylabel(r"$S_3$ (astigmatism) [$\times 10^{-4}$]", fontsize=10)
    ax_sei.set_zlabel(r"$S_4$ (Petzval) [$\times 10^{-4}$]", fontsize=10)
    ax_sei.set_title(
        "Seidel descriptor: $(S_2, S_3, S_4)$ system-sum\n"
        "→ iso-performance lenses spread across the archive",
        fontsize=12)
    ax_sei.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Compute spread ratios
    q_rms = np.array([e["rms_um"] for e in quartet])
    q_efl = np.array([abs(e["efl_err_pct"]) for e in quartet])
    q_s = np.array([e["seidel_full"] for e in quartet])
    std_rms = all_rms.std() + 1e-12
    std_efl = all_efl.std() + 1e-12
    std_s2 = all_s[:, 1].std() + 1e-18
    std_s3 = all_s[:, 2].std() + 1e-18
    std_s4 = all_s[:, 3].std() + 1e-18
    beh_span_sigma = math.hypot(np.ptp(q_rms) / std_rms, np.ptp(q_efl) / std_efl)
    sei_span_sigma = math.sqrt(
        (np.ptp(q_s[:, 1]) / std_s2) ** 2 +
        (np.ptp(q_s[:, 2]) / std_s3) ** 2 +
        (np.ptp(q_s[:, 3]) / std_s4) ** 2
    )

    ax_beh.text(0.98, 0.05,
                f"iso-RMS quartet span:\n{beh_span_sigma:.2f} σ (archive)",
                transform=ax_beh.transAxes, ha="right", va="bottom",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9))

    fig.suptitle(
        "Same spot, different optical character — four Cooke triplets "
        "with the same spot RMS\n"
        f"Seidel spreads them {sei_span_sigma/beh_span_sigma:.2f}× wider than "
        "generic behaviour descriptors do",
        fontsize=14, fontweight="bold",
    )
    out_png = os.path.join(OUT_DIR, "fig_iso_spot_different_character.png")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_png}")
    print(f"behaviour-space span  = {beh_span_sigma:.3f} sigma")
    print(f"Seidel-space span     = {sei_span_sigma:.3f} sigma")
    print(f"ratio (Seidel/behav.) = {sei_span_sigma/beh_span_sigma:.2f}x")

    # Also save numerical quartet data
    quartet_data = []
    for i, e in enumerate(quartet):
        quartet_data.append({
            "label": f"Lens {i+1}",
            "name": e["name"],
            "rms_um": e["rms_um"],
            "efl_err_pct": e["efl_err_pct"],
            "seidel_S1": float(e["seidel_full"][0]),
            "seidel_S2": float(e["seidel_full"][1]),
            "seidel_S3": float(e["seidel_full"][2]),
            "seidel_S4": float(e["seidel_full"][3]),
            "seidel_S5": float(e["seidel_full"][4]),
        })
    with open(os.path.join(OUT_DIR, "quartet_data.json"), "w") as f:
        json.dump({
            "quartet": quartet_data,
            "behaviour_span_sigma": beh_span_sigma,
            "seidel_span_sigma": sei_span_sigma,
            "ratio": sei_span_sigma / beh_span_sigma,
        }, f, indent=2)


if __name__ == "__main__":
    main()
