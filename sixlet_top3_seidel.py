"""
Top-3 design families: full Seidel signatures + landscape with only top-3 points.

Loads the 3 best-RMS refined elites from the sixlet design-family gallery
(one rep per basin), computes their full (S1..S5) using the descriptor on
the refined theta, then produces:

  Figure 1: 3 lens diagrams + their Seidel signatures as bar charts
  Figure 2: hero landscape with ONLY the top-3 family anchors plotted
"""
import glob
import json
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
from optical_rl.qd.lensmega.descriptors.seidel import SeidelDescriptor
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic


CFG = ("/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
       "config/examples/sixlet_discovery.yaml")
RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "sixlet_discovery_20260423_132001")
GALLERY_JSON = ("/home/hisham26/Desktop/RL/paper_figures/sixlet_design_gallery/"
                "design_family_summary.json")
GRIDS_NPZ = ("/home/hisham26/Desktop/RL/paper_figures/landscape_sixlet/"
             "grids.npz")
OUT = "/home/hisham26/Desktop/RL/paper_figures/sixlet_top3"
os.makedirs(OUT, exist_ok=True)


def load_top3():
    """Pick top 3 STRUCTURALLY distinct families: one per sign pattern, taking
    the best-RMS member of each, then sort by RMS and keep the top 3."""
    with open(GALLERY_JSON) as f:
        d = json.load(f)
    by_pattern = {}
    for b in d["basins"]:
        sp = b["sign_pattern"]
        if sp not in by_pattern or b["rms_um_refined"] < by_pattern[sp]["rms_um_refined"]:
            by_pattern[sp] = b
    reps = sorted(by_pattern.values(), key=lambda b: b["rms_um_refined"])
    return reps[:3]


def compute_full_seidel_for(thetas):
    """Use the same descriptor the run used, but enabled with all 5 terms,
    to compute the full Seidel signature of each refined theta."""
    cfg = load_config(CFG)
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

    out = []
    for theta in thetas:
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
        try:
            hfov_deg = float(np.rad2deg(float(lens.rfov)))
        except Exception:
            hfov_deg = 15.0
        s = desc.compute(
            lens,
            field_y_deg=hfov_deg,
            object_depth_mm=p.object_depth_mm,
            wavelength_um=p.wavelength_um,
            device="cpu",
        )
        out.append(s.detach().numpy())
    return np.array(out)


def main():
    top3 = load_top3()
    print("Top 3 design families:")
    for r in top3:
        print(f"  basin {r['id']}  RMS={r['rms_um_refined']:.2f}μm  "
              f"sign={r['sign_pattern']}  ({r['rep_name']})")

    # Find the lens-rendered PNG path for each
    png_paths = []
    for r in top3:
        candidate = os.path.join(RUN_DIR, "elite_designs",
                                  f"{r['rep_name']}_refined.png")
        png_paths.append(candidate if os.path.exists(candidate) else None)

    # Need full theta — re-load each elite JSON to get refined theta
    thetas = []
    for r in top3:
        path = os.path.join(RUN_DIR, "elite_designs", f"{r['rep_name']}.json")
        with open(path) as f:
            d = json.load(f)
        thetas.append(np.array(d["refined"]["theta"], dtype=float))
    seidels = compute_full_seidel_for(thetas)
    print(f"\nFull Seidel signatures (S1..S5):")
    for r, s in zip(top3, seidels):
        print(f"  basin {r['id']}  S1={s[0]:+.2e}  S2={s[1]:+.2e}  "
              f"S3={s[2]:+.2e}  S4={s[3]:+.2e}  S5={s[4]:+.2e}")

    family_colors = ["#E45756", "#4C78A8", "#54A24B"]

    # ===== Figure 1: lens diagrams + Seidel bar charts =====
    fig, axes = plt.subplots(2, 3, figsize=(14, 7),
                             gridspec_kw=dict(height_ratios=[1, 1.0],
                                              hspace=0.4, wspace=0.25))
    # Top row: lens diagrams
    for i, (r, png, color) in enumerate(zip(top3, png_paths, family_colors)):
        ax = axes[0, i]
        if png:
            ax.imshow(imread(png))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3)
        ax.set_title(
            f"Family {i+1} (basin {r['id']})\n"
            f"refined RMS = {r['rms_um_refined']:.2f} μm  ·  sign {r['sign_pattern']}",
            fontsize=11, color=color, fontweight="bold")
    # Bottom row: Seidel bar charts
    coeff_names = ["S₁\nspherical", "S₂\ncoma", "S₃\nastig", "S₄\nPetzval", "S₅\ndistortion"]
    x = np.arange(5)
    for i, (r, s, color) in enumerate(zip(top3, seidels, family_colors)):
        ax = axes[1, i]
        s_scaled = s * 1e4   # × 10⁻⁴ for readable axis
        bars = ax.bar(x, s_scaled, color=color, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", lw=0.6)
        for j, val in enumerate(s_scaled):
            offset = 0.04 * (max(abs(s_scaled)) + 0.1)
            ax.text(j, val + (offset if val >= 0 else -offset),
                    f"{val:+.2f}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(coeff_names, fontsize=8)
        ax.set_ylabel(r"Seidel value [$\times 10^{-4}$]")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(f"Aberration signature — Family {i+1}", fontsize=10)
    fig.suptitle(
        "Top-3 sixlet design families — lens cross-section and Seidel signature",
        fontsize=13, fontweight="bold",
    )
    out1 = os.path.join(OUT, "top3_lens_and_seidel.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out1}")

    # ===== Figure 2: landscape with only top-3 anchors =====
    grids = np.load(GRIDS_NPZ)
    AA = grids["AA"]; BB = grids["BB"]; rms_grid = grids["rms_grid"]
    a_lo = AA.min(); a_hi = AA.max()
    b_lo = BB.min(); b_hi = BB.max()

    fig, ax = plt.subplots(figsize=(10, 8))
    rms_vis = np.log10(np.clip(rms_grid, 1e-4, 1e3))
    cf = ax.contourf(AA, BB, rms_vis, levels=30, cmap="magma_r",
                     vmin=np.nanpercentile(rms_vis, 2),
                     vmax=np.nanpercentile(rms_vis, 98))
    ax.contour(AA, BB, rms_vis, levels=10, colors="white",
               linewidths=0.4, alpha=0.35)

    # Stagger label offsets per family so they don't clobber each other.
    # F1 at (0,0): label up-left.  F2 at (10.8,0): label down.  F3 at (13.6,6.6): label up-right.
    label_offsets = [(-50, 22), (0, -42), (60, 25)]
    for i, (r, color) in enumerate(zip(top3, family_colors)):
        ax.scatter(r["plane_a"], r["plane_b"], s=560, c=color, marker="*",
                   edgecolor="black", linewidths=2.0, zorder=20)
        # Print to console so the user can verify positions
        print(f"  plotting Family {i+1}: (a, b) = ({r['plane_a']:.2f}, "
              f"{r['plane_b']:.2f}), residual = {r.get('plane_residual', 0):.2f}")
        dx, dy = label_offsets[i]
        ax.annotate(
            f"Family {i+1}\nbasin {r['id']}\n{r['rms_um_refined']:.2f} μm",
            (r["plane_a"], r["plane_b"]),
            fontsize=10, color="white",
            ha="center" if dx == 0 else ("right" if dx < 0 else "left"),
            xytext=(dx, dy), textcoords="offset points",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=1.5),
            bbox=dict(facecolor=color, alpha=0.92, pad=2.5,
                      boxstyle="round", edgecolor="black"))

    ax.set_xlabel("plane coord a")
    ax.set_ylabel("plane coord b")
    ax.set_xlim(a_lo, a_hi); ax.set_ylim(b_lo, b_hi); ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, shrink=0.85,
                 label="log₁₀(RMS / mm)  ·  yellow = better")
    fig.suptitle(
        "Sixlet hero plane — only the top-3 design families plotted\n"
        "(red stars are the basin anchors; landscape is the actual fitness "
        "function on the plane)",
        fontsize=12, fontweight="bold",
    )
    out2 = os.path.join(OUT, "top3_landscape_only.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
