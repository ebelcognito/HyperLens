"""
Paper intro figure:
  "Same spot, different aberration character."

Pick N elites with nearly identical refined spot RMS from a clean archive.
Show:
  (top)   their lens layouts side-by-side
  (bottom left)  scatter of ALL elites in a generic behaviour space
                 (rms_refined_um, |efl_err_%|) — iso-set collapses onto a point
  (bottom right) scatter of ALL elites in Seidel space (S2, S3) —
                 iso-set spreads across the archive

The figure makes the intro argument in one frame:
  "a generic behaviour descriptor can't see that these are different lenses.
   Seidel can."
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.image import imread


ARCHIVE = "/home/hisham26/Desktop/RL/results/lensmega/cooke_triplet_materials_20260414_173935"
OUT = "/home/hisham26/Desktop/RL/paper_figures"
os.makedirs(OUT, exist_ok=True)


def load_elites(archive_dir):
    paths = sorted(glob.glob(os.path.join(archive_dir, "elite_designs", "*.json")))
    elites = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if "refined" not in d or not d["refined"]:
            continue
        ref = d["refined"]
        name = os.path.basename(p)[:-5]
        efl = ref.get("efl_mm", 0)
        try:
            efl_f = float(efl)
        except Exception:
            efl_f = 50.0
        efl_err = abs(efl_f - 50.0) / 50.0 * 100.0
        elites.append({
            "name": name,
            "path": p,
            "s2": float(d["measures"][0]),
            "s3": float(d["measures"][1]),
            "rms_um": float(ref["rms_mm"]) * 1000.0,
            "efl_err_pct": efl_err,
            "lens_png": os.path.join(archive_dir, "elite_designs", f"{name}_refined.png"),
        })
    return elites


def pick_iso_rms_quartet(elites, rms_band=(11.5, 13.5)):
    """From elites within the RMS band, pick 4 maximally spread in (S2, S3).
    Greedy farthest-point selection."""
    pool = [e for e in elites if rms_band[0] <= e["rms_um"] <= rms_band[1]]
    if len(pool) < 4:
        return pool
    coords = np.array([[e["s2"], e["s3"]] for e in pool])
    # Normalize for fair distance
    coords_n = coords / (np.abs(coords).max(axis=0) + 1e-12)

    # Start with the elite having the largest |S|
    idx0 = int(np.argmax(np.linalg.norm(coords_n, axis=1)))
    selected = [idx0]
    while len(selected) < 4:
        d_to_sel = np.min(
            np.stack([np.linalg.norm(coords_n - coords_n[i], axis=1) for i in selected]),
            axis=0,
        )
        d_to_sel[selected] = -1
        selected.append(int(np.argmax(d_to_sel)))
    return [pool[i] for i in selected]


def main():
    elites = load_elites(ARCHIVE)
    print(f"loaded {len(elites)} refined elites from Cooke archive")

    quartet = pick_iso_rms_quartet(elites, rms_band=(11.5, 13.5))
    quartet = sorted(quartet, key=lambda e: e["s2"])
    print(f"\niso-RMS quartet (RMS {quartet[0]['rms_um']:.2f} - {quartet[-1]['rms_um']:.2f} um):")
    for e in quartet:
        print(f"  {e['name']:<28}  rms={e['rms_um']:>5.2f} um  "
              f"S2={e['s2']:+.2e}  S3={e['s3']:+.2e}  EFL_err={e['efl_err_pct']:.3f}%")

    # --- figure ---
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.35, wspace=0.30,
        left=0.06, right=0.98, top=0.92, bottom=0.08,
    )

    # Top row: 4 lens layouts
    colors = ["#E45756", "#4C78A8", "#54A24B", "#F58518"]
    for i, e in enumerate(quartet):
        ax = fig.add_subplot(gs[0, i])
        if os.path.exists(e["lens_png"]):
            img = imread(e["lens_png"])
            ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[i])
            spine.set_linewidth(3)
        title = f"Lens {i+1}\nspot RMS = {e['rms_um']:.2f} μm"
        ax.set_title(title, fontsize=11, color=colors[i], fontweight="bold")

    # Middle row: two BC-space scatters spanning 2 columns each
    ax_beh = fig.add_subplot(gs[1:, :2])
    ax_sei = fig.add_subplot(gs[1:, 2:])

    all_rms = np.array([e["rms_um"] for e in elites])
    all_efl = np.array([e["efl_err_pct"] for e in elites])
    all_s2 = np.array([e["s2"] for e in elites])
    all_s3 = np.array([e["s3"] for e in elites])

    # Generic behaviour space (rms, efl_err)
    ax_beh.scatter(all_rms, all_efl, s=35, c="lightgray", edgecolor="#999", linewidths=0.5,
                   alpha=0.7, label="archive elites")
    for i, e in enumerate(quartet):
        ax_beh.scatter(e["rms_um"], e["efl_err_pct"], s=260, c=colors[i],
                       edgecolor="black", linewidths=1.5, zorder=5,
                       label=f"Lens {i+1}")
    ax_beh.axvspan(11.5, 13.5, color="gold", alpha=0.15, zorder=0,
                   label="iso-RMS band (11.5-13.5 μm)")
    ax_beh.set_xlabel("refined spot RMS (μm)", fontsize=12)
    ax_beh.set_ylabel("|EFL error| (%)", fontsize=12)
    ax_beh.set_title("Generic behaviour descriptor: (RMS, EFL_err)\n"
                     "→ the iso-RMS lenses are ~indistinguishable", fontsize=12)
    ax_beh.grid(True, alpha=0.3)
    ax_beh.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Seidel space
    ax_sei.scatter(all_s2 * 1e4, all_s3 * 1e4, s=35, c="lightgray",
                   edgecolor="#999", linewidths=0.5, alpha=0.7, label="archive elites")
    for i, e in enumerate(quartet):
        ax_sei.scatter(e["s2"] * 1e4, e["s3"] * 1e4, s=260, c=colors[i],
                       edgecolor="black", linewidths=1.5, zorder=5,
                       label=f"Lens {i+1}")
    ax_sei.axhline(0, color="#666", linewidth=0.5, alpha=0.6)
    ax_sei.axvline(0, color="#666", linewidth=0.5, alpha=0.6)
    ax_sei.set_xlabel(r"$S_2$ (coma)  [$\times 10^{-4}$]", fontsize=12)
    ax_sei.set_ylabel(r"$S_3$ (astigmatism)  [$\times 10^{-4}$]", fontsize=12)
    ax_sei.set_title("Seidel descriptor: $(S_2, S_3)$\n"
                     "→ the iso-RMS lenses spread across the archive", fontsize=12)
    ax_sei.grid(True, alpha=0.3)
    ax_sei.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Compute + annotate spread statistics
    quartet_rms = np.array([e["rms_um"] for e in quartet])
    quartet_efl = np.array([e["efl_err_pct"] for e in quartet])
    quartet_s2 = np.array([e["s2"] for e in quartet])
    quartet_s3 = np.array([e["s3"] for e in quartet])

    # Standardize each BC by the full-archive standard deviation, then measure quartet range
    std_rms = all_rms.std();  std_efl = all_efl.std()
    std_s2 = all_s2.std();    std_s3 = all_s3.std()
    beh_range = np.hypot((quartet_rms.ptp() / max(std_rms, 1e-9)),
                         (quartet_efl.ptp() / max(std_efl, 1e-9)))
    sei_range = np.hypot((quartet_s2.ptp() / max(std_s2, 1e-9)),
                         (quartet_s3.ptp() / max(std_s3, 1e-9)))

    ax_beh.text(0.97, 0.04,
                f"iso-RMS quartet span:\n{beh_range:.2f}σ (of the archive)",
                transform=ax_beh.transAxes, ha="right", va="bottom",
                fontsize=10, color="black",
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))
    ax_sei.text(0.97, 0.04,
                f"iso-RMS quartet span:\n{sei_range:.2f}σ (of the archive)",
                transform=ax_sei.transAxes, ha="right", va="bottom",
                fontsize=10, color="black",
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))

    fig.suptitle(
        "Four Cooke triplets with the same spot RMS — the BC that sees the difference",
        fontsize=14, fontweight="bold",
    )

    out_png = os.path.join(OUT, "fig_same_spot_different_character.png")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"\nsaved {out_png}")
    plt.close(fig)

    # Ratio line for the caption
    if beh_range > 0:
        ratio = sei_range / beh_range
        print(f"\nSpread ratio (Seidel / generic-behaviour) for the iso-RMS set: {ratio:.2f}x")


if __name__ == "__main__":
    main()
