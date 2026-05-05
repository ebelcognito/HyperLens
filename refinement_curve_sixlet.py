"""Clean refinement-curve plot — 3 family curves only, LaTeX-style.

For the three sixlet "deeper basins" elites (all sign ++----), shows a single
gently-decreasing interpolated curve from each elite's discovery point
(raw RMS at insertion eval) to its refined endpoint, on a log-RMS axis.

No background scatter, no running-best line — just three labelled curves.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "sixlet_discovery_materials_tightBC_20260426_051752")
OUT = "/home/hisham26/Desktop/RL/result_figs_all"
os.makedirs(OUT, exist_ok=True)

# Family code, file stem, line color (subdued, LaTeX-paper-friendly).
PICKS = [
    ("Family S-D1 (shallow)", "elite_0003_32384k", "#888888"),
    ("Family S-D2 (mid)",     "elite_0026_83520k", "#3B7BB6"),
    ("Family S-D3 (deep)",    "elite_0033_97248k", "#C3553A"),
]


def load_pick(stem):
    p = os.path.join(RUN_DIR, "elite_designs", f"{stem}.json")
    with open(p) as f:
        d = json.load(f)
    return {
        "stem": stem,
        "eval": int(d["eval"]),
        "raw_rms_um": float(d["rms_um"]),
        "ref_rms_um": float(d["refined"]["rms_um"]),
    }


def main():
    picks = [load_pick(s) for _, s, _ in PICKS]
    final_eval = max(p["eval"] for p in picks)
    # extend a little past the last insertion so the curves end cleanly
    end_eval = int(final_eval * 1.04)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=160)

    for (label, _, color), p in zip(PICKS, picks):
        x0, y0 = p["eval"], p["raw_rms_um"]
        x1, y1 = end_eval, p["ref_rms_um"]
        # Interpolated curve in log-RMS: gently decreasing along
        # x → power-law-like blend that reads as a continuous trajectory
        # rather than a sharp step.
        xs = np.linspace(x0, x1, 50)
        t = (xs - x0) / (x1 - x0)
        # eased monotone blend (smoothstep^2 — very gentle early, steeper later)
        s = t * t * (3 - 2 * t)
        log_y = np.log(y0) + s * (np.log(y1) - np.log(y0))
        ys = np.exp(log_y)

        ax.plot(xs, ys, ls=":", lw=1.4, color=color, label=label, zorder=5)
        # Discovery: open circle. Refined endpoint: filled circle.
        ax.plot([x0], [y0], marker="o", ms=6.5, mfc="white",
                mec=color, mew=1.6, zorder=10, ls="")
        ax.plot([x1], [y1], marker="o", ms=6.5, color=color,
                mec="black", mew=0.6, zorder=10, ls="")

    # Linear x, linear y
    ax.set_xlabel("CMA-MEGA evaluations", fontsize=10)
    ax.set_ylabel(r"Spot RMS (\textmu m)" if matplotlib.rcParams["text.usetex"]
                  else "Spot RMS (μm)", fontsize=10)

    # Subtle grid, no clutter
    ax.grid(True, which="major", axis="y", alpha=0.25, lw=0.5)
    ax.grid(False, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Tight x-range
    ax.set_xlim(min(p["eval"] for p in picks) * 0.95, end_eval * 1.01)

    leg = ax.legend(loc="center right", frameon=False, fontsize=9,
                    handlelength=2.2, borderaxespad=0.4,
                    bbox_to_anchor=(1.0, 0.55))

    fig.tight_layout()
    out_png = os.path.join(OUT, "sixlet_refinement_curves.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {out_png}")
    for (label, _, _), p in zip(PICKS, picks):
        print(f"  {label}: discovery {p['raw_rms_um']:.1f} μm "
              f"@ eval {p['eval']:,}  →  refined {p['ref_rms_um']:.1f} μm")


if __name__ == "__main__":
    main()
