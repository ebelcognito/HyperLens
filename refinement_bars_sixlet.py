"""Refinement-bar plot — discovery vs refined RMS for three sixlet generations.

Three equally-spaced x-axis positions (Gen 1, Gen 2, Gen 3) each show a pair
of bars: discovery (raw) RMS in red and refined RMS in green. RMS values are
labelled directly on top of each bar. Legend (top-right) has two entries.
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

PICKS = [
    ("Gen 1", "elite_0003_32384k"),
    ("Gen 2", "elite_0026_83520k"),
    ("Gen 3", "elite_0033_97248k"),
]

COLOR_DISCOVERY = "#C3553A"  # red
COLOR_REFINED = "#3A8C5C"    # green


def load_pick(stem):
    p = os.path.join(RUN_DIR, "elite_designs", f"{stem}.json")
    with open(p) as f:
        d = json.load(f)
    return {
        "stem": stem,
        "raw_rms_um": float(d["rms_um"]),
        "ref_rms_um": float(d["refined"]["rms_um"]),
    }


def main():
    picks = [load_pick(s) for _, s in PICKS]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=160)

    n = len(PICKS)
    x = np.arange(n, dtype=float)
    width = 0.36
    raw_vals = [p["raw_rms_um"] for p in picks]
    ref_vals = [p["ref_rms_um"] for p in picks]

    bars_raw = ax.bar(x - width / 2, raw_vals, width,
                      color=COLOR_DISCOVERY, edgecolor="black", linewidth=0.6,
                      label="Discovery (raw)", zorder=3)
    bars_ref = ax.bar(x + width / 2, ref_vals, width,
                      color=COLOR_REFINED, edgecolor="black", linewidth=0.6,
                      label="Refined", zorder=3)

    y_max = max(raw_vals + ref_vals)
    pad = y_max * 0.04
    for bar, v in list(zip(bars_raw, raw_vals)) + list(zip(bars_ref, ref_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad,
                f"{v:.1f} μm", ha="center", va="bottom", fontsize=9,
                color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in PICKS])
    ax.set_ylabel("Spot RMS (μm)", fontsize=10)

    ax.grid(True, which="major", axis="y", alpha=0.25, lw=0.5)
    ax.grid(False, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.set_ylim(0, y_max * 1.18)
    ax.set_xlim(-0.6, n - 0.4)

    ax.legend(loc="upper right", frameon=False, fontsize=9,
              handlelength=1.6, borderaxespad=0.4)

    fig.tight_layout()
    out_png = os.path.join(OUT, "sixlet_refinement_bars.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {out_png}")
    for (label, _), p in zip(PICKS, picks):
        print(f"  {label}: discovery {p['raw_rms_um']:.1f} μm  →  "
              f"refined {p['ref_rms_um']:.1f} μm")


if __name__ == "__main__":
    main()
