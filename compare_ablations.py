#!/usr/bin/env python3
"""Compare ablation runs: Seidel BCs vs Random BCs vs Single-Objective.

Usage:
    python compare_ablations.py <seidel_dir> <random_dir> <single_dir> [--out <output_dir>]

Example:
    python compare_ablations.py \
        results/lensmega/ablation_seidel_s2s3_20260416_225121 \
        results/lensmega/ablation_random_bc_20260416_225125 \
        results/lensmega/ablation_single_objective_20260416_225130
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ── helpers ──────────────────────────────────────────────────────────────

LABELS = {
    "seidel": "Seidel S₂+S₃",
    "random": "Random Projection",
    "single": "Single-Objective",
}
COLORS = {
    "seidel": "#2196F3",
    "random": "#FF9800",
    "single": "#F44336",
}


def load_trajectory(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "trajectory.csv")


def load_elites(run_dir: Path) -> list:
    elite_dir = run_dir / "elite_designs"
    if not elite_dir.exists():
        return []
    elites = []
    for jp in sorted(elite_dir.glob("*.json")):
        with open(jp) as f:
            d = json.load(f)
        d["_file"] = jp.name
        elites.append(d)
    return elites


def load_archive(run_dir: Path):
    with open(run_dir / "archive.pkl", "rb") as f:
        return pickle.load(f)


# ── plotting ─────────────────────────────────────────────────────────────

def plot_comparison(dirs: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    trajs = {k: load_trajectory(d) for k, d in dirs.items()}
    elites = {k: load_elites(d) for k, d in dirs.items()}

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ── Panel 1: Best fitness over evaluations ───────────────────────
    ax = axes[0, 0]
    for key in ["seidel", "random", "single"]:
        df = trajs[key]
        ax.plot(df["evals"] / 1000, df["best_fitness"],
                color=COLORS[key], linewidth=2, label=LABELS[key])
    ax.set_xlabel("Evaluations (×1000)", fontsize=11)
    ax.set_ylabel("Best Fitness (higher = better)", fontsize=11)
    ax.set_title("Best Fitness Over Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Coverage over evaluations ───────────────────────────
    ax = axes[0, 1]
    for key in ["seidel", "random", "single"]:
        df = trajs[key]
        ax.plot(df["evals"] / 1000, df["coverage"] * 100,
                color=COLORS[key], linewidth=2, label=LABELS[key])
    ax.set_xlabel("Evaluations (×1000)", fontsize=11)
    ax.set_ylabel("Archive Coverage (%)", fontsize=11)
    ax.set_title("Archive Coverage Over Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Cumulative new cells (exploration rate) ─────────────
    ax = axes[0, 2]
    for key in ["seidel", "random", "single"]:
        df = trajs[key]
        ax.plot(df["evals"] / 1000, df["new_cells"].cumsum(),
                color=COLORS[key], linewidth=2, label=LABELS[key])
    ax.set_xlabel("Evaluations (×1000)", fontsize=11)
    ax.set_ylabel("Cumulative New Cells", fontsize=11)
    ax.set_title("Exploration Rate", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Elite refinement comparison ─────────────────────────
    ax = axes[1, 0]
    bar_data = []
    for key in ["seidel", "random", "single"]:
        for e in elites[key]:
            ref = e.get("refined", {})
            if "rms_um" in ref and "rms_um" in e:
                raw_rms = float(e["rms_um"]) if isinstance(e["rms_um"], (int, float)) else float(e["rms_um"])
                ref_rms_str = ref["rms_um"]
                ref_rms = float(ref_rms_str) if isinstance(ref_rms_str, (int, float, str)) else 0
                bar_data.append({
                    "condition": LABELS[key],
                    "color": COLORS[key],
                    "raw_rms": raw_rms,
                    "refined_rms": ref_rms,
                    "label": e["_file"].replace(".json", ""),
                    "sign": e.get("sign_pattern", "?"),
                })

    if bar_data:
        x = np.arange(len(bar_data))
        width = 0.35
        ax.bar(x - width/2, [d["raw_rms"] for d in bar_data], width,
               color=[d["color"] for d in bar_data], alpha=0.4,
               edgecolor="k", linewidth=0.5, label="Raw")
        ax.bar(x + width/2, [d["refined_rms"] for d in bar_data], width,
               color=[d["color"] for d in bar_data], alpha=1.0,
               edgecolor="k", linewidth=0.5, label="Refined")
        ax.set_xticks(x)
        ax.set_xticklabels([d["sign"] for d in bar_data],
                           fontsize=8, rotation=45, ha="right")
        ax.legend(fontsize=9)
    ax.set_ylabel("RMS Spot (μm)", fontsize=11)
    ax.set_title("Elite Quality: Raw vs Refined", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 5: Restarts over time ──────────────────────────────────
    ax = axes[1, 1]
    for key in ["seidel", "random", "single"]:
        df = trajs[key]
        if "restarts" in df.columns:
            ax.plot(df["evals"] / 1000, df["restarts"],
                    color=COLORS[key], linewidth=2, label=LABELS[key])
    ax.set_xlabel("Evaluations (×1000)", fontsize=11)
    ax.set_ylabel("Cumulative Restarts", fontsize=11)
    ax.set_title("CMA Restarts (Search Instability)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 6: Summary table ───────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")

    table_data = []
    for key in ["seidel", "random", "single"]:
        df = trajs[key]
        el = elites[key]
        best_raw = df["best_fitness"].max()
        final_coverage = df["coverage"].iloc[-1] * 100
        total_evals = df["evals"].iloc[-1]
        n_elites = len(el)
        restarts = int(df["restarts"].iloc[-1]) if "restarts" in df.columns else 0

        # Best refined RMS
        best_ref_rms = None
        for e in el:
            ref = e.get("refined", {})
            if "rms_um" in ref:
                rms = float(ref["rms_um"])
                if best_ref_rms is None or rms < best_ref_rms:
                    best_ref_rms = rms

        table_data.append([
            LABELS[key],
            f"{best_raw:.4f}",
            f"{best_ref_rms:.1f} μm" if best_ref_rms else "—",
            f"{final_coverage:.1f}%",
            str(n_elites),
            str(restarts),
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=["Condition", "Best Raw\nFitness", "Best Refined\nRMS", "Coverage", "Elites", "Restarts"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Color the condition cells
    for i, key in enumerate(["seidel", "random", "single"]):
        table[i + 1, 0].set_facecolor(COLORS[key])
        table[i + 1, 0].set_text_props(color="white", fontweight="bold")

    fig.suptitle(
        "Ablation Study: Seidel BCs vs Random BCs vs Single-Objective\n"
        f"(Cooke Triplet with Materials, {int(trajs['seidel']['evals'].iloc[-1]/1000)}k evals)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = out_dir / "ablation_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # ── Bonus: per-condition archive maps side by side ────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6))
    for i, key in enumerate(["seidel", "random", "single"]):
        ax = axes2[i]
        archive = load_archive(dirs[key])
        cells = archive.cells if hasattr(archive, "cells") else archive._cells

        if len(cells) == 0:
            ax.text(0.5, 0.5, "Empty", ha="center", va="center", fontsize=14)
            ax.set_title(LABELS[key])
            continue

        m0 = [np.array(r.measures)[0] for r in cells.values()]
        m1 = [np.array(r.measures)[1] for r in cells.values()]
        fits = [r.fitness for r in cells.values()]

        sc = ax.scatter(m0, m1, c=fits, cmap="viridis", s=3, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="fitness", shrink=0.8)
        ax.set_title(f"{LABELS[key]}\n({len(cells)} cells)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Descriptor axis 0", fontsize=10)
        ax.set_ylabel("Descriptor axis 1", fontsize=10)

    fig2.suptitle("Archive Maps by Condition", fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    path2 = out_dir / "ablation_archive_maps.png"
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {path2}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seidel_dir", type=Path, help="Seidel S2+S3 run directory")
    p.add_argument("random_dir", type=Path, help="Random BC run directory")
    p.add_argument("single_dir", type=Path, help="Single-objective run directory")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: next to seidel_dir)")
    args = p.parse_args()

    dirs = {
        "seidel": args.seidel_dir,
        "random": args.random_dir,
        "single": args.single_dir,
    }

    for key, d in dirs.items():
        if not d.exists():
            print(f"ERROR: {d} does not exist")
            return
        if not (d / "trajectory.csv").exists():
            print(f"ERROR: {d}/trajectory.csv not found")
            return

    out_dir = args.out or args.seidel_dir.parent / "ablation_comparison"
    plot_comparison(dirs, out_dir)
    print(f"\nDone! Results in {out_dir}")


if __name__ == "__main__":
    main()
