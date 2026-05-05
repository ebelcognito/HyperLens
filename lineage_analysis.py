"""Reconstruct per-elite lineages from a LensMEGA run and plot how the
five Seidel coefficients evolve along each lineage.

Usage
-----

    python lineage_analysis.py <run_dir> [--k 3] [--stride 5] \
                               [--render-lenses] [--out <dir>]

What it does
------------

1. Load ``<run_dir>/lineage.jsonl`` (per-emission log) and
   ``<run_dir>/archive.pkl`` (final archive).
2. Among the current archive elites, pick *K* geometrically distinct
   high-fitness exemplars (k-means on theta, best-fitness member per
   cluster).
3. For each exemplar, walk the ``parent_theta_hash`` chain back to the
   root (a seed/calibration sample).
4. For each lineage, produce one composite figure showing:
   - fitness vs. generation
   - |S1|..|S5| magnitudes vs. generation (the core ask)
   - trajectory in the BC space (S2, S3)
5. Optionally render the lens at a few points along the lineage
   (``--render-lenses``).

Outputs are written to ``<run_dir>/lineage_analysis/`` by default.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


SEIDEL_LABELS = ("S1", "S2", "S3", "S4", "S5")


# ======================================================================
# Loading
# ======================================================================

@dataclass
class LineageRow:
    source: str
    gen: int
    emission: int
    parent_hash: Optional[str]
    theta_hash: str
    theta: np.ndarray
    fitness: Optional[float]
    rejected: bool
    insert_status: int
    bc_cell: Optional[Tuple[int, ...]]
    measures: np.ndarray
    seidel_all: Optional[np.ndarray]  # shape (5,) or None

    @classmethod
    def from_json(cls, d: dict) -> "LineageRow":
        return cls(
            source=d["source"],
            gen=int(d["gen"]),
            emission=int(d["emission"]),
            parent_hash=d.get("parent_theta_hash"),
            theta_hash=str(d["theta_hash"]),
            theta=np.asarray(d["theta"], dtype=np.float64),
            fitness=(None if d.get("fitness") is None else float(d["fitness"])),
            rejected=bool(d.get("rejected", False)),
            insert_status=int(d.get("insert_status", 0)),
            bc_cell=(tuple(d["bc_cell"]) if d.get("bc_cell") is not None else None),
            measures=np.asarray(d.get("measures", []), dtype=np.float64),
            seidel_all=(
                np.asarray(d["seidel_all"], dtype=np.float64)
                if d.get("seidel_all") is not None
                else None
            ),
        )


def load_lineage(path: Path) -> Tuple[List[LineageRow], Dict[str, LineageRow]]:
    rows: List[LineageRow] = []
    by_hash: Dict[str, LineageRow] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = LineageRow.from_json(d)
            rows.append(row)
            # Keep the FIRST occurrence of a theta_hash; duplicate hashes can
            # happen if the same theta is re-evaluated (rare, but possible
            # because project_theta is deterministic).
            by_hash.setdefault(row.theta_hash, row)
    return rows, by_hash


def load_archive(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


# ======================================================================
# Exemplar selection
# ======================================================================

def _kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    """Tiny k-means (labels only). Works without sklearn."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if k >= n:
        return np.arange(n)
    # k-means++ init
    first = int(rng.integers(0, n))
    centers = [X[first]]
    for _ in range(1, k):
        dists = np.min(
            np.stack([np.sum((X - c) ** 2, axis=1) for c in centers], axis=1),
            axis=1,
        )
        probs = dists / (dists.sum() + 1e-12)
        idx = int(rng.choice(n, p=probs))
        centers.append(X[idx])
    C = np.stack(centers)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                C[j] = X[mask].mean(axis=0)
    return labels


def pick_exemplars(
    archive,
    by_hash: Dict[str, LineageRow],
    *,
    k: int,
    min_fitness: Optional[float] = None,
    seed: int = 0,
) -> List[LineageRow]:
    """Pick *k* geometrically distinct exemplars among archive elites.

    Uses k-means on normalized theta vectors; within each cluster, picks
    the member with highest fitness that also exists in the lineage log.
    """
    elites: List[Tuple[np.ndarray, float]] = []
    for _, r in archive.cells.items():
        if min_fitness is not None and float(r.fitness) < min_fitness:
            continue
        elites.append((np.asarray(r.theta, dtype=np.float64), float(r.fitness)))
    if len(elites) == 0:
        raise RuntimeError("archive is empty (or all elites filtered out)")

    thetas = np.stack([e[0] for e in elites])
    fits = np.array([e[1] for e in elites])

    # Normalize each theta dimension to [0, 1] by its empirical range.
    rng_per_dim = thetas.max(axis=0) - thetas.min(axis=0) + 1e-9
    X = (thetas - thetas.min(axis=0)) / rng_per_dim

    labels = _kmeans(X, k=k, seed=seed)

    exemplars: List[LineageRow] = []
    for j in range(k):
        mask = labels == j
        if not mask.any():
            continue
        idx_cluster = np.where(mask)[0]
        # Sort cluster members by fitness, highest first.
        order = idx_cluster[np.argsort(-fits[idx_cluster])]
        picked: Optional[LineageRow] = None
        for idx in order:
            theta = thetas[idx]
            h = _theta_hash(theta)
            if h in by_hash:
                picked = by_hash[h]
                break
        if picked is None:
            print(
                f"[exemplars] cluster {j}: no lineage row for any member; skipping"
            )
            continue
        exemplars.append(picked)
    return exemplars


def _theta_hash(theta: np.ndarray) -> str:
    import hashlib
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


# ======================================================================
# Lineage walking
# ======================================================================

def walk_ancestry(
    start: LineageRow, by_hash: Dict[str, LineageRow], *, max_depth: int = 10000
) -> List[LineageRow]:
    """Return the ancestry chain root-first: [root, ..., parent, start]."""
    chain = [start]
    current = start
    for _ in range(max_depth):
        if current.parent_hash is None:
            break
        parent = by_hash.get(current.parent_hash)
        if parent is None:
            break
        chain.append(parent)
        current = parent
    chain.reverse()
    return chain


# ======================================================================
# Plotting
# ======================================================================

def plot_lineage(
    chain: List[LineageRow],
    *,
    title: str,
    out_path: Path,
    stride: int = 1,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sub-sample: keep every `stride`-th ancestor, but always include the
    # first and last.
    if stride > 1 and len(chain) > 2:
        idxs = list(range(0, len(chain), stride))
        if (len(chain) - 1) not in idxs:
            idxs.append(len(chain) - 1)
        chain = [chain[i] for i in idxs]

    gens = np.array([r.gen for r in chain])
    fits = np.array(
        [r.fitness if r.fitness is not None else np.nan for r in chain]
    )
    seidel_mat = []
    for r in chain:
        if r.seidel_all is None or r.seidel_all.size != 5:
            seidel_mat.append(np.full(5, np.nan))
        else:
            seidel_mat.append(np.abs(r.seidel_all))
    seidel_mat = np.stack(seidel_mat)  # shape (T, 5)

    # BC path (measures are S2, S3 in this problem).
    meas = np.stack(
        [
            (r.measures if r.measures.size else np.full(2, np.nan))
            for r in chain
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: fitness
    axes[0].plot(gens, fits, "-o", color="#222", ms=3)
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("fitness")
    axes[0].set_title("Fitness along lineage")
    axes[0].grid(alpha=0.3)

    # Panel 2: |Sk| magnitudes
    colors = ["#D62728", "#1F77B4", "#2CA02C", "#FF7F0E", "#9467BD"]
    for j, label in enumerate(SEIDEL_LABELS):
        axes[1].plot(
            gens, seidel_mat[:, j], "-", color=colors[j], label=f"|{label}|", lw=1.5
        )
    axes[1].set_xlabel("generation")
    axes[1].set_ylabel("|Sₖ|")
    axes[1].set_yscale("log")
    axes[1].set_title("Seidel magnitudes along lineage")
    axes[1].grid(which="both", alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    # Panel 3: BC path (S2 vs S3)
    sc = axes[2].scatter(
        meas[:, 0], meas[:, 1], c=gens, cmap="viridis", s=22
    )
    axes[2].plot(meas[:, 0], meas[:, 1], "-", color="#888", lw=0.8, alpha=0.6)
    axes[2].plot(meas[0, 0], meas[0, 1], "s", color="red", ms=9, label="root")
    axes[2].plot(meas[-1, 0], meas[-1, 1], "*", color="gold", ms=14,
                 markeredgecolor="black", label="final")
    axes[2].set_xlabel("S2 (coma)")
    axes[2].set_ylabel("S3 (astigmatism)")
    axes[2].set_title("BC-space path")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="best", fontsize=8)
    fig.colorbar(sc, ax=axes[2], shrink=0.85).set_label("generation")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_overlay(
    chains: List[List[LineageRow]],
    labels: Sequence[str],
    *,
    out_path: Path,
) -> None:
    """One figure comparing all K exemplars overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10.colors

    # Panel 1: fitness curves overlaid.
    for i, chain in enumerate(chains):
        gens = np.array([r.gen for r in chain])
        fits = np.array(
            [r.fitness if r.fitness is not None else np.nan for r in chain]
        )
        axes[0].plot(gens, fits, "-", color=colors[i % len(colors)],
                     label=labels[i], lw=1.2)
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("fitness")
    axes[0].set_title("Fitness trajectories — all exemplars")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    # Panel 2: BC paths overlaid.
    for i, chain in enumerate(chains):
        meas = np.stack(
            [
                (r.measures if r.measures.size else np.full(2, np.nan))
                for r in chain
            ]
        )
        axes[1].plot(meas[:, 0], meas[:, 1], "-", color=colors[i % len(colors)],
                     label=labels[i], lw=1.0, alpha=0.8)
        axes[1].plot(meas[-1, 0], meas[-1, 1], "*",
                     color=colors[i % len(colors)], ms=12,
                     markeredgecolor="black")
    axes[1].set_xlabel("S2 (coma)")
    axes[1].set_ylabel("S3 (astigmatism)")
    axes[1].set_title("BC-space paths — all exemplars")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="LensMEGA run directory")
    ap.add_argument(
        "--k", type=int, default=3, help="number of exemplar lineages (default 3)"
    )
    ap.add_argument(
        "--stride", type=int, default=1,
        help="plot every Nth ancestor (default 1)"
    )
    ap.add_argument(
        "--min-fitness", type=float, default=None,
        help="only consider archive elites with fitness >= this value",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="k-means seed for exemplar clustering",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="output directory (default: <run_dir>/lineage_analysis)",
    )
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    lineage_path = run_dir / "lineage.jsonl"
    archive_path = run_dir / "archive.pkl"
    if not lineage_path.exists():
        sys.exit(f"missing lineage log: {lineage_path}")
    if not archive_path.exists():
        sys.exit(f"missing archive pickle: {archive_path}")

    out_dir: Path = args.out or (run_dir / "lineage_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[lineage] loading {lineage_path} ...", flush=True)
    rows, by_hash = load_lineage(lineage_path)
    print(
        f"[lineage] {len(rows)} rows, {len(by_hash)} unique theta hashes",
        flush=True,
    )

    archive = load_archive(archive_path)
    print(f"[lineage] archive: {len(archive.cells)} elites", flush=True)

    exemplars = pick_exemplars(
        archive, by_hash,
        k=args.k,
        min_fitness=args.min_fitness,
        seed=args.seed,
    )
    print(f"[lineage] picked {len(exemplars)} exemplars", flush=True)
    if not exemplars:
        sys.exit("no exemplars found")

    chains: List[List[LineageRow]] = []
    labels: List[str] = []
    for i, ex in enumerate(exemplars):
        chain = walk_ancestry(ex, by_hash)
        chains.append(chain)
        fit_str = f"fit={ex.fitness:.4f}" if ex.fitness is not None else "fit=?"
        label = (
            f"E{i+1} | depth={len(chain)} | {fit_str} | "
            f"hash={ex.theta_hash[:8]}"
        )
        labels.append(label)
        print(f"  [{i+1}] {label}", flush=True)

        out_png = out_dir / f"exemplar_{i+1}_{ex.theta_hash[:8]}.png"
        plot_lineage(chain, title=label, out_path=out_png, stride=args.stride)
        print(f"    wrote {out_png}", flush=True)

        # Also dump the chain as JSON for downstream analysis.
        chain_json = out_dir / f"exemplar_{i+1}_{ex.theta_hash[:8]}.json"
        with chain_json.open("w") as f:
            json.dump(
                [
                    {
                        "gen": r.gen,
                        "emission": r.emission,
                        "theta_hash": r.theta_hash,
                        "parent_hash": r.parent_hash,
                        "fitness": r.fitness,
                        "measures": r.measures.tolist(),
                        "seidel_all": (
                            r.seidel_all.tolist()
                            if r.seidel_all is not None else None
                        ),
                    }
                    for r in chain
                ],
                f,
                indent=2,
            )

    overlay_out = out_dir / "overlay.png"
    plot_overlay(chains, labels, out_path=overlay_out)
    print(f"[lineage] wrote overlay: {overlay_out}", flush=True)


if __name__ == "__main__":
    main()
