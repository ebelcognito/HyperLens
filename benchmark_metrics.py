"""Unified quality + diversity metrics for comparing lens-design methods.

Every method's output is summarised as a *design set*: a JSON file with

    {
      "method": "LensMEGA",
      "n_evaluations": 50000,
      "wall_seconds": 1234.5,            (optional)
      "designs": [
        {"theta": [...], "fitness": ..., "metrics": {...}},
        ...
      ]
    }

This script loads one or more such files and computes comparable metrics
across methods:

  - best_fitness
  - n_elites (fitness thresholds)
  - n_families (curvature sign pattern)
  - n_families_clustered (theta-space k-medoids with silhouette cutoff)
  - pareto front of (family_id, best_fitness_within_family)

Usage:

    python benchmark_metrics.py --num-curvatures 6 \
                                lensmega=run_dir/archive_cells.json \
                                lbfgs=/tmp/m4.json \
                                nsga2=/tmp/m3.json

Special syntax: if the value ends in ``archive_cells.json``, the script
auto-adapts the LensMEGA archive schema into the design-set schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def _load_lensmega_refined(run_dir: Path, default_name: str) -> dict:
    """Load LensMEGA's actual outputs: each elite_designs/*.json has been
    post-refined by L-BFGS; use its ``refined`` dict as the real design.

    Prefers refined.fitness / refined.rms_mm; falls back to raw.
    """
    import glob
    elite_files = sorted(glob.glob(str(run_dir / "elite_designs" / "*.json")))
    designs = []
    for ef in elite_files:
        try:
            d = json.load(open(ef))
        except Exception:
            continue
        refined = d.get("refined") or {}
        # prefer refined values
        fit = refined.get("fitness")
        rms = refined.get("rms_mm")
        theta = refined.get("theta") or d.get("theta")
        # Build metrics dict mimicking the archive schema
        metrics = {
            "rms_spot": float(rms) if rms is not None else (
                d.get("rms_mm") if d.get("rms_mm") is not None else None
            ),
            "invalid_fraction": d.get("invalid_fraction", 0.0),
        }
        if refined.get("efl_mm") is not None:
            metrics["efl_actual"] = refined["efl_mm"]
        designs.append({
            "theta": theta,
            "fitness": float(fit) if fit is not None else d.get("fitness"),
            "metrics": metrics,
            "is_refined": refined.get("converged", False),
        })
    # n_evals from trajectory.csv
    n_eval = None
    traj = run_dir / "trajectory.csv"
    if traj.exists():
        import csv
        with traj.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            try:
                n_eval = int(rows[-1]["evals"])
            except (KeyError, ValueError):
                n_eval = None
    return {
        "name": default_name,
        "n_evaluations": n_eval,
        "designs": designs,
    }


def _load_method(path: Path, default_name: str) -> dict:
    """Return {name, n_evaluations, designs}.  Adapts archive_cells.json or
    points to a run directory (in which case loads refined elite_designs).
    """
    # Directory? Load refined elite_designs.
    if path.is_dir():
        return _load_lensmega_refined(path, default_name)
    with path.open() as f:
        payload = json.load(f)
    # archive_cells.json schema from LensMEGA?
    if "cells" in payload:
        designs = []
        for c in payload["cells"]:
            designs.append({
                "theta": c["theta"],
                "fitness": c["fitness"],
                "metrics": c.get("metrics", {}),
            })
        n_eval = payload.get("stats", {}).get("total_evaluations")
        # Fall back to trajectory.csv in the same directory.
        if n_eval is None:
            traj = path.parent / "trajectory.csv"
            if traj.exists():
                import csv
                with traj.open() as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    try:
                        n_eval = int(rows[-1]["evals"])
                    except (KeyError, ValueError):
                        n_eval = None
        return {
            "name": default_name,
            "n_evaluations": int(n_eval) if n_eval else None,
            "designs": designs,
        }
    # Otherwise assume our own method-output schema.
    payload.setdefault("name", default_name)
    return payload


# ----------------------------------------------------------------------
# Metric utilities
# ----------------------------------------------------------------------

def _filter_invalid(designs: List[dict], max_inv: Optional[float],
                    min_rms_mm: Optional[float]) -> List[dict]:
    out = []
    for d in designs:
        m = d.get("metrics", {})
        if max_inv is not None and m.get("invalid_fraction", 0.0) > max_inv:
            continue
        if min_rms_mm is not None and m.get("rms_spot", 0.0) < min_rms_mm:
            continue
        out.append(d)
    return out


def _sign_pattern(theta: List[float], num_curvatures: int) -> str:
    signs = []
    for v in theta[:num_curvatures]:
        signs.append("+" if float(v) >= 0 else "-")
    return "".join(signs)


def _count_families_signs(
    designs: List[dict],
    *,
    num_curvatures: int,
    fitness_threshold: float,
) -> Tuple[int, Dict[str, dict]]:
    """Count curvature-sign patterns where at least one design passes the
    fitness threshold.  Returns (count, per-family summary)."""
    by_pattern: Dict[str, List[dict]] = defaultdict(list)
    for d in designs:
        p = _sign_pattern(d["theta"], num_curvatures)
        by_pattern[p].append(d)
    summary: Dict[str, dict] = {}
    good = 0
    for pat, ds in by_pattern.items():
        fits = np.array([d["fitness"] for d in ds])
        best = float(fits.max())
        summary[pat] = {
            "n_designs": len(ds),
            "best_fitness": best,
            "n_above_threshold": int((fits >= fitness_threshold).sum()),
            "above_threshold": best >= fitness_threshold,
        }
        if best >= fitness_threshold:
            good += 1
    return good, summary


def _cluster_families_theta(
    designs: List[dict],
    *,
    k_range: Tuple[int, int] = (2, 12),
    fitness_threshold: float,
) -> Tuple[int, dict]:
    """k-medoids-style clustering on normalised θ; pick k by silhouette.
    Count clusters with at least one design above fitness threshold."""
    good = [d for d in designs if d["fitness"] >= fitness_threshold]
    if len(good) < max(k_range[0], 3):
        return 0, {"note": "too few good designs to cluster"}
    thetas = np.array([d["theta"] for d in good], dtype=np.float64)
    # Normalize per-dim.
    lo = thetas.min(axis=0, keepdims=True)
    hi = thetas.max(axis=0, keepdims=True)
    rng = np.clip(hi - lo, 1e-12, None)
    X = (thetas - lo) / rng

    best = None
    for k in range(k_range[0], min(k_range[1] + 1, X.shape[0])):
        labels = _kmeans(X, k, seed=0)
        sil = _silhouette(X, labels)
        if best is None or sil > best[1]:
            best = (k, sil, labels)
    if best is None:
        return 0, {"note": "clustering failed"}
    k, sil, labels = best
    return int(k), {"k": int(k), "silhouette": float(sil)}


def _kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 80
            ) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx0 = int(rng.integers(0, n))
    centers = [X[idx0]]
    for _ in range(k - 1):
        d2 = np.min(
            np.stack([np.sum((X - c) ** 2, axis=1) for c in centers], axis=1),
            axis=1,
        )
        p = d2 / (d2.sum() + 1e-12)
        centers.append(X[int(rng.choice(n, p=p))])
    C = np.stack(centers)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(axis=0)
    return labels


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    # Basic silhouette on small sets.  Fine for n < 500.
    n = X.shape[0]
    if n > 1500:
        idx = np.random.default_rng(0).choice(n, 1500, replace=False)
        X = X[idx]; labels = labels[idx]; n = 1500
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    sils = []
    for i in range(n):
        same = labels == labels[i]
        if same.sum() <= 1:
            continue
        a = d[i, same].sum() / (same.sum() - 1)
        b_candidates = []
        for lab in np.unique(labels):
            if lab == labels[i]:
                continue
            other = labels == lab
            if not other.any():
                continue
            b_candidates.append(d[i, other].mean())
        if not b_candidates:
            continue
        b = min(b_candidates)
        sils.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(sils)) if sils else 0.0


# ----------------------------------------------------------------------
# Per-method summary
# ----------------------------------------------------------------------

def summarise(
    method: dict,
    *,
    num_curvatures: int,
    thresholds: List[float],
    primary_threshold: float,
    max_invalid_frac: Optional[float],
    min_rms_mm: Optional[float],
) -> dict:
    name = method["name"]
    designs = method["designs"]
    n_evals = method.get("n_evaluations")
    all_n = len(designs)
    filtered = _filter_invalid(designs, max_invalid_frac, min_rms_mm)
    fits = np.array([d["fitness"] for d in filtered]) if filtered else np.array([])
    best = float(fits.max()) if fits.size else float("-inf")

    counts = {f"n_fit_gt_{t:+g}": int((fits >= t).sum()) for t in thresholds}
    n_sign, family_summary = _count_families_signs(
        filtered, num_curvatures=num_curvatures,
        fitness_threshold=primary_threshold,
    )
    n_cluster, cluster_info = _cluster_families_theta(
        filtered, fitness_threshold=primary_threshold,
    )

    # Useful-efficiency: good elites / evaluations.
    eff = (counts.get(f"n_fit_gt_{primary_threshold:+g}", 0) / max(n_evals, 1)
           ) if n_evals else None

    return {
        "name": name,
        "n_designs_total": all_n,
        "n_designs_filtered": len(filtered),
        "n_evaluations": n_evals,
        "best_fitness": best,
        **counts,
        "n_families_signs": n_sign,
        "n_families_clustered": n_cluster,
        "cluster_info": cluster_info,
        "useful_efficiency": eff,
        "family_summary": family_summary,
    }


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_comparison(summaries: List[dict], primary_threshold: float,
                    out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(summaries)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = [s["name"] for s in summaries]
    xs = np.arange(n)
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    # (0, 0): best fitness
    bests = [s["best_fitness"] for s in summaries]
    axes[0, 0].bar(xs, bests, color=colors)
    for x, v in zip(xs, bests):
        axes[0, 0].text(x, v, f"{v:+.3f}", ha="center", va="bottom", fontsize=9)
    axes[0, 0].set_xticks(xs)
    axes[0, 0].set_xticklabels(names, rotation=15, fontsize=9)
    axes[0, 0].set_ylabel("best fitness (higher = better)")
    axes[0, 0].set_title("Best fitness")
    axes[0, 0].grid(axis="y", alpha=0.3)

    # (0, 1): # good elites at various thresholds
    thr_keys = [k for k in summaries[0] if k.startswith("n_fit_gt_")]
    thr_values = sorted(
        [(k, float(k.replace("n_fit_gt_", ""))) for k in thr_keys],
        key=lambda x: x[1], reverse=True,
    )
    w = 0.8 / len(thr_values)
    for t_i, (k, v) in enumerate(thr_values):
        vals = [s[k] for s in summaries]
        axes[0, 1].bar(xs + (t_i - (len(thr_values)-1) / 2) * w, vals, w,
                       label=f"fit ≥ {v:+g}", alpha=0.85)
    axes[0, 1].set_xticks(xs)
    axes[0, 1].set_xticklabels(names, rotation=15, fontsize=9)
    axes[0, 1].set_title("# elites above threshold")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(axis="y", alpha=0.3)

    # (1, 0): # families discovered
    signs = [s["n_families_signs"] for s in summaries]
    clus = [s["n_families_clustered"] for s in summaries]
    w = 0.4
    axes[1, 0].bar(xs - w/2, signs, w, color=colors, alpha=0.85,
                   label="sign-pattern families")
    axes[1, 0].bar(xs + w/2, clus, w, color=colors, alpha=0.5,
                   label="θ-clustered families",
                   hatch="///")
    for x, v in zip(xs - w/2, signs):
        axes[1, 0].text(x, v, str(v), ha="center", va="bottom", fontsize=9)
    for x, v in zip(xs + w/2, clus):
        axes[1, 0].text(x, v, str(v), ha="center", va="bottom", fontsize=9)
    axes[1, 0].set_xticks(xs)
    axes[1, 0].set_xticklabels(names, rotation=15, fontsize=9)
    axes[1, 0].set_ylabel("# families")
    axes[1, 0].set_title(f"# distinct families (fit ≥ {primary_threshold:+g})")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.3)

    # (1, 1): useful-efficiency
    effs = [s.get("useful_efficiency") or 0 for s in summaries]
    axes[1, 1].bar(xs, effs, color=colors)
    for x, v in zip(xs, effs):
        if v > 0:
            axes[1, 1].text(x, v, f"{v:.2e}", ha="center", va="bottom",
                            fontsize=9)
    axes[1, 1].set_xticks(xs)
    axes[1, 1].set_xticklabels(names, rotation=15, fontsize=9)
    axes[1, 1].set_ylabel(f"# (fit ≥ {primary_threshold:+g}) / evaluations")
    axes[1, 1].set_title("Useful efficiency")
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Benchmark comparison  (fit threshold = {primary_threshold:+g})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("methods", nargs="+", help="name=path pairs")
    ap.add_argument("--num-curvatures", type=int, default=6)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[-0.5, -0.2, -0.1, -0.05])
    ap.add_argument("--primary-threshold", type=float, default=-0.2)
    ap.add_argument("--max-invalid-frac", type=float, default=0.1)
    ap.add_argument("--min-rms-mm", type=float, default=0.001)
    ap.add_argument("--out", type=Path, default=Path("/tmp/benchmark"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for spec in args.methods:
        if "=" not in spec:
            sys.exit(f"expected name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        method = _load_method(Path(path), default_name=name)
        method["name"] = name
        s = summarise(
            method,
            num_curvatures=args.num_curvatures,
            thresholds=args.thresholds,
            primary_threshold=args.primary_threshold,
            max_invalid_frac=args.max_invalid_frac,
            min_rms_mm=args.min_rms_mm,
        )
        summaries.append(s)
        print(f"\n=== {name} ===")
        for k, v in s.items():
            if k in ("family_summary", "cluster_info"):
                continue
            print(f"  {k}: {v}")

    # Save summaries + plot.
    with (args.out / "summaries.json").open("w") as f:
        json.dump(summaries, f, indent=2)
    plot_comparison(summaries, args.primary_threshold,
                    args.out / "comparison.png")
    print(f"\n[benchmark] wrote {args.out / 'comparison.png'}")
    print(f"[benchmark] wrote {args.out / 'summaries.json'}")


if __name__ == "__main__":
    main()
