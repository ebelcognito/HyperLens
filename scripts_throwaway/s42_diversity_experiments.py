#!/usr/bin/env python
"""Experiments 1-5 for supplementary §S4.2 on the diversity metric.

E1: PCA explained variance on parameter vectors (min-max + z-score).
E2: 2D PCA scatter + hull, HL vs SA+GA, count-controlled.
E3: pairwise behavior-distance vs parameter-distance scatter + corrs.
E4: ε-sweep N_ε(ε) for both methods, both topologies.
E5: sanity: reproduce Table-3-style ε-distinct counts; verify ε-separation.

All experiments share one normalization (min-max from the canonical
champion config), one HL pool per topology (refined_top500_seed42_hl
single-run), one SA+GA pool per topology (refined_above_-1.0).
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("MPMATH_NOGMPY", "1")

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import ConvexHull

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DeepLens"))

OUT = ROOT / "figures" / "s42_experiments"
OUT.mkdir(parents=True, exist_ok=True)

POOLS = {
    "sixlet": dict(
        bounds_cfg=ROOT / "experiments/gradient_alignment/sixlet_apsc_champion_config.yaml",
        hl=ROOT / "results/lensmega/sixlet_apsc_discovery_v2_seed42_20260506_034601/refined_top500_seed42_hl.json",
        saga=ROOT / "results/diversity_run_20260506_033447/sixlet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json",
    ),
    "octuplet": dict(
        bounds_cfg=ROOT / "experiments/gradient_alignment/octuplet_apsc_champion_config.yaml",
        hl=ROOT / "results/lensmega/octuplet_apsc_2cement_v2_seed42_20260506_040025/refined_top500_seed42_hl.json",
        saga=ROOT / "results/diversity_run_20260506_033447/octuplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json",
    ),
}
EPS_TABLE3 = (0.20, 0.30)
RMS_CAPS = (None, 30.0, 20.0, 10.0)


# ============================================================
# Loaders
# ============================================================

def load_bounds(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text())
    return np.asarray(cfg["problem"]["theta_bounds"], dtype=np.float64)


def load_pool(json_path: Path) -> list[dict]:
    """Returns list of {theta, rms_um, fitness, S2, S3 (or None)}."""
    data = json.loads(json_path.read_text())
    designs = data["designs"] if isinstance(data, dict) else data
    out = []
    for des in designs:
        th = des.get("theta")
        if th is None:
            continue
        m = des.get("metrics") or {}
        s2 = m.get("seidel_S2", des.get("seidel_S2"))
        s3 = m.get("seidel_S3", des.get("seidel_S3"))
        out.append({
            "theta": np.asarray(th, dtype=np.float64),
            "rms_um": float(des.get("rms_um", des.get("refined_rms_um", float("nan")))),
            "refined_fitness": float(des.get("refined_fitness", des.get("fitness", 0.0))),
            "S2": float(s2) if s2 is not None else None,
            "S3": float(s3) if s3 is not None else None,
        })
    return out


def normalize_minmax(thetas: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    return (thetas - lo) / span


def zscore(thetas: np.ndarray) -> np.ndarray:
    mu = thetas.mean(axis=0)
    sd = thetas.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (thetas - mu) / sd


# ============================================================
# E5: Sanity (reproduce Table 3 + check ε-separation)
# ============================================================

def eps_net(thetas_norm: np.ndarray, fitness: np.ndarray, eps: float,
             sort_by_rms: np.ndarray | None = None):
    """User-spec procedure: sort ascending by RMS (best first), greedy
    accept if min dist > eps (dist normalized by sqrt(D))."""
    D = thetas_norm.shape[1]
    sqrtD = np.sqrt(D)
    if sort_by_rms is not None:
        order = np.argsort(sort_by_rms)
    else:
        order = np.argsort(-fitness)
    selected_u = []
    selected_i = []
    for i in order:
        u = thetas_norm[i]
        if not selected_u:
            selected_u.append(u); selected_i.append(int(i)); continue
        d = min(np.linalg.norm(u - su) / sqrtD for su in selected_u)
        if d > eps:
            selected_u.append(u); selected_i.append(int(i))
    return selected_i


def pairwise_min_dist(thetas_norm: np.ndarray) -> float:
    if len(thetas_norm) < 2:
        return float("inf")
    D = thetas_norm.shape[1]
    sqrtD = np.sqrt(D)
    diffs = thetas_norm[:, None, :] - thetas_norm[None, :, :]
    dist = np.linalg.norm(diffs, axis=2) / sqrtD
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def e5_sanity(system: str, hl_pool, bounds: np.ndarray, results: dict):
    print(f"\n[E5] {system}: reproduce Table-3-style ε-distinct counts", flush=True)
    thetas = np.stack([d["theta"] for d in hl_pool])
    rms = np.array([d["rms_um"] for d in hl_pool])
    fits = np.array([d["refined_fitness"] for d in hl_pool])
    thetas_n = normalize_minmax(thetas, bounds)

    rows = []
    for cap_um in RMS_CAPS:
        if cap_um is None:
            mask = np.ones(len(hl_pool), dtype=bool)
            label = "all"
        else:
            mask = rms < cap_um
            label = f"sub-{int(cap_um)}µm"
        idx = np.where(mask)[0]
        for eps in EPS_TABLE3:
            picks = eps_net(thetas_n[idx], fits[idx], eps,
                            sort_by_rms=rms[idx])
            sel_u = thetas_n[idx][picks]
            min_d = pairwise_min_dist(sel_u)
            rows.append({"rms_cap": label, "eps": eps, "n_pool": int(mask.sum()),
                         "n_eps_distinct": len(picks), "min_pairwise_dist": min_d,
                         "separation_ok": bool(min_d > eps or len(picks) <= 1)})
    for r in rows:
        ok = "✓" if r["separation_ok"] else "✗"
        print(f"  {ok} {r['rms_cap']:>10s}  ε={r['eps']:.2f}  pool={r['n_pool']:>4d}  "
              f"N_ε={r['n_eps_distinct']:>3d}  min_pair_dist={r['min_pairwise_dist']:.3f}",
              flush=True)
    results.setdefault("E5", {})[system] = rows


# ============================================================
# E1: PCA explained variance
# ============================================================

def pca_cumvar(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    # SVD-based PCA for stability.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / (Xc.shape[0] - 1)
    cum = np.cumsum(var) / var.sum()
    return cum


def n_components_for(cumvar: np.ndarray, frac: float) -> int:
    return int(np.searchsorted(cumvar, frac) + 1)


def e1_pca_variance(system: str, hl_pool, bounds: np.ndarray, results: dict):
    print(f"\n[E1] {system}: PCA explained variance on θ", flush=True)
    thetas = np.stack([d["theta"] for d in hl_pool])
    out = {"n_designs": len(hl_pool), "dim": int(thetas.shape[1])}
    for name, X in (("minmax", normalize_minmax(thetas, bounds)),
                     ("zscore", zscore(thetas))):
        cum = pca_cumvar(X)
        out[name] = {
            "cum_pc1": float(cum[0]),
            "cum_pc2": float(cum[1]) if len(cum) > 1 else 1.0,
            "cum_pc3": float(cum[2]) if len(cum) > 2 else 1.0,
            "n_for_90pct": n_components_for(cum, 0.90),
            "n_for_95pct": n_components_for(cum, 0.95),
        }
        print(f"  {name}: PC1={cum[0]*100:.1f}%  cum2={cum[1]*100:.1f}%  "
              f"cum3={cum[2]*100:.1f}%  k90%={out[name]['n_for_90pct']}  "
              f"k95%={out[name]['n_for_95pct']}", flush=True)
    results.setdefault("E1", {})[system] = out
    # Scree.
    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor="white")
    for name, color in (("minmax", "#3b5b8a"), ("zscore", "#c97a3d")):
        X = normalize_minmax(thetas, bounds) if name == "minmax" else zscore(thetas)
        cum = pca_cumvar(X)
        ax.plot(np.arange(1, len(cum) + 1), cum * 100, marker="o", ms=3,
                color=color, label=f"{name}")
    ax.axhline(90, color="#666", ls="--", lw=0.6)
    ax.set_xlabel("# principal components")
    ax.set_ylabel("cumulative explained variance (%)")
    ax.set_title(f"E1 PCA scree ({system}, n={len(hl_pool)}, D={thetas.shape[1]})")
    ax.grid(True, lw=0.4, color="#dddddd")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(OUT / f"e1_scree_{system}.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ============================================================
# E2: 2D PCA scatter + hull
# ============================================================

def hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)  # 2D: volume == area
    except Exception:
        return 0.0


def mean_pairwise_dist(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = points[:, None, :] - points[None, :, :]
    d = np.linalg.norm(diffs, axis=2)
    iu = np.triu_indices_from(d, k=1)
    return float(d[iu].mean())


def e2_pca_scatter(system: str, hl_pool, saga_pool, bounds: np.ndarray,
                    results: dict, n_subsample_trials: int = 100):
    print(f"\n[E2] {system}: 2D PCA scatter HL vs SA+GA", flush=True)
    hl_th  = np.stack([d["theta"] for d in hl_pool])
    sg_th  = np.stack([d["theta"] for d in saga_pool])
    hl_n = normalize_minmax(hl_th, bounds)
    sg_n = normalize_minmax(sg_th, bounds)
    combined = np.vstack([hl_n, sg_n])
    mu = combined.mean(axis=0)
    U, S, Vt = np.linalg.svd(combined - mu, full_matrices=False)
    proj = (combined - mu) @ Vt.T[:, :2]
    hl_p = proj[:len(hl_n)]
    sg_p = proj[len(hl_n):]
    cum2 = float((S[:2] ** 2).sum() / (S ** 2).sum())
    print(f"  combined-pool PC1+PC2 var: {cum2*100:.1f}%", flush=True)

    # Raw spread numbers.
    hl_hull = hull_area(hl_p); sg_hull = hull_area(sg_p)
    hl_mpd  = mean_pairwise_dist(hl_p); sg_mpd = mean_pairwise_dist(sg_p)
    # Count-control: subsample HL to SA+GA size, average hull area + MPD.
    rng = np.random.default_rng(seed=42)
    n_sg = len(sg_n)
    sub_areas, sub_mpds = [], []
    for _ in range(n_subsample_trials):
        idx = rng.choice(len(hl_p), size=n_sg, replace=False)
        sub_areas.append(hull_area(hl_p[idx]))
        sub_mpds.append(mean_pairwise_dist(hl_p[idx]))
    out = {
        "n_hl": len(hl_n), "n_sg": n_sg,
        "combined_pc12_var": cum2,
        "hl_hull_area_full": hl_hull, "sg_hull_area": sg_hull,
        "hl_hull_area_subsampled_mean": float(np.mean(sub_areas)),
        "hl_hull_area_subsampled_p5_p95": [float(np.percentile(sub_areas, 5)),
                                            float(np.percentile(sub_areas, 95))],
        "hl_mpd_full": hl_mpd, "sg_mpd": sg_mpd,
        "hl_mpd_subsampled_mean": float(np.mean(sub_mpds)),
        "hl_mpd_per_point_full": hl_mpd, "sg_mpd_per_point": sg_mpd,
    }
    print(f"  HL full hull-area={hl_hull:.4f} (n={len(hl_n)})", flush=True)
    print(f"  SA+GA hull-area={sg_hull:.4f} (n={n_sg})", flush=True)
    print(f"  HL subsampled-to-{n_sg} hull-area: mean={out['hl_hull_area_subsampled_mean']:.4f} "
          f"(5/95% [{out['hl_hull_area_subsampled_p5_p95'][0]:.4f}, "
          f"{out['hl_hull_area_subsampled_p5_p95'][1]:.4f}])", flush=True)
    print(f"  mean pairwise distance: HL full={hl_mpd:.3f}  SA+GA={sg_mpd:.3f}  "
          f"HL subsampled mean={out['hl_mpd_subsampled_mean']:.3f}", flush=True)
    results.setdefault("E2", {})[system] = out

    # Plot.
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor="white")
    ax.scatter(hl_p[:, 0], hl_p[:, 1], s=10, c="#3b5b8a", alpha=0.45,
               edgecolors="none", label=f"HyperLens (n={len(hl_n)})")
    ax.scatter(sg_p[:, 0], sg_p[:, 1], s=22, c="#c97a3d", alpha=0.85,
               edgecolors="#7a4b25", linewidths=0.4, label=f"SA+GA (n={n_sg})")
    for pts, color in ((hl_p, "#3b5b8a"), (sg_p, "#c97a3d")):
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
                poly = np.vstack([poly, poly[:1]])
                ax.plot(poly[:, 0], poly[:, 1], color=color, lw=1.2, alpha=0.85)
            except Exception:
                pass
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"E2 2D PCA of θ ({system})  "
                 f"PC1+PC2 = {cum2*100:.1f}%")
    ax.grid(True, lw=0.4, color="#dddddd"); ax.set_axisbelow(True)
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()
    fig.savefig(OUT / f"e2_pca_scatter_{system}.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ============================================================
# E3: pairwise behavior vs parameter
# ============================================================

def e3_behavior_vs_param(system: str, hl_pool, bounds: np.ndarray, results: dict,
                          n_max_pairs: int = 200_000):
    print(f"\n[E3] {system}: pairwise behavior-distance vs parameter-distance",
          flush=True)
    keep = [d for d in hl_pool if d["S2"] is not None and d["S3"] is not None]
    n = len(keep)
    print(f"  {n}/{len(hl_pool)} designs have Seidel S2,S3", flush=True)
    if n < 5:
        print("  too few designs, skipping", flush=True); return
    thetas = np.stack([d["theta"] for d in keep])
    seidel = np.stack([[d["S2"], d["S3"]] for d in keep])
    thetas_n = normalize_minmax(thetas, bounds)
    # Behavior: min-max-normalize S2,S3 to [0,1].
    bh_lo = seidel.min(axis=0); bh_hi = seidel.max(axis=0)
    bh_span = np.where((bh_hi - bh_lo) > 0, bh_hi - bh_lo, 1.0)
    bh_n = (seidel - bh_lo) / bh_span

    sqrtD = np.sqrt(thetas_n.shape[1])
    rng = np.random.default_rng(seed=42)
    n_pairs_total = n * (n - 1) // 2
    if n_pairs_total <= n_max_pairs:
        ii, jj = np.triu_indices(n, k=1)
    else:
        ii = rng.integers(0, n, size=n_max_pairs)
        jj = rng.integers(0, n, size=n_max_pairs)
        mask = ii != jj
        ii = ii[mask]; jj = jj[mask]
    p_dist = np.linalg.norm(thetas_n[ii] - thetas_n[jj], axis=1) / sqrtD
    b_dist = np.linalg.norm(bh_n[ii] - bh_n[jj], axis=1) / np.sqrt(2)

    pear = pearsonr(p_dist, b_dist)
    sp = spearmanr(p_dist, b_dist)
    # Critical checks.
    zero_p_nonzero_b = int(((p_dist < 1e-12) & (b_dist > 1e-6)).sum())
    out = {
        "n_designs": n, "n_pairs": int(len(p_dist)),
        "pearson_r": float(pear[0]), "pearson_p": float(pear[1]),
        "spearman_rho": float(sp[0]), "spearman_p": float(sp[1]),
        "zero_param_distinct_behavior_count": zero_p_nonzero_b,
    }
    print(f"  n_pairs={out['n_pairs']}  Pearson r={pear[0]:.3f}  "
          f"Spearman ρ={sp[0]:.3f}", flush=True)
    print(f"  pairs with param-dist=0 yet behavior-dist>0: {zero_p_nonzero_b}",
          flush=True)
    results.setdefault("E3", {})[system] = out

    # Plot density (hexbin) of param vs behavior distance.
    fig, ax = plt.subplots(figsize=(6.0, 5.2), facecolor="white")
    hb = ax.hexbin(p_dist, b_dist, gridsize=60, cmap="viridis", mincnt=1, bins="log")
    ax.set_xlabel(r"parameter distance  $\|θ_i-θ_j\|/\sqrt{D}$")
    ax.set_ylabel(r"behavior distance  $\|(S_2,S_3)_{i}-(S_2,S_3)_{j}\|/\sqrt{2}$ "
                  r"(min-max-normalized)")
    ax.set_title(f"E3 ({system}): r={pear[0]:.2f}, ρ={sp[0]:.2f}, "
                 f"n_pairs={out['n_pairs']}")
    fig.colorbar(hb, ax=ax, label=r"$\log_{10}(N_{\mathrm{pairs}})$")
    ax.grid(True, lw=0.3, color="#dddddd"); ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(OUT / f"e3_pair_scatter_{system}.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# E4: ε-sweep
# ============================================================

def eps_sweep(thetas_n: np.ndarray, fitness: np.ndarray, rms: np.ndarray,
               eps_grid: np.ndarray) -> list[int]:
    counts = []
    for eps in eps_grid:
        picks = eps_net(thetas_n, fitness, float(eps), sort_by_rms=rms)
        counts.append(len(picks))
    return counts


def e4_eps_sweep(system: str, hl_pool, saga_pool, bounds: np.ndarray,
                  results: dict):
    print(f"\n[E4] {system}: ε-sweep N_ε(ε) for HL and SA+GA", flush=True)
    hl_th = np.stack([d["theta"] for d in hl_pool])
    sg_th = np.stack([d["theta"] for d in saga_pool])
    hl_n = normalize_minmax(hl_th, bounds); sg_n = normalize_minmax(sg_th, bounds)
    hl_f = np.array([d["refined_fitness"] for d in hl_pool])
    sg_f = np.array([d["refined_fitness"] for d in saga_pool])
    hl_r = np.array([d["rms_um"] for d in hl_pool])
    sg_r = np.array([d["rms_um"] for d in saga_pool])
    eps_grid = np.linspace(0.00, 0.50, 26)
    hl_curve = eps_sweep(hl_n, hl_f, hl_r, eps_grid)
    sg_curve = eps_sweep(sg_n, sg_f, sg_r, eps_grid)
    out = {"eps_grid": eps_grid.tolist(), "hl_counts": hl_curve,
           "sg_counts": sg_curve, "n_hl": len(hl_n), "n_sg": len(sg_n)}
    results.setdefault("E4", {})[system] = out
    # Plot.
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor="white")
    ax.plot(eps_grid, hl_curve, color="#3b5b8a", marker="o", ms=3,
            label=f"HyperLens (n={len(hl_n)})")
    ax.plot(eps_grid, sg_curve, color="#c97a3d", marker="s", ms=3,
            label=f"SA+GA (n={len(sg_n)})")
    for eps_mark in (0.20, 0.30):
        ax.axvline(eps_mark, color="#666", ls=":", lw=0.7)
        ax.text(eps_mark, ax.get_ylim()[1] * 0.9, f"ε={eps_mark}",
                ha="center", fontsize=8, color="#444",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_xlabel("ε (normalized θ, dist/√D)")
    ax.set_ylabel(r"$N_ε$ (ε-distinct designs)")
    ax.set_yscale("log")
    ax.set_title(f"E4 ε-sweep ({system})")
    ax.grid(True, lw=0.4, color="#dddddd", which="both")
    ax.set_axisbelow(True)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(OUT / f"e4_eps_sweep_{system}.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# Driver
# ============================================================

def main():
    results = {}
    for system, cfg in POOLS.items():
        print(f"\n========= {system} =========", flush=True)
        bounds = load_bounds(cfg["bounds_cfg"])
        hl_pool = load_pool(cfg["hl"])
        sg_pool = load_pool(cfg["saga"])
        print(f"  HL pool: n={len(hl_pool)}  SA+GA pool: n={len(sg_pool)}",
              flush=True)
        e5_sanity(system, hl_pool, bounds, results)
        e1_pca_variance(system, hl_pool, bounds, results)
        e2_pca_scatter(system, hl_pool, sg_pool, bounds, results)
        e3_behavior_vs_param(system, hl_pool, bounds, results)
        e4_eps_sweep(system, hl_pool, sg_pool, bounds, results)
    out_json = OUT / "s42_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
