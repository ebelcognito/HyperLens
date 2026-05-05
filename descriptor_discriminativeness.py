"""
Descriptor discriminativeness analysis.

Tests the claim:
  "Designs in the discovery regime exhibit aberration signatures that vary
   substantially across geometric families. Well-corrected designs cluster
   near origin in aberration space (by definition). Therefore descriptor
   discriminativeness is highest exactly where the search needs to separate
   basins."

For all 39 sixlet refined elites:
  1. Cluster in refined-theta space (geometric families).
  2. Compute full Seidel (S1..S5) at *raw* theta (discovery regime) AND at
     *refined* theta (post-L-BFGS).
  3. Compute silhouette score for the basin labels in BC space, both at
     discovery and at refinement.
  4. Render a 2-panel scatter (S2,S3) discovery vs refined, colored by basin.
  5. Per-family Seidel-sign-pattern table (raw vs refined).

Hypotheses to test:
  - Discovery silhouette >> refined silhouette  (geometric separation
    contracts under refinement).
  - Sign patterns are stable across raw -> refined within each family
    (categorical separability persists).
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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, silhouette_samples

sys.path.insert(0, "/home/hisham26/Desktop/RL")
sys.path.insert(0, "/home/hisham26/Desktop/RL/DeepLens")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.lensmega.descriptors.seidel import SeidelDescriptor
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic


RUN_DIR = ("/home/hisham26/Desktop/RL/results/lensmega/"
           "cooke_triplet_materials_20260414_173935")
CFG = os.path.join(RUN_DIR, "config.yaml")
OUT = "/home/hisham26/Desktop/RL/paper_figures/descriptor_discriminativeness_cooke"
os.makedirs(OUT, exist_ok=True)

REFINED_RMS_MAX_UM = 30.0   # only "actually refined" designs (per user)
CLUSTER_FRAC = 0.25


def load_elites():
    paths = sorted(glob.glob(os.path.join(RUN_DIR, "elite_designs", "*.json")))
    out = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        ref = d.get("refined") or {}
        if "theta" not in ref or "rms_mm" not in ref:
            continue
        rms_um_ref = float(ref["rms_mm"]) * 1000.0
        if rms_um_ref > REFINED_RMS_MAX_UM:
            continue
        out.append({
            "name": os.path.basename(p)[:-5],
            "theta_raw": np.array(d["theta"], dtype=float),
            "theta_refined": np.array(ref["theta"], dtype=float),
            "rms_um_raw": float(d.get("rms_mm", 0.0)) * 1000.0,
            "rms_um_refined": rms_um_ref,
            "sign_pattern": d.get("sign_pattern", ""),
        })
    return out


def cluster_in_refined_theta(elites, frac=CLUSTER_FRAC):
    thetas = np.array([e["theta_refined"] for e in elites])
    mu = thetas.mean(0); sd = thetas.std(0) + 1e-9
    X = (thetas - mu) / sd
    d = pdist(X)
    thresh = max(float(np.median(d) * frac), 0.5)
    Z = linkage(X, method="single", metric="euclidean")
    return fcluster(Z, t=thresh, criterion="distance")


def compute_full_seidel(thetas, cfg):
    p = cfg.problem
    template, lens = load_lens_template(
        p.lens_path, wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index, device="cpu",
    )
    solver = GeneralizedParaxialSolver(template)
    desc = SeidelDescriptor(["S1", "S2", "S3", "S4", "S5"])
    desc.prepare(template, device="cpu")

    out = []
    for theta in thetas:
        theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=False)
        try:
            _apply_theta_generic(
                lens, theta_t, template=template, solver=solver,
                focal_length_target_mm=p.focal_length_target_mm, device="cpu",
                parameterization=p.parameterization,
                free_curvature_indices=p.free_curvature_indices,
                free_gap_indices=p.free_gap_indices,
                variable_flange=p.variable_flange,
                variable_materials=p.variable_materials,
                free_material_indices=p.free_material_indices,
            )
            hfov_deg = float(np.rad2deg(float(lens.rfov)))
            s = desc.compute(lens, field_y_deg=hfov_deg,
                             object_depth_mm=p.object_depth_mm,
                             wavelength_um=p.wavelength_um, device="cpu")
            out.append(s.detach().numpy())
        except Exception as e:
            print(f"  apply/compute failed: {e}")
            out.append(np.full(5, np.nan))
    return np.array(out)


def silhouette_safe(X, labels):
    """Silhouette score, robust to single-member clusters."""
    keep = np.array([np.sum(labels == l) > 1 for l in labels])
    if keep.sum() < 2 or len(set(labels[keep])) < 2:
        return float("nan")
    return float(silhouette_score(X[keep], labels[keep]))


def main():
    cfg = load_config(CFG)
    elites = load_elites()
    print(f"loaded {len(elites)} refined elites with refined RMS ≤ {REFINED_RMS_MAX_UM}μm")
    rms_vals = sorted(e["rms_um_refined"] for e in elites)
    print(f"  RMS range: {rms_vals[0]:.2f} – {rms_vals[-1]:.2f} μm")

    labels = cluster_in_refined_theta(elites)
    n_basins = len(set(labels))
    print(f"clustered into {n_basins} basins (frac={CLUSTER_FRAC})")

    print("\ncomputing full Seidel signatures at RAW theta...")
    seidel_raw = compute_full_seidel([e["theta_raw"] for e in elites], cfg)
    print("computing full Seidel signatures at REFINED theta...")
    seidel_ref = compute_full_seidel([e["theta_refined"] for e in elites], cfg)

    valid = ~np.any(np.isnan(seidel_raw), 1) & ~np.any(np.isnan(seidel_ref), 1)
    print(f"\n{valid.sum()}/{len(elites)} elites with valid Seidel at both raw and refined")
    seidel_raw = seidel_raw[valid]; seidel_ref = seidel_ref[valid]
    labels = labels[valid]
    elites = [e for e, v in zip(elites, valid) if v]

    # ============== silhouette scores ==============
    def stdz(X):
        sd = X.std(0); sd[sd < 1e-15] = 1.0
        return (X - X.mean(0)) / sd

    s23_raw = seidel_raw[:, [1, 2]]
    s23_ref = seidel_ref[:, [1, 2]]

    # Unstandardized (S2, S3) - reflects how CMA-MEGA actually sees BC space:
    # whichever Seidel has bigger spread dominates the Euclidean distance.
    sil_s23_raw = silhouette_safe(s23_raw, labels)
    sil_s23_ref = silhouette_safe(s23_ref, labels)
    # Standardized (S2, S3) - scale-fair: each coefficient contributes equally.
    sil_s23_std_raw = silhouette_safe(stdz(s23_raw), labels)
    sil_s23_std_ref = silhouette_safe(stdz(s23_ref), labels)
    # Standardized (S1..S5) - full-aberration view, scale-fair across all 5 dims.
    sil_s5_raw = silhouette_safe(stdz(seidel_raw), labels)
    sil_s5_ref = silhouette_safe(stdz(seidel_ref), labels)

    # Per-Seidel spread + magnitude report.
    # If a coefficient has high MEAN MAGNITUDE but low STD across elites,
    # it's a uniform "residual error" not a discriminator (e.g. S5 may behave this way).
    print("\n========== PER-COEFFICIENT MAG vs SPREAD ==========")
    print(f"  {'name':<3} {'raw mean|.|':>13} {'raw std':>13} {'ref mean|.|':>13} {'ref std':>13}")
    for j, name in enumerate(["S1", "S2", "S3", "S4", "S5"]):
        print(f"  {name:<3} "
              f"{np.mean(np.abs(seidel_raw[:, j])):>13.3e} "
              f"{seidel_raw[:, j].std():>13.3e} "
              f"{np.mean(np.abs(seidel_ref[:, j])):>13.3e} "
              f"{seidel_ref[:, j].std():>13.3e}")

    # ============== all-pairs silhouette scan ==============
    # Tests the hypothesis: any pair drawn from the family-discriminating
    # Seidels should give competitive silhouette at discovery.
    print("\n========== ALL-PAIRS SILHOUETTE (z-standardized) ==========")
    print(f"  {'pair':<10} {'discovery':>10} {'refined':>10}  {'≈ contract':>12}")
    pair_results = {}
    pair_names = ["S1", "S2", "S3", "S4", "S5"]
    pairs = [(i, j) for i in range(5) for j in range(i+1, 5)]
    rows = []
    for i, j in pairs:
        Xr = stdz(seidel_raw[:, [i, j]])
        Xf = stdz(seidel_ref[:, [i, j]])
        sr = silhouette_safe(Xr, labels)
        sf = silhouette_safe(Xf, labels)
        contract = sr - sf
        pair_str = f"({pair_names[i]},{pair_names[j]})"
        rows.append((pair_str, sr, sf, contract))
        pair_results[pair_str] = {"raw": sr, "ref": sf, "contract": contract}
    rows.sort(key=lambda r: -r[1])  # rank by discovery silhouette
    for pair_str, sr, sf, contract in rows:
        marker = "  ← used by run" if pair_str == "(S2,S3)" else ""
        print(f"  {pair_str:<10} {sr:>+10.4f} {sf:>+10.4f}  {contract:>+12.4f}{marker}")

    print("\n========== SILHOUETTE SCORES ==========")
    print(f"  (S2, S3) unstd raw     :  {sil_s23_raw:+.4f}   (algorithm's actual BC view)")
    print(f"  (S2, S3) unstd refined :  {sil_s23_ref:+.4f}")
    print(f"  (S2, S3) z-std raw     :  {sil_s23_std_raw:+.4f}   (scale-fair)")
    print(f"  (S2, S3) z-std refined :  {sil_s23_std_ref:+.4f}")
    print(f"  (S1..S5) z-std raw     :  {sil_s5_raw:+.4f}   (full Seidel, scale-fair)")
    print(f"  (S1..S5) z-std refined :  {sil_s5_ref:+.4f}")

    # ============== sign-pattern persistence ==============
    sign_raw = np.sign(seidel_raw)
    sign_ref = np.sign(seidel_ref)
    persist = np.all(sign_raw == sign_ref, axis=1).mean()
    print(f"\nsign-pattern fully persists raw->refined: {persist*100:.1f}% of elites")

    # Per-family modal sign pattern (raw and refined)
    family_signs = {}
    for c in sorted(set(labels)):
        mask = labels == c
        signs_r = sign_raw[mask]
        signs_f = sign_ref[mask]
        def mode_signs(arr):
            return tuple(int(np.sign(np.sum(arr[:, k]))) for k in range(5))
        family_signs[int(c)] = {
            "n": int(mask.sum()),
            "mode_raw": mode_signs(signs_r),
            "mode_ref": mode_signs(signs_f),
            "mean_S1_raw": float(seidel_raw[mask, 0].mean()),
            "mean_S2_raw": float(seidel_raw[mask, 1].mean()),
            "mean_S3_raw": float(seidel_raw[mask, 2].mean()),
            "mean_S4_raw": float(seidel_raw[mask, 3].mean()),
            "mean_S5_raw": float(seidel_raw[mask, 4].mean()),
            "mean_S1_ref": float(seidel_ref[mask, 0].mean()),
            "mean_S2_ref": float(seidel_ref[mask, 1].mean()),
            "mean_S3_ref": float(seidel_ref[mask, 2].mean()),
            "mean_S4_ref": float(seidel_ref[mask, 3].mean()),
            "mean_S5_ref": float(seidel_ref[mask, 4].mean()),
            "median_rms_um_refined": float(np.median([
                e["rms_um_refined"] for e, ll in zip(elites, labels) if ll == c
            ])),
        }

    print("\n========== PER-FAMILY MODAL SIGN PATTERN ==========")
    print(f"{'basin':>6} {'n':>3}  {'mode raw':<22} {'mode ref':<22} {'med RMS μm':>10}")
    for c, f in family_signs.items():
        sr = " ".join("+" if s > 0 else "−" if s < 0 else "0" for s in f["mode_raw"])
        sf = " ".join("+" if s > 0 else "−" if s < 0 else "0" for s in f["mode_ref"])
        match = "✓" if f["mode_raw"] == f["mode_ref"] else "≠"
        print(f"  {c:>4} {f['n']:>3}  S1..S5 [{sr}]      [{sf}] {match}  "
              f"{f['median_rms_um_refined']:>9.2f}")

    # ============== two-panel scatter ==============
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, n_basins))
    color_map = {int(c): cluster_colors[i] for i, c in enumerate(sorted(set(labels)))}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    for ax, X, sil, title in [
        (axes[0], s23_raw, sil_s23_raw, "Discovery regime  (raw θ)"),
        (axes[1], s23_ref, sil_s23_ref, "Refined  (post-L-BFGS)"),
    ]:
        for c in sorted(set(labels)):
            m = labels == c
            ax.scatter(X[m, 0], X[m, 1], s=80, c=[color_map[int(c)]],
                       edgecolor="black", linewidths=0.7,
                       label=f"family {c} (n={m.sum()})")
        ax.axhline(0, c="grey", lw=0.5, alpha=0.5)
        ax.axvline(0, c="grey", lw=0.5, alpha=0.5)
        ax.set_xlabel("S₂ (coma)")
        ax.set_ylabel("S₃ (astigmatism)")
        ax.set_title(f"{title}\nsilhouette = {sil:+.3f}",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)

    # Use same legend (basin colours) only on right panel
    axes[1].legend(loc="best", fontsize=8, ncol=2, framealpha=0.9)

    fig.suptitle(
        "Descriptor discriminativeness: aberration space at discovery vs refinement\n"
        f"39 sixlet elites, {n_basins} geometric families (clusters in refined-θ space)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = os.path.join(OUT, "descriptor_discriminativeness_scatter.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_png}")

    # ============== bootstrap CI on silhouette ==============
    # Resample 28 elites with replacement, recompute silhouette × N_BOOT.
    # Reports 95% CI on each silhouette score.
    print("\n========== BOOTSTRAP 95% CI (1000 resamples) ==========")
    rng = np.random.default_rng(42)
    N_BOOT = 1000
    n = len(labels)

    def boot_one(X, lbl, rng):
        idx = rng.integers(0, n, size=n)
        Xb, lb = X[idx], lbl[idx]
        return silhouette_safe(Xb, lb)

    bootstrap_keys = {
        "S2S3_unstd_raw": (s23_raw, labels),
        "S2S3_unstd_ref": (s23_ref, labels),
        "S2S3_zstd_raw":  (stdz(s23_raw), labels),
        "S2S3_zstd_ref":  (stdz(s23_ref), labels),
        "S1to5_zstd_raw": (stdz(seidel_raw), labels),
        "S1to5_zstd_ref": (stdz(seidel_ref), labels),
    }
    bootstrap_results = {}
    for key, (X, lbl) in bootstrap_keys.items():
        vals = []
        for _ in range(N_BOOT):
            v = boot_one(X, lbl, rng)
            if not np.isnan(v):
                vals.append(v)
        vals = np.array(vals)
        lo, med, hi = np.percentile(vals, [2.5, 50, 97.5])
        bootstrap_results[key] = {
            "median": float(med), "lo95": float(lo), "hi95": float(hi),
            "mean": float(vals.mean()), "std": float(vals.std()),
            "n_valid": int(len(vals)),
        }
        print(f"  {key:<22}  median={med:+.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]  "
              f"std={vals.std():.3f}  n_valid={len(vals)}/{N_BOOT}")

    # ============== silhouette bar comparison ==============
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    cats = ["(S₂,S₃) unstd\n(algo's BC view)",
            "(S₂,S₃) z-std\n(scale-fair)",
            "(S₁…S₅) z-std\n(full)"]
    raw_vals = [sil_s23_raw, sil_s23_std_raw, sil_s5_raw]
    ref_vals = [sil_s23_ref, sil_s23_std_ref, sil_s5_ref]
    raw_keys = ["S2S3_unstd_raw", "S2S3_zstd_raw", "S1to5_zstd_raw"]
    ref_keys = ["S2S3_unstd_ref", "S2S3_zstd_ref", "S1to5_zstd_ref"]
    raw_err = np.array([
        [bootstrap_results[k]["median"] - bootstrap_results[k]["lo95"] for k in raw_keys],
        [bootstrap_results[k]["hi95"] - bootstrap_results[k]["median"] for k in raw_keys],
    ])
    ref_err = np.array([
        [bootstrap_results[k]["median"] - bootstrap_results[k]["lo95"] for k in ref_keys],
        [bootstrap_results[k]["hi95"] - bootstrap_results[k]["median"] for k in ref_keys],
    ])
    x = np.arange(len(cats)); w = 0.35
    ax.bar(x - w/2, raw_vals, w, yerr=raw_err, capsize=4,
           label="discovery (raw θ)", color="#4C78A8", edgecolor="black",
           error_kw=dict(ecolor="black", lw=1.2))
    ax.bar(x + w/2, ref_vals, w, yerr=ref_err, capsize=4,
           label="refined (post L-BFGS)", color="#F58518", edgecolor="black",
           error_kw=dict(ecolor="black", lw=1.2))
    for i, (rv, rfv) in enumerate(zip(raw_vals, ref_vals)):
        ax.text(i - w/2, rv + 0.04 if rv >= 0 else rv - 0.06,
                f"{rv:+.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(i + w/2, rfv + 0.04 if rfv >= 0 else rfv - 0.06,
                f"{rfv:+.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, c="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("silhouette score")
    ax.set_title("Family separation in aberration space\nhigh at discovery → contracts toward 0 at refinement\n(error bars = bootstrap 95% CI, 1000 resamples)",
                 fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    out_png2 = os.path.join(OUT, "silhouette_comparison.png")
    fig.savefig(out_png2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png2}")

    # ============== per-family signature heatmap (raw vs refined) ==============
    families_sorted = sorted(family_signs.keys(),
                             key=lambda c: family_signs[c]["median_rms_um_refined"])
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3, 0.4 * len(families_sorted) + 1)))
    seidel_labels = ["S₁", "S₂", "S₃", "S₄", "S₅"]
    for ax, regime, label in [
        (axes[0], "raw", "Discovery (raw θ)"),
        (axes[1], "ref", "Refined"),
    ]:
        M = np.array([
            [family_signs[c][f"mean_S{j+1}_{regime}"] for j in range(5)]
            for c in families_sorted
        ])
        vmax = np.nanpercentile(np.abs(M), 95)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(5)); ax.set_xticklabels(seidel_labels, fontsize=11)
        ax.set_yticks(range(len(families_sorted)))
        ax.set_yticklabels(
            [f"b{c} (RMS={family_signs[c]['median_rms_um_refined']:.0f}μm)"
             for c in families_sorted], fontsize=9,
        )
        ax.set_title(label, fontweight="bold")
        for i, c in enumerate(families_sorted):
            for j in range(5):
                v = family_signs[c][f"mean_S{j+1}_{regime}"]
                txt = f"{v*1e4:+.1f}" if abs(v) > 1e-8 else "0"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 * vmax else "black")
        plt.colorbar(im, ax=ax, fraction=0.04, label="value × 10⁻⁴")
    fig.suptitle("Mean Seidel signature per family (raw vs refined)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png3 = os.path.join(OUT, "per_family_seidel_heatmap.png")
    fig.savefig(out_png3, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png3}")

    # ============== save numbers as JSON ==============
    summary = {
        "n_elites": int(len(elites)),
        "n_basins": int(n_basins),
        "silhouette": {
            "S2S3_unstd_raw": sil_s23_raw,
            "S2S3_unstd_ref": sil_s23_ref,
            "S2S3_zstd_raw": sil_s23_std_raw,
            "S2S3_zstd_ref": sil_s23_std_ref,
            "S1to5_zstd_raw": sil_s5_raw,
            "S1to5_zstd_ref": sil_s5_ref,
        },
        "per_coeff_std_raw": [float(seidel_raw[:, j].std()) for j in range(5)],
        "per_coeff_std_ref": [float(seidel_ref[:, j].std()) for j in range(5)],
        "per_coeff_meanabs_raw": [float(np.mean(np.abs(seidel_raw[:, j]))) for j in range(5)],
        "per_coeff_meanabs_ref": [float(np.mean(np.abs(seidel_ref[:, j]))) for j in range(5)],
        "all_pair_silhouettes_zstd": pair_results,
        "bootstrap_95ci": bootstrap_results,
        "sign_pattern_fully_persists_pct": float(persist * 100),
        "family_signatures": family_signs,
    }
    out_json = os.path.join(OUT, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
