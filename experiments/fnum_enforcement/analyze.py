"""Enforced-vs-soft f/# comparison (supplementary).

For sixlet and octuplet, compares three matched arms:
  soft    = enforce_fnum:false  (the paper's pools)
  ENF_W1  = enforce_fnum:true, fnum_penalty_weight=1
  ENF_W10 = enforce_fnum:true, fnum_penalty_weight=10
Everything else identical (template, 2048²/512² bins+bounds, 100k iters, seed42).

Reads each arm's refined_top500 JSON (post-L-BFGS pool) and archive_cells.json
(raw QD archive). Emits a metrics table (md + csv) and figures into
experiments/fnum_enforcement/analysis/.
"""
import os, sys, json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nas/longleaf/home/hisham26/Desktop/hyperlens"
sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts_throwaway"))
from fig5_extract import eps_net_select, norm_theta, load_bounds  # paper's ε-net

OUT = os.path.join(ROOT, "experiments/fnum_enforcement/analysis")
os.makedirs(OUT, exist_ok=True)

FN_LO, FN_HI = 3.6, 4.4          # "on-spec" band around f/4
RMS_SHARP = 20.0                  # µm, "sharp" threshold used in S9
EPS = 0.30                        # ε-net radius (matches the eps030 figures)
TOP_N = 100                       # compare every arm at equal top-N by fitness
                                  # (soft pool is 500, enforced is 100) -> fair
DISPLAY = {"soft": "none", "ENF_W1": "enforced w=1", "ENF_W10": "enforced w=10"}

SOFT = {
    "sixlet":  "results/lensmega/sixlet_REFSWEEP2_G_grid2048_seed42_20260509_043849",
    "octuplet":"results/lensmega/octuplet_REFSWEEP_D_long_lbfgs_seed42_20260509_121606",
}
def enf_dir(system, w):
    base = ("sixlet_REFSWEEP2_G_grid2048_seed42" if system=="sixlet"
            else "octuplet_REFSWEEP_D_long_lbfgs_seed42")
    hits = sorted(glob.glob(f"results/lensmega/{base}_ENF_W{w}_*"))
    return hits[-1] if hits else None

def refined_path(run_dir):
    h = glob.glob(os.path.join(run_dir, "refined_top*.json"))
    return h[0] if h else None

def load_refined(run_dir):
    p = refined_path(run_dir)
    if not p: return None
    d = json.load(open(p))
    des = d["designs"] if isinstance(d, dict) and "designs" in d else d
    # rank by refined fitness (best first) and keep top-N for a fair, equal-size
    # comparison across arms (soft pool=500, enforced=100).
    des = sorted(des, key=lambda x: -float(x.get("refined_fitness", -1e18)))[:TOP_N]
    fn = np.array([float(x.get("fnum_actual", np.nan)) for x in des])
    rms = np.array([float(x.get("rms_um", np.nan)) for x in des])
    th = [x["theta"] for x in des]
    fit = np.array([float(x.get("refined_fitness", np.nan)) for x in des])
    return dict(fn=fn, rms=rms, theta=th, fit=fit, n=len(des))

def load_archive(run_dir):
    p = os.path.join(run_dir, "archive_cells.json")
    if not os.path.exists(p): return None
    d = json.load(open(p))
    cells = d["cells"]
    fit = np.array([c["fitness"] for c in cells])
    meas = np.array([c["measures"] for c in cells])          # (N,2) S2,S3
    fn = np.array([c["metrics"].get("fnum_actual", np.nan) for c in cells])
    return dict(fit=fit, meas=meas, fn=fn, n=len(cells), bins=d["archive_bins"])

def eps_distinct_count(theta_list, fit, bounds, eps=EPS):
    if not theta_list: return 0
    D = bounds.shape[0]
    pool = []
    order = np.argsort(-fit)   # best fitness first
    for i in order:
        th = np.asarray(theta_list[i], dtype=np.float64)
        if len(th) != D: continue
        pool.append({"theta_norm": norm_theta(th, bounds), "fitness": float(fit[i])})
    if not pool: return 0
    keep = eps_net_select(pool, eps, D)
    return len(keep)

def pct(a, lo, hi):
    a = a[~np.isnan(a)]
    return 100.0*np.mean((a>=lo)&(a<=hi)) if len(a) else float("nan")

rows = []
arms = [("soft", None), ("ENF_W1", 1), ("ENF_W10", 10)]
fig_data = {}   # system -> arm -> dict

for system in ("sixlet", "octuplet"):
    bounds = load_bounds(Path(SOFT[system]) / "config.yaml")
    fig_data[system] = {}
    for arm, w in arms:
        rd = SOFT[system] if arm == "soft" else enf_dir(system, w)
        if rd is None:
            print(f"[skip] {system} {arm}: no run dir yet"); continue
        ref = load_refined(rd)
        arc = load_archive(rd)
        if ref is None:
            print(f"[skip] {system} {arm}: no refined pool yet ({rd})"); continue
        fn, rms = ref["fn"], ref["rms"]
        onspec_sharp = int(np.sum((fn>=FN_LO)&(fn<=FN_HI)&(rms<RMS_SHARP)))
        under4_sharp = int(np.sum((fn<4.0)&(rms<RMS_SHARP)))
        nd = eps_distinct_count(ref["theta"], ref["fit"], bounds)      # ungated
        # quality-GATED diversity: ε-distinct among designs with RMS < bar.
        n_sharp = int(np.sum(rms < RMS_SHARP))
        smask = rms < RMS_SHARP
        th_sharp = [ref["theta"][i] for i in range(len(rms)) if smask[i]]
        nd_sharp = eps_distinct_count(th_sharp, ref["fit"][smask], bounds)
        row = dict(
            system=system, arm=arm,
            refined_N=ref["n"],
            fn_med=np.nanmedian(fn), fn_p10=np.nanpercentile(fn,10),
            fn_p90=np.nanpercentile(fn,90), fn_in_band_pct=pct(fn,FN_LO,FN_HI),
            rms_med=np.nanmedian(rms), rms_min=np.nanmin(rms),
            onspec_sharp=onspec_sharp, under4_sharp=under4_sharp,
            n_sharp=n_sharp, eps_distinct_sharp=nd_sharp,
            eps_distinct_ungated=nd,
            archive_cells=(arc["n"] if arc else -1),
            qd_score=(float(np.sum(arc["fit"])) if arc else float("nan")),
            mean_cell_fit=(float(np.mean(arc["fit"])) if arc else float("nan")),
            archive_fn_in_band_pct=(pct(arc["fn"],FN_LO,FN_HI) if arc else float("nan")),
        )
        rows.append(row)
        fig_data[system][arm] = dict(fn=fn, rms=rms, arc=arc)
        print(f"[ok] {system:8s} {DISPLAY.get(arm,arm):13s} N={ref['n']:4d} "
              f"f/#med={row['fn_med']:.2f} inband={row['fn_in_band_pct']:.0f}% "
              f"bestRMS={row['rms_min']:.1f} n_sharp={n_sharp} "
              f"eps_dist<20={nd_sharp} (ungated {nd})")

# ---- write table ----
import csv
cols = ["system","arm","refined_N","fn_med","fn_p10","fn_p90","fn_in_band_pct",
        "rms_med","rms_min","onspec_sharp","under4_sharp",
        "n_sharp","eps_distinct_sharp","eps_distinct_ungated",
        "archive_cells","qd_score","mean_cell_fit","archive_fn_in_band_pct"]
with open(os.path.join(OUT,"summary.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({k:r.get(k) for k in cols})

with open(os.path.join(OUT,"summary.md"),"w") as f:
    f.write("# Enforced vs soft f/# — matched A/B (seed42, 100k iters)\n\n")
    f.write(f"On-spec band f/# ∈ [{FN_LO},{FN_HI}]; sharp = RMS < {RMS_SHARP} µm; ε-net radius {EPS}.\n\n")
    hdr = ["system","constraint","refined N","f/# med","f/# p10–p90","% in band",
           "RMS med","RMS min (best)","n sharp(<20µm)","ε-distinct(<20µm)",
           "ε-distinct(ungated)","on-spec&sharp","archive cells"]
    f.write("| "+" | ".join(hdr)+" |\n")
    f.write("|"+"|".join(["---"]*len(hdr))+"|\n")
    for r in rows:
        f.write("| "+" | ".join([
            r["system"], DISPLAY.get(r["arm"], r["arm"]), str(r["refined_N"]),
            f"{r['fn_med']:.2f}", f"{r['fn_p10']:.2f}–{r['fn_p90']:.2f}",
            f"{r['fn_in_band_pct']:.0f}%", f"{r['rms_med']:.1f}", f"{r['rms_min']:.1f}",
            str(r["n_sharp"]), str(r["eps_distinct_sharp"]),
            str(r["eps_distinct_ungated"]), str(r["onspec_sharp"]),
            str(r["archive_cells"]),
        ])+" |\n")
print("\nwrote", os.path.join(OUT,"summary.md"))

# ---- figures ----
ARM_C = {"soft":"#888888", "ENF_W1":"#1565C0", "ENF_W10":"#C62828"}
# Fig 1: f/# histograms (refined pool)
fig, axes = plt.subplots(1, 2, figsize=(12,4.2))
for ax, system in zip(axes, ("sixlet","octuplet")):
    for arm,_ in arms:
        if arm not in fig_data[system]: continue
        fn = fig_data[system][arm]["fn"]; fn=fn[~np.isnan(fn)]
        ax.hist(fn, bins=np.arange(2.0,9.01,0.2), histtype="step", lw=2,
                color=ARM_C[arm], label=f"{DISPLAY.get(arm,arm)} (n={len(fn)})", density=True)
    ax.axvspan(FN_LO,FN_HI,color="gold",alpha=0.15)
    ax.axvline(4.0,color="k",ls=":",lw=1)
    ax.set_title(f"{system} — refined-pool f/#"); ax.set_xlabel("f/#"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_fnum_hist.png"),dpi=160); plt.close(fig)

# Fig 2: f/# vs RMS scatter
fig, axes = plt.subplots(1, 2, figsize=(12,4.6))
for ax, system in zip(axes, ("sixlet","octuplet")):
    for arm,_ in arms:
        if arm not in fig_data[system]: continue
        fn=fig_data[system][arm]["fn"]; rms=fig_data[system][arm]["rms"]
        ax.scatter(fn, rms, s=10, alpha=0.5, color=ARM_C[arm], label=DISPLAY.get(arm,arm))
    ax.axvspan(FN_LO,FN_HI,color="gold",alpha=0.15); ax.axhline(RMS_SHARP,color="k",ls=":",lw=1)
    ax.set_yscale("log"); ax.set_title(f"{system} — f/# vs RMS")
    ax.set_xlabel("f/#"); ax.set_ylabel("poly RMS (µm)"); ax.set_xlim(2,9); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_fnum_vs_rms.png"),dpi=160); plt.close(fig)

# Fig 3: archive descriptor coverage (S2,S3) per arm
fig, axes = plt.subplots(2, 3, figsize=(13,7.5))
for i, system in enumerate(("sixlet","octuplet")):
    for j,(arm,_) in enumerate(arms):
        ax=axes[i,j]
        d=fig_data[system].get(arm)
        if d and d["arc"]:
            m=d["arc"]["meas"]
            ax.scatter(m[:,0]*1e3, m[:,1]*1e3, s=1, alpha=0.25, color=ARM_C[arm])
            ax.set_title(f"{system} {DISPLAY.get(arm,arm)} — {d['arc']['n']} cells", fontsize=9)
        else:
            ax.set_title(f"{system} {DISPLAY.get(arm,arm)} — (pending)", fontsize=9)
        ax.set_xlabel("S2 (×1e-3)"); ax.set_ylabel("S3 (×1e-3)")
fig.suptitle("Raw QD-archive coverage in Seidel (S2,S3) space", y=1.0)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_archive_coverage.png"),dpi=150); plt.close(fig)

print("figures ->", OUT)
