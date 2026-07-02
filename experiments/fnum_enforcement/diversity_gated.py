"""Quality-GATED diversity: ε-distinct count among refined designs that clear an
RMS bar. The ungated count (analyze.py) rewards parameter-space scatter even for
optically bad designs; gating answers the real question — how many DIVERSE and
SHARP designs does each arm give you.

For each (system, arm) over the top-100-by-fitness refined pool, and for a sweep
of RMS thresholds T, report  n(RMS<T)  and  ε-distinct(RMS<T)  (ε=0.30).
"""
import os, sys, json, glob
from pathlib import Path
import numpy as np

ROOT = "/nas/longleaf/home/hisham26/Desktop/hyperlens"
sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts_throwaway"))
from fig5_extract import eps_net_select, norm_theta, load_bounds

TOP_N = 100
EPS = 0.30
THRESH = [20, 30, 50, 1e9]   # µm; 1e9 = no gate (reproduces the old number)

SOFT = {
    "sixlet":  "results/lensmega/sixlet_REFSWEEP2_G_grid2048_seed42_20260509_043849",
    "octuplet":"results/lensmega/octuplet_REFSWEEP_D_long_lbfgs_seed42_20260509_121606",
}
def enf_dir(system, w):
    base = ("sixlet_REFSWEEP2_G_grid2048_seed42" if system=="sixlet"
            else "octuplet_REFSWEEP_D_long_lbfgs_seed42")
    h = sorted(glob.glob(f"results/lensmega/{base}_ENF_W{w}_*"))
    return h[-1] if h else None

def load_pool(run_dir):
    h = glob.glob(os.path.join(run_dir, "refined_top*.json"))
    if not h: return None
    d = json.load(open(h[0]))
    des = d["designs"] if isinstance(d, dict) and "designs" in d else d
    des = sorted(des, key=lambda x:-float(x.get("refined_fitness",-1e18)))[:TOP_N]
    return [(float(x.get("rms_um",np.nan)), float(x.get("refined_fitness",-1e18)),
             x["theta"]) for x in des]

def eps_distinct(items, bounds, eps=EPS):
    """items: list of (rms, fit, theta); already RMS-filtered. ε-net by fitness."""
    if not items: return 0
    D = bounds.shape[0]
    items = sorted(items, key=lambda t:-t[1])  # best fitness first
    pool = [{"theta_norm": norm_theta(np.asarray(t[2],float), bounds),
             "fitness": t[1]} for t in items if len(t[2])==D]
    return len(eps_net_select(pool, eps, D)) if pool else 0

arms = [("none","soft",None), ("ENF w=1","ENF_W1",1), ("ENF w=10","ENF_W10",10)]
print(f"ε-distinct (ε={EPS}) among refined top-{TOP_N} designs with RMS < T\n")
hdr = f"{'system':9s} {'arm':9s} {'bestRMS':>7s} | " + " ".join(
    [f"T<{t:>4}: n/εd" if t<1e9 else "  no-gate: n/εd" for t in THRESH])
print(hdr); print("-"*len(hdr))
rows=[]
for system in ("sixlet","octuplet"):
    bounds = load_bounds(Path(SOFT[system])/"config.yaml")
    for disp,key,w in arms:
        rd = SOFT[system] if key=="soft" else enf_dir(system,w)
        pool = load_pool(rd) if rd else None
        if not pool: print(f"{system:9s} {disp:9s}  (pending)"); continue
        best = min(p[0] for p in pool)
        cells=[]
        for T in THRESH:
            sub=[p for p in pool if p[0] < T]
            cells.append(f"{len(sub):>3d}/{eps_distinct(sub,bounds):>2d}")
        print(f"{system:9s} {disp:9s} {best:7.1f} | " + "   ".join(f"{c:>9s}" for c in cells))
        rows.append((system,disp,best,cells))
