"""Unified picks_raw builder for triplet / sixlet / octuplet.

For each top-32 mono-RMS pick, resolves its raw (pre-refinement) (S2, S3) and
raw fitness via:
  1. archive_cells.json[pool_idx] — fastest path for Seidel-descriptor runs
  2. archive_cells.json[pool_idx].metrics for non-Seidel runs (RANDBC etc.)
  3. checkpoint.pkl matched by raw_fitness — fallback when archive_cells absent

Drops picks whose raw (S2, S3) is outside the system's canonical square window.
"""
import json, glob, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")

SYSTEMS = {
    "triplet": {
        "src": "scripts_throwaway/mono587_top32_triplet_eps030_sub10.json",
        "square_R": 0.001,
        "out": "mono587_top32_triplet_picks_raw.json",
    },
    "sixlet": {
        "src": "scripts_throwaway/mono587_top40_sixlet_eps030_sub10.json",
        "square_R": 0.008,
        "out": "mono587_top32_sixlet_picks_raw.json",
    },
    "octuplet": {
        "src": "scripts_throwaway/mono587_top32_octuplet_eps030_sub20.json",
        "square_R": 0.005,
        "out": "mono587_top32_octuplet_picks_raw.json",
    },
}

_arc_cache, _ckpt_cache = {}, {}
def archive_cells(run):
    if run not in _arc_cache:
        p = ROOT / f"results/lensmega/{run}/archive_cells.json"
        _arc_cache[run] = json.load(open(p)) if p.exists() else None
    return _arc_cache[run]

def checkpoint(run):
    if run not in _ckpt_cache:
        p = ROOT / f"results/lensmega/{run}/checkpoint.pkl"
        _ckpt_cache[run] = pickle.load(open(p, "rb")) if p.exists() else None
    return _ckpt_cache[run]


def cell_to_s2_s3(cell, ac_kind):
    """Resolve a cell to (S2, S3) in physical units regardless of descriptor."""
    if ac_kind == "seidel":
        return float(cell["measures"][0]), float(cell["measures"][1])
    m = cell.get("metrics", {})
    if "seidel_S2" in m and "seidel_S3" in m:
        return float(m["seidel_S2"]), float(m["seidel_S3"])
    return None


def fetch_raw(run, idx):
    fps = sorted(glob.glob(str(ROOT / f"results/lensmega/{run}/refined_top*.json")))
    if not fps: return None
    rt = json.load(open(fps[0]))
    if idx >= len(rt["designs"]): return None
    des = rt["designs"][idx]
    pool_idx = des.get("pool_idx")
    raw_fit = des.get("raw_fitness")
    ac = archive_cells(run)
    if ac is not None and pool_idx is not None and pool_idx < len(ac["cells"]):
        cell = ac["cells"][pool_idx]
        coords = cell_to_s2_s3(cell, ac.get("descriptor_kind"))
        if coords is not None:
            return {"s2": coords[0], "s3": coords[1],
                    "raw_fit": float(cell["fitness"]),
                    "source": f"archive_cells.json (kind={ac.get('descriptor_kind')})"}
    ckpt = checkpoint(run)
    if ckpt is None or raw_fit is None:
        return None
    matches = [(ij, c) for ij, c in ckpt.archive.cells.items()
               if abs(c.fitness - raw_fit) < 1e-9]
    if len(matches) != 1:
        return None
    (i, j), cell = matches[0]
    m = getattr(cell, "metrics", {}) or {}
    if "seidel_S2" in m and "seidel_S3" in m:
        s2, s3 = float(m["seidel_S2"]), float(m["seidel_S3"])
    else:
        meas = np.asarray(cell.measures, dtype=float)
        s2, s3 = float(meas[0]), float(meas[1])
    return {"s2": s2, "s3": s3, "raw_fit": float(cell.fitness),
            "source": f"checkpoint.pkl[cell({i},{j})]"}


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["triplet", "octuplet"]
    for s in targets:
        cfg = SYSTEMS[s]
        SR = cfg["square_R"]
        bundle = json.load(open(ROOT / cfg["src"]))
        designs = bundle["designs"][:32]
        out_picks, oob, unresolved = [], [], []
        for d in designs:
            rank = int(d["rank"])
            r = fetch_raw(d["run"], int(d["idx"]))
            if r is None:
                unresolved.append(rank); continue
            if abs(r["s2"]) > SR or abs(r["s3"]) > SR:
                oob.append((rank, r["s2"], r["s3"], r["raw_fit"]))
                continue
            out_picks.append({
                "rank": rank, "run": d["run"], "idx": d["idx"],
                "mono_fitness": float(d["mono_fitness"]),
                "mono_rms_um": float(d["mono_rms_um"]),
                "poly_rms_um": float(d["poly_rms_um"]),
                "raw_S2": r["s2"], "raw_S3": r["s3"],
                "raw_fitness": r["raw_fit"],
                "source": r["source"],
            })
        print(f"\n[{s}] picks_in_square (|S2|≤{SR}, |S3|≤{SR}): {len(out_picks)}")
        print(f"  dropped OOB ({len(oob)}):")
        for rank, s2, s3, rf in oob:
            print(f"    rank {rank}: S2={s2:+.3e} S3={s3:+.3e} fit={rf:+.4f}")
        if unresolved:
            print(f"  unresolved ({len(unresolved)}): {unresolved}")
        out = {
            "system": s,
            "square_R": SR,
            "n_picks_resolved": len(out_picks),
            "n_oob_dropped": len(oob),
            "n_unresolved": len(unresolved),
            "oob_dropped": [{"rank": r, "S2": s2, "S3": s3, "raw_fit": rf}
                            for r, s2, s3, rf in oob],
            "unresolved_ranks": unresolved,
            "picks": sorted(out_picks, key=lambda x: x["rank"]),
        }
        (ROOT / cfg["out"]).write_text(json.dumps(out, indent=2))
        print(f"  wrote {ROOT / cfg['out']}")


if __name__ == "__main__":
    main()
