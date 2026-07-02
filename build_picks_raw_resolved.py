"""Resolve the 32 mono-RMS sixlet picks to their raw (S2, S3) and raw fitness:
  - 25 via parent run's archive_cells.json (pool_idx)
  - 4 via parent run's checkpoint.pkl (raw_fitness match) for runs that didn't
    save archive_cells.json (CHAMPION, grid1024)
  - 3 dropped: ranks 23, 30, 32 — raw measures are far outside the canonical
    Seidel bounds (S2/S3 in ±5 to ±15), pathological invalid designs whose
    refinement happened to rescue them. Their plot position would just be
    edge-clamped, distorting the figure.

Output: mono587_top32_sixlet_picks_raw.json with 29 entries (the in-bounds set).
"""
import json, glob, pickle
from pathlib import Path
import numpy as np

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
TOP32 = ROOT / "scripts_throwaway/mono587_top40_sixlet_eps030_sub10.json"
OUT = ROOT / "mono587_top32_sixlet_picks_raw.json"

BOUNDS = np.array([[-0.008, 0.008], [-0.012, 0.012]])

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


def fetch_raw(run, idx):
    """Returns dict(s2, s3, raw_fit, source) or None."""
    fps = sorted(glob.glob(str(ROOT / f"results/lensmega/{run}/refined_top*.json")))
    if not fps: return None
    rt = json.load(open(fps[0]))
    if idx >= len(rt["designs"]): return None
    des = rt["designs"][idx]
    pool_idx = des.get("pool_idx")
    raw_fit = des.get("raw_fitness")
    # Try archive_cells first
    ac = archive_cells(run)
    if ac is not None and pool_idx is not None and pool_idx < len(ac["cells"]):
        cell = ac["cells"][pool_idx]
        return {"s2": float(cell["measures"][0]), "s3": float(cell["measures"][1]),
                "raw_fit": float(cell["fitness"]), "source": "archive_cells.json"}
    # Fall back to checkpoint by raw_fitness match
    ckpt = checkpoint(run)
    if ckpt is None: return None
    matches = [(ij, c) for ij, c in ckpt.archive.cells.items()
               if abs(c.fitness - raw_fit) < 1e-9]
    if len(matches) != 1: return None
    (i, j), cell = matches[0]
    s2, s3 = float(cell.measures[0]), float(cell.measures[1])
    return {"s2": s2, "s3": s3, "raw_fit": float(cell.fitness),
            "source": f"checkpoint.pkl[cell({i},{j})]"}


def main():
    bundle = json.load(open(TOP32))
    designs = bundle["designs"][:32]
    out_picks, oob, unresolved = [], [], []
    for d in designs:
        rank = int(d["rank"])
        r = fetch_raw(d["run"], int(d["idx"]))
        if r is None:
            unresolved.append(rank); continue
        if not (BOUNDS[0,0] <= r["s2"] <= BOUNDS[0,1]
                and BOUNDS[1,0] <= r["s3"] <= BOUNDS[1,1]):
            oob.append((rank, r["s2"], r["s3"], r["raw_fit"]))
            continue
        out_picks.append({
            "rank": rank,
            "run": d["run"],
            "idx": d["idx"],
            "mono_fitness": float(d["mono_fitness"]),
            "mono_rms_um": float(d["mono_rms_um"]),
            "poly_rms_um": float(d["poly_rms_um"]),
            "raw_S2": r["s2"],
            "raw_S3": r["s3"],
            "raw_fitness": r["raw_fit"],
            "source": r["source"],
        })

    print(f"Resolved in-bounds: {len(out_picks)}")
    print(f"Dropped OOB:        {len(oob)}")
    for rank, s2, s3, rf in oob:
        print(f"  rank {rank}: S2={s2:+.3e}  S3={s3:+.3e}  raw_fit={rf:+.4f}")
    print(f"Unresolved:         {len(unresolved)} ({unresolved})")

    out = {
        "system": "sixlet_apsc_v2",
        "n_picks_resolved": len(out_picks),
        "n_oob_dropped": len(oob),
        "n_unresolved": len(unresolved),
        "oob_dropped": [{"rank": r, "S2": s2, "S3": s3, "raw_fit": rf}
                        for r, s2, s3, rf in oob],
        "picks": sorted(out_picks, key=lambda x: x["rank"]),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
