"""Trajectory-level analysis of the 3 sixlet 'deeper basin' elites.

For elites at evals 32384 (S-D1), 83520 (S-D2), 97248 (S-D3), look at:
  1. What was σ doing around their insertion?
  2. Was there a restart near each?
  3. What was the running cumulative restart count at each insertion?
  4. From lineage.jsonl: what was each elite's parent_theta_hash, and where
     did it sit in the search tree (which emitter run, how many generations
     in)?
  5. Cosine similarity between consecutive ++---- archive insertions to see
     whether elites are evolving from each other.
"""
import csv
import json
import os
from collections import defaultdict

RUN = ("/home/hisham26/Desktop/RL/results/lensmega/"
       "sixlet_discovery_materials_tightBC_20260426_051752")

PICKS = {
    "S-D1": ("elite_0003_32384k", 32384, 179.6),
    "S-D2": ("elite_0026_83520k", 83520, 59.3),
    "S-D3": ("elite_0033_97248k", 97248, 12.7),
}


def load_trajectory():
    """trajectory.csv has per-generation aggregate stats."""
    rows = []
    with open(os.path.join(RUN, "trajectory.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({
                "gen": int(r["gen"]),
                "evals": int(r["evals"]),
                "coverage": float(r["coverage"]),
                "best_fitness": float(r["best_fitness"]),
                "new_cells": int(r["new_cells"]),
                "replaced_elites": int(r["replaced_elites"]),
                "rejected": int(r["rejected"]),
                "sigma": float(r["sigma"]),
                "restarts": int(r["restarts"]),
            })
    return rows


def load_elite_tracker():
    """elite_tracker.csv: per-elite-event stats (eval, fitness, sign_pattern)."""
    rows = []
    with open(os.path.join(RUN, "elite_tracker.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({
                "eval": int(r["eval"]),
                "fitness": float(r["fitness"]),
                "rms_um": float(r["rms_um"]),
                "sign_pattern": r["sign_pattern"],
                "m0": float(r["m0"]), "m1": float(r["m1"]),
                "cosine_sim": float(r["cosine_sim"]),
                "sign_match": int(r["sign_match"]),
            })
    return rows


def trajectory_window(traj, target_eval, window=400):
    """Return trajectory rows whose evals fall within ±window of target."""
    return [r for r in traj if abs(r["evals"] - target_eval) <= window]


def find_restart_near(traj, target_eval, window=2000):
    """Look for a restart event near target_eval (where 'restarts' counter
    increments). Return (gen, eval, before_restarts, after_restarts) of any."""
    last_r = traj[0]["restarts"]
    events = []
    for r in traj:
        if r["restarts"] > last_r:
            events.append((r["gen"], r["evals"], last_r, r["restarts"]))
        last_r = r["restarts"]
    nearby = [e for e in events if abs(e[1] - target_eval) <= window]
    return nearby, len(events)


def lookup_lineage(theta_hash_target):
    """Find a specific elite's lineage record."""
    if not os.path.exists(os.path.join(RUN, "lineage.jsonl")):
        return None
    with open(os.path.join(RUN, "lineage.jsonl")) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("theta_hash") == theta_hash_target:
                    return rec
            except Exception:
                continue
    return None


def grep_lineage_around(target_eval, window=200):
    """Return lineage records around target eval (eval = gen × pop_size,
    approximately). Walk lineage.jsonl, count emissions until we hit target."""
    if not os.path.exists(os.path.join(RUN, "lineage.jsonl")):
        return []
    out = []
    with open(os.path.join(RUN, "lineage.jsonl")) as f:
        for i, line in enumerate(f):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if abs(i - target_eval) <= window:
                out.append((i, rec.get("source", "?"), rec.get("gen", -1),
                           rec.get("emission", -1),
                           (rec.get("parent_theta_hash") or "")[:10]))
    return out


def main():
    traj = load_trajectory()
    elites = load_elite_tracker()
    print(f"Loaded {len(traj)} trajectory rows ({traj[-1]['gen']} gens, "
          f"{traj[-1]['evals']} evals, {traj[-1]['restarts']} total restarts).\n")

    # ------------------------------------------------------------
    # 1) For each pick, σ + restart context at its insertion
    # ------------------------------------------------------------
    print("=" * 78)
    print("1. SIGMA + RESTART CONTEXT AT EACH PICK'S INSERTION")
    print("=" * 78)
    for label, (stem, ev, refrms) in PICKS.items():
        win = trajectory_window(traj, ev, window=200)
        if not win:
            print(f"  {label} @ eval {ev}: no trajectory rows near it")
            continue
        sigmas = [r["sigma"] for r in win]
        restarts = [r["restarts"] for r in win]
        new_cells = sum(r["new_cells"] for r in win)
        replaced = sum(r["replaced_elites"] for r in win)
        print(f"  {label}  ({stem}, eval {ev}, refined {refrms} μm):")
        print(f"     σ around insertion : min={min(sigmas):.4f}  "
              f"max={max(sigmas):.4f}  mean={sum(sigmas)/len(sigmas):.4f}")
        print(f"     restart count      : {min(restarts)} → {max(restarts)} "
              f"(±200 evals window)")
        print(f"     archive activity   : {new_cells} new cells, "
              f"{replaced} replacements in this window")

    # ------------------------------------------------------------
    # 2) Restart events near each pick
    # ------------------------------------------------------------
    print()
    print("=" * 78)
    print("2. RESTARTS NEAR EACH PICK (within ±2000 evals)")
    print("=" * 78)
    total_restarts = 0
    for label, (stem, ev, _) in PICKS.items():
        nearby, total = find_restart_near(traj, ev, window=2000)
        total_restarts = total
        if nearby:
            for g, e, before, after in nearby:
                print(f"  {label} (eval {ev}):  RESTART at gen {g}, eval {e}  "
                      f"(restart count {before}→{after})  "
                      f"Δeval = {e - ev:+d}")
        else:
            print(f"  {label} (eval {ev}):  no restart within ±2000 evals")
    print(f"\n  Total restarts in run: {total_restarts}")

    # ------------------------------------------------------------
    # 3) ++---- elite events around each pick — are they all coming from
    #    the same exploration trajectory or independent restarts?
    # ------------------------------------------------------------
    print()
    print("=" * 78)
    print("3. ++---- ELITE EVENTS NEAR EACH PICK")
    print("=" * 78)
    target_signs = "++----"
    same_sign = [e for e in elites if e["sign_pattern"] == target_signs]
    print(f"  Total ++---- elite events in run: {len(same_sign)}")
    for label, (stem, ev, _) in PICKS.items():
        near = [e for e in same_sign if abs(e["eval"] - ev) <= 4000]
        print(f"\n  Around {label} (eval {ev}):")
        for e in near:
            mark = "  ★ THIS PICK" if e["eval"] == ev else ""
            print(f"     eval {e['eval']:>6d}  rms={e['rms_um']:>6.1f}μm  "
                  f"m=({e['m0']:+.2e}, {e['m1']:+.2e})  cossim={e['cosine_sim']:.3f}"
                  f"{mark}")

    # ------------------------------------------------------------
    # 4) Cosine similarity sequence over all ++---- elites
    # ------------------------------------------------------------
    print()
    print("=" * 78)
    print("4. ALL ++---- ELITE EVENTS — TRAJECTORY (sorted by eval)")
    print("=" * 78)
    same_sign.sort(key=lambda e: e["eval"])
    pick_evals = {ev for _, (_, ev, _) in PICKS.items()}
    for i, e in enumerate(same_sign):
        mark = ""
        for label, (_, ev, _) in PICKS.items():
            if e["eval"] == ev:
                mark = f"  ★ {label}"
        print(f"  [{i:>3d}]  eval {e['eval']:>6d}  rms={e['rms_um']:>7.1f}μm  "
              f"m=({e['m0']:+.2e}, {e['m1']:+.2e}){mark}")


if __name__ == "__main__":
    main()
