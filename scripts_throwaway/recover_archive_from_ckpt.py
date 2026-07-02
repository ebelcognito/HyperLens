#!/usr/bin/env python3
"""Recover archive.pkl from checkpoint.pkl when search died mid-write.

Per RUNBOOK_triplet_hl_ablation.md pitfall #2 / #6.

Usage: recover_archive_from_ckpt.py <run_dir>
"""
import pickle
import sys
from pathlib import Path

# Project root must be on sys.path so pickle can resolve `optical_rl.*` classes
# (GridArchive, CMAMEGACheckpoint, etc.) when loading checkpoint.pkl.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

run_dir = Path(sys.argv[1]).resolve()
ckpt_path = run_dir / 'checkpoint.pkl'
arc_path = run_dir / 'archive.pkl'

if arc_path.exists():
    print(f"[recover] archive.pkl already exists at {arc_path}, skipping")
    sys.exit(0)

if not ckpt_path.exists():
    print(f"[recover] ERROR: no checkpoint.pkl at {ckpt_path}", file=sys.stderr)
    sys.exit(1)

with open(ckpt_path, 'rb') as f:
    ckpt = pickle.load(f)

archive = ckpt.archive
with open(arc_path, 'wb') as f:
    pickle.dump(archive, f)

print(f"[recover] wrote {arc_path}  (gen={ckpt.generation}, evals={ckpt.completed_evaluations}, best_fit={ckpt.best_fitness:.4f})")
