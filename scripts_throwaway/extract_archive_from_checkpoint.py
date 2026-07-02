#!/usr/bin/env python3
"""Extract archive.pkl from checkpoint.pkl when the search died mid-final-write.

Usage: extract_archive_from_checkpoint.py <run_dir>
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "/nas/longleaf/home/hisham26/Desktop/hyperlens")

run_dir = Path(sys.argv[1])
ckpt_path = run_dir / "checkpoint.pkl"
archive_path = run_dir / "archive.pkl"

if archive_path.exists():
    print(f"[skip] archive.pkl already exists: {archive_path}")
    sys.exit(0)

if not ckpt_path.exists():
    print(f"[error] checkpoint.pkl not found: {ckpt_path}", file=sys.stderr)
    sys.exit(1)

print(f"[extract] {ckpt_path} -> {archive_path}")
ckpt = pickle.load(open(ckpt_path, "rb"))
archive = getattr(ckpt, "archive", None)
if archive is None and isinstance(ckpt, dict):
    archive = ckpt.get("archive")
if archive is None:
    print(f"[error] no .archive attribute on checkpoint object (type={type(ckpt).__name__})", file=sys.stderr)
    sys.exit(2)
pickle.dump(archive, open(archive_path, "wb"))
print(f"[ok] wrote {archive_path}")
