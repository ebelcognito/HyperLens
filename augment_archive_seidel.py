"""Post-hoc add ``seidel_S1..S5`` to every cell in ``archive_cells.json``.

Older runs (before the descriptor side-channel was wired up) stored only
the 2-D BC measures, not the full 5-Seidel fingerprint.  This script
reconstructs each cell's Seidel vector deterministically from its
``theta`` using the same paraxial code the run used.  No re-evaluation
of spot RMS is needed — it's already in ``metrics.rms_spot``.

Writes ``<run_dir>/archive_cells.json`` in-place (with backup).

Usage
-----

    python augment_archive_seidel.py <run_dir> [--backup]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict

import numpy as np


def _load_cfg(run_dir: Path):
    from optical_rl.qd.lensmega.config.loader import load_config
    from optical_rl.qd.lensmega.config.schema import RunConfig
    yaml_path = run_dir / "config.yaml"
    if yaml_path.exists():
        return load_config(yaml_path)
    json_path = run_dir / "config_resolved.json"
    if json_path.exists():
        with json_path.open() as f:
            return RunConfig.model_validate(json.load(f))
    raise FileNotFoundError(f"no config in {run_dir}")


def _build_setup(cfg):
    from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
    from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
    p = cfg.problem
    template, lens = load_lens_template(
        p.lens_path,
        wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        device="cpu",
    )
    solver = GeneralizedParaxialSolver(template)
    return template, lens, solver


def _apply(lens, theta, cfg, template, solver):
    from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
    p = cfg.problem
    _apply_theta_generic(
        lens, theta,
        template=template, solver=solver,
        focal_length_target_mm=p.focal_length_target_mm,
        device="cpu",
        parameterization=p.parameterization,
        free_curvature_indices=p.free_curvature_indices,
        free_gap_indices=p.free_gap_indices,
        variable_flange=p.variable_flange,
        variable_materials=p.variable_materials,
        free_material_indices=p.free_material_indices,
    )


def _hfov_deg(lens, template, efl_target: float) -> float:
    try:
        if hasattr(lens, "rfov") and float(lens.rfov) > 0:
            return float(np.rad2deg(float(lens.rfov)))
    except Exception:
        pass
    half = template.sensor_size_mm[0] / 2.0
    return math.degrees(math.atan(half / max(abs(efl_target), 1.0)))


_LABEL_TO_SK = {
    "S1_spherical": "S1",
    "S2_coma": "S2",
    "S3_astigmatism": "S3",
    "S4_petzval": "S4",
    "S5_distortion": "S5",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument(
        "--backup", action="store_true",
        help="write archive_cells.json.bak before overwriting",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="write to this file instead of overwriting (optional)",
    )
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    archive_path = run_dir / "archive_cells.json"
    if not archive_path.exists():
        sys.exit(f"missing {archive_path}")

    from optical_rl.qd.lensmega.descriptors.seidel import (
        compute_per_element_seidel,
    )

    cfg = _load_cfg(run_dir)
    template, lens, solver = _build_setup(cfg)

    with archive_path.open() as f:
        payload = json.load(f)
    cells = payload.get("cells", [])
    print(f"[augment] {len(cells)} cells in {archive_path.name}", flush=True)

    if cells and all(
        f"seidel_S{k}" in cells[0].get("metrics", {}) for k in (1, 2, 3, 4, 5)
    ):
        print("[augment] cells already have seidel_S1..S5; nothing to do",
              flush=True)
        return

    efl_target = float(cfg.problem.focal_length_target_mm or 50.0)
    wvln = float(cfg.problem.wavelength_um)
    field_index = int(cfg.problem.descriptor_field_index)
    n_done = 0
    n_fail = 0
    for i, cell in enumerate(cells):
        theta = np.asarray(cell.get("theta", []), dtype=np.float64)
        if theta.size == 0:
            n_fail += 1
            continue
        try:
            _apply(lens, theta, cfg, template, solver)
            hfov = _hfov_deg(lens, template, efl_target)
            # Descriptor field (usually index 1 = 0.5 * hfov)
            if field_index == 0:
                field_y = 0.0
            elif field_index == 1:
                field_y = 0.5 * hfov
            else:
                field_y = hfov
            fp = compute_per_element_seidel(
                lens, template,
                field_y_deg=field_y, wavelength_um=wvln, device="cpu",
            )
            sys_total = fp["system_total"]
            metrics = cell.setdefault("metrics", {})
            for label, short in _LABEL_TO_SK.items():
                metrics[f"seidel_{short}"] = float(sys_total[label])
            n_done += 1
        except Exception as e:
            n_fail += 1
            if n_fail < 5:
                print(f"  cell {i} failed: {type(e).__name__}: {e}",
                      flush=True)
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(cells)}]  done={n_done}  fail={n_fail}",
                  flush=True)

    print(f"[augment] done={n_done} fail={n_fail}", flush=True)

    out_path = args.out or archive_path
    if out_path == archive_path and args.backup:
        backup = archive_path.with_suffix(".json.bak")
        shutil.copy2(archive_path, backup)
        print(f"[augment] backup written: {backup}", flush=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[augment] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
