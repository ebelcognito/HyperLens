"""High-N Fibonacci + random verification of the top refined designs.

For each picked design, evaluates the lensmega-aggregation RMS using:
  - Fibonacci (deterministic) at N = 256, 1024, 4096
  - Uniform random at N = 4096 (3 seeds, mean ± std)

Prints a table per design + a summary.

Picks the top-K designs by `rms_match_unpruned_um` from the paper_diagnostics
summary.csv.

Usage:
    python verify_top_designs_highn.py <pruned_dir> [--top-k 5]
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float32)

from deeplens.optics import GeoLens
from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
from optical_rl.qd.lensmega.problem.tracing import (
    trace_fields_batch, rms_spot_batch, disk_samples,
)


def fibonacci_disk(n):
    return disk_samples(int(n))   # lensmega's exact pupil pattern


def random_disk(n, seed):
    rng = np.random.default_rng(int(seed))
    theta = rng.uniform(0, 2*np.pi, size=n)
    r = np.sqrt(rng.uniform(0, 1, size=n))
    return np.stack([r*np.cos(theta), r*np.sin(theta)], axis=-1)


def build_unpruned(theta, run_dir, device="cpu"):
    cfg = load_config(str(run_dir / "config.yaml"))
    p = cfg.problem
    template, _ = load_lens_template(
        p.lens_path, wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        device=device,
        cemented_interfaces=tuple(p.cemented_interfaces),
    )
    solver = GeneralizedParaxialSolver(template)
    lens = GeoLens(filename=p.lens_path, device=device)
    _apply_theta_generic(
        lens, np.asarray(theta, dtype=float),
        template=template, solver=solver,
        focal_length_target_mm=p.focal_length_target_mm,
        device=device,
        parameterization=p.parameterization,
        free_curvature_indices=p.free_curvature_indices,
        free_gap_indices=p.free_gap_indices,
        variable_flange=p.variable_flange,
        variable_materials=p.variable_materials,
        free_material_indices=p.free_material_indices,
    )
    return lens, cfg


WAVELENGTHS_UM = [0.486, 0.587, 0.656]
FIELD_WEIGHTS = np.array([1.0, 2.0, 4.0]) / 7.0


def eval_lensmega_rms(lens, cfg, unit_disk_xy_np, n_rays):
    """Replicate lensmega's exact rms_spot aggregation:
    3 fields × 3 wavelengths × N rays via Fibonacci or random pupil pattern.
    Returns RMS in mm."""
    p = cfg.problem
    pupilz, pupilr = lens.calc_entrance_pupil()
    pupilz, pupilr = float(pupilz), float(pupilr)
    aperture_xy = torch.tensor(unit_disk_xy_np, dtype=torch.float32) * pupilr

    # hfov from EFL and r_sensor
    f = float(lens.calc_foclen())
    hfov_rad = math.atan(float(lens.r_sensor) / f)
    hfov_deg = math.degrees(hfov_rad)
    field_pts = ((0.0, 0.0), (0.0, 0.5*hfov_deg), (0.0, hfov_deg))

    rms_per_w = []
    for wvln in WAVELENGTHS_UM:
        ray = trace_fields_batch(
            lens, field_pts, aperture_xy, pupilz,
            object_depth_mm=p.object_depth_mm,
            wavelength_um=float(wvln),
            device="cpu", dtype=torch.float32,
        )
        rms, _ = rms_spot_batch(ray)   # shape (F=3,)
        rms_per_w.append(rms.detach().cpu().numpy())
    rms_all = np.stack(rms_per_w)   # (W=3, F=3)
    # lensmega aggregation: weighted-field-sum then mean across W
    rms_weighted = (rms_all * FIELD_WEIGHTS[None, :]).sum(axis=1)   # (W,)
    return float(rms_weighted.mean())   # in mm (lens units)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pruned_dir", type=str)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    pruned_dir = Path(args.pruned_dir)
    diag = pruned_dir / "paper_diagnostics" / "summary.csv"
    rows = list(csv.DictReader(open(diag)))
    def _f(r, k):
        try: return float(r[k])
        except: return float("inf")
    rows.sort(key=lambda r: _f(r, "rms_match_unpruned_um"))
    top = rows[:args.top_k]
    print(f"Picked top-{args.top_k} by rms_match_unpruned_um:")
    for r in top:
        print(f"  {r['tag']:<25}  meta={_f(r,'refined_rms_um_meta'):>7.2f}μm  "
              f"unp_match={_f(r,'rms_match_unpruned_um'):>7.2f}μm")
    print()

    run_dir = pruned_dir.parent

    headers = ["tag", "meta", "fib256", "fib1024", "fib4096",
               "rnd4k_mean", "rnd4k_std", "fib_drift", "rnd-fib"]
    widths  = [25, 7, 8, 8, 8, 10, 7, 9, 9]
    print("  ".join(f"{h:<{w}}" for h, w in zip(headers, widths)))
    print("-" * sum(widths))

    fib256  = fibonacci_disk(256)
    fib1024 = fibonacci_disk(1024)
    fib4096 = fibonacci_disk(4096)

    for r in top:
        tag = r["tag"]
        meta_path = pruned_dir / f"{tag}_meta.json"
        if not meta_path.exists():
            print(f"{tag:<25}  (no meta file, skipping)"); continue
        meta = json.load(open(meta_path))
        if "theta_refined" not in meta:
            print(f"{tag:<25}  (no theta_refined, skipping)"); continue
        lens, cfg = build_unpruned(meta["theta_refined"], run_dir)

        rms_f256  = eval_lensmega_rms(lens, cfg, fib256,  256)  * 1000
        rms_f1024 = eval_lensmega_rms(lens, cfg, fib1024, 1024) * 1000
        rms_f4096 = eval_lensmega_rms(lens, cfg, fib4096, 4096) * 1000
        rnd_vals = []
        for seed in [11, 22, 33]:
            rms_r = eval_lensmega_rms(lens, cfg, random_disk(4096, seed), 4096) * 1000
            rnd_vals.append(rms_r)
        rnd_mean = float(np.mean(rnd_vals))
        rnd_std = float(np.std(rnd_vals))

        meta_rms = _f(r, "refined_rms_um_meta")
        fib_drift = rms_f4096 - rms_f256        # how much Fibonacci converges with more rays
        rnd_minus_fib = rnd_mean - rms_f4096    # disagreement between methods

        cells = [tag, meta_rms, rms_f256, rms_f1024, rms_f4096,
                 rnd_mean, rnd_std, fib_drift, rnd_minus_fib]
        out = []
        for c, w in zip(cells, widths):
            if isinstance(c, str): out.append(f"{c:<{w}}")
            else: out.append(f"{c:<{w}.2f}")
        print("  ".join(out))


if __name__ == "__main__":
    main()
