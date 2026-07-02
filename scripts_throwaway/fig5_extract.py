#!/usr/bin/env python
"""Figure 5 data extraction + sanity render. Throwaway script.

Three deliverables substantiate the diversity-at-scale claims:
  (a) lens-form grids (sixlet/octuplet x HL/SAGA), 6 reps per pool by
      greedy eps-net at eps=0.20 in normalised theta space
  (b) sixlet aberration radar from per-design Seidel S1..S5
  (c) eps-distinct curves for triplet, sixlet, octuplet x {HL, SAGA}

All Seidel coefficients are read from pre-stored design metrics (no
recomputation). Lens geometry is reconstructed by replaying theta
through the existing FullDOFLensProblem / GeoLens machinery. Spot
diagrams use trace_fields_batch from the canonical tracing utilities.

Note on HL paths: the brief points to results/diversity_run_*/<sys>/
seed42/hl/ but the diversity_run dir holds only configs+logs. The
actual HL run dirs are under results/lensmega/. Steelman SA+GA paths
match the brief.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Match runner threading discipline (used by the Seidel/full_dof code).
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import yaml

import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

OUT = ROOT / "figures" / "fig5_data"
OUT.mkdir(parents=True, exist_ok=True)

# -------------------- per-system data sources --------------------
# (system_name, theta_dim, hl_dir, saga_dir, hl_config_yaml, lens_json)
SYSTEMS = {
    "triplet": dict(
        hl_json=ROOT / "results/lensmega/cooke_triplet_materials_modernstack_seed42_20260506_033503/refined_top500_seed42_hl.json",
        saga_json=ROOT / "results/steelman_div020_20260506_123801/triplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json",
        cfg=ROOT / "results/diversity_run_20260506_033447/configs/hl_triplet_seed42.yaml",
    ),
    "sixlet": dict(
        hl_json=ROOT / "results/lensmega/sixlet_apsc_discovery_v2_seed42_20260506_034601/refined_top500_seed42_hl.json",
        saga_json=ROOT / "results/steelman_div020_20260506_123801/sixlet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json",
        cfg=ROOT / "results/diversity_run_20260506_033447/configs/hl_sixlet_seed42.yaml",
    ),
    "octuplet": dict(
        hl_json=ROOT / "results/lensmega/octuplet_apsc_2cement_v2_seed42_20260506_040025/refined_top500_seed42_hl.json",
        saga_json=ROOT / "results/steelman_div020_20260506_123801/octuplet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json",
        cfg=ROOT / "results/diversity_run_20260506_033447/configs/hl_octuplet_seed42.yaml",
    ),
}

EPS_GRID = np.arange(0.05, 0.5 + 1e-9, 0.025)
EPS_TARGET = 0.20
RMS_LIMIT_UM = 1000.0
N_REP = 6

# -------------------- bounds-aware normalisation --------------------

def load_bounds(cfg_path: Path) -> np.ndarray:
    cfg = yaml.safe_load(cfg_path.read_text())
    return np.array(cfg["problem"]["theta_bounds"], dtype=np.float64)


def norm_theta(theta: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = np.where((hi - lo) > 0.0, hi - lo, 1.0)
    return (np.asarray(theta, dtype=np.float64).reshape(-1) - lo) / span


# -------------------- pool loaders --------------------

def load_pool(json_path: Path, bounds: np.ndarray) -> List[dict]:
    with json_path.open() as f:
        d = json.load(f)
    designs = d["designs"]
    pool: List[dict] = []
    for i, des in enumerate(designs):
        rms_um = float(des["rms_um"])
        if rms_um >= RMS_LIMIT_UM:
            continue
        theta = np.asarray(des["theta"], dtype=np.float64).reshape(-1)
        m = des["metrics"]
        pool.append({
            "idx": i,
            "theta": theta,
            "theta_norm": norm_theta(theta, bounds),
            "rms_um": rms_um,
            "refined_fitness": float(des["refined_fitness"]),
            "raw_fitness": float(des["raw_fitness"]),
            "S1": float(m["seidel_S1"]),
            "S2": float(m["seidel_S2"]),
            "S3": float(m["seidel_S3"]),
            "S4": float(m["seidel_S4"]),
            "S5": float(m["seidel_S5"]),
            "efl_mm": float(des["efl_mm"]),
            "fnum_actual": float(des["fnum_actual"]),
            "raw_rms_um": float(des.get("raw_rms_um", float("nan"))),
        })
    # Sort best-first by refined_fitness DESC (matches fig4 convention).
    pool.sort(key=lambda r: -r["refined_fitness"])
    return pool


# -------------------- eps-net selection --------------------

def eps_net_select(pool: List[dict], eps: float, dim: int):
    """Greedy eps-net. Returns (accepted_indices_in_pool, distances_to_nearest_at_accept_time)."""
    accepted: List[int] = []
    accepted_norms: List[np.ndarray] = []
    sqrtD = float(np.sqrt(dim))
    for i, r in enumerate(pool):
        v = r["theta_norm"]
        ok = True
        for a in accepted_norms:
            if np.linalg.norm(v - a) / sqrtD <= eps:
                ok = False
                break
        if ok:
            accepted.append(i)
            accepted_norms.append(v)
    return accepted


def eps_distinct_curve(pool: List[dict], eps_grid: np.ndarray, dim: int) -> List[int]:
    return [len(eps_net_select(pool, float(e), dim)) for e in eps_grid]


# -------------------- Selection of 6 representatives (with padding) --------------------

def select_representatives(pool: List[dict], dim: int) -> Tuple[List[dict], List[bool]]:
    """Pick N_REP designs. First fill via eps-net at EPS_TARGET; if fewer than
    N_REP eps-distinct, pad with the next best-fitness designs that were
    rejected (i.e. nearest to already-selected). Returns (selected, is_padded).
    """
    accepted_idx = eps_net_select(pool, EPS_TARGET, dim)
    selected = [pool[i] for i in accepted_idx[:N_REP]]
    is_padded = [False] * len(selected)
    if len(selected) >= N_REP:
        return selected[:N_REP], is_padded[:N_REP]
    # Pad with rejected designs in fitness order, skipping already-accepted.
    accepted_set = set(accepted_idx[:N_REP])
    for i, r in enumerate(pool):
        if i in accepted_set:
            continue
        selected.append(r)
        is_padded.append(True)
        if len(selected) == N_REP:
            break
    return selected, is_padded


# -------------------- lens geometry reconstruction --------------------

def build_lens_template_and_solver(cfg_path: Path):
    """Return (lens_template, solver, prob_cfg, lens_json_path) for use by both
    SeidelDescriptor (recompute fallback) and the rendering apply-theta path.
    """
    from optical_rl.qd.lensmega.config.loader import load_config
    from optical_rl.qd.lensmega.problem.lens_problem import LensProblemConfig
    from optical_rl.qd.lensmega.problem.full_dof import FullDOFParaxialSolver
    from optical_rl.qd.lensmega.problem.lens_template import load_lens_template

    cfg = load_config(cfg_path)
    p = cfg.problem
    device = "cpu"
    lens_path = ROOT / p.lens_path
    template, _ = load_lens_template(
        str(lens_path),
        wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        device=device,
        cemented_interfaces=tuple(p.cemented_interfaces),
    )
    solver = FullDOFParaxialSolver(template)
    prob_cfg = LensProblemConfig(
        lens_path=str(lens_path),
        wavelength_um=p.wavelength_um,
        fitness_wavelengths_um=tuple(p.fitness_wavelengths_um),
        object_depth_mm=p.object_depth_mm,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        focal_length_target_mm=p.focal_length_target_mm,
        theta_bounds=tuple(tuple(b) for b in p.theta_bounds),
        stage=p.stage,
        num_rays=p.num_rays,
        invalid_tol=p.invalid_tol,
        invalid_penalty_weight=p.invalid_penalty_weight,
        intersect_penalty_weight=p.intersect_penalty_weight,
        air_min_mm=p.air_min_mm,
        thick_min_mm=p.thick_min_mm,
        strict_entrance_pupil=p.strict_entrance_pupil,
        descriptor_field_index=p.descriptor_field_index,
        parameterization=p.parameterization,
        efl_penalty_weight=p.efl_penalty_weight,
        free_gap_indices=tuple(p.free_gap_indices),
        variable_flange=p.variable_flange,
        flange_bounds=tuple(p.flange_bounds) if p.flange_bounds else None,
        variable_materials=p.variable_materials,
        free_material_indices=tuple(p.free_material_indices),
        cemented_interfaces=tuple(p.cemented_interfaces),
        field_weights=tuple(p.field_weights) if p.field_weights else None,
        device=device,
        enforce_fnum=p.enforce_fnum,
        f_number_target=p.f_number_target,
        fnum_penalty_weight=p.fnum_penalty_weight,
    )
    return template, solver, prob_cfg, lens_path, p


def make_geolens(prob_cfg, p_problem, lens_path, theta: np.ndarray, template, solver, device="cpu"):
    """Apply theta to a fresh GeoLens via _apply_theta_generic."""
    from deeplens.optics import GeoLens
    from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
    lens = GeoLens(filename=str(lens_path), device=device)
    _apply_theta_generic(
        lens, theta,
        template=template,
        solver=solver,
        focal_length_target_mm=p_problem.focal_length_target_mm,
        device=device,
        parameterization=p_problem.parameterization,
        free_curvature_indices=p_problem.free_curvature_indices,
        free_gap_indices=p_problem.free_gap_indices,
        variable_flange=p_problem.variable_flange,
        variable_materials=p_problem.variable_materials,
        free_material_indices=p_problem.free_material_indices,
    )
    return lens


def extract_geometry(lens) -> dict:
    """Return curvatures, axial positions, gaps, semi-apertures, materials, sensor info."""
    surfs = []
    for i, s in enumerate(lens.surfaces):
        m1 = getattr(s, "mat1", None)
        m2 = getattr(s, "mat2", None)
        c_attr = getattr(s, "c", None)
        if c_attr is None:
            c_val = 0.0  # stop / aperture surfaces are flat
        elif torch.is_tensor(c_attr):
            c_val = float(c_attr.detach().cpu().item())
        else:
            c_val = float(c_attr)
        rec = {
            "i": i,
            "type": type(s).__name__,
            "c": c_val,
            "d": float(s.d.detach().cpu().item()) if torch.is_tensor(s.d) else float(s.d),
            "r": float(s.r),
            "is_stop": "Aperture" in type(s).__name__ or getattr(s, "is_stop", False),
        }
        for tag, m in [("mat1", m1), ("mat2", m2)]:
            if m is None:
                rec[f"{tag}_name"] = None
                rec[f"{tag}_n"] = None
                rec[f"{tag}_V"] = None
                continue
            n_attr = getattr(m, "n", None)
            V_attr = getattr(m, "V", None)
            try:
                n_val = float(n_attr.detach().cpu().item()) if torch.is_tensor(n_attr) else float(n_attr)
            except Exception:
                n_val = None
            try:
                V_val = float(V_attr.detach().cpu().item()) if torch.is_tensor(V_attr) else float(V_attr)
            except Exception:
                V_val = None
            rec[f"{tag}_name"] = getattr(m, "name", None)
            rec[f"{tag}_n"] = n_val
            rec[f"{tag}_V"] = V_val
        surfs.append(rec)
    d_sensor = float(lens.d_sensor.detach().cpu().item()) if torch.is_tensor(lens.d_sensor) else float(lens.d_sensor)
    return {
        "surfaces": surfs,
        "d_sensor_mm": d_sensor,
        "r_sensor_mm": float(getattr(lens, "r_sensor", float("nan"))),
        "foclen_mm": float(getattr(lens, "foclen", float("nan"))),
    }


# -------------------- spot diagrams --------------------

def compute_spot_endpoints(lens, prob_cfg, p_problem, num_rays: int = 256) -> np.ndarray:
    """3 fields x 3 wavelengths x num_rays. Returns (3, 3, num_rays, 2) sensor-plane xy."""
    from optical_rl.qd.lensmega.problem.tracing import (
        disk_samples, trace_fields_batch,
    )
    import math

    device = "cpu"
    dtype = torch.float64
    aperture = torch.from_numpy(disk_samples(num_rays)).to(dtype=dtype, device=device)

    # entrance pupil for batched targeting
    with torch.no_grad():
        pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)
    pupil_z = float(pupilz)
    pupil_r = float(pupilr)
    aperture_xy = aperture * pupil_r

    # 3 fields = (0, 0.5*hfov, hfov)
    half_sensor = float(lens.r_sensor)  # half-diagonal already (APS-C r_sensor=17 in templates)
    efl = float(getattr(lens, "foclen", float(p_problem.focal_length_target_mm)))
    # use sensor_size_mm[0]/2 same as runner (matches stage2 convention)
    template_sensor_half = lens.sensor_size[0] / 2.0 if hasattr(lens, "sensor_size") else half_sensor
    hfov_deg = math.degrees(math.atan(template_sensor_half / abs(efl)))
    fields = ((0.0, 0.0), (0.0, 0.5 * hfov_deg), (0.0, hfov_deg))

    out = np.full((3, 3, num_rays, 2), np.nan, dtype=np.float32)
    wvlns = list(p_problem.fitness_wavelengths_um)
    for wi, w in enumerate(wvlns):
        ray = trace_fields_batch(
            lens, fields, aperture_xy=aperture_xy, pupil_z=pupil_z,
            object_depth_mm=p_problem.object_depth_mm, wavelength_um=float(w),
            device=device, dtype=dtype,
        )
        valid = ray.is_valid.detach().cpu().numpy()  # (F, N)
        xy = ray.o[..., :2].detach().cpu().numpy()  # (F, N, 2)
        # zero-out invalid rays
        for f in range(3):
            mask = valid[f]
            out[f, wi, :, :] = np.where(mask[:, None], xy[f], np.nan)
    return out


# -------------------- main --------------------

def main():
    expected_eps20 = {
        "triplet": {"hl": 39, "saga": 24},
        "sixlet": {"hl": 86, "saga": 16},
        "octuplet": {"hl": 31, "saga": 13},
    }
    expected_pool_size = {
        "sixlet": {"hl": 500, "saga": 16},
        "octuplet": {"hl": 310, "saga": 13},  # spec said ~500 for HL, actual is 310 (= n_refined_ok)
    }

    print("=" * 70)
    print("Fig 5 data extraction")
    print("=" * 70)

    # Pre-flight: existence
    for sys_name, info in SYSTEMS.items():
        for k in ("hl_json", "saga_json", "cfg"):
            if not info[k].exists():
                sys.exit(f"MISSING: {info[k]}")

    # ---- Load all 6 pools ----
    pools = {}
    bounds_by_sys = {}
    for sys_name, info in SYSTEMS.items():
        bounds = load_bounds(info["cfg"])
        bounds_by_sys[sys_name] = bounds
        pools[(sys_name, "hl")] = load_pool(info["hl_json"], bounds)
        pools[(sys_name, "saga")] = load_pool(info["saga_json"], bounds)
        D = bounds.shape[0]
        print(f"  {sys_name:9s}  D={D:2d}  HL pool={len(pools[(sys_name,'hl')]):4d}  "
              f"SAGA pool={len(pools[(sys_name,'saga')]):4d}")

    # ---- Part C: eps-distinct curves ----
    print("\nPART C: eps-distinct curves")
    eps_curves = {}
    for sys_name, info in SYSTEMS.items():
        D = bounds_by_sys[sys_name].shape[0]
        for method in ("hl", "saga"):
            pool = pools[(sys_name, method)]
            counts = eps_distinct_curve(pool, EPS_GRID, D)
            eps_curves[(sys_name, method)] = counts
            csv = OUT / f"eps_curves_{sys_name}_{method}.csv"
            with csv.open("w") as f:
                f.write("epsilon,count\n")
                for e, c in zip(EPS_GRID, counts):
                    f.write(f"{e:.6f},{c}\n")
            # find count at eps=0.20
            i20 = int(np.argmin(np.abs(EPS_GRID - EPS_TARGET)))
            print(f"  {sys_name:9s} {method:5s}  curve[0.20]={counts[i20]:4d} "
                  f"(expected {expected_eps20[sys_name][method]})")

    # ---- Sanity: pool sizes & eps-distinct match ----
    print("\nSanity checks (pool sizes and eps@0.20 vs experiment log)")
    mismatches = []
    for sys_name in ("sixlet", "octuplet"):
        for method in ("hl", "saga"):
            got = len(pools[(sys_name, method)])
            exp = expected_pool_size[sys_name][method]
            tol = 50 if method == "hl" else 1
            ok = abs(got - exp) <= tol
            print(f"  pool size {sys_name:9s} {method:5s}: {got} (expected ~{exp}) {'OK' if ok else 'MISMATCH'}")
            if not ok:
                mismatches.append(f"pool {sys_name}/{method}: {got} vs {exp}")
    for sys_name in ("triplet", "sixlet", "octuplet"):
        for method in ("hl", "saga"):
            i20 = int(np.argmin(np.abs(EPS_GRID - EPS_TARGET)))
            got = eps_curves[(sys_name, method)][i20]
            exp = expected_eps20[sys_name][method]
            ok = abs(got - exp) <= 2
            print(f"  eps@0.20 {sys_name:9s} {method:5s}: {got} (expected {exp}) {'OK' if ok else 'MISMATCH'}")
            if not ok:
                mismatches.append(f"eps@0.20 {sys_name}/{method}: {got} vs {exp}")

    if mismatches:
        print("\n!!! MISMATCHES !!!")
        for m in mismatches:
            print("  ", m)
        sys.exit(2)

    # SA+GA flatness check below saturation
    print("\nSaturation check (SA+GA curves should be flat below eps=0.20)")
    for sys_name in ("triplet", "sixlet", "octuplet"):
        counts = eps_curves[(sys_name, "saga")]
        sub = [c for e, c in zip(EPS_GRID, counts) if e <= 0.20 + 1e-9]
        is_flat = len(set(sub)) == 1
        print(f"  {sys_name:9s} SAGA below 0.20: {sub} -> {'FLAT' if is_flat else 'NOT FLAT'}")

    # ---- Part B: per-design Seidel (sixlet) + envelopes ----
    print("\nPART B: sixlet Seidel envelopes")
    sd_rows = []
    for method in ("hl", "saga"):
        for r in pools[("sixlet", method)]:
            sd_rows.append({
                "system": "sixlet", "method": method, "design_id": r["idx"],
                "S_1": r["S1"], "S_2": r["S2"], "S_3": r["S3"],
                "S_4": r["S4"], "S_5": r["S5"], "refined_RMS_um": r["rms_um"],
            })
    csv = OUT / "seidel_per_design.csv"
    with csv.open("w") as f:
        f.write("system,method,design_id,S_1,S_2,S_3,S_4,S_5,refined_RMS_um\n")
        for row in sd_rows:
            f.write(",".join([
                row["system"], row["method"], str(row["design_id"]),
                f"{row['S_1']:.6e}", f"{row['S_2']:.6e}", f"{row['S_3']:.6e}",
                f"{row['S_4']:.6e}", f"{row['S_5']:.6e}", f"{row['refined_RMS_um']:.4f}",
            ]) + "\n")
    print(f"  wrote {csv}")

    # Envelope stats
    abs_pools = {}
    for method in ("hl", "saga"):
        arr = np.array([
            [abs(r[k]) for k in ("S1", "S2", "S3", "S4", "S5")]
            for r in pools[("sixlet", method)]
        ])
        abs_pools[method] = arr  # (N, 5)
    # Normalisation constants: max across both methods per axis
    norm_const = np.maximum(abs_pools["hl"].max(axis=0), abs_pools["saga"].max(axis=0))
    norm_const = np.where(norm_const > 0, norm_const, 1.0)

    radar = {}
    terms = ["S_1", "S_2", "S_3", "S_4", "S_5"]
    for method in ("hl", "saga"):
        arr = abs_pools[method]
        d = {}
        for j, t in enumerate(terms):
            col = arr[:, j]
            d[t] = {
                "p5": float(np.percentile(col, 5)),
                "p50": float(np.percentile(col, 50)),
                "p95": float(np.percentile(col, 95)),
            }
        d["normalisation_constants"] = {t: float(norm_const[j]) for j, t in enumerate(terms)}
        radar[f"sixlet_{method}"] = d
    (OUT / "radar_envelopes.json").write_text(json.dumps(radar, indent=2))
    print(f"  wrote {OUT / 'radar_envelopes.json'}")

    # Envelope check: HL p95 > SA+GA p95 on >=3 of 5 axes
    p95_diff = [
        radar["sixlet_hl"][t]["p95"] - radar["sixlet_saga"][t]["p95"] for t in terms
    ]
    larger = sum(1 for x in p95_diff if x > 0)
    print(f"  HL p95 > SAGA p95 on {larger}/5 axes: {'OK' if larger >= 3 else 'MISMATCH'}")
    print(f"  per-axis (HL p95 - SAGA p95): " +
          ", ".join(f"{t}={p95_diff[j]:+.3e}" for j, t in enumerate(terms)))

    # ---- Part A: select 6 reps per pool (sixlet, octuplet) x (hl, saga) ----
    print("\nPART A: rep selection + lens geometry + spot diagrams")
    selected_designs = {}
    spot_data = {}

    for sys_name in ("sixlet", "octuplet"):
        bounds = bounds_by_sys[sys_name]
        D = bounds.shape[0]
        template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(SYSTEMS[sys_name]["cfg"])
        for method in ("hl", "saga"):
            pool = pools[(sys_name, method)]
            sel, padded_flags = select_representatives(pool, D)
            print(f"  {sys_name:9s} {method:5s}: pool={len(pool)}  selected={len(sel)}  padded={sum(padded_flags)}")

            entries = []
            for k, (r, padded) in enumerate(zip(sel, padded_flags)):
                # Build lens, extract geometry
                try:
                    lens = make_geolens(prob_cfg, p_problem, lens_path, r["theta"], template, solver)
                    geom = extract_geometry(lens)
                    spot_xy = compute_spot_endpoints(lens, prob_cfg, p_problem, num_rays=256)
                    spot_ok = True
                except Exception as e:
                    print(f"    WARNING: design {r['idx']} ({sys_name}/{method}) failed lens build: {e}")
                    geom = None
                    spot_xy = np.full((3, 3, 256, 2), np.nan, dtype=np.float32)
                    spot_ok = False

                entries.append({
                    "rank": k,
                    "padded": bool(padded),
                    "design_id": r["idx"],
                    "theta": r["theta"].tolist(),
                    "S_1": r["S1"], "S_2": r["S2"], "S_3": r["S3"],
                    "S_4": r["S4"], "S_5": r["S5"],
                    "refined_RMS_um": r["rms_um"],
                    "refined_fitness": r["refined_fitness"],
                    "efl_mm": r["efl_mm"],
                    "fnum_actual": r["fnum_actual"],
                    "lens_geometry": geom,
                })
                key = f"{sys_name}_{method}_{k}"
                spot_data[key] = spot_xy

            # Pairwise diversity check
            if len(sel) >= 2:
                pdists = []
                for a in range(len(sel)):
                    for b in range(a + 1, len(sel)):
                        d = np.linalg.norm(sel[a]["theta_norm"] - sel[b]["theta_norm"]) / np.sqrt(D)
                        pdists.append(d)
                min_pd = min(pdists)
                if method == "hl" and not any(padded_flags):
                    ok = min_pd > EPS_TARGET - 1e-6
                    print(f"    pairwise min dist = {min_pd:.4f}  (expect > {EPS_TARGET}: {'OK' if ok else 'WARN'})")
                else:
                    print(f"    pairwise min dist = {min_pd:.4f}  (padded pool, expected to be smaller)")

            selected_designs[f"{sys_name}_{method}"] = entries

    (OUT / "selected_designs.json").write_text(json.dumps(selected_designs, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else None))
    print(f"  wrote {OUT / 'selected_designs.json'}")

    # ---- Save spot diagrams ----
    np.savez(OUT / "spot_diagrams.npz", **spot_data)
    print(f"  wrote {OUT / 'spot_diagrams.npz'} ({len(spot_data)} entries)")

    # ---- Done ----
    print("\nDone with extraction.")
    print(f"All outputs under {OUT}")
    return selected_designs, eps_curves, radar, pools


if __name__ == "__main__":
    main()
