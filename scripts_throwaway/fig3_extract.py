#!/usr/bin/env python
"""Figure 3 data extraction. Throwaway script.

Steered-descent demonstration on canonical Cooke triplet:
  Branch A: descent direction = -sign(M_S2) * grad_M_S2 / ||grad_M_S2||
  Branch B: descent direction = -sign(M_S3) * grad_M_S3 / ||grad_M_S3||
  K=30 steps in unit-cube parameter space, then L-BFGS refine each
  endpoint with paper's standard refinement penalty config.

Outputs go to figures/fig3_data/.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import sys, json, copy, argparse
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem

CFG_PATH = ROOT / "optical_rl/qd/lensmega/config/examples/cooke_triplet_materials_modernstack.yaml"
ARCHIVE_PATH = ROOT / "results/lensmega/cooke_triplet_materials_modernstack_seed42_20260506_033503/archive.pkl"
OUT = ROOT / "figures/fig3_data"
OUT.mkdir(parents=True, exist_ok=True)

# fields/wavelengths for spot diagrams (3 fields x 3 wavelengths)
FIELD_FRACS = (0.0, 0.5, 1.0)
N_RAYS_PER_FW = 256

# steered descent params
K_STEPS = 30
ETA = 0.020  # nominal upper-bound step. run_descent backtracks per-step (halves up
             # to 7x) until the step keeps L < 2x current; aborts the walk if no
             # feasible step exists or L exceeds 5.0. So this is an upper cap; each
             # branch walks as far as the local landscape permits.

# selection thresholds for theta_0
THRESH_S2 = 5.0e-5
THRESH_S3 = 5.0e-5
THRESH_L_MAX = 30.0  # random init RMS spans ~3..3000+; <30 means basin is reachable
# Both branches must refine to spot RMS < this (L = fitness_neg ≈ rms_mm)
REFINED_L_MAX = 0.030   # ~30 um RMS


def load_problem(refine: bool = False):
    cfg = load_config(CFG_PATH)
    cfg.descriptor.params = {"terms": ["S1", "S2", "S3", "S4", "S5"]}
    if refine:
        io = cfg.io
        if io.refine_efl_penalty_weight is not None:
            cfg.problem.efl_penalty_weight = io.refine_efl_penalty_weight
        if io.refine_invalid_penalty_weight is not None:
            cfg.problem.invalid_penalty_weight = io.refine_invalid_penalty_weight
        if io.refine_intersect_penalty_weight is not None:
            cfg.problem.intersect_penalty_weight = io.refine_intersect_penalty_weight
        if io.refine_field_weights is not None:
            cfg.problem.field_weights = tuple(io.refine_field_weights)
        if (cfg.problem.enforce_fnum
                and io.refine_fnum_penalty_weight is not None):
            cfg.problem.fnum_penalty_weight = io.refine_fnum_penalty_weight
    desc = _build_descriptor(cfg)
    prob = _build_problem(cfg, desc)
    prob._ensure_backend()
    return cfg, prob


def cos_sim(u, v):
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu == 0 or nv == 0: return 0.0
    return float(np.dot(u, v) / (nu * nv))


def run_descent(prob, theta_0, lo, hi, k_idx, K, eta,
                max_step_grow=2.0, l_abort=5.0):
    """k_idx is index in descriptor: m1=S2, m2=S3.
    Per-step backtracking line search: shrink eta until the step's L is
    < max_step_grow * current L. If even eta/64 still violates that, OR
    if L > l_abort, stop the walk and return where we are.
    """
    range_ = hi - lo
    theta = theta_0.astype(np.float64).copy()
    rows = []
    aborted = False
    for step in range(K + 1):
        r = prob.evaluate(theta, with_grads=(step < K and not aborted))
        M = float(r.measures[k_idx])
        L = -float(r.fitness)
        S2 = float(r.measures[1])
        S3 = float(r.measures[2])
        dist = float(np.linalg.norm((theta - theta_0) / range_))
        rows.append({
            "step": step, "L": L, "S_2": S2, "S_3": S3,
            "M_target": M, "M_target_abs": abs(M),
            "distance_from_start_unit": dist,
        })
        if step == K or aborted:
            if aborted and step < K:
                continue
            break
        if L > l_abort:
            aborted = True
            continue
        gM = r.grads[f"m{k_idx}"]
        gM_unit = gM * range_
        norm = np.linalg.norm(gM_unit)
        if norm < 1e-30:
            break
        direction = -np.sign(M) * gM_unit / norm
        u = (theta - lo) / range_
        eta_try = eta
        accepted = False
        for _ in range(7):  # halve down to eta/64
            u_new = np.clip(u + eta_try * direction, 0.0, 1.0)
            theta_new = lo + u_new * range_
            r_try = prob.evaluate(theta_new, with_grads=False)
            L_try = -float(r_try.fitness)
            if L_try < max_step_grow * max(L, 0.1):
                theta = theta_new
                accepted = True
                break
            eta_try *= 0.5
        if not accepted:
            aborted = True
    return theta, rows


def lbfgs_refine(prob_r, theta_init, cfg, lo, hi, max_steps=100):
    """Mirrors optical_rl/qd/lensmega/core/runner.py:236-269."""
    io = cfg.io
    theta_t = torch.tensor(theta_init.copy(), dtype=torch.float32, device="cpu",
                           requires_grad=True)
    optimizer = torch.optim.LBFGS([theta_t], lr=io.refine_lbfgs_lr, max_iter=10,
                                  line_search_fn="strong_wolfe")
    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)
    converge_tol = io.refine_converge_tol
    best_loss = float("inf")
    best_theta = theta_init.copy()
    lbfgs_steps = 0
    converged = False
    no_improve = 0
    for step in range(max_steps):
        def closure():
            optimizer.zero_grad()
            with torch.no_grad():
                theta_t.clamp_(lo_t, hi_t)
            fitness, _, _ = prob_r._fitness_and_metrics(theta_t)
            loss = -fitness
            if loss.requires_grad:
                loss.backward()
            return loss
        loss_val = optimizer.step(closure)
        lbfgs_steps = step + 1
        if loss_val is not None:
            lv = float(loss_val)
            if lv < best_loss - 1e-8:
                best_loss = lv
                no_improve = 0
                with torch.no_grad():
                    best_theta = theta_t.detach().cpu().numpy().copy()
            else:
                no_improve += 1
        if theta_t.grad is not None:
            gn = float(theta_t.grad.norm().item())
            if gn < converge_tol:
                converged = True; break
        if no_improve >= 5:
            converged = True; break
    best_theta = np.clip(best_theta.astype(np.float64), lo, hi)
    return best_theta, lbfgs_steps, converged


def fibonacci_circle(n, r, z):
    """Golden-angle Fibonacci spiral pupil sample. Returns (n,3) tensor."""
    golden = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n)
    rho = np.sqrt((i + 0.5) / n) * r
    phi = i * golden
    x = rho * np.cos(phi); y = rho * np.sin(phi)
    z_arr = np.full(n, z)
    pts = np.stack([x, y, z_arr], axis=-1)
    return torch.tensor(pts, dtype=torch.float32)


def trace_spot(theta, cfg):
    """Trace 256 rays per (field, wavelength) on Fibonacci spiral.
    Returns dict of arrays {x_sensor, y_sensor, field_index, wavelength_index, valid}."""
    from deeplens.optics import GeoLens  # noqa: F401
    from deeplens.optics.light import Ray
    from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
    from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
    from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver

    p = cfg.problem
    template, lens = load_lens_template(
        p.lens_path, wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        device="cpu", cemented_interfaces=tuple(p.cemented_interfaces),
    )
    solver = GeneralizedParaxialSolver(template)
    _apply_theta_generic(
        lens, theta, template=template, solver=solver,
        focal_length_target_mm=p.focal_length_target_mm, device="cpu",
        parameterization=p.parameterization,
        free_curvature_indices=p.free_curvature_indices,
        free_gap_indices=p.free_gap_indices,
        variable_flange=p.variable_flange,
        variable_materials=p.variable_materials,
        free_material_indices=p.free_material_indices,
    )

    # half FOV from sensor + EFL
    half_sensor = template.sensor_size_mm[0] / 2.0
    efl = float(lens.foclen)
    hfov_deg = float(np.degrees(np.arctan(half_sensor / max(abs(efl), 1.0))))
    pupilz, pupilr = lens.get_entrance_pupil()
    pupilz = float(pupilz); pupilr = float(pupilr)
    depth = -20000.0
    wvlns = list(p.fitness_wavelengths_um)

    xs, ys, fi, wi, valid = [], [], [], [], []
    for f_idx, frac in enumerate(FIELD_FRACS):
        fov_y_deg = frac * hfov_deg
        # field point in object space (y axis)
        y0 = np.tan(np.radians(fov_y_deg)) * depth
        ray_o = np.array([0.0, y0, depth], dtype=np.float32)
        for w_idx, wvln in enumerate(wvlns):
            pupil_pts = fibonacci_circle(N_RAYS_PER_FW, pupilr, pupilz)  # (N,3)
            ray_o_t = torch.tensor(ray_o, dtype=torch.float32).unsqueeze(0).repeat(N_RAYS_PER_FW, 1)
            ray_d = pupil_pts - ray_o_t
            ray = Ray(ray_o_t, ray_d, float(wvln), device="cpu")
            with torch.no_grad():
                ray_out = lens.trace2sensor(ray)
            o = ray_out.o.detach().cpu().numpy()
            v = ray_out.is_valid.detach().cpu().numpy().astype(bool)
            xs.append(o[..., 0]); ys.append(o[..., 1])
            fi.append(np.full(N_RAYS_PER_FW, f_idx))
            wi.append(np.full(N_RAYS_PER_FW, w_idx))
            valid.append(v)

    return {
        "x_sensor": np.concatenate(xs),
        "y_sensor": np.concatenate(ys),
        "field_index": np.concatenate(fi),
        "wavelength_index": np.concatenate(wi),
        "valid": np.concatenate(valid),
        "wavelengths_um": np.array(wvlns, dtype=np.float64),
        "field_fractions": np.array(FIELD_FRACS, dtype=np.float64),
        "hfov_deg": hfov_deg,
        "d_sensor_mm": float(lens.d_sensor),
        "efl_mm": efl,
    }


def lens_geometry(theta, cfg):
    """Surface curvatures, thicknesses, materials, clear apertures."""
    from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
    from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
    from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver

    p = cfg.problem
    template, lens = load_lens_template(
        p.lens_path, wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        device="cpu", cemented_interfaces=tuple(p.cemented_interfaces),
    )
    solver = GeneralizedParaxialSolver(template)
    _apply_theta_generic(
        lens, theta, template=template, solver=solver,
        focal_length_target_mm=p.focal_length_target_mm, device="cpu",
        parameterization=p.parameterization,
        free_curvature_indices=p.free_curvature_indices,
        free_gap_indices=p.free_gap_indices,
        variable_flange=p.variable_flange,
        variable_materials=p.variable_materials,
        free_material_indices=p.free_material_indices,
    )
    surfaces = []
    for i, s in enumerate(lens.surfaces):
        cls = type(s).__name__
        d_axial = float(s.d.item()) if hasattr(s, "d") else None
        c = None
        if hasattr(s, "c"):
            c_attr = s.c
            try: c = float(c_attr.item() if hasattr(c_attr,'item') else c_attr)
            except: c = None
        n_after = float(s.n_after) if hasattr(s, "n_after") else None
        v_after = float(s.v_after) if hasattr(s, "v_after") else None
        r = float(s.r) if hasattr(s, "r") else None
        is_stop = bool(getattr(s, "is_stop", False))
        surfaces.append({
            "index": i, "type": cls, "is_stop": is_stop,
            "curvature_inv_mm": c,
            "d_axial_mm": d_axial,
            "n_after": n_after,
            "v_after": v_after,
            "clear_aperture_mm": r,
        })
    return {
        "foclen_mm": float(lens.foclen),
        "d_sensor_mm": float(lens.d_sensor),
        "sensor_size_mm": list(template.sensor_size_mm),
        "surfaces": surfaces,
    }


# ----------------- driver -----------------
def try_seed(seed, cfg, prob, prob_r, lo, hi, eta=ETA, K=K_STEPS, max_steps_lbfgs=100,
             theta0_override=None, label=None):
    """If theta0_override is provided, use that; otherwise sample randomly."""
    if theta0_override is not None:
        theta0 = np.asarray(theta0_override, dtype=np.float64)
        r0 = prob.evaluate(theta0, with_grads=False)
        L0 = -r0.fitness
        cand = 0
        if not (abs(r0.measures[1]) > THRESH_S2 and abs(r0.measures[2]) > THRESH_S3):
            return None
    else:
        rng = np.random.default_rng(seed)
        chosen = None
        for cand in range(5):
            u0 = rng.random(len(lo))
            theta0 = lo + u0 * (hi - lo)
            r0 = prob.evaluate(theta0, with_grads=False)
            L0 = -r0.fitness
            S2_0 = r0.measures[1]; S3_0 = r0.measures[2]
            if abs(S2_0) > THRESH_S2 and abs(S3_0) > THRESH_S3 and L0 < THRESH_L_MAX:
                chosen = (theta0, r0, cand); break
        if chosen is None:
            return None
        theta0, r0, cand = chosen
        L0 = -r0.fitness
    S = r0.measures
    if label is None:
        label = f"seed={seed}"
    print(f"\n=== {label} (cand={cand}) ===")
    print(f"  L0={L0:.4f}  S1..S5 = {[f'{x:+.3e}' for x in S]}")

    # Compute gradients at start
    rg = prob.evaluate(theta0, with_grads=True)
    grad_L = -rg.grads['fitness']  # since grads['fitness'] = grad of fitness = -grad of L
    grad_M_S2 = rg.grads['m1']
    grad_M_S3 = rg.grads['m2']
    cos_S2 = cos_sim(-grad_L, grad_M_S2)
    cos_S3 = cos_sim(-grad_L, grad_M_S3)
    print(f"  cos(-grad_L, grad_M_S2)={cos_S2:+.4f}")
    print(f"  cos(-grad_L, grad_M_S3)={cos_S3:+.4f}")

    # Seidel signs are conventional, so |cos| is the meaningful alignment.
    # Descent direction includes -sign(M_k) factor to absorb sign convention.
    if abs(cos_S2) < 0.5 or abs(cos_S3) < 0.5:
        print("  |cos(start)| too low; skip")
        return None

    # Branch A: target S2 (k_idx=1)
    thetaA_pre, trajA = run_descent(prob, theta0, lo, hi, k_idx=1, K=K, eta=eta)
    LA_pre = trajA[-1]['L']
    # Branch B: target S3 (k_idx=2)
    thetaB_pre, trajB = run_descent(prob, theta0, lo, hi, k_idx=2, K=K, eta=eta)
    LB_pre = trajB[-1]['L']
    print(f"  branch A: L {L0:.4f} -> {LA_pre:.4f}  (target |M_S2| {abs(S[1]):.3e} -> {trajA[-1]['M_target_abs']:.3e})")
    print(f"  branch B: L {L0:.4f} -> {LB_pre:.4f}  (target |M_S3| {abs(S[2]):.3e} -> {trajB[-1]['M_target_abs']:.3e})")

    # L-BFGS refine each
    thetaA_ref, A_steps, A_conv = lbfgs_refine(prob_r, thetaA_pre, cfg, lo, hi, max_steps_lbfgs)
    thetaB_ref, B_steps, B_conv = lbfgs_refine(prob_r, thetaB_pre, cfg, lo, hi, max_steps_lbfgs)
    rA = prob.evaluate(thetaA_ref, with_grads=False)
    rB = prob.evaluate(thetaB_ref, with_grads=False)
    LA_post = -rA.fitness
    LB_post = -rB.fitness
    print(f"  refined: L_A={LA_post:.4f} (~{LA_post*1000:.0f} um, {A_steps}st, conv={A_conv})  "
          f"L_B={LB_post:.4f} (~{LB_post*1000:.0f} um, {B_steps}st, conv={B_conv})")

    # Sanity checks
    range_ = hi - lo
    sep = float(np.linalg.norm((thetaA_ref - thetaB_ref) / range_) / np.sqrt(len(lo)))
    # Visible-difference proxy on the 6 free curvatures
    fci = list(cfg.problem.free_curvature_indices)
    cdiff = []
    for ci in fci:
        # find the surface with this index
        # Curvature in raw lens param. We have theta — first 6 entries are curvatures by convention.
        pass
    # Easier proxy: compare theta surface curvatures directly via theta indices 0..5
    nfc = len(fci)
    diffs = []
    for i in range(nfc):
        a = thetaA_ref[i]; b = thetaB_ref[i]
        denom = max(abs(a), abs(b))
        if denom < 1e-12: continue
        diffs.append(abs(a-b)/denom)
    n_above_20 = sum(1 for x in diffs if x > 0.20)
    print(f"  branch separation / sqrt(D) = {sep:.4f}")
    print(f"  curvature diffs (rel): {[f'{x:.2f}' for x in diffs]}")
    print(f"  n surfaces with rel diff > 0.20: {n_above_20}/{nfc}")

    rA_ratio = abs(rA.measures[1])/max(abs(S[1]), 1e-30)
    rB_ratio = abs(rB.measures[2])/max(abs(S[2]), 1e-30)

    return {
        "seed": seed, "cand": cand,
        "theta0": theta0, "r0": r0,
        "grad_L": grad_L, "grad_M_S2": grad_M_S2, "grad_M_S3": grad_M_S3,
        "cos_S2": cos_S2, "cos_S3": cos_S3,
        "thetaA_pre": thetaA_pre, "thetaA_ref": thetaA_ref,
        "thetaB_pre": thetaB_pre, "thetaB_ref": thetaB_ref,
        "trajA": trajA, "trajB": trajB,
        "LA_pre": LA_pre, "LB_pre": LB_pre,
        "LA_post": LA_post, "LB_post": LB_post,
        "A_steps": A_steps, "A_conv": A_conv,
        "B_steps": B_steps, "B_conv": B_conv,
        "rA": rA, "rB": rB,
        "sep": sep, "n_above_20": n_above_20,
        "curvature_diffs": diffs,
        "rA_ratio_S2": rA_ratio,
        "rB_ratio_S3": rB_ratio,
    }


def archive_warm_starts(arch, prob, lo, hi, max_candidates=30):
    """Pick archive cells with substantial S2/S3 residual AND viable cosines."""
    cells = []
    for k, c in arch.cells.items():
        if not np.all(np.isfinite(c.theta)): continue
        s2, s3 = c.measures[0], c.measures[1]
        L = -c.fitness
        # require substantial residual S2 and S3 + raw L < 0.5
        if abs(s2) > 1e-4 and abs(s3) > 1e-4 and L < 0.5:
            cells.append((L, k, np.asarray(c.theta, dtype=np.float64)))
    cells.sort()
    # screen by |cos|>0.5
    out = []
    for L, k, theta in cells:
        rg = prob.evaluate(theta, with_grads=True)
        gL = -rg.grads['fitness']; gS2 = rg.grads['m1']; gS3 = rg.grads['m2']
        cs2 = cos_sim(-gL, gS2); cs3 = cos_sim(-gL, gS3)
        if abs(cs2) > 0.5 and abs(cs3) > 0.5:
            out.append((L, k, theta, cs2, cs3))
            if len(out) >= max_candidates: break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*",
                    default=list(range(20)))
    ap.add_argument("--eta", type=float, default=ETA)
    ap.add_argument("--K", type=int, default=K_STEPS)
    ap.add_argument("--archive-warm-start", action="store_true",
                    help="Enumerate archive cells with substantial S2/S3 + good cos")
    args = ap.parse_args()

    cfg, prob = load_problem(refine=False)
    cfg_r, prob_r = load_problem(refine=True)
    bounds = np.array(cfg.problem.theta_bounds)
    lo = bounds[:,0]; hi = bounds[:,1]
    D = len(lo)
    print(f"D={D}, bounds loaded")

    candidates = []
    if args.archive_warm_start:
        import pickle
        arch = pickle.load(open(ARCHIVE_PATH, "rb"))
        ws = archive_warm_starts(arch, prob, lo, hi, max_candidates=20)
        print(f"\n{len(ws)} archive cells pass screening")
        for L0, k, theta0, cs2, cs3 in ws:
            label = f"cell={k}"
            res = try_seed(seed=hash(str(k))%10000, cfg=cfg, prob=prob, prob_r=prob_r,
                           lo=lo, hi=hi, eta=args.eta, K=args.K,
                           theta0_override=theta0, label=label)
            if res is None: continue
            res['cell_key'] = k
            res['label'] = label
            L0_actual = -float(res['r0'].fitness)
            # also count surfaces with rel curvature diff > 0.50 — proxy for
            # "lens looks visibly different to the eye"
            n_above_50 = sum(1 for x in res['curvature_diffs'] if x > 0.50)
            res['n_above_50'] = n_above_50
            ok = (
                abs(res['cos_S2']) > 0.5 and abs(res['cos_S3']) > 0.5 and
                res['rA_ratio_S2'] < 1.0 and res['rB_ratio_S3'] < 1.0 and
                res['LA_post'] < REFINED_L_MAX and
                res['LB_post'] < REFINED_L_MAX and
                res['sep'] > 0.05 and
                res['n_above_50'] >= 3   # at least 3 of 6 surfaces differ by >50%
            )
            res['ok'] = ok
            candidates.append(res)
            print(f"  ALL OK = {ok}")
        if not candidates:
            sys.exit("no archive cell produced a viable run")
        # Sort by ok-first, then by combined visible-difference + separation score
        def score(r):
            return -(r.get('n_above_50', 0) * 1.0 + r['sep'] * 5.0)
        candidates.sort(key=lambda r: (not r['ok'], score(r)))
        best = candidates[0]
        print(f"\n>>> chose {best['label']} (sep={best['sep']:.4f}, ok={best['ok']}, "
              f"n>50={best.get('n_above_50',0)}, "
              f"L_A={best['LA_post']*1000:.1f}um, L_B={best['LB_post']*1000:.1f}um) <<<")
        if not best['ok']:
            print("\nNo cell passed all sanity checks. Diagnostic:")
            for r in candidates:
                print(f"  {r['label']}: cos|S2|={abs(r['cos_S2']):.3f} cos|S3|={abs(r['cos_S3']):.3f} "
                      f"sep={r['sep']:.3f} n>20={r['n_above_20']} n>50={r.get('n_above_50',0)} "
                      f"LA={r['LA_post']*1000:.1f}um LB={r['LB_post']*1000:.1f}um")
            sys.exit(2)
        write_outputs(cfg, best, lo, hi)
        return

    for s in args.seeds:
        res = try_seed(s, cfg, prob, prob_r, lo, hi, eta=args.eta, K=args.K)
        if res is None:
            continue
        # require all sanity checks
        L0 = -float(res['r0'].fitness)
        ok = (
            abs(res['cos_S2']) > 0.5 and abs(res['cos_S3']) > 0.5 and
            res['rA_ratio_S2'] < 1.0 and res['rB_ratio_S3'] < 1.0 and
            res['LA_post'] < REFINED_L_MAX and  # both branches refine to <= 30um RMS
            res['LB_post'] < REFINED_L_MAX and
            res['sep'] > 0.10 and
            res['n_above_20'] >= 2
        )
        res['ok'] = ok
        candidates.append(res)
        print(f"  ALL OK = {ok}")

    if not candidates:
        sys.exit("no seed produced a viable run")

    # Pick best: ok first, then largest separation
    candidates.sort(key=lambda r: (not r['ok'], -r['sep']))
    best = candidates[0]
    print(f"\n>>> chose seed={best['seed']} (sep={best['sep']:.4f}, ok={best['ok']}) <<<")

    if not best['ok']:
        # report and stop before write/plot
        print("No seed passed all sanity checks. Diagnostic:")
        for r in candidates:
            print(f"  seed={r['seed']}: cos_S2={r['cos_S2']:.3f} cos_S3={r['cos_S3']:.3f} "
                  f"sep={r['sep']:.3f} n>20={r['n_above_20']} "
                  f"L0={-r['r0'].fitness:.3f} LA={r['LA_post']:.3f} LB={r['LB_post']:.3f}")
        sys.exit(2)

    # ---------- write outputs ----------
    write_outputs(cfg, best, lo, hi)


def write_outputs(cfg, b, lo, hi):
    seed = b['seed']
    cell_key = b.get('cell_key')
    label = b.get('label', f'seed={seed}')
    # start.json
    r0 = b['r0']
    grad_L = b['grad_L']; gS2 = b['grad_M_S2']; gS3 = b['grad_M_S3']
    start = {
        "theta": b['theta0'].tolist(),
        "L": -float(r0.fitness),
        "S_1": float(r0.measures[0]),
        "S_2": float(r0.measures[1]),
        "S_3": float(r0.measures[2]),
        "S_4": float(r0.measures[3]),
        "S_5": float(r0.measures[4]),
        "grad_L_norm": float(np.linalg.norm(grad_L)),
        "grad_M_S2_norm": float(np.linalg.norm(gS2)),
        "grad_M_S3_norm": float(np.linalg.norm(gS3)),
        "cos_minus_grad_L_grad_M_S2": float(b['cos_S2']),
        "cos_minus_grad_L_grad_M_S3": float(b['cos_S3']),
        "random_seed": int(seed),
        "candidate_index": int(b['cand']),
    }
    with open(OUT/"start.json", "w") as f:
        json.dump(start, f, indent=2)

    # trajectories
    import csv
    for tag, traj in [("A", b['trajA']), ("B", b['trajB'])]:
        with open(OUT/f"branch_{tag}_trajectory.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step","L","S_2","S_3","M_S2_magnitude","M_S3_magnitude","distance_from_start"])
            w.writeheader()
            for row in traj:
                # remap field names for output schema
                w.writerow({
                    "step": row['step'],
                    "L": row['L'],
                    "S_2": row['S_2'],
                    "S_3": row['S_3'],
                    "M_S2_magnitude": abs(row['S_2']),
                    "M_S3_magnitude": abs(row['S_3']),
                    "distance_from_start": row['distance_from_start_unit'],
                })

    # branch refined JSONs
    for tag, theta_pre, theta_post, L_pre, L_post, steps, conv, r_post in [
        ("A", b['thetaA_pre'], b['thetaA_ref'], b['LA_pre'], b['LA_post'], b['A_steps'], b['A_conv'], b['rA']),
        ("B", b['thetaB_pre'], b['thetaB_ref'], b['LB_pre'], b['LB_post'], b['B_steps'], b['B_conv'], b['rB']),
    ]:
        with open(OUT/f"branch_{tag}_refined.json", "w") as f:
            json.dump({
                "theta_pre_lbfgs": theta_pre.tolist(),
                "theta_post_lbfgs": theta_post.tolist(),
                "L_pre": float(L_pre),
                "L_post": float(L_post),
                "S_2_post": float(r_post.measures[1]),
                "S_3_post": float(r_post.measures[2]),
                "lbfgs_steps_used": int(steps),
                "lbfgs_converged": bool(conv),
            }, f, indent=2)

    # spot diagrams + lens geometries
    print("tracing rays for 3 designs...")
    spots = {}
    geoms = {}
    for tag, theta in [
        ("start", b['theta0']),
        ("branch_A_refined", b['thetaA_ref']),
        ("branch_B_refined", b['thetaB_ref']),
    ]:
        spots[tag] = trace_spot(theta, cfg)
        geoms[tag] = lens_geometry(theta, cfg)

    # save spot_diagrams.npz: flatten dict-of-dict into arrays + a manifest
    flat = {}
    for tag, d in spots.items():
        for k, v in d.items():
            flat[f"{tag}__{k}"] = v
    np.savez(OUT/"spot_diagrams.npz", **flat,
             tags=np.array(list(spots.keys()), dtype=object))
    with open(OUT/"lens_geometries.json", "w") as f:
        json.dump(geoms, f, indent=2)

    # summary for sanity-check report
    with open(OUT/"chosen_seed.json", "w") as f:
        json.dump({
            "label": label,
            "archive_cell_key": list(cell_key) if cell_key is not None else None,
            "seed": int(seed), "candidate_index": int(b['cand']),
            "cos_S2": float(b['cos_S2']), "cos_S3": float(b['cos_S3']),
            "M_S2_at_start_abs": float(abs(r0.measures[1])),
            "M_S2_at_branchA_end_abs": float(b['trajA'][-1]['M_target_abs']),
            "M_S2_ratio_A": float(b['trajA'][-1]['M_target_abs'] / max(abs(r0.measures[1]),1e-30)),
            "M_S3_at_start_abs": float(abs(r0.measures[2])),
            "M_S3_at_branchB_end_abs": float(b['trajB'][-1]['M_target_abs']),
            "M_S3_ratio_B": float(b['trajB'][-1]['M_target_abs'] / max(abs(r0.measures[2]),1e-30)),
            "L_start": float(-r0.fitness),
            "L_A_post": float(b['LA_post']),
            "L_B_post": float(b['LB_post']),
            "branch_separation_norm_per_sqrtD": float(b['sep']),
            "n_curvatures_with_rel_diff_gt_0.20": int(b['n_above_20']),
            "curvature_relative_diffs": list(map(float, b['curvature_diffs'])),
        }, f, indent=2)

    print(f"\nwrote outputs to {OUT}")


if __name__ == "__main__":
    main()
