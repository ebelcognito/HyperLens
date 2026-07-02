"""Refine top-5 diverse archive cells above a fitness floor.

Picks 5 cells with fitness > --fitness-floor by farthest-first traversal on the
(S2_coma, S3_astigmatism) descriptor plane (seeded by the highest-fitness cell),
then runs the same LBFGS + stability pipeline as refine_posthoc.py.

Usage:
    python refine_top5_diverse.py <run_dir> [--fitness-floor -6.0] [--n 5]
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from refine_posthoc import (
    build_lens_from_theta,
    make_refine_problem,
    refine_one,
    render_png,
    stability_analysis,
)
from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.descriptors.seidel import compute_per_element_seidel


def farthest_first(points: np.ndarray, fitnesses: np.ndarray, n: int) -> list[int]:
    """Return n indices: seed with argmax(fitness), then greedily pick the
    point whose min-distance to the chosen set is largest."""
    assert len(points) == len(fitnesses)
    if len(points) <= n:
        return list(range(len(points)))
    chosen = [int(np.argmax(fitnesses))]
    # min distance from each candidate to the chosen set
    d = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        new_d = np.linalg.norm(points - points[i], axis=1)
        d = np.minimum(d, new_d)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--fitness-floor", type=float, default=-6.0)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--rms-threshold-um", type=float, default=50.0,
                    help="Render PNGs only when refined RMS is below this (μm).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    elite_dir = run_dir / "elite_designs"
    elite_dir.mkdir(exist_ok=True)
    cfg = load_config(str(run_dir / "config.yaml"))
    io = cfg.io

    archive = pickle.load(open(run_dir / "archive.pkl", "rb"))
    print(f"[top5] archive cells: {len(archive.cells)}")

    candidates = [
        (k, v) for k, v in archive.cells.items() if v.fitness > args.fitness_floor
    ]
    print(f"[top5] cells with fitness > {args.fitness_floor}: {len(candidates)}")
    if not candidates:
        print("[top5] nothing to refine.")
        return

    measures = np.array([list(v.measures) for _, v in candidates], dtype=np.float64)
    fits = np.array([v.fitness for _, v in candidates], dtype=np.float64)

    # Normalize descriptor axes to archive bounds so neither dimension dominates.
    bounds = cfg.descriptor.archive_bounds if hasattr(cfg.descriptor, "archive_bounds") else None
    if bounds is None:
        # Fall back to per-axis range from candidate set.
        lo_m = measures.min(axis=0)
        hi_m = measures.max(axis=0)
        rng = np.maximum(hi_m - lo_m, 1e-9)
        norm = (measures - lo_m) / rng
    else:
        b = np.array(bounds, dtype=np.float64)
        lo_m, hi_m = b[:, 0], b[:, 1]
        rng = np.maximum(hi_m - lo_m, 1e-9)
        norm = (measures - lo_m) / rng

    picks = farthest_first(norm, fits, args.n)
    print(f"[top5] picked {len(picks)} diverse cells:")
    for rank, i in enumerate(picks, 1):
        (row, col), v = candidates[i]
        print(f"  #{rank}: cell=({row},{col})  fit={v.fitness:.4f}  "
              f"measures=({v.measures[0]:+.5f}, {v.measures[1]:+.5f})")

    theta_bounds = list(cfg.problem.theta_bounds)
    lo = np.array([b[0] for b in theta_bounds], dtype=np.float64)
    hi = np.array([b[1] for b in theta_bounds], dtype=np.float64)
    prob = make_refine_problem(cfg)

    n_done = n_render = n_err = 0
    t0 = time.time()
    for rank, i in enumerate(picks, 1):
        (row, col), ev = candidates[i]
        theta_raw = np.array(ev.theta, dtype=np.float64)
        tag = f"top5diverse_{rank:02d}_r{row}_c{col}"
        try:
            best_theta, ref_fit, lb_steps, converged, gnorm = refine_one(
                theta_raw, prob, io, lo, hi
            )
            ref_eval = prob.evaluate(best_theta, with_grads=False)
            ref_rms_mm = ref_eval.metrics.get("rms_spot", abs(ref_fit))
            ref_efl = ref_eval.metrics.get("efl_actual", 0.0)
            stability = stability_analysis(prob, best_theta, ref_fit, lo, hi, io)

            raw_rms_mm = ev.metrics.get("rms_spot", abs(ev.fitness))
            info = {
                "tag": tag,
                "cell": [int(row), int(col)],
                "fitness": float(ev.fitness),
                "rms_mm": float(raw_rms_mm),
                "rms_um": f"{raw_rms_mm * 1000:.1f}",
                "efl_mm": f"{ev.metrics.get('efl_actual', 0):.2f}",
                "theta": [float(v) for v in theta_raw],
                "measures": [float(v) for v in ev.measures],
                "metrics": {k: float(v) for k, v in ev.metrics.items()},
                "refined": {
                    "theta": [float(v) for v in best_theta],
                    "fitness": float(ref_fit),
                    "rms_mm": float(ref_rms_mm),
                    "rms_um": f"{ref_rms_mm * 1000:.1f}",
                    "efl_mm": f"{ref_efl:.2f}",
                    "lbfgs_steps": int(lb_steps),
                    "converged": bool(converged),
                    "final_grad_norm": round(float(gnorm), 8),
                    "improvement_ratio": round(
                        float(ev.fitness / ref_fit) if abs(ref_fit) > 1e-12 else 0.0, 4
                    ),
                    "raw_fitness": float(ev.fitness),
                    "stability": stability,
                    "metrics": {k: float(v) for k, v in ref_eval.metrics.items()},
                },
            }

            ref_rms_um = ref_rms_mm * 1000.0
            if ref_rms_um < args.rms_threshold_um:
                try:
                    lens_raw, _ = build_lens_from_theta(theta_raw, cfg)
                    lens_ref, template_ref = build_lens_from_theta(best_theta, cfg)
                    raw_signs = "".join("+" if v > 0 else "-" for v in theta_raw[:12])
                    ref_signs = "".join("+" if v > 0 else "-" for v in best_theta[:12])
                    render_png(
                        lens_raw, elite_dir / f"{tag}_raw.png",
                        f"{tag} RAW | fitness={ev.fitness:.6f}  "
                        f"RMS={raw_rms_mm*1000:.1f}μm  EFL={ev.metrics.get('efl_actual',0):.1f}mm  "
                        f"[{raw_signs}]"
                    )
                    render_png(
                        lens_ref, elite_dir / f"{tag}_refined.png",
                        f"{tag} REFINED | fitness={ref_fit:.6f}  "
                        f"RMS={ref_rms_um:.1f}μm  EFL={ref_efl:.1f}mm  "
                        f"[{ref_signs}]  stability={stability['sensitivity']:.6f}  "
                        f"survived={stability['pct_survived']:.0f}%"
                    )
                    try:
                        info["fingerprint_refined"] = compute_per_element_seidel(
                            lens_ref, template_ref,
                            field_y_deg=7.0,
                            wavelength_um=cfg.problem.wavelength_um,
                            device="cpu",
                        )
                    except Exception as e:
                        info["fingerprint_refined"] = {"error": str(e)}
                    n_render += 1
                except Exception as e:
                    print(f"  [render-fail] {tag}: {e}", flush=True)

            (elite_dir / f"{tag}.json").write_text(json.dumps(info, indent=2))
            n_done += 1

            print(
                f"[#{rank}] cell=({row:>3},{col:>3}) "
                f"raw_fit={ev.fitness:.4f} -> ref_fit={ref_fit:.4f}  "
                f"raw_rms={raw_rms_mm*1000:6.1f}μm -> ref_rms={ref_rms_um:6.1f}μm  "
                f"({lb_steps} steps, {'CONVERGED' if converged else 'maxsteps'})  "
                f"stability={stability.get('sensitivity', 0):.4f} "
                f"survived={stability.get('pct_survived', 0):.0f}%",
                flush=True,
            )
        except Exception as e:
            print(f"  [refine-fail] {tag}: {e}", flush=True)
            n_err += 1

    print()
    print(f"[top5] DONE  refined={n_done}  rendered={n_render}  errors={n_err}")
    print(f"[top5] wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
