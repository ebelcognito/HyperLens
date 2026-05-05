#!/usr/bin/env python3
"""Refine + render the single best elite from the live checkpoint of a
running LensMEGA job. Uses _EliteRenderWorker._render synchronously."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.core.runner import _EliteRenderWorker


RUN_DIR = Path(
    "results/lensmega/septlet_apsc_e2e3cement_s4s2_20260502_224444"
)


def main() -> None:
    cfg = load_config(RUN_DIR / "config.yaml")
    # Disable EFL penalty for refinement (both the base problem weight and the
    # heavier refine override). The refiner reads io.refine_efl_penalty_weight
    # if set, else falls back to problem.efl_penalty_weight.
    cfg.io.refine_efl_penalty_weight = 0.0
    cfg.problem.efl_penalty_weight = 0.0

    with open(RUN_DIR / "checkpoint.pkl", "rb") as f:
        cp = pickle.load(f)

    arch = cp.archive
    best_idx, best = max(arch.cells.items(), key=lambda kv: kv[1].fitness)

    theta = np.asarray(best.theta, dtype=np.float64)
    rms_um_str = f"{best.metrics.get('rms_spot', 0.0) * 1000:.1f}"
    efl_str = f"{best.metrics.get('efl_actual', 0.0):.2f}"
    sign_pattern = "".join(
        "+" if v > 0 else "-" for v in np.sign(theta[:6])
    )

    info = {
        "eval": int(cp.completed_evaluations),
        "elite_index": 9999,
        "fitness": float(best.fitness),
        "rms_mm": float(best.metrics.get("rms_spot", 0.0)),
        "rms_um": rms_um_str,
        "efl_mm": efl_str,
        "sign_pattern": sign_pattern,
        "theta": [float(v) for v in theta],
        "measures": [float(v) for v in np.asarray(best.measures)],
        "metrics_raw": {k: float(v) for k, v in best.metrics.items()},
        "archive_cell": [int(best_idx[0]), int(best_idx[1])],
    }

    out_dir = RUN_DIR / "elite_designs"
    out_dir.mkdir(exist_ok=True)
    worker = _EliteRenderWorker(out_dir, cfg)

    tag = f"BEST_LIVE_eval{int(cp.completed_evaluations)}k_eflpen0"
    print(f"[best] cell={best_idx} fitness={best.fitness:.6f} "
          f"RMS={rms_um_str}μm EFL={efl_str}mm signs=[{sign_pattern}]")
    print(f"[best] running synchronous refine+render -> {tag} ...")

    # Call _render directly (synchronous) instead of going through the queue.
    worker._render(tag, theta, info)

    print(f"[best] done. JSON + PNGs in {out_dir}/{tag}.*")


if __name__ == "__main__":
    main()
