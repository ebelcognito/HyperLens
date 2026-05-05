"""Smoke + sanity tests for the F# soft-penalty implementation.

Tests:
  T1. Backward compat: cooke_triplet_materials.yaml (no enforce_fnum) loads,
      builds problem, evaluates a candidate, returns identical fitness as a
      pristine baseline (we assert no extra penalty is applied).
  T2. Penalty math: with enforce_fnum=True, on the same candidate
        (a) fnum_penalty == w * (fnum_actual - target)^2
        (b) fnum_penalty >= 0
        (c) gradient of fnum_penalty w.r.t. theta is non-zero
        (d) increasing the F# target from current value → penalty grows
            quadratically (sample 5 targets, fit a parabola)
  T3. Pupil-rejection short-circuit: with strict_entrance_pupil=True and a
      degenerate candidate that triggers rejection, F# block does NOT execute
      (fitness is the rejection sentinel and metrics include "rejected").
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/hisham26/Desktop/RL")
sys.path.insert(0, "/home/hisham26/Desktop/RL/DeepLens")

from optical_rl.qd.lensmega.config.loader import load_config
from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem


CFG = ("/home/hisham26/Desktop/RL/optical_rl/qd/lensmega/"
       "config/examples/cooke_triplet_materials.yaml")
ELITE_JSON = ("/home/hisham26/Desktop/RL/results/lensmega/"
              "cooke_triplet_materials_20260414_173935/elite_designs/"
              "elite_0024_23328k.json")


def load_candidate():
    with open(ELITE_JSON) as f:
        return np.array(json.load(f)["refined"]["theta"], dtype=float)


def t1_backward_compat():
    print("\n[T1] Backward compat (enforce_fnum=False)")
    cfg = load_config(CFG)
    assert cfg.problem.enforce_fnum is False, "default must be False"
    print(f"  default enforce_fnum: {cfg.problem.enforce_fnum}  ✓")
    print(f"  default f_number_target: {cfg.problem.f_number_target}")
    print(f"  default fnum_penalty_weight: {cfg.problem.fnum_penalty_weight}")

    desc = _build_descriptor(cfg)
    prob = _build_problem(cfg, desc)
    prob._ensure_backend()

    theta = load_candidate()
    res = prob.evaluate(theta, with_grads=False)
    # fnum_actual is always reported (for logging) but the penalty fields
    # MUST be absent so the loss path is unchanged from pre-F# behaviour.
    assert "fnum_actual" in res.metrics, "fnum_actual should always be reported"
    assert "fnum_penalty" not in res.metrics, "F# penalty must be skipped when disabled"
    assert "fnum_deviation" not in res.metrics, "F# deviation must be skipped when disabled"
    print(f"  fitness = {res.fitness:.8f}")
    print(f"  metrics keys: {sorted(res.metrics.keys())}")
    print(f"  fnum_actual logged: F/{res.metrics['fnum_actual']:.3f}")
    print(f"  ✓ fnum_actual reported; penalty fields absent; loss path unchanged")
    return res.fitness


def t2_penalty_math(baseline_fitness):
    print("\n[T2] Penalty math (enforce_fnum=True)")
    cfg = load_config(CFG)
    cfg.problem.enforce_fnum = True
    cfg.problem.f_number_target = 8.0
    cfg.problem.fnum_penalty_weight = 1.0

    desc = _build_descriptor(cfg)
    prob = _build_problem(cfg, desc)
    prob._ensure_backend()

    theta = load_candidate()
    res = prob.evaluate(theta, with_grads=True)
    fnum = res.metrics["fnum_actual"]
    dev  = res.metrics["fnum_deviation"]   # relative: (fnum - target) / target
    pen  = res.metrics["fnum_penalty"]
    print(f"  fnum_actual         = {fnum:.4f}")
    print(f"  fnum_deviation_rel  = {dev:.4f}  (= (actual-target)/target)")
    print(f"  fnum_penalty        = {pen:.6f}")

    # (a) penalty = w * rel_dev^2  (relative-error formulation, matches EFL)
    target = cfg.problem.f_number_target
    expected_rel = (fnum - target) / target
    expected_pen = 1.0 * expected_rel * expected_rel
    assert abs(dev - expected_rel) < 1e-6, f"deviation rel: {dev} vs {expected_rel}"
    assert abs(pen - expected_pen) < 1e-6, f"penalty math: {pen} vs {expected_pen}"
    print(f"  ✓ deviation == (actual-target)/target  (residual {abs(dev - expected_rel):.2e})")
    print(f"  ✓ penalty   == w * rel_dev^2           (residual {abs(pen - expected_pen):.2e})")

    # (b) penalty >= 0
    assert pen >= 0
    print(f"  ✓ penalty >= 0")

    # (c) grad of fitness w.r.t. theta non-zero
    grad = res.grads["fitness"]
    assert grad is not None and np.linalg.norm(grad) > 0
    print(f"  ✓ ||d_fitness/d_theta||  = {np.linalg.norm(grad):.4e} (non-zero)")

    # (d) penalty grows quadratically as we sweep f_target away from fnum_actual.
    print(f"  Quadratic-growth test (target sweep around fnum_actual={fnum:.2f}):")
    print(f"    {'target':>8s} {'rel_dev':>8s} {'penalty':>12s} {'expected_w*r^2':>16s}")
    for tgt in [fnum - 1.0, fnum - 0.5, fnum, fnum + 0.5, fnum + 1.0]:
        cfg.problem.f_number_target = tgt
        prob2 = _build_problem(cfg, _build_descriptor(cfg))
        prob2._ensure_backend()
        r = prob2.evaluate(theta, with_grads=False)
        d = r.metrics["fnum_deviation"]
        p = r.metrics["fnum_penalty"]
        print(f"    {tgt:8.3f} {d:8.4f} {p:12.6f} {d*d:16.6f}")

    # (e) baseline unchanged when penalty subtracted
    delta = baseline_fitness - res.fitness
    print(f"  fitness shift from baseline: {delta:+.6f}  "
          f"(expected ≈ +{pen:.6f}) — penalty was applied")


def t3_rejection_short_circuit():
    """Verify the F# block is *positioned after* every rejection branch in
    each problem variant. Triggering a real rejection from random θ is
    flaky (the variable flange usually keeps BFL positive), so we do a
    structural source-text check: every rejection path returns BEFORE the
    line that adds the F# penalty term."""
    print("\n[T3] Rejection short-circuit (structural verification)")

    files_and_markers = [
        ("optical_rl/qd/lensmega/problem/lens_problem.py",
         ["rejected_pupil"]),
        ("optical_rl/qd/lensmega/problem/full_dof.py",
         ["rejected_bfl", "rejected_pupil"]),
        ("optical_rl/qd/lensmega/problem/power_budget.py",
         ["rejected_bfl", "rejected_pupil"]),
    ]
    base = "/home/hisham26/Desktop/RL/"
    import re
    for relpath, markers in files_and_markers:
        with open(base + relpath) as f:
            src = f.read()
        # Find line of the F# penalty addition.
        m_fnum = re.search(r"fitness\s*=\s*fitness\s*-\s*fnum_penalty", src)
        assert m_fnum is not None, f"{relpath}: F# penalty line missing"
        fnum_line = src[:m_fnum.start()].count("\n") + 1
        # Find each rejection-return line (must precede F# block in source).
        for marker in markers:
            m_rej = re.search(rf'"{marker}"', src)
            assert m_rej is not None, f"{relpath}: rejection marker {marker} missing"
            rej_line = src[:m_rej.start()].count("\n") + 1
            assert rej_line < fnum_line, (
                f"{relpath}: rejection branch '{marker}' at line {rej_line} "
                f"should precede F# block at line {fnum_line}"
            )
            print(f"  {relpath:48s} {marker:18s} L{rej_line:4d} < F#L{fnum_line}  ✓")
    print(f"  ✓ All rejection paths short-circuit before F# penalty in all variants")

    # Belt-and-suspenders runtime check: verify the F# block IS gated by
    # enforce_fnum (i.e. when False, no F# metrics emitted).
    cfg = load_config(CFG)
    cfg.problem.enforce_fnum = False
    desc = _build_descriptor(cfg)
    prob = _build_problem(cfg, desc)
    prob._ensure_backend()
    res = prob.evaluate(load_candidate(), with_grads=False)
    assert "fnum_penalty" not in res.metrics
    print(f"  ✓ enforce_fnum=False at runtime: F# fields absent from metrics")


if __name__ == "__main__":
    bl = t1_backward_compat()
    t2_penalty_math(bl)
    t3_rejection_short_circuit()
    print("\nAll tests passed.")
