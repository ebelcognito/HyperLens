"""Unit tests for the canonical CMA-ES engine and checkpoint format.

These cover the inner CMA-ES math (convergence + reset) and the
checkpoint/resume round-trip used by every CMA-MEGA runner. They are
intentionally pure-numpy and do not touch DeepLens, so they run in a
fraction of a second.
"""
from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from optical_rl.qd._cma import GradientCoefficientCMA
from optical_rl.qd._checkpoint import (
    CMAMEGACheckpoint,
    INSERT_NEW,
    INSERT_NONE,
    INSERT_REPLACED,
    TrajectoryLogger,
    hash_config,
    load_checkpoint,
    save_checkpoint,
)


class CMAESEngineTests(unittest.TestCase):
    """Math-level tests for the canonical (mu/mu_w, lambda)-CMA-ES emitter."""

    def test_initial_state(self):
        e = GradientCoefficientCMA(dim=4, seed=1, sigma=0.5, population_size=12)
        self.assertEqual(e.dim, 4)
        self.assertEqual(e.sigma, 0.5)
        self.assertEqual(e.generation, 0)
        np.testing.assert_array_equal(e.mean, np.zeros(4))
        np.testing.assert_allclose(e.C, np.eye(4))
        np.testing.assert_array_equal(e.p_sigma, np.zeros(4))
        np.testing.assert_array_equal(e.p_c, np.zeros(4))

    def test_ask_returns_correct_shape(self):
        e = GradientCoefficientCMA(dim=3, seed=2, sigma=0.7, population_size=10)
        c = e.ask(10)
        self.assertEqual(c.shape, (10, 3))
        self.assertTrue(np.all(np.isfinite(c)))

    def test_ask_zero_returns_empty(self):
        e = GradientCoefficientCMA(dim=3, seed=0, sigma=0.5, population_size=8)
        out = e.ask(0)
        self.assertEqual(out.shape, (0, 3))

    def test_sphere_convergence(self):
        """Optimize -||x - target||^2 (10-D sphere). Canonical CMA-ES
        must drive the mean to the target and shrink sigma over time.
        """
        target = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 0.0, 2.0, -0.5, 1.5, -1.5])
        e = GradientCoefficientCMA(dim=10, seed=7, sigma=1.0, population_size=20)
        for _ in range(150):
            c = e.ask(20)
            scores = -np.sum((c - target) ** 2, axis=1)  # higher is better
            e.tell(c, scores)
        err = float(np.linalg.norm(e.mean - target))
        self.assertLess(err, 1e-3, f"mean error {err} too large after 150 gens")
        self.assertLess(e.sigma, 0.01, f"sigma {e.sigma} did not decay")

    def test_reset_restores_initial(self):
        e = GradientCoefficientCMA(dim=3, seed=4, sigma=0.6, population_size=12)
        # Drive state away from initial.
        for _ in range(20):
            c = e.ask(12)
            e.tell(c, np.linspace(-1, 1, 12))
        self.assertGreater(e.generation, 0)
        e.reset()
        self.assertEqual(e.generation, 0)
        np.testing.assert_array_equal(e.mean, np.zeros(3))
        np.testing.assert_allclose(e.C, np.eye(3))
        np.testing.assert_array_equal(e.p_sigma, np.zeros(3))
        np.testing.assert_array_equal(e.p_c, np.zeros(3))
        self.assertEqual(e.sigma, 0.6)

    def test_get_set_state_roundtrip(self):
        """After get_state -> set_state, two emitters must produce the
        exact same next sample (RNG and CMA state both restored)."""
        e1 = GradientCoefficientCMA(dim=3, seed=42, sigma=0.5, population_size=16)
        rng = np.random.default_rng(0)
        for _ in range(7):
            c = e1.ask(16)
            scores = rng.standard_normal(16)
            e1.tell(c, scores)
        state = e1.get_state()

        e2 = GradientCoefficientCMA(dim=3, seed=999, sigma=0.1)
        e2.set_state(state)
        # Both must agree on next ask AND on subsequent tell.
        a1, a2 = e1.ask(16), e2.ask(16)
        np.testing.assert_allclose(a1, a2, atol=0.0)

    def test_set_state_dim_mismatch_raises(self):
        e = GradientCoefficientCMA(dim=3, seed=0, sigma=0.5, population_size=8)
        state = e.get_state()
        wrong = GradientCoefficientCMA(dim=4, seed=0, sigma=0.5, population_size=8)
        with self.assertRaises(ValueError):
            wrong.set_state(state)

    def test_set_state_schema_mismatch_raises(self):
        e = GradientCoefficientCMA(dim=2, seed=0, sigma=0.5, population_size=8)
        state = e.get_state()
        state["schema_version"] = 999
        with self.assertRaises(ValueError):
            e.set_state(state)

    def test_should_restart_on_collapsed_sigma(self):
        e = GradientCoefficientCMA(dim=3, seed=0, sigma=0.5, population_size=10)
        # Initialize constants by running one generation, then force tiny sigma.
        e.tell(e.ask(10), np.linspace(-1, 1, 10))
        e.sigma = 1e-30
        flag, reason = e.should_restart()
        self.assertTrue(flag)
        self.assertEqual(reason, "tol_sigma")

    def test_should_restart_returns_false_for_healthy_state(self):
        e = GradientCoefficientCMA(dim=3, seed=0, sigma=0.5, population_size=10)
        for _ in range(5):
            e.tell(e.ask(10), np.random.default_rng(0).standard_normal(10))
        flag, _reason = e.should_restart()
        self.assertFalse(flag)


class CheckpointFormatTests(unittest.TestCase):
    """Round-trip tests for the CMAMEGACheckpoint pickle format."""

    def test_save_and_load(self):
        e = GradientCoefficientCMA(dim=3, seed=11, sigma=0.5, population_size=8)
        for _ in range(5):
            e.tell(e.ask(8), np.random.default_rng(1).standard_normal(8))
        ckpt = CMAMEGACheckpoint(
            archive={"fake": "cells"},
            emitter_state=e.get_state(),
            completed_evaluations=40,
            generation=5,
            restart_count=0,
            best_fitness=-0.5,
            config_hash="abc123",
            timestamp="2026-04-10T00:00:00",
            wallclock_seconds=12.3,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint.pkl"
            save_checkpoint(ckpt, path)
            self.assertTrue(path.exists())
            loaded = load_checkpoint(path)

        self.assertEqual(loaded.completed_evaluations, 40)
        self.assertEqual(loaded.generation, 5)
        self.assertEqual(loaded.config_hash, "abc123")
        self.assertEqual(loaded.archive, {"fake": "cells"})
        # The emitter state must round-trip and be reusable.
        e2 = GradientCoefficientCMA(dim=3, seed=0, sigma=0.1)
        e2.set_state(loaded.emitter_state)
        a1, a2 = e.ask(8), e2.ask(8)
        np.testing.assert_allclose(a1, a2, atol=0.0)

    def test_atomic_save_no_partial_file(self):
        """save_checkpoint must not leave a stray .tmp file behind on success."""
        e = GradientCoefficientCMA(dim=2, seed=0, sigma=0.5, population_size=4)
        e.tell(e.ask(4), np.array([0.0, 1.0, -1.0, 2.0]))
        ckpt = CMAMEGACheckpoint(
            archive=None, emitter_state=e.get_state(),
            completed_evaluations=1, generation=1, restart_count=0,
            best_fitness=0.0, config_hash="x", timestamp="t", wallclock_seconds=0.0,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint.pkl"
            save_checkpoint(ckpt, path)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            self.assertTrue(path.exists())
            self.assertFalse(tmp_path.exists(), "stray .tmp file left behind")


class TrajectoryLoggerTests(unittest.TestCase):
    """Per-generation CSV survives kills (one row per call, header once)."""

    def test_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trajectory.csv"
            logger = TrajectoryLogger(path)
            logger.log(
                gen=1, evals=16, coverage=0.05, best_fitness=-0.8,
                new_cells=10, replaced_elites=2, wasted_elites=4,
                sigma=0.5, restarts=0, wallclock_s=1.0,
            )
            logger.log(
                gen=2, evals=32, coverage=0.07, best_fitness=-0.7,
                new_cells=8, replaced_elites=4, wasted_elites=4,
                sigma=0.55, restarts=0, wallclock_s=2.0,
            )
            text = path.read_text()
            lines = text.strip().splitlines()
            self.assertEqual(len(lines), 3)  # header + 2 rows
            self.assertIn("gen,evals,coverage", lines[0])
            self.assertTrue(lines[1].startswith("1,16,"))
            self.assertTrue(lines[2].startswith("2,32,"))

    def test_append_does_not_duplicate_header(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trajectory.csv"
            l1 = TrajectoryLogger(path)
            l1.log(
                gen=1, evals=16, coverage=0.05, best_fitness=-0.8,
                new_cells=10, replaced_elites=2, wasted_elites=4,
                sigma=0.5, restarts=0, wallclock_s=1.0,
            )
            # Simulate process restart: new logger instance, same file.
            l2 = TrajectoryLogger(path)
            l2.log(
                gen=2, evals=32, coverage=0.07, best_fitness=-0.7,
                new_cells=8, replaced_elites=4, wasted_elites=4,
                sigma=0.55, restarts=0, wallclock_s=2.0,
            )
            lines = path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 3)  # header + 2 rows, header NOT duplicated


class InsertEnumTests(unittest.TestCase):
    """The three-way enum must be backwards-compatible: 0 falsy, 1/2 truthy."""

    def test_constants(self):
        self.assertEqual(INSERT_NONE, 0)
        self.assertEqual(INSERT_REPLACED, 1)
        self.assertEqual(INSERT_NEW, 2)

    def test_truthiness(self):
        self.assertFalse(bool(INSERT_NONE))
        self.assertTrue(bool(INSERT_REPLACED))
        self.assertTrue(bool(INSERT_NEW))


class ConfigHashTests(unittest.TestCase):
    """Config-drift detection used by --resume."""

    def test_same_config_same_hash(self):
        from dataclasses import dataclass
        @dataclass
        class C:
            a: int = 1
            b: float = 2.0
        self.assertEqual(hash_config(C()), hash_config(C()))

    def test_different_config_different_hash(self):
        from dataclasses import dataclass
        @dataclass
        class C:
            a: int = 1
            b: float = 2.0
        self.assertNotEqual(hash_config(C(a=2)), hash_config(C(a=1)))


if __name__ == "__main__":
    unittest.main()
