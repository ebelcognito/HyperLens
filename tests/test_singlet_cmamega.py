import math
import unittest

import numpy as np

from optical_rl.qd.singlet_cmamega import (
    CMAMEGAArchive2D,
    GradientCoefficientCMA,
    SingletEvalResult,
    project_box,
    theta_to_unit,
    unit_to_theta,
)


class TestSingletCMAMEGAHelpers(unittest.TestCase):
    def test_project_box_clamps_componentwise(self):
        bounds = ((-0.05, 0.05), (-0.05, 0.05), (5.0, 100.0))
        theta = np.array([0.08, -0.09, 140.0], dtype=np.float64)
        projected = project_box(theta, bounds)
        np.testing.assert_allclose(projected, np.array([0.05, -0.05, 100.0]))

    def test_unit_round_trip(self):
        bounds = ((-0.05, 0.05), (-0.05, 0.05), (5.0, 100.0))
        theta = np.array([0.01, -0.02, 33.0], dtype=np.float64)
        unit = theta_to_unit(theta, bounds)
        restored = unit_to_theta(unit, bounds)
        np.testing.assert_allclose(restored, theta)

    def test_archive_prefers_higher_fitness(self):
        archive = CMAMEGAArchive2D((4, 4), ((-1.0, 1.0), (-1.0, 1.0)), seed=0)
        low = SingletEvalResult(
            theta=np.array([0.0, 0.0, 10.0]),
            fitness=-2.0,
            measures=(0.1, 0.1),
            metrics={"rms_spot": 2.0},
            grads=None,
        )
        high = SingletEvalResult(
            theta=np.array([0.0, 0.0, 20.0]),
            fitness=-1.0,
            measures=(0.1, 0.1),
            metrics={"rms_spot": 1.0},
            grads=None,
        )
        self.assertTrue(archive.insert(low))
        self.assertTrue(archive.insert(high))
        self.assertAlmostEqual(archive.stats()["best_fitness"], -1.0)

    def test_gradient_coefficient_cma_shapes(self):
        # Canonical CMA-ES (post-2026-04 rewrite): the covariance attribute
        # is `C` (Hansen notation), not `cov`. The old EMA-tracked emitter
        # called it `cov`; this test was updated to match the new engine.
        cma = GradientCoefficientCMA(dim=3, seed=123, sigma=0.5, population_size=6)
        coeffs = cma.ask(6)
        self.assertEqual(coeffs.shape, (6, 3))
        scores = np.linspace(-1.0, 1.0, 6, dtype=np.float64)
        cma.tell(coeffs, scores)
        self.assertEqual(cma.mean.shape, (3,))
        self.assertEqual(cma.C.shape, (3, 3))
        self.assertTrue(math.isfinite(cma.sigma))


if __name__ == "__main__":
    unittest.main()
