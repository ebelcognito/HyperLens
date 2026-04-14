import unittest

import numpy as np

from optical_rl.qd.cmamega_ext.archive_grid2d import ArchiveGrid2D
from optical_rl.qd.cmamega_ext.types import EvalResult


class TestCMAMEGAExtArchive(unittest.TestCase):
    def _result(self, theta, fitness, measures):
        return EvalResult(
            theta=np.asarray(theta, dtype=np.float64),
            fitness=float(fitness),
            measures=(float(measures[0]), float(measures[1])),
            descriptor_vector=np.zeros((4,), dtype=np.float64),
            metrics={},
            grads=None,
        )

    def test_insert_and_replace(self):
        arc = ArchiveGrid2D((8, 8), ((-1.0, 1.0), (-1.0, 1.0)), seed=0)
        a = self._result([0, 0, 0, 0, 4, 40], -2.0, (0.2, -0.3))
        b = self._result([0, 0, 0, 0, 4, 50], -1.0, (0.2, -0.3))
        self.assertTrue(arc.insert(a))
        self.assertTrue(arc.insert(b))
        stats = arc.stats()
        self.assertEqual(stats.filled, 1.0)
        self.assertEqual(stats.best_fitness, -1.0)

    def test_out_of_bounds_rejected(self):
        arc = ArchiveGrid2D((8, 8), ((-1.0, 1.0), (-1.0, 1.0)), seed=0)
        a = self._result([0, 0, 0, 0, 4, 40], -1.0, (2.0, 0.0))
        self.assertFalse(arc.insert(a))


if __name__ == "__main__":
    unittest.main()
