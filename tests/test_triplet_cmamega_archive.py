import unittest

import numpy as np

from optical_rl.qd.cmamega_triplet.archive_grid2d import ArchiveGrid2D
from optical_rl.qd.cmamega_triplet.types import EvalResult


class TestTripletCMAMEGAArchive(unittest.TestCase):
    def _result(self, theta, fitness, measures):
        return EvalResult(
            theta=np.asarray(theta, dtype=np.float64),
            fitness=float(fitness),
            measures=(float(measures[0]), float(measures[1])),
            descriptor_vector=np.zeros((5,), dtype=np.float64),
            metrics={},
            grads=None,
        )

    def test_insert_and_replace(self):
        archive = ArchiveGrid2D((8, 8), ((-1.0, 1.0), (-1.0, 1.0)), seed=0)
        a = self._result([0, 0, 0, 0, 0, 0, 35], -2.0, (0.2, -0.3))
        b = self._result([0, 0, 0, 0, 0, 0, 40], -1.0, (0.2, -0.3))
        self.assertTrue(archive.insert(a))
        self.assertTrue(archive.insert(b))
        stats = archive.stats()
        self.assertEqual(stats.filled, 1.0)
        self.assertEqual(stats.best_fitness, -1.0)

    def test_out_of_bounds_rejected(self):
        archive = ArchiveGrid2D((8, 8), ((-1.0, 1.0), (-1.0, 1.0)), seed=0)
        out = self._result([0, 0, 0, 0, 0, 0, 40], -1.0, (2.0, 0.0))
        self.assertFalse(archive.insert(out))


if __name__ == "__main__":
    unittest.main()
