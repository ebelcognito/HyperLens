import unittest

import numpy as np

from optical_rl.qd.map_elites import MapElites, MapElitesConfig


class TestMapElites(unittest.TestCase):
    def test_fills_archive_and_improves_best(self):
        # Genotype: 2D point. Descriptor: same 2D point. Fitness: highest at origin.
        cfg = MapElitesConfig(
            bins=(10, 10),
            desc_bounds=((-1.0, 1.0), (-1.0, 1.0)),
            sigma=0.15,
            seed=123,
        )

        def sample(rng: np.random.Generator) -> np.ndarray:
            return rng.uniform(-1.0, 1.0, size=(2,))

        def evaluate(g: np.ndarray):
            x, y = float(g[0]), float(g[1])
            fitness = -(x * x + y * y)  # best is 0 at (0,0)
            desc = (x, y)
            return fitness, desc

        me = MapElites(cfg, sample_genotype=sample, evaluate=evaluate)

        stats_seed = me.seed_archive(50)
        self.assertGreater(stats_seed.filled, 0)
        self.assertGreater(stats_seed.coverage, 0.0)

        best_after_seed = stats_seed.best_fitness
        stats_final = me.run(num_evals=400, seed=0)

        self.assertGreaterEqual(stats_final.best_fitness, best_after_seed)
        self.assertGreater(stats_final.filled, stats_seed.filled)
        self.assertGreater(stats_final.coverage, stats_seed.coverage)


if __name__ == "__main__":
    unittest.main()

