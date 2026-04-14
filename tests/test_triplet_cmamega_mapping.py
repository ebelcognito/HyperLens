import unittest

import numpy as np

from optical_rl.qd.cmamega_triplet.problem_triplet_cooke import FixedCookeTripletProblem, TripletCookeConfig


class TestTripletCMAMEGAMapping(unittest.TestCase):
    def test_triplet_mapping_and_flat_init(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch is not available in this environment")

        cfg = TripletCookeConfig(
            lens_path="/home/hisham26/Desktop/RL/DeepLens/datasets/lenses/cooke.json",
            device="cpu",
            num_rays=32,
            theta_bounds=(
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-0.05, 0.05),
                (5.0, 120.0),
            ),
            active_descriptor_indices=(0, 1),
        )
        problem = FixedCookeTripletProblem(cfg)
        theta0 = problem.initial_theta()
        self.assertEqual(theta0.shape, (7,))
        self.assertTrue(np.allclose(theta0[:6], 0.0))

        res = problem.evaluate(theta0, with_grads=False)
        self.assertTrue(np.isfinite(res.fitness))
        self.assertEqual(res.descriptor_vector.shape, (5,))
        self.assertTrue(np.isfinite(res.measures[0]))
        self.assertTrue(np.isfinite(res.measures[1]))


if __name__ == "__main__":
    unittest.main()
