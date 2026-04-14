import unittest

import numpy as np

from optical_rl.qd.cmamega_ext.problem_doublet_cooke import DoubletCookeConfig, FixedCookeDoubletProblem


class TestCMAMEGAExtDoubletMapping(unittest.TestCase):
    def test_template_transform_and_eval(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch is not available in this environment")

        cfg = DoubletCookeConfig(
            device="cpu",
            num_rays=32,
            theta_bounds=(
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-0.05, 0.05),
                (2.5, 7.5),
                (5.0, 120.0),
            ),
            active_descriptor_indices=(0, 1),
        )
        problem = FixedCookeDoubletProblem(cfg)
        theta0 = problem.initial_theta()
        self.assertEqual(theta0.shape, (6,))
        res = problem.evaluate(theta0, with_grads=False)
        self.assertTrue(np.isfinite(res.fitness))
        self.assertEqual(res.descriptor_vector.shape, (4,))
        self.assertTrue(np.isfinite(res.measures[0]))
        self.assertTrue(np.isfinite(res.measures[1]))


if __name__ == "__main__":
    unittest.main()
