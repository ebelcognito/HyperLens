import unittest

from optical_rl.qd.cmamega_triplet.problem_triplet_cooke import FixedCookeTripletProblem, TripletCookeConfig
from optical_rl.qd.cmamega_triplet.runner_triplet_cmamega import TripletCMAMEGARunner
from optical_rl.qd.cmamega_triplet.types import RunnerConfig


class TestTripletCMAMEGARunnerSmoke(unittest.TestCase):
    def test_runner_smoke(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch is not available in this environment")

        problem_cfg = TripletCookeConfig(
            lens_path="/home/hisham26/Desktop/RL/DeepLens/datasets/lenses/cooke.json",
            device="cpu",
            num_rays=24,
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
        problem = FixedCookeTripletProblem(problem_cfg)
        runner_cfg = RunnerConfig(
            theta_bounds=problem_cfg.theta_bounds,
            archive_bins=(12, 12),
            archive_bounds=None,
            cma_eta=0.05,
            seed_count=8,
            calibration_count=32,
            initial_sigma=0.5,
            population_size=4,
            cma_seed=0,
            log_every=100,
            active_descriptor_indices=(0, 1),
        )
        runner = TripletCMAMEGARunner(runner_cfg, problem)
        seed_stats = runner.seed_archive()
        self.assertGreater(seed_stats["filled"], 0.0)
        final_stats = runner.run(16)
        self.assertGreaterEqual(final_stats["coverage"], seed_stats["coverage"])


if __name__ == "__main__":
    unittest.main()
