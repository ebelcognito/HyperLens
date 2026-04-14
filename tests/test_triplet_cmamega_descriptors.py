import unittest

import numpy as np

from optical_rl.qd.cmamega_triplet.triplet_descriptors import DESCRIPTOR_NAMES, build_descriptor_dict


class TestTripletCMAMEGADescriptors(unittest.TestCase):
    def test_descriptor_shape_and_finiteness(self):
        desc = build_descriptor_dict(
            mean_offaxis_y=1.0,
            alpha_deg=0.5,
            phi=[0.10, -0.05, 0.02, -0.01, 0.03, -0.02],
            aberr_proxy=0.2,
            invalid_fraction=0.03,
        )
        self.assertEqual(tuple(desc.keys()), DESCRIPTOR_NAMES)
        vec = np.array([float(desc[k]) for k in DESCRIPTOR_NAMES], dtype=np.float64)
        self.assertEqual(vec.shape, (5,))
        self.assertTrue(np.all(np.isfinite(vec)))

    def test_surface_power_skew_bounded(self):
        desc = build_descriptor_dict(
            mean_offaxis_y=0.5,
            alpha_deg=0.5,
            phi=[0.2, 0.1, -0.05, -0.2, -0.1, 0.05],
            aberr_proxy=0.1,
            invalid_fraction=0.0,
        )
        d1 = float(desc["surface_power_skew"])
        self.assertGreaterEqual(d1, -1.0 - 1e-6)
        self.assertLessEqual(d1, 1.0 + 1e-6)

    def test_descriptor_spread_non_degenerate(self):
        a = build_descriptor_dict(
            mean_offaxis_y=0.2,
            alpha_deg=0.5,
            phi=[0.02, 0.02, 0.02, 0.01, 0.01, 0.01],
            aberr_proxy=0.05,
            invalid_fraction=0.02,
        )
        b = build_descriptor_dict(
            mean_offaxis_y=2.0,
            alpha_deg=0.5,
            phi=[0.30, 0.20, 0.10, -0.10, -0.20, -0.30],
            aberr_proxy=0.40,
            invalid_fraction=0.20,
        )
        v_a = np.array([float(a[k]) for k in DESCRIPTOR_NAMES], dtype=np.float64)
        v_b = np.array([float(b[k]) for k in DESCRIPTOR_NAMES], dtype=np.float64)
        self.assertGreater(float(np.linalg.norm(v_a - v_b)), 1e-3)


if __name__ == "__main__":
    unittest.main()
