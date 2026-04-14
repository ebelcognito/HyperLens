import unittest

import numpy as np

from optical_rl.qd.cmamega_ext.descriptors import DESCRIPTOR_NAMES, build_descriptor_dict


class TestCMAMEGAExtDescriptors(unittest.TestCase):
    def test_descriptor_shape_and_names(self):
        desc = build_descriptor_dict(
            mean_offaxis_y=1.0,
            alpha_deg=0.5,
            phi=[0.1, -0.05, 0.02, -0.01],
            aberr_proxy=0.2,
        )
        self.assertEqual(tuple(desc.keys()), DESCRIPTOR_NAMES)
        vec = np.array([float(desc[k]) for k in DESCRIPTOR_NAMES], dtype=np.float64)
        self.assertEqual(vec.shape, (4,))
        self.assertTrue(np.all(np.isfinite(vec)))


if __name__ == "__main__":
    unittest.main()
