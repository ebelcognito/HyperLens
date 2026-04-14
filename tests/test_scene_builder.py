import unittest

from optical_rl.utils.scene_builder import scene_ray_bundle_with_custom_glass


class TestSceneBuilder(unittest.TestCase):
    def test_scene_ray_bundle_builds_n_rays(self):
        points = [
            (130.0, 130.0),
            (150.0, 120.0),
            (170.0, 130.0),
            (180.0, 150.0),
            (170.0, 170.0),
            (150.0, 180.0),
            (130.0, 170.0),
            (120.0, 150.0),
        ]
        ray_targets = [(290.0, 120.0), (290.0, 150.0), (290.0, 180.0)]

        scene = scene_ray_bundle_with_custom_glass(
            source_x=10.0,
            source_y=150.0,
            ray_targets=ray_targets,
            detector_x=290.0,
            detector_center_y=150.0,
            detector_half_height=3.0,
            points=points,
            ref_index=1.5,
        )

        self.assertIn("objs", scene)
        objs = scene["objs"]
        rays = [o for o in objs if o.get("type") == "SingleRay"]
        glass = [o for o in objs if o.get("type") == "Glass"]
        det = [o for o in objs if o.get("type") == "Detector"]

        self.assertEqual(len(rays), len(ray_targets))
        self.assertEqual(len(glass), 1)
        self.assertEqual(len(det), 1)


if __name__ == "__main__":
    unittest.main()

