"""Mass conservation checks for the fixed-sigma FFNet target."""

import unittest

import numpy as np

from vehicle_ffnet.gaussian_targets import gaussian_density_d8


class GaussianTargetTests(unittest.TestCase):
    def test_one_unit_per_vehicle_including_image_boundary(self):
        points = np.array([[0.0, 0.0], [1279.0, 719.0], [410.5, 350.5]])
        target = gaussian_density_d8(points, 720, 1280)
        self.assertEqual(target.shape, (90, 160))
        self.assertAlmostEqual(float(target.sum(dtype=np.float64)), 3.0, places=4)
        self.assertGreater(float(target[0, 0]), 0.0)
        self.assertGreater(float(target[-1, -1]), 0.0)

    def test_empty_and_non_divisible_image(self):
        empty = gaussian_density_d8(np.empty((0, 2)), 37, 65)
        self.assertEqual(empty.shape, (5, 9))
        self.assertEqual(float(empty.sum()), 0.0)
        corner = gaussian_density_d8(np.array([[64.0, 36.0]]), 37, 65)
        self.assertAlmostEqual(float(corner.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
