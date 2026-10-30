"""Check the mass conservation and valid-region geometry used by diagnosis."""

import unittest

import torch

from vehicle_ffnet.diagnose_instance_mass import assign_mass, predict_valid_d8


class ConstantDensityModel(torch.nn.Module):
    def forward(self, image):
        rows, cols = image.shape[-2] // 8, image.shape[-1] // 8
        density = image.new_ones((1, 1, rows, cols))
        return density, density


class InstanceMassDiagnosisTest(unittest.TestCase):
    def test_nearest_point_partition_conserves_density(self):
        density = torch.tensor([[0.4, 0.6]])
        points = torch.tensor([[3.5, 3.5], [11.5, 3.5]])
        assigned, local, far = assign_mass(density, points, radius_px=4)
        torch.testing.assert_close(assigned, torch.tensor([0.4, 0.6]))
        torch.testing.assert_close(local, assigned)
        self.assertAlmostEqual(float(far), 0.0)

    def test_far_mass_not_counted_as_local(self):
        density = torch.tensor([[1.0, 2.0]])
        points = torch.tensor([[3.5, 3.5]])
        assigned, local, far = assign_mass(density, points, radius_px=4)
        self.assertAlmostEqual(float(assigned[0]), 3.0)
        self.assertAlmostEqual(float(local[0]), 1.0)
        self.assertAlmostEqual(float(far), 2.0)

    def test_padded_output_cells_are_removed(self):
        image = torch.zeros((1, 3, 73, 129))
        density = predict_valid_d8(ConstantDensityModel(), image)
        self.assertEqual(tuple(density.shape), (10, 17))
        self.assertEqual(float(density.sum()), 170.0)


if __name__ == "__main__":
    unittest.main()
