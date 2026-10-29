"""Focused checks for the two independent FFNet ablations."""

import unittest

import torch

from vehicle_ffnet.model import upstream_modules
from vehicle_ffnet.pml_loss import ProximalMappingLoss
from vehicle_ffnet.spatial_gate import SpatialGateFusion
from vehicle_ffnet.train import compute_loss


class PMLVehicleTests(unittest.TestCase):
    def test_carpk_edge_points_map_to_valid_d8_cells_with_unit_mass(self):
        density = torch.zeros((1, 1, 90, 160), dtype=torch.float32)
        density[0, 0, 0, 0] = 1.0
        density[0, 0, 89, 159] = 1.0
        points = [torch.tensor([[0.0, 0.0], [1279.0, 719.0]])]
        total, point_loss, transport = ProximalMappingLoss()(density, points, return_components=True)
        self.assertAlmostEqual(float(density.sum()), len(points[0]))
        self.assertAlmostEqual(float(point_loss), 0.0, places=6)
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(transport))

    def test_point_beyond_valid_d8_shape_is_rejected(self):
        density = torch.ones((1, 1, 2, 3), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "outside the valid stride-8"):
            ProximalMappingLoss()(density, [torch.tensor([[4.0, 16.0]])])

    def test_pml_count_uses_only_unpadded_prediction(self):
        class FixedPrediction(torch.nn.Module):
            def forward(self, images):
                prediction = images[:, :1]
                return prediction, prediction / prediction.flatten(1).sum(1)[:, None, None, None]

        image = torch.zeros((1, 3, 4, 4))
        image[0, 0, 0, 0] = 1.5
        image[0, 0, 3, 3] = 999.0
        points = [torch.tensor([[0.0, 0.0]])]
        target = torch.zeros((1, 1, 2, 3))
        target[0, 0, 0, 0] = 1.0
        _, _, predicted, counts = compute_loss(
            FixedPrediction(), ProximalMappingLoss(), image, points, target,
            wot=0.0, wtv=0.0, valid_shape=(2, 3), supervision="pml",
        )
        self.assertAlmostEqual(float(predicted[0]), 1.5)
        self.assertAlmostEqual(float(counts[0]), 1.0)

    def test_pml_training_uses_local_loss_without_extra_global_count(self):
        class FixedPrediction(torch.nn.Module):
            def forward(self, images):
                prediction = images[:, :1]
                return prediction, prediction / prediction.flatten(1).sum(1)[:, None, None, None]

        density = torch.tensor([[[[3.0, 0.0, 0.0, 0.0]]]])
        points = [torch.tensor([[4.0, 4.0], [20.0, 4.0]])]
        criterion = ProximalMappingLoss(stride=8)
        total, terms, _, _ = compute_loss(
            FixedPrediction(), criterion, density, points, torch.zeros_like(density),
            wot=0.0, wtv=0.0, supervision="pml", pml_count_weight=0.0,
        )
        self.assertAlmostEqual(float(total.detach()), float(criterion(density, points).detach()))
        self.assertAlmostEqual(float(terms[1].detach()), 1.0)

    def test_local_mass_cancellation_is_penalized_even_when_global_count_is_right(self):
        criterion = ProximalMappingLoss(stride=8)
        points = [torch.tensor([[4.0, 4.0], [20.0, 4.0]])]
        correct = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        displaced = torch.tensor([[[[2.0, 0.0, 0.0, 0.0]]]], requires_grad=True)
        self.assertEqual(float(correct.sum()), float(displaced.sum()))
        self.assertAlmostEqual(
            float((criterion(displaced, points) - criterion(correct, points)).detach()),
            2.0, places=5,
        )
        criterion(displaced, points).backward()
        self.assertTrue(torch.isfinite(displaced.grad).all())
        self.assertGreater(float(displaced.grad.abs().sum()), 0.0)

    def test_empty_annotation_penalizes_false_positive_mass(self):
        density = torch.tensor([[[[0.25, 0.75]]]], requires_grad=True)
        loss = ProximalMappingLoss()(density, [torch.empty((0, 2))])
        self.assertAlmostEqual(float(loss.detach()), 2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(density.grad).all())


class SpatialGateTests(unittest.TestCase):
    def test_gate_starts_exactly_as_original_fusion_and_receives_gradient(self):
        upstream, _ = upstream_modules()
        original = upstream.Fusion(16, 32, 64)
        with torch.no_grad():
            original.final[0].bias.fill_(1.0)
        features = (
            torch.randn((1, 16, 8, 8)),
            torch.randn((1, 32, 4, 4)),
            torch.randn((1, 64, 2, 2)),
        )
        with torch.no_grad():
            baseline = original(*features)
        gated = SpatialGateFusion(original)
        with torch.no_grad():
            self.assertTrue(torch.allclose(baseline, gated(*features), atol=1e-7))
        gated(*features).sum().backward()
        self.assertIsNotNone(gated.residual.weight.grad)
        self.assertGreater(float(gated.residual.weight.grad.abs().sum()), 0.0)
        with torch.no_grad():
            gated.residual.weight -= 1e-4 * gated.residual.weight.grad
        gated.zero_grad(set_to_none=True)
        gated(*features).sum().backward()
        self.assertGreater(float(gated.route.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
