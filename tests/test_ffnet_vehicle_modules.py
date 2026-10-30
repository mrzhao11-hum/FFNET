"""Checks for the point-only FFNet vehicle feature variants."""

import unittest

import torch

from vehicle_ffnet.model import build_model, upstream_modules
from vehicle_ffnet.pml_loss import ProximalMappingLoss
from vehicle_ffnet.vehicle_modules import (
    LocalDetailEnhancement, ScaleGateFusion, VEHICLE_VARIANTS,
)


class VehicleModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_variant_switches_are_separate(self):
        self.assertEqual(VEHICLE_VARIANTS["a1_lde"], (True, False, False))
        self.assertEqual(VEHICLE_VARIANTS["a2_vftm"], (False, True, False))
        self.assertEqual(VEHICLE_VARIANTS["a3_lde_vftm"], (True, True, False))
        self.assertEqual(VEHICLE_VARIANTS["a4_lde_vftm_gate"], (True, True, True))

    def test_f4_detail_is_aligned_and_receives_gradient(self):
        torch.manual_seed(11)
        detail = LocalDetailEnhancement()
        f4 = torch.randn((1, 96, 16, 20), requires_grad=True)
        f8 = torch.randn((1, 192, 8, 10), requires_grad=True)
        output = detail(f4, f8)
        self.assertEqual(tuple(output.shape), tuple(f8.shape))
        output.square().mean().backward()
        self.assertGreater(float(f4.grad.abs().sum()), 0.0)
        self.assertGreater(float(detail.project[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(detail.gate[-1].weight.grad.abs().sum()), 0.0)

    def test_cross_scale_gate_starts_as_original_and_has_first_step_gradient(self):
        torch.manual_seed(12)
        upstream, _ = upstream_modules()
        original = upstream.Fusion(16, 32, 64)
        with torch.no_grad():
            original.final[0].weight.fill_(0.01)
            original.final[0].bias.fill_(1.0)
        features = (torch.randn((1, 16, 8, 12)),
                    torch.randn((1, 32, 4, 6)),
                    torch.randn((1, 64, 2, 3)))
        with torch.no_grad():
            baseline = original(*features)
        gated = ScaleGateFusion(original)
        predicted = gated(*features)
        self.assertTrue(torch.allclose(baseline, predicted, atol=1e-6))
        weights = gated.route_weights(gated.aligned_features(*features))
        self.assertTrue(torch.allclose(weights, torch.full_like(weights, 1 / 3)))
        predicted.square().mean().backward()
        self.assertGreater(float(gated.route[-1].weight.grad.abs().sum()), 0.0)

    def test_a3_and_a4_have_identical_initial_density(self):
        image = torch.randn((1, 3, 96, 128))
        torch.manual_seed(42)
        a3 = build_model(pretrained=False, attention_norm="group_all",
                         vehicle_variant="a3_lde_vftm").eval()
        torch.manual_seed(42)
        a4 = build_model(pretrained=False, attention_norm="group_all",
                         vehicle_variant="a4_lde_vftm_gate").eval()
        with torch.no_grad():
            d3, _ = a3(image)
            d4, _ = a4(image)
        self.assertEqual(tuple(d3.shape), (1, 1, 12, 16))
        self.assertTrue(torch.allclose(d3, d4, atol=1e-5))
        self.assertEqual(sum(isinstance(m, torch.nn.BatchNorm2d) for m in a4.modules()), 0)

    def test_point_loss_reaches_all_three_modules(self):
        torch.manual_seed(43)
        model = build_model(pretrained=False, attention_norm="group_all",
                            vehicle_variant="a4_lde_vftm_gate")
        with torch.no_grad():
            model.fusion.final[0].weight.normal_(0.0, 0.001)
            model.fusion.final[0].bias.fill_(1.0)
        image = torch.randn((1, 3, 96, 128))
        density, _ = model(image)
        loss = ProximalMappingLoss()(density, [torch.tensor([[64.0, 48.0]])])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (model.lde.project[0].weight,
                          model.ccsm1.experts[0][0].weight,
                          model.fusion.route[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
