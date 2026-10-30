"""Prevent a hard nonnegative-output dead zone in Gaussian regression."""
import unittest
import torch
from vehicle_ffnet.model import build_model, initialize_density_head


class DensityHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_negative_logits_still_receive_recovery_gradient(self):
        for activation, has_gradient in ((torch.nn.ReLU(), False),
                                         (torch.nn.Softplus(), True)):
            logits = torch.full((1, 1, 2, 2), -6., requires_grad=True)
            target = torch.full_like(logits, .02)
            prediction = activation(logits)
            loss = (prediction-target).square().sum() + .002*(prediction.sum()-.08).abs()
            loss.backward()
            self.assertEqual(bool((logits.grad.abs() > 0).all()), has_gradient)
            if has_gradient:
                self.assertTrue((logits.grad < 0).all())

    def test_calibrated_head_and_checkpoint_loading_keep_activation(self):
        model = build_model(pretrained=False, attention_norm='group_all',
                            vehicle_variant='a3_lde_vftm', density_activation='softplus')
        mass = 42.92865779927449 / 14400
        bias = initialize_density_head(model, mass, weight_std=0.)
        with torch.no_grad():
            prediction = model.fusion.final(torch.randn(1, 112, 90, 160))
        self.assertAlmostEqual(float(prediction.sum()), 42.92865779927449, places=3)
        self.assertLess(bias, -5.)
        other = build_model(pretrained=False, attention_norm='group_all',
                            vehicle_variant='a3_lde_vftm', density_activation='softplus')
        other.load_state_dict(model.state_dict(), strict=True)
        self.assertIsInstance(other.fusion.final[1], torch.nn.Softplus)


if __name__ == '__main__':
    unittest.main()
