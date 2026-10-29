"""Check that pixel and global-count losses both reach the FFNet output."""

import unittest

import torch

from vehicle_ffnet.train import compute_loss


class GaussianLossTests(unittest.TestCase):
    def test_gaussian_and_count_terms_are_independent_and_differentiable(self):
        class FixedPrediction(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.density = torch.nn.Parameter(torch.full((1, 1, 4, 4), 0.1))

            def forward(self, images):
                prediction = self.density.expand(len(images), -1, -1, -1)
                return prediction, prediction / prediction.sum()

        model = FixedPrediction()
        images = torch.zeros((1, 3, 32, 32))
        points = [torch.tensor([[4.0, 4.0]])]
        discrete = torch.zeros((1, 1, 4, 4))
        no_count, terms, _, _ = compute_loss(
            model, None, images, points, discrete, 0.0, 0.0,
            supervision="gaussian", gaussian_count_weight=0.0,
        )
        with_count, weighted_terms, _, _ = compute_loss(
            model, None, images, points, discrete, 0.0, 0.0,
            supervision="gaussian", gaussian_count_weight=1.0,
        )
        self.assertAlmostEqual(float((with_count - no_count).detach()),
                               float(weighted_terms[1].detach()), places=5)
        self.assertGreater(float(terms[6].detach()), 0.0)
        with_count.backward()
        self.assertTrue(torch.isfinite(model.density.grad).all())
        self.assertGreater(float(model.density.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
