"""Exercise H5 alignment, partial edge cells and the actual training loss."""
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from vehicle_ffnet.data import VehicleDataset, full_image_collate
from vehicle_ffnet.gaussian_targets import gaussian_density_d8
from vehicle_ffnet.prepare_gaussian_h5 import prepare_record, load_gaussian_h5
from vehicle_ffnet.train import compute_loss


class GaussianH5Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        Image.new("RGB", (65, 37)).save(self.root / "image.png")
        (self.root / "boxes.txt").write_text("0 0 2 2 1\n62 34 64 36 1\n")
        self.points = np.array([[1., 1.], [63., 35.]], dtype=np.float32)
        self.record = dict(id="sample", image="image.png", annotation="boxes.txt",
                           width=65, height=37, count=2)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"samples": [self.record]}))
        prepare_record(self.record, self.root, self.root / "gt")

    def test_gt_matches_online_and_flip_before_partial_block_sum(self):
        for flip in (0., 1.):
            ds = VehicleDataset(self.root, self.manifest, train=True, full_image=True,
                                flip_probability=flip, gaussian_target_root=self.root / "gt")
            image, points, target, shape = ds[0]
            expected = gaussian_density_d8(points.numpy(), 37, 65)
            np.testing.assert_allclose(target[0].numpy(), expected, atol=1e-6)
            self.assertEqual(shape, (5, 9))
            self.assertEqual(tuple(image.shape), (3, 64, 96))
            self.assertAlmostEqual(float(target.sum()), 2., places=5)

    def test_reject_stale_annotation_and_corrupt_mass(self):
        path = self.root / "gt/sample.h5"
        with self.assertRaisesRegex(ValueError, "annotation mismatch"):
            load_gaussian_h5(path, self.points + 1, 37, 65)
        with h5py.File(path, "r+") as f:
            f["density"][:] = 0
        with self.assertRaisesRegex(ValueError, "mass mismatch"):
            load_gaussian_h5(path, self.points, 37, 65)

    def test_loss_masks_model_padding_and_matches_formula(self):
        ds = VehicleDataset(self.root, self.manifest, train=True, full_image=True,
                            flip_probability=0., gaussian_target_root=self.root / "gt")
        images, points, target, valid = full_image_collate([ds[0]])
        class Fixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.output = torch.nn.Parameter(torch.full((1, 1, 8, 12), 0.1))
            def forward(self, x):
                return self.output, self.output / self.output.sum()
        model = Fixed()
        total, terms, prediction, count = compute_loss(
            model, None, images, points, target, 0., 0., valid_shape=valid,
            supervision="gaussian", gaussian_target_source="h5")
        cropped = model.output[:, :, :5, :9]
        expected = (cropped-target).square().sum() + .002*(cropped.sum()-2).abs()
        torch.testing.assert_close(total, expected)
        self.assertAlmostEqual(float(prediction[0]), 4.5, places=5)
        total.backward()
        self.assertTrue(torch.isfinite(model.output.grad).all())
        self.assertGreater(float(model.output.grad[:, :, :5, :9].abs().sum()), 0)
        self.assertEqual(float(model.output.grad[:, :, 5:, :].abs().sum()), 0.)
        self.assertEqual(float(model.output.grad[:, :, :, 9:].abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
