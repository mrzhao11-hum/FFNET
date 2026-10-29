import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from vehicle_ffnet.data import VehicleDataset, box_centers, discrete_map, discrete_map_rect
from vehicle_ffnet.evaluate import use_current_image_bn_statistics
from vehicle_ffnet.inference import predict_density, predict_density_full
from vehicle_ffnet.model import build_model
from vehicle_ffnet.rectangular_ot import RectangularOTLoss
from vehicle_ffnet.prepare import partition_pucpr_train
from vehicle_ffnet.scene_split import scene_of


class ManifestProtocolTests(unittest.TestCase):
    def test_pucpr_variants_of_one_index_stay_together(self):
        train, val = partition_pucpr_train(["24_Cloudy", "24_Sunny", "25_Cloudy", "25_Sunny"])
        self.assertEqual(train, ["24_Cloudy", "24_Sunny"])
        self.assertEqual(val, ["25_Cloudy", "25_Sunny"])

    def test_existing_manifests_are_disjoint_and_complete(self):
        root = Path("data/manifests")
        for dataset, sizes in (("CARPK", (827, 162, 459, 989)),
                               ("CARPK_GF2_val", (834, 155, 459, 989)),
                               ("PUCPR+", (75, 25, 25, 100))):
            with self.subTest(dataset=dataset):
                ids = {}
                for split in ("train", "val", "test", "train_full"):
                    path = root / dataset / f"{split}.json"
                    rows = json.loads(path.read_text(encoding="utf-8"))["samples"]
                    ids[split] = {row["id"] for row in rows}
                    self.assertEqual(len(ids[split]), len(rows))
                self.assertEqual(tuple(len(ids[name]) for name in ("train", "val", "test", "train_full")), sizes)
                self.assertFalse(ids["train"] & ids["val"])
                self.assertFalse(ids["test"] & ids["train_full"])
                self.assertEqual(ids["train"] | ids["val"], ids["train_full"])

    def test_gf2_candidate_holds_out_one_complete_sequence(self):
        base = Path("data/manifests/CARPK_GF2_val")
        val = json.loads((base / "val.json").read_text(encoding="utf-8"))["samples"]
        train = json.loads((base / "train.json").read_text(encoding="utf-8"))["samples"]
        self.assertEqual({scene_of(row["id"]) for row in val}, {"20161030_GF2"})
        self.assertNotIn("20161030_GF2", {scene_of(row["id"]) for row in train})
        self.assertEqual(sum(row["count"] for row in val), 6328)


class DensityProtocolTests(unittest.TestCase):
    def test_rectangular_point_map_keeps_edge_mass(self):
        points = torch.tensor([[0.0, 0.0], [129.0, 89.0]])
        density = discrete_map_rect(points, height=90, width=130)
        self.assertEqual(tuple(density.shape), (1, 12, 17))
        self.assertEqual(float(density.sum()), 2.0)

    def test_full_image_dataset_pads_input_but_not_target(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            (root / "Images").mkdir()
            (root / "Annotations").mkdir()
            Image.new("RGB", (130, 90)).save(root / "Images" / "two.png")
            (root / "Annotations" / "two.txt").write_text(
                "0 0 16 16\n120 80 129 89\n", encoding="utf-8"
            )
            manifest = root / "train.json"
            manifest.write_text(json.dumps({"samples": [{"id": "two", "image": "Images/two.png",
                "annotation": "Annotations/two.txt", "width": 130, "height": 90,
                "count": 2}]}), encoding="utf-8")
            dataset = VehicleDataset(
                root, manifest, crop_size=256, train=True, flip_probability=0,
                full_image=True,
            )
            image, points, density, valid_shape = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 96, 160))
            self.assertEqual(tuple(density.shape), (1, 12, 17))
            self.assertEqual(valid_shape, (12, 17))
            self.assertEqual(float(density.sum()), float(len(points)))

    def test_rectangular_ot_has_finite_nonzero_gradient(self):
        prediction = torch.ones((1, 1, 4, 6), requires_grad=True)
        normalized = prediction / prediction.sum()
        points = [torch.tensor([[12.0, 16.0]])]
        loss, _, _ = RectangularOTLoss(
            8, torch.device("cpu"), iterations=10, reference_size=16.0
        )(
            normalized, prediction, points
        )
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)

    def test_full_image_padding_preserves_constant_density(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, image):
                n, _, h, w = image.shape
                density = torch.full((n, 1, h // 8, w // 8), 64.0, device=image.device)
                return density, density

        image = torch.zeros((1, 3, 96, 80))
        density = predict_density_full(ConstantModel(), image)
        self.assertEqual(tuple(density.shape), (1, 1, 96, 80))
        self.assertTrue(torch.allclose(density, torch.ones_like(density)))
        self.assertAlmostEqual(float(density.sum()), 96 * 80)

    def test_full_image_count_excludes_padded_feature_cells(self):
        class PaddedResponseModel(torch.nn.Module):
            def forward(self, image):
                n, _, h, w = image.shape
                density = torch.ones((n, 1, h // 8, w // 8), device=image.device)
                density[:, :, 10:, :] = 100
                density[:, :, :, 10:] = 100
                return density, density

        image = torch.zeros((1, 3, 80, 80))
        density = predict_density_full(PaddedResponseModel(), image)
        self.assertEqual(tuple(density.shape), (1, 1, 80, 80))
        self.assertAlmostEqual(float(density.sum()), 100.0, places=3)

    def test_full_image_spatial_map_ignores_padded_output_cells(self):
        class EdgeModel(torch.nn.Module):
            def __init__(self, height, width, pad_value):
                super().__init__()
                self.height = height
                self.width = width
                self.pad_value = pad_value

            def forward(self, image):
                n, _, h, w = image.shape
                density = image.new_zeros((n, 1, h // 8, w // 8))
                valid_rows = (self.height + 7) // 8
                valid_cols = (self.width + 7) // 8
                density[:, :, valid_rows - 1, valid_cols - 1] = 1.0
                density[:, :, valid_rows:, :] = self.pad_value
                density[:, :, :, valid_cols:] = self.pad_value
                return density, density

        for height, width in ((720, 1280), (90, 130)):
            with self.subTest(size=(height, width)):
                image = torch.zeros((1, 3, height, width))
                expected = predict_density_full(EdgeModel(height, width, 0.0), image)
                actual = predict_density_full(EdgeModel(height, width, 100.0), image)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
                self.assertAlmostEqual(float(actual.sum()), 1.0, places=5)
                self.assertEqual(tuple(actual.shape), (1, 1, height, width))

    def test_group_norm_replaces_only_pooled_attention_bn(self):
        model = build_model(pretrained=False, attention_norm="group")
        attention_norms = [
            module for name, module in model.named_modules()
            if name.endswith("attention.bn")
        ]
        self.assertEqual(len(attention_norms), 12)
        self.assertTrue(all(isinstance(module, torch.nn.GroupNorm) for module in attention_norms))
        self.assertEqual(
            sum(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules()),
            12,
        )
        attention = dict(model.named_modules())["ccsm1.conv1.0.attention"]
        sample = torch.randn(2, attention.fc.in_channels, 4, 4)
        with torch.no_grad():
            attention.train()
            train_channel_weight = attention(sample)[0]
            attention.eval()
            eval_channel_weight = attention(sample)[0]
        self.assertTrue(torch.allclose(train_channel_weight, eval_channel_weight, atol=1e-6))

    def test_group_all_removes_every_batch_norm(self):
        model = build_model(pretrained=False, attention_norm="group_all")
        self.assertEqual(
            sum(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules()),
            0,
        )
        ccsm_norms = [
            dict(model.named_modules())[f"ccsm{block}.conv{stage}.2"]
            for block in (1, 2, 3) for stage in (1, 2, 3, 4)
        ]
        self.assertTrue(all(isinstance(module, torch.nn.GroupNorm) for module in ccsm_norms))
        self.assertTrue(all(module.num_groups == 8 for module in ccsm_norms))

    def test_current_image_bn_diagnostic_keeps_saved_statistics(self):
        bn = torch.nn.BatchNorm2d(2)
        bn.running_mean.fill_(10)
        model = torch.nn.Sequential(bn).eval()
        image = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
        original_mean = bn.running_mean.clone()
        original_batches = bn.num_batches_tracked.clone()
        with torch.no_grad():
            normal = model(image)
            with use_current_image_bn_statistics(model) as changed:
                current = model(image)
            restored = model(image)
        self.assertEqual(changed, 1)
        self.assertFalse(bn.training)
        self.assertTrue(bn.track_running_stats)
        self.assertTrue(torch.equal(bn.running_mean, original_mean))
        self.assertTrue(torch.equal(bn.num_batches_tracked, original_batches))
        self.assertFalse(torch.allclose(normal, current))
        self.assertTrue(torch.allclose(normal, restored))

    def test_points_keep_one_count_each_even_with_collision(self):
        points = box_centers([(0, 0, 6, 6), (2, 2, 8, 8), (120, 120, 127, 127)])
        density = discrete_map(points, 128)
        self.assertEqual(tuple(density.shape), (1, 16, 16))
        self.assertEqual(float(density.sum()), 3.0)
        self.assertEqual(float(density[0, 0, 0]), 2.0)

    def test_overlap_average_does_not_double_count(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, image):
                n, _, h, w = image.shape
                density = torch.full((n, 1, h // 8, w // 8), 64.0, device=image.device)
                return density, density

        image = torch.zeros((1, 3, 96, 80))
        density = predict_density(ConstantModel(), image, 64, patch_batch_size=2)
        self.assertEqual(tuple(density.shape), (1, 1, 96, 80))
        self.assertTrue(torch.allclose(density, torch.ones_like(density)))
        self.assertAlmostEqual(float(density.sum()), 96 * 80)

    def test_training_crop_count_matches_discrete_target(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            (root / "Images").mkdir()
            (root / "Annotations").mkdir()
            Image.new("RGB", (128, 128)).save(root / "Images" / "one.png")
            (root / "Annotations" / "one.txt").write_text("0 0 16 16\n", encoding="utf-8")
            manifest = root / "train.json"
            manifest.write_text(json.dumps({"samples": [{"id": "one", "image": "Images/one.png",
                "annotation": "Annotations/one.txt", "width": 128, "height": 128,
                "count": 1}]}), encoding="utf-8")
            image, points, density = VehicleDataset(root, manifest, crop_size=128, train=True)[0]
            self.assertEqual(tuple(image.shape), (3, 128, 128))
            self.assertEqual(float(density.sum()), float(len(points)))

    def test_horizontal_flip_matches_upstream_discrete_map_order(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            (root / "Images").mkdir()
            (root / "Annotations").mkdir()
            Image.new("RGB", (128, 128)).save(root / "Images" / "one.png")
            (root / "Annotations" / "one.txt").write_text("7 10 8 10 1\n", encoding="utf-8")
            manifest = root / "train.json"
            manifest.write_text(json.dumps({"samples": [{"id": "one", "image": "Images/one.png",
                "annotation": "Annotations/one.txt", "width": 128, "height": 128,
                "count": 1}]}), encoding="utf-8")
            with patch("vehicle_ffnet.data.random.random", return_value=0.8):
                _, points, density = VehicleDataset(root, manifest, crop_size=128, train=True)[0]
            self.assertAlmostEqual(float(points[0, 0]), 119.5)
            self.assertEqual(float(density[0, 1, 14]), 1.0)


if __name__ == "__main__":
    unittest.main()
