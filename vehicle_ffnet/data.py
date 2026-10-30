"""Point-supervised vehicle data in the format expected by official FFNet."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .prepare import load_boxes


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def box_centers(boxes: list[tuple[float, float, float, float]]) -> torch.Tensor:
    if not boxes:
        return torch.empty((0, 2), dtype=torch.float32)
    return torch.tensor([((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes])


def discrete_map(points: torch.Tensor, crop_size: int, stride: int = 8) -> torch.Tensor:
    """Each accepted center contributes exactly one unit at output stride."""
    if crop_size % stride:
        raise ValueError("crop size must be divisible by output stride")
    side = crop_size // stride
    target = torch.zeros((1, side, side), dtype=torch.float32)
    if len(points):
        pixel = points.round().long().clamp(0, crop_size - 1)
        row = pixel[:, 1] // stride
        col = pixel[:, 0] // stride
        flat_index = row * side + col
        target.view(-1).scatter_add_(0, flat_index, torch.ones(len(points)))
    if target.sum().item() != len(points):
        raise RuntimeError("discrete target lost point mass")
    return target


def discrete_map_rect(points: torch.Tensor, height: int, width: int, stride: int = 8) -> torch.Tensor:
    """Build a rectangular point map with one unit per annotated center."""
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    rows = (height + stride - 1) // stride
    cols = (width + stride - 1) // stride
    target = torch.zeros((1, rows, cols), dtype=torch.float32)
    if len(points):
        pixel = points.round().long()
        row = pixel[:, 1].clamp(0, height - 1) // stride
        col = pixel[:, 0].clamp(0, width - 1) // stride
        target.view(-1).scatter_add_(
            0, row * cols + col, torch.ones(len(points), dtype=torch.float32)
        )
    if target.sum().item() != len(points):
        raise RuntimeError("rectangular target lost point mass")
    return target


def normalize_image(image: Image.Image) -> torch.Tensor:
    return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)


class VehicleDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        manifest: str | Path,
        crop_size: int = 384,
        train: bool = False,
        flip_probability: float = 0.5,
        limit: int | None = None,
        full_image: bool = False,
        gaussian_target_root: str | Path | None = None,
        gaussian_sigma: float = 8.0,
        gaussian_truncate: float = 3.0,
    ) -> None:
        self.data_root = Path(data_root)
        document = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.records = document["samples"]
        self.crop_size = crop_size
        self.train = train
        self.flip_probability = flip_probability
        self.full_image = full_image
        self.gaussian_target_root = Path(gaussian_target_root) if gaussian_target_root else None
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_truncate = gaussian_truncate
        if self.gaussian_target_root is not None and (not train or not full_image):
            raise ValueError("H5 density targets require full-image training")
        if limit is not None:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f"empty manifest: {manifest}")
        if crop_size % 32:
            raise ValueError("official FFNet requires crop size divisible by 32")
        if self.gaussian_target_root is not None:
            for record in self.records:
                path = self.gaussian_target_root / f"{record['id']}.h5"
                if not path.is_file():
                    raise FileNotFoundError(f"missing Gaussian GT: {path}; run prepare_gaussian_h5")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(self.data_root / record["image"]) as source:
            image = source.convert("RGB")
        points = box_centers(load_boxes(self.data_root / record["annotation"]))
        if image.size != (record["width"], record["height"]) or len(points) != record["count"]:
            raise ValueError(f"image or annotation changed since manifest creation: {record['id']}")
        if not self.train:
            return normalize_image(image), float(len(points)), record["id"]

        width, height = image.size
        if self.full_image:
            density = None
            if self.gaussian_target_root is not None:
                from .prepare_gaussian_h5 import load_gaussian_h5
                density = load_gaussian_h5(
                    self.gaussian_target_root / f"{record['id']}.h5", points.numpy(),
                    height, width, self.gaussian_sigma, self.gaussian_truncate)
            if random.random() < self.flip_probability:
                image = TF.hflip(image)
                points[:, 0] = width - 1 - points[:, 0]
                if density is not None:
                    density = np.flip(density, axis=1).copy()
            if density is None:
                target = discrete_map_rect(points, height, width)
            else:
                from .gaussian_targets import block_sum_density
                target = torch.from_numpy(block_sum_density(density))[None]
            normalized = normalize_image(image)
            padded_height = (height + 31) // 32 * 32
            padded_width = (width + 31) // 32 * 32
            normalized = F.pad(
                normalized, (0, padded_width - width, 0, padded_height - height),
                mode="replicate",
            )
            return normalized, points, target, (target.shape[-2], target.shape[-1])

        size = self.crop_size
        if min(width, height) < size:
            raise ValueError(f"image too small for {size}px crop: {record['id']}")
        left = random.randint(0, width - size)
        top = random.randint(0, height - size)
        image = image.crop((left, top, left + size, top + size))
        points = points - torch.tensor((left, top), dtype=torch.float32)
        inside = ((points >= 0) & (points < size)).all(dim=1)
        points = points[inside]
        target = discrete_map(points, size)
        if random.random() > 1.0 - self.flip_probability:
            image = TF.hflip(image)
            points[:, 0] = size - 1 - points[:, 0]
            target = torch.flip(target, dims=(-1,))
        return normalize_image(image), points, target


def train_collate(batch):
    images, points, maps = zip(*batch)
    return torch.stack(images), list(points), torch.stack(maps)


def full_image_collate(batch):
    """Full-image training uses batch one so source image sizes may vary."""
    if len(batch) != 1:
        raise ValueError("full-image training requires batch size 1")
    image, points, target, valid_shape = batch[0]
    return image.unsqueeze(0), [points], target.unsqueeze(0), valid_shape
