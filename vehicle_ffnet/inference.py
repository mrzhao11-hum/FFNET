"""Official FFNet overlapping-patch inference, separated for reuse and tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def predict_density(model, image: torch.Tensor, crop_size: int, patch_batch_size: int = 4) -> torch.Tensor:
    """Return an input-resolution density map using upstream bilinear /64 stitching."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("inference expects one image with shape [1, 3, H, W]")
    _, _, height, width = image.shape
    if min(height, width) < crop_size or crop_size % 32:
        raise ValueError("image must cover the crop, and crop size must be divisible by 32")
    stride = getattr(model, "output_stride", 8)
    windows = []
    for i in range(0, height, crop_size):
        top = max(min(height - crop_size, i), 0)
        for j in range(0, width, crop_size):
            left = max(min(width - crop_size, j), 0)
            windows.append((top, left))

    density = torch.zeros((1, 1, height, width), device=image.device)
    coverage = torch.zeros_like(density)
    for start in range(0, len(windows), patch_batch_size):
        positions = windows[start : start + patch_batch_size]
        patches = torch.cat(
            [image[:, :, top : top + crop_size, left : left + crop_size] for top, left in positions]
        )
        predicted, _ = model(patches)
        if predicted.shape[-2:] != (crop_size // stride, crop_size // stride):
            raise ValueError(f"unexpected FFNet output shape: {tuple(predicted.shape)}")
        upsampled = F.interpolate(
            predicted, size=(crop_size, crop_size), mode="bilinear", align_corners=True
        ) / float(stride ** 2)
        for patch, (top, left) in zip(upsampled, positions):
            density[:, :, top : top + crop_size, left : left + crop_size] += patch
            coverage[:, :, top : top + crop_size, left : left + crop_size] += 1
    if (coverage == 0).any():
        raise RuntimeError("patch inference left uncovered pixels")
    return density / coverage


@torch.no_grad()
def predict_density_full(model, image: torch.Tensor) -> torch.Tensor:
    """Evaluate one original image after padding to FFNet's stride-32 requirement."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("full-image inference expects one image with shape [1, 3, H, W]")
    height, width = image.shape[-2:]
    stride = getattr(model, "output_stride", 8)
    padded_height = (height + 31) // 32 * 32
    padded_width = (width + 31) // 32 * 32
    padded = F.pad(
        image, (0, padded_width - width, 0, padded_height - height), mode="replicate"
    )
    prediction, _ = model(padded)
    expected = (padded_height // stride, padded_width // stride)
    if prediction.shape[-2:] != expected:
        raise ValueError(f"unexpected FFNet output shape: {tuple(prediction.shape)}")
    valid_rows = (height + stride - 1) // stride
    valid_cols = (width + stride - 1) // stride
    valid_prediction = prediction[:, :, :valid_rows, :valid_cols]
    # Remove padded output cells before interpolation so they cannot blur into
    # the density map near the original image boundary.
    fine = F.interpolate(
        valid_prediction,
        size=(valid_rows * stride, valid_cols * stride),
        mode="bilinear",
        align_corners=True,
    ) / float(stride ** 2)
    fine = fine[:, :, :height, :width]
    valid_mass = valid_prediction.sum()
    return fine * (valid_mass / fine.sum().clamp_min(1e-8))
