"""Mass-preserving fixed-Gaussian targets for the FFNet density output.

Each CARPK box center contributes exactly one unit on the original image.
The image-resolution target is block-summed to FFNet's stride-8 output;
ordinary image resizing would lose the count interpretation.
"""

from __future__ import annotations

import math

import numpy as np


def gaussian_density_d8(
    points: np.ndarray,
    height: int,
    width: int,
    sigma: float = 8.0,
    truncate: float = 3.0,
    stride: int = 8,
) -> np.ndarray:
    """Return a [ceil(H/stride), ceil(W/stride)] target whose sum is #points."""
    if height <= 0 or width <= 0 or sigma <= 0 or truncate <= 0 or stride <= 0:
        raise ValueError("image size, sigma, truncate and stride must be positive")
    centers = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    density = np.zeros((height, width), dtype=np.float32)
    radius = max(1, math.ceil(truncate * sigma))
    for center_x, center_y in centers:
        if not (0 <= center_x < width and 0 <= center_y < height):
            raise ValueError(f"point outside image: {(center_x, center_y)}")
        x_min = max(0, math.floor(float(center_x) - radius))
        x_max = min(width - 1, math.ceil(float(center_x) + radius))
        y_min = max(0, math.floor(float(center_y) - radius))
        y_max = min(height - 1, math.ceil(float(center_y) + radius))
        xs = np.arange(x_min, x_max + 1, dtype=np.float32)[None, :]
        ys = np.arange(y_min, y_max + 1, dtype=np.float32)[:, None]
        kernel = np.exp(-((xs - center_x) ** 2 + (ys - center_y) ** 2)
                        / np.float32(2.0 * sigma * sigma))
        kernel /= np.float32(kernel.sum(dtype=np.float64))
        density[y_min:y_max + 1, x_min:x_max + 1] += kernel

    rows = (height + stride - 1) // stride
    cols = (width + stride - 1) // stride
    padded = np.pad(density, ((0, rows * stride - height),
                              (0, cols * stride - width)))
    target = padded.reshape(rows, stride, cols, stride).sum(axis=(1, 3))
    if not np.isclose(target.sum(dtype=np.float64), len(centers), atol=1e-3):
        raise RuntimeError("Gaussian target lost vehicle mass")
    return target.astype(np.float32, copy=False)
