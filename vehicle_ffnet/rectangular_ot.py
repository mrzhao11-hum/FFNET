"""FFNet's point OT surrogate on a valid rectangular stride-8 density map."""

from __future__ import annotations

import torch
from torch import nn

from .model import upstream_modules


class RectangularOTLoss(nn.Module):
    def __init__(self, stride: int, device: torch.device, iterations: int = 100,
                 regularization: float = 10.0, reference_size: float = 256.0) -> None:
        super().__init__()
        upstream_modules()
        from losses.bregman_pytorch import sinkhorn

        self.sinkhorn = sinkhorn
        self.stride = stride
        self.device = device
        self.iterations = iterations
        self.regularization = regularization
        self.reference_size = reference_size

    def forward(self, normalized: torch.Tensor, prediction: torch.Tensor,
                points: list[torch.Tensor]):
        if normalized.shape != prediction.shape or normalized.ndim != 4:
            raise ValueError("OT maps must have matching [B,1,H,W] shapes")
        batch_size, channels, rows, cols = prediction.shape
        if channels != 1 or len(points) != batch_size:
            raise ValueError("OT expects one density channel and one point set per image")
        scale = self.reference_size / max(rows * self.stride, cols * self.stride)
        ys = (torch.arange(rows, device=prediction.device, dtype=torch.float32) + 0.5)
        xs = (torch.arange(cols, device=prediction.device, dtype=torch.float32) + 0.5)
        ys = ys * self.stride * scale
        xs = xs * self.stride * scale
        loss = prediction.new_zeros(1)
        objective = prediction.new_zeros(1)
        wd = 0.0
        for index, image_points in enumerate(points):
            if len(image_points) == 0:
                continue
            scaled_points = image_points.to(prediction.device) * scale
            y_distance = (scaled_points[:, 1, None] - ys[None, :]).square()
            x_distance = (scaled_points[:, 0, None] - xs[None, :]).square()
            cost = (y_distance[:, :, None] + x_distance[:, None, :]).reshape(
                len(image_points), -1
            )
            source_prob = normalized[index, 0].reshape(-1).detach()
            target_prob = torch.full(
                (len(image_points),), 1.0 / len(image_points),
                dtype=prediction.dtype, device=prediction.device,
            )
            plan, log = self.sinkhorn(
                target_prob, source_prob, cost, self.regularization,
                maxIter=self.iterations, log=True,
            )
            beta = log["beta"]
            if not torch.isfinite(beta).all():
                raise FloatingPointError("nonfinite rectangular OT dual variables")
            objective += torch.sum(normalized[index, 0] * beta.reshape(rows, cols))
            source = prediction[index, 0].reshape(-1).detach()
            source_count = source.sum()
            denominator = source_count.square() + 1e-8
            gradient = source_count / denominator * beta - (source * beta).sum() / denominator
            loss += torch.sum(prediction[index, 0] * gradient.detach().reshape(rows, cols))
            wd += float(torch.sum(cost * plan).item())
        return loss, wd, objective
