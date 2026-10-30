"""PML's nearest-point and proximal-map loss for a stride-8 vehicle density.

This follows the two terms in the authors' ``losses/pmloss.py``: each point
receives one unit of density mass, while a centered distance cost moves mass
towards its nearest point.  The original full-image box centers are used as
point targets; no Gaussian target or vehicle-size proxy is introduced.
"""

from __future__ import annotations

import torch
from torch import nn


class ProximalMappingLoss(nn.Module):
    def __init__(self, stride: int = 8, radius_px: float = 32.0,
                 density_factor: float = 1.0) -> None:
        super().__init__()
        if stride <= 0 or radius_px <= 0 or density_factor <= 0:
            raise ValueError("stride, radius, and density factor must be positive")
        self.stride = stride
        self.radius_px = radius_px
        self.density_factor = density_factor

    def forward(self, density: torch.Tensor, points: list[torch.Tensor],
                return_components: bool = False):
        if density.ndim != 4 or density.shape[1] != 1 or len(points) != len(density):
            raise ValueError("PML expects [B,1,H,W] density and one point set per image")
        batch, _, rows, cols = density.shape
        # Work in output-cell coordinates: an original-image point (x, y)
        # maps to (x/stride, y/stride). Cell centres retain PML's
        # (stride-1)/2 original-pixel convention.
        cell_offset = (self.stride - 1) / (2 * self.stride)
        ys = torch.arange(rows, device=density.device, dtype=torch.float32) + cell_offset
        xs = torch.arange(cols, device=density.device, dtype=torch.float32) + cell_offset
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid_xy = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
        point_total = density.new_zeros(())
        transport_total = density.new_zeros(())

        for index, image_points in enumerate(points):
            flat = density[index, 0].flatten()
            count = len(image_points)
            if count == 0:
                # Match the authors' empty-image branch (both terms penalize mass).
                excess = flat.abs().sum()
                point_total = point_total + excess
                transport_total = transport_total + excess
                continue

            with torch.no_grad():
                if image_points.ndim != 2 or image_points.shape[1] < 2:
                    raise ValueError("PML points must have x,y coordinates")
                point_xy = image_points[:, :2].to(device=density.device, dtype=torch.float32) / self.stride
                if (not bool(torch.isfinite(point_xy).all())
                        or bool(((point_xy < 0) | (point_xy[:, 0:1] >= cols)
                                 | (point_xy[:, 1:2] >= rows)).any())):
                    raise ValueError(f"PML point is outside the valid stride-{self.stride} density map")
            image_mass = flat.sum()
            if bool((image_mass.detach() <= self.density_factor).item()):
                # The original implementation uses a global count fallback until
                # there is enough predicted mass for per-point assignment.
                point_total = point_total + (image_mass - count * self.density_factor).abs()
                continue

            with torch.no_grad():
                distances = torch.cdist(grid_xy, point_xy)
                nearest_distance, assignment = distances.min(dim=1)
                # The hard nearest-point partition is exactly the author's M.
                partition = torch.zeros_like(distances).scatter_(1, assignment[:, None], 1.0)
                region_max = (partition * nearest_distance[:, None]).amax(dim=0)
                region_max = region_max.clamp_min(self.radius_px / self.stride)
                cost = torch.expm1(nearest_distance / region_max[assignment])
                region_size = partition.sum(dim=0).clamp_min(1.0)
                mean_cost = (partition * cost[:, None]).sum(dim=0) / region_size
                centered_cost = cost - mean_cost[assignment]

            point_mass = (partition * flat[:, None]).sum(dim=0)
            point_loss = (point_mass - self.density_factor).abs().sum()
            transport_loss = (flat * centered_cost).sum()
            point_total = point_total + point_loss
            transport_total = transport_total + transport_loss

        point_total = point_total / batch
        transport_total = transport_total / batch
        total_loss = point_total + transport_total
        if return_components:
            return total_loss, point_total, transport_total
        return total_loss
