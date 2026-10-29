"""Small spatial scale gate at FFNet's existing three-branch fusion point."""

from __future__ import annotations

import torch
from torch import nn


class SpatialGateFusion(nn.Module):
    """Add a zero-initialized, spatially routed residual before the FFNet head.

    The original 16/32/64-channel concatenation and density head remain in
    place.  The gate therefore tests whether location-specific scale selection
    adds anything beyond the author's CCSM attention and concatenation.
    """

    def __init__(self, original_fusion: nn.Module) -> None:
        super().__init__()
        self.upsample_1 = original_fusion.upsample_1
        self.upsample_2 = original_fusion.upsample_2
        self.final = original_fusion.final
        channels = (16, 32, 64)
        width = 16
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(source, width, kernel_size=1, bias=False),
                nn.GroupNorm(4, width),
                nn.GELU(),
            )
            for source in channels
        )
        self.route = nn.Conv2d(3 * width, 3, kernel_size=3, padding=1)
        self.residual = nn.Conv2d(width, sum(channels), kernel_size=1)
        nn.init.zeros_(self.route.weight)
        nn.init.zeros_(self.route.bias)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        x2 = self.upsample_1(x2)
        x3 = self.upsample_2(x3)
        if x1.shape[-2:] != x2.shape[-2:] or x1.shape[-2:] != x3.shape[-2:]:
            raise ValueError("FFNet branch shapes do not align at stride 8")
        aligned = [projection(features) for projection, features in zip(
            self.projections, (x1, x2, x3)
        )]
        alpha = torch.softmax(self.route(torch.cat(aligned, dim=1)), dim=1)
        selected = sum(alpha[:, index:index + 1] * features
                       for index, features in enumerate(aligned))
        original = torch.cat((x1, x2, x3), dim=1)
        return self.final(original + self.residual(selected))
