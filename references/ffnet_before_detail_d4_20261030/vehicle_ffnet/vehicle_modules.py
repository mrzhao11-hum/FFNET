"""Point-supervised vehicle adaptations of FFNet's feature path.

The three switches are intentionally separate: F4 local detail, F8 spatial
receptive-field selection, and F8/F16/F32 cross-scale selection.  No box,
visibility, or scale annotation enters these modules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BoundedResidualScale(nn.Module):
    """A learnable residual coefficient in (0, 0.2), initially 0.1."""

    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(()))

    def forward(self) -> torch.Tensor:
        return 0.2 * torch.sigmoid(self.logit)


class LocalDetailEnhancement(nn.Module):
    """Transfer local F4 evidence to aligned F8 cells without a box mask."""

    def __init__(self, f4_channels: int = 96, f8_channels: int = 192) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(f4_channels, f4_channels, 3, padding=1,
                      groups=f4_channels, bias=False),
            nn.GroupNorm(8, f4_channels),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(2 * f4_channels, 48, 1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, f8_channels, 1, bias=False),
        )
        self.neighborhood = nn.Conv2d(
            f8_channels, f8_channels, 3, padding=1,
            groups=f8_channels, bias=False,
        )
        self.gate = nn.Sequential(
            nn.Conv2d(2 * f8_channels, 48, 1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 1, 3, padding=1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.scale = BoundedResidualScale()

    def forward(self, f4: torch.Tensor, f8: torch.Tensor) -> torch.Tensor:
        if f4.shape[-2:] != tuple(2 * size for size in f8.shape[-2:]):
            raise ValueError("F4 must have twice F8's height and width")
        local = self.local(f4)
        pooled = torch.cat((F.avg_pool2d(local, 2), F.max_pool2d(local, 2)), dim=1)
        detail = self.project(pooled)
        detail = detail + 0.1 * self.neighborhood(detail)
        gate = torch.sigmoid(self.gate(torch.cat((f8, detail), dim=1)))
        return f8 + self.scale() * gate * detail


class VehicleFTM(nn.Module):
    """Keep FFNet's ODConv/channel path; add local F8 receptive-field choices."""

    def __init__(self, original_ftm: nn.Module) -> None:
        super().__init__()
        self.base = original_ftm
        channels = self.base.conv2[0].out_planes
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation,
                          dilation=dilation, groups=channels, bias=False),
                nn.GroupNorm(8, channels),
            )
            for dilation in (1, 2)
        )
        self.route = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, 3, 1),
        )
        nn.init.zeros_(self.route[-1].weight)
        with torch.no_grad():
            self.route[-1].bias.copy_(torch.tensor([0.5, 0.0, 0.0]))
        self.expert_scale = BoundedResidualScale()
        self.spatial_scale = BoundedResidualScale()

    def route_weights(self, feature: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.route(feature), dim=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        base = self.base
        x = base.ch_att_s(feature) * feature
        x = base.conv1(x) + x
        x = base.conv2(x)
        x = base.conv3(x) + x
        weights = self.route_weights(x)
        correction = (weights[:, 1:2] * self.experts[0](x)
                      + weights[:, 2:3] * self.experts[1](x))
        x = x + self.expert_scale() * correction
        x = base.conv4(x)
        attention = base.sa_s(x)
        # A small floor preserves the original FTM's attention scale while
        # preventing near-zero attention from fully erasing weak evidence.
        return x * (attention + self.spatial_scale() * (1 - attention))


class ScaleGateFusion(nn.Module):
    """Spatial scale weights with exactly the original concat at initialization.

    For a uniform softmax each multiplier is one.  Unlike a zero-initialized
    output projection, the route's final convolution receives a gradient on
    the first backward pass whenever the density head is active.
    """

    def __init__(self, original_fusion: nn.Module) -> None:
        super().__init__()
        self.upsample_1 = original_fusion.upsample_1
        self.upsample_2 = original_fusion.upsample_2
        self.final = original_fusion.final
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, 16, 1, bias=False),
                          nn.GroupNorm(4, 16), nn.GELU())
            for channels in (16, 32, 64)
        )
        self.route = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 3, 1),
        )
        nn.init.zeros_(self.route[-1].weight)
        nn.init.zeros_(self.route[-1].bias)
        self.scale = BoundedResidualScale()

    def aligned_features(
        self, f8: torch.Tensor, f16: torch.Tensor, f32: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f16 = self.upsample_1(f16)
        f32 = self.upsample_2(f32)
        if f8.shape[-2:] != f16.shape[-2:] or f8.shape[-2:] != f32.shape[-2:]:
            raise ValueError("FFNet features do not align at stride 8")
        return f8, f16, f32

    def route_weights(
        self, aligned: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        projected = [projection(feature) for projection, feature in
                     zip(self.projections, aligned)]
        return torch.softmax(self.route(torch.cat(projected, dim=1)), dim=1)

    def forward(self, f8: torch.Tensor, f16: torch.Tensor, f32: torch.Tensor) -> torch.Tensor:
        aligned = self.aligned_features(f8, f16, f32)
        weights = self.route_weights(aligned)
        scale = self.scale()
        adjusted = [feature * (1 + scale * (3 * weights[:, i:i + 1] - 1))
                    for i, feature in enumerate(aligned)]
        return self.final(torch.cat(adjusted, dim=1))


VEHICLE_VARIANTS = {
    "baseline": (False, False, False),
    "a1_lde": (True, False, False),
    "a2_vftm": (False, True, False),
    "a3_lde_vftm": (True, True, False),
    "a4_lde_vftm_gate": (True, True, True),
}


class VehicleFFNet(nn.Module):
    """Expose ConvNeXt F4 while retaining FFNet's D8 density semantics."""

    def __init__(self, original: nn.Module, variant: str) -> None:
        super().__init__()
        if variant not in VEHICLE_VARIANTS or variant == "baseline":
            raise ValueError(f"unsupported vehicle variant: {variant}")
        use_lde, use_vftm, use_gate = VEHICLE_VARIANTS[variant]
        self.variant = variant
        self.backbone = original.backbone
        self.lde = LocalDetailEnhancement() if use_lde else None
        self.ccsm1 = VehicleFTM(original.ccsm1) if use_vftm else original.ccsm1
        self.ccsm2 = original.ccsm2
        self.ccsm3 = original.ccsm3
        self.fusion = ScaleGateFusion(original.fusion) if use_gate else original.fusion

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f4 = self.backbone.stem(image.float())
        f8 = self.backbone.stage1(f4)
        f16 = self.backbone.stage2(f8)
        f32 = self.backbone.stage3(f16)
        if self.lde is not None:
            f8 = self.lde(f4, f8)
        t8 = self.ccsm1(f8)
        t16 = self.ccsm2(f16)
        t32 = self.ccsm3(f32)
        density = self.fusion(t8, t16, t32)
        normalized = density / (density.flatten(1).sum(1)[:, None, None, None] + 1e-6)
        return density, normalized
