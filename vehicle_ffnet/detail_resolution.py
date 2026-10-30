"""Late F4 detail controls; all three original FTM blocks stay unchanged."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn


DETAIL_VARIANTS = ("b1_detail", "b2_self_gate", "b3_semantic_gate", "d4_detail")


class DetailScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(math.log(1 / 3)))

    def forward(self):
        return .2 * self.logit.sigmoid()  # initially .05


def detail_extractor():
    return nn.Sequential(
        nn.Conv2d(96, 32, 1, bias=False),
        nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
        nn.GroupNorm(8, 32), nn.GELU(),
    )


def gate_predictor():
    layer = nn.Sequential(nn.Conv2d(64, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 3, padding=1))
    nn.init.zeros_(layer[-1].weight)
    nn.init.zeros_(layer[-1].bias)
    return layer


class LateDetail(nn.Module):
    """Add detail after T8, preserving the original multiscale semantic path.

    Gates are latent feature weights, not supervised visibility masks.
    A .1 floor allows weak local evidence to receive gradients.
    """
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.local = detail_extractor()
        self.down = nn.Conv2d(64, 16, 1, bias=False)
        self.scale = DetailScale()
        # Optional gate parameters must not perturb shared initialization or
        # the data-loader RNG. B1/B2/B3 start with identical effective masks.
        with torch.random.fork_rng(devices=[]):
            if mode != "b1_detail":
                self.guide = nn.Conv2d(32 if mode == "b2_self_gate" else 16, 32, 1)
                self.gate = gate_predictor()

    def forward(self, f4, t8):
        local = self.local(f4)
        if local.shape[-2:] != tuple(2 * x for x in t8.shape[-2:]):
            raise ValueError("late detail requires aligned F4/T8")
        if self.mode == "b1_detail":
            mask = local.new_full((local.shape[0], 1, *local.shape[-2:]), .55)
        else:
            source = local if self.mode == "b2_self_gate" else t8
            guide = self.guide(source)
            if guide.shape[-2:] != local.shape[-2:]:
                guide = F.interpolate(guide, size=local.shape[-2:], mode="bilinear", align_corners=False)
            mask = .1 + .9 * self.gate(torch.cat((local, guide), 1)).sigmoid()
        evidence = local * mask
        evidence = self.down(torch.cat((F.avg_pool2d(evidence, 2), F.max_pool2d(evidence, 2)), 1))
        return t8 + self.scale() * evidence


class D4Fusion(nn.Module):
    """Upsample semantic FEATURES, then add gated F4 detail and predict D4.

    No D8 density is predicted or supervised. No /4 applies to features.
    The output has units of vehicle mass per stride-4 cell.
    """
    def __init__(self, original):
        super().__init__()
        self.upsample_1 = original.upsample_1
        self.upsample_2 = original.upsample_2
        self.semantic = nn.Sequential(nn.Conv2d(112, 32, 1, bias=False), nn.GroupNorm(8, 32), nn.GELU())
        self.local = detail_extractor()
        self.gate = gate_predictor()
        self.scale = DetailScale()
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(), nn.Conv2d(32, 32, 1),
        )
        self.final = nn.Sequential(nn.Conv2d(32, 1, 1), nn.ReLU())
        # Fresh ReLU head: positive bias and small weights keep it live.
        # Training refines this bias from training-set counts only.
        nn.init.normal_(self.final[0].weight, std=1e-5)
        nn.init.constant_(self.final[0].bias, .001)

    def forward(self, t8, t16, t32, f4):
        semantic = self.semantic(torch.cat((t8, self.upsample_1(t16), self.upsample_2(t32)), 1))
        semantic = F.interpolate(semantic, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        local = self.local(f4)
        mask = .1 + .9 * self.gate(torch.cat((local, semantic), 1)).sigmoid()
        feature = semantic + self.scale() * mask * local
        feature = feature + .1 * self.refine(feature)
        return self.final(feature)


class DetailResolutionFFNet(nn.Module):
    def __init__(self, original, variant):
        super().__init__()
        self.variant = variant
        self.output_stride = 4 if variant == "d4_detail" else 8
        self.backbone = original.backbone
        self.ccsm1, self.ccsm2, self.ccsm3 = original.ccsm1, original.ccsm2, original.ccsm3
        # Additions do not change official FTM/head initial weights or RNG.
        with torch.random.fork_rng(devices=[]):
            self.late_detail = LateDetail(variant) if self.output_stride == 8 else None
            self.fusion = D4Fusion(original.fusion) if self.output_stride == 4 else original.fusion

    def forward(self, image):
        f4 = self.backbone.stem(image.float())
        f8 = self.backbone.stage1(f4)
        f16 = self.backbone.stage2(f8)
        f32 = self.backbone.stage3(f16)
        t8, t16, t32 = self.ccsm1(f8), self.ccsm2(f16), self.ccsm3(f32)
        if self.late_detail is not None:
            density = self.fusion(self.late_detail(f4, t8), t16, t32)
        else:
            density = self.fusion(t8, t16, t32, f4)
        normalized = density / (density.flatten(1).sum(1)[:, None, None, None] + 1e-6)
        return density, normalized
