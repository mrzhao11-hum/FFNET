"""Controlled FFNet fusion-only experiments; original FTM/head stay intact.

C1: nonlinear channel residual after concat.
C2: same residual with a depthwise spatial operator before normalization.
C3: nonlinear high-level context, conditioned on T8, before concat.
"""
from __future__ import annotations

import torch
from torch import nn

FUSION_MODES = ("c1_channel", "c2_spatial", "c3_semantic")


class ResidualScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(()))

    def forward(self):
        return .2 * self.logit.sigmoid()  # initially .1, never identically zero


class CollaborativeFusion(nn.Module):
    def __init__(self, original, mode):
        super().__init__()
        if mode not in FUSION_MODES:
            raise ValueError(f"unsupported collaborative fusion: {mode}")
        self.mode = mode
        self.upsample_1 = original.upsample_1
        self.upsample_2 = original.upsample_2
        self.final = original.final
        # Preserve all original parameter values and the sampling RNG state.
        with torch.random.fork_rng(devices=[]):
            self.scale = ResidualScale()
            if mode in ("c1_channel", "c2_spatial"):
                self.input_projection = nn.Conv2d(112, 56, 1, bias=False)
                self.norm = nn.GroupNorm(8, 56)
                self.activation = nn.GELU()
                self.output_projection = nn.Conv2d(56, 112, 1, bias=False)
                if mode == "c2_spatial":
                    self.spatial = nn.Conv2d(56, 56, 3, padding=1, groups=56, bias=False)
                    nn.init.dirac_(self.spatial.weight, groups=56)
                else:
                    self.spatial = nn.Identity()
            else:
                self.context = nn.Sequential(
                    nn.Conv2d(96, 16, 1, bias=False), nn.GroupNorm(4, 16), nn.GELU(),
                )
                self.output_projection = nn.Conv2d(16, 16, 1, bias=False)
                self.gate = nn.Sequential(
                    nn.Conv2d(32, 16, 1, bias=False), nn.GroupNorm(4, 16),
                    nn.GELU(), nn.Conv2d(16, 1, 1),
                )
                nn.init.normal_(self.gate[-1].weight, std=.01)
                nn.init.zeros_(self.gate[-1].bias)
            # Exact original concat at initialization. Upstream residual/gate
            # gradients start after this projection takes its first update.
            nn.init.zeros_(self.output_projection.weight)

    def aligned_features(self, t8, t16, t32):
        aligned = (t8, self.upsample_1(t16), self.upsample_2(t32))
        if any(x.shape[-2:] != t8.shape[-2:] for x in aligned):
            raise ValueError("FFNet fusion inputs must align at stride 8")
        return aligned

    def gate_weights(self, t8, context):
        return .1 + .9 * self.gate(torch.cat((t8, context), dim=1)).sigmoid()

    def fused_features(self, t8, t16, t32):
        t8, t16, t32 = self.aligned_features(t8, t16, t32)
        if self.mode == "c3_semantic":
            context = self.context(torch.cat((t16, t32), dim=1))
            correction = self.output_projection(context)
            t8 = t8 + self.scale() * self.gate_weights(t8, context) * correction
            return torch.cat((t8, t16, t32), dim=1)
        concat = torch.cat((t8, t16, t32), dim=1)
        hidden = self.spatial(self.input_projection(concat))
        correction = self.output_projection(self.activation(self.norm(hidden)))
        return concat + self.scale() * correction

    def forward(self, t8, t16, t32):
        return self.final(self.fused_features(t8, t16, t32))

    def probe_parameters(self):
        parameters = {"fusion_output": self.output_projection.weight}
        if self.mode == "c3_semantic":
            parameters.update(fusion_context=self.context[0].weight,
                              fusion_gate=self.gate[-1].weight)
        else:
            parameters["fusion_input"] = self.input_projection.weight
            if self.mode == "c2_spatial":
                parameters["fusion_spatial"] = self.spatial.weight
        return parameters
