"""Load the official FFNet class without changing its source files."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


UPSTREAM = Path(__file__).resolve().parents[1] / "upstream" / "Fuss-Free-structure-main"


def upstream_modules():
    if not (UPSTREAM / "Networks" / "FFNet.py").is_file():
        raise FileNotFoundError(f"official FFNet source missing at {UPSTREAM}")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from Networks import FFNet as network_module
    from losses.ot_loss import OT_Loss

    return network_module, OT_Loss


def replace_pooled_attention_bn(model: nn.Module) -> int:
    """Use per-image channel normalization after ODConv global pooling."""
    replaced = 0
    for name, module in model.named_modules():
        if not name.endswith(".attention"):
            continue
        old = getattr(module, "bn", None)
        if not isinstance(old, nn.BatchNorm2d):
            continue
        norm = nn.GroupNorm(1, old.num_features, eps=old.eps, affine=True)
        with torch.no_grad():
            norm.weight.copy_(old.weight)
            norm.bias.copy_(old.bias)
        module.bn = norm
        replaced += 1
    if replaced != 12:
        raise RuntimeError(f"expected 12 pooled ODConv attention BN layers, found {replaced}")
    return replaced


def replace_ccsm_bn(model: nn.Module) -> int:
    """Normalize each CCSM feature map per image, independent of running stats."""
    replaced = 0
    for block_name in ("ccsm1", "ccsm2", "ccsm3"):
        block = getattr(model, block_name)
        for stage_name in ("conv1", "conv2", "conv3", "conv4"):
            stage = getattr(block, stage_name)
            old = stage[2]
            if not isinstance(old, nn.BatchNorm2d):
                raise RuntimeError(f"expected BatchNorm2d at {block_name}.{stage_name}.2")
            if old.num_features % 8:
                raise RuntimeError(f"CCSM channel count is not divisible by 8: {old.num_features}")
            norm = nn.GroupNorm(8, old.num_features, eps=old.eps, affine=True)
            with torch.no_grad():
                norm.weight.copy_(old.weight)
                norm.bias.copy_(old.bias)
            stage[2] = norm
            replaced += 1
    if replaced != 12:
        raise RuntimeError(f"expected 12 CCSM BN layers, found {replaced}")
    return replaced


def build_model(pretrained: bool = True, attention_norm: str = "batch",
                fusion_mode: str = "original", vehicle_variant: str = "baseline"):
    from .vehicle_modules import VEHICLE_VARIANTS, VehicleFFNet

    if attention_norm not in {"batch", "group", "group_all"}:
        raise ValueError(f"unsupported attention_norm: {attention_norm}")
    if fusion_mode not in {"original", "spatial_gate"}:
        raise ValueError(f"unsupported FFNet fusion mode: {fusion_mode}")
    if vehicle_variant not in VEHICLE_VARIANTS:
        raise ValueError(f"unsupported vehicle variant: {vehicle_variant}")
    if vehicle_variant != "baseline" and fusion_mode != "original":
        raise ValueError("vehicle variants require fusion_mode=original")

    def configure(model):
        if attention_norm in {"group", "group_all"}:
            replace_pooled_attention_bn(model)
        if attention_norm == "group_all":
            replace_ccsm_bn(model)
        if fusion_mode == "spatial_gate":
            from .spatial_gate import SpatialGateFusion
            model.fusion = SpatialGateFusion(model.fusion)
        if vehicle_variant != "baseline":
            model = VehicleFFNet(model, vehicle_variant)
        return model

    module, _ = upstream_modules()
    if pretrained:
        return configure(module.FFNet())
    # Keep the official architecture but skip its hard-coded ImageNet download
    # for offline smoke tests and checkpoint evaluation.
    constructor = module.convnext_tiny

    def without_weights(*args, **kwargs):
        kwargs["weights"] = None
        return constructor(*args, **kwargs)

    module.convnext_tiny = without_weights
    try:
        return configure(module.FFNet())
    finally:
        module.convnext_tiny = constructor
