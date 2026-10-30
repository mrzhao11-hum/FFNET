"""One full-image PML forward/backward on the densest training image."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .data import VehicleDataset
from .model import build_model
from .pml_loss import ProximalMappingLoss
from .train import compute_loss, set_seed
from .vehicle_modules import ScaleGateFusion, VehicleFTM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("supervision") != "pml" or config.get("train_mode") != "full":
        raise ValueError("probe requires a full-image PML training config")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the PML memory probe requires a visible CUDA device")

    train_manifest = Path(config["manifest_dir"]) / "train.json"
    records = json.loads(train_manifest.read_text(encoding="utf-8"))["samples"]
    index = max(range(len(records)), key=lambda i: records[i]["count"])
    record = records[index]
    dataset = VehicleDataset(
        config["data_root"], train_manifest, crop_size=config["crop_size"],
        train=True, flip_probability=0.0, full_image=True,
    )
    image, points, target, valid_shape = dataset[index]
    set_seed(config["seed"])
    model = build_model(
        pretrained=config["pretrained"], attention_norm=config["attention_norm"],
        fusion_mode=config.get("fusion_mode", "original"),
        vehicle_variant=config.get("vehicle_variant", "baseline"),
    ).to(device)
    model.train()
    criterion = ProximalMappingLoss(
        stride=8, radius_px=config["pml_radius"],
        density_factor=config["pml_density_factor"],
    )
    images = image.unsqueeze(0).to(device)
    point_list = [points.to(device)]
    target = target.unsqueeze(0).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    total, terms, predicted, _ = compute_loss(
        model, criterion, images, point_list, target, config["wot"], config["wtv"],
        valid_shape=valid_shape, supervision="pml",
        pml_count_weight=config["pml_count_weight"],
    )
    if not torch.isfinite(total):
        raise FloatingPointError("PML loss is nonfinite on the densest training image")
    total.backward()
    gradient_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("nonfinite model gradient in PML probe")
        gradient_norm += float(parameter.grad.abs().sum().item())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError("PML probe produced no usable model gradient")
    module_gradients = {}
    if getattr(model, "lde", None) is not None:
        module_gradients["f4_detail"] = model.lde.project[0].weight.grad
        module_gradients["f4_gate"] = model.lde.gate[-1].weight.grad
    if isinstance(model.ccsm1, VehicleFTM):
        module_gradients["f8_expert"] = model.ccsm1.experts[0][0].weight.grad
        module_gradients["f8_route"] = model.ccsm1.route[-1].weight.grad
    if isinstance(model.fusion, ScaleGateFusion):
        module_gradients["scale_route"] = model.fusion.route[-1].weight.grad
    measured = {}
    for name, gradient in module_gradients.items():
        norm = 0.0 if gradient is None else float(gradient.abs().sum().item())
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"PML probe found no usable gradient in {name}")
        measured[name] = norm
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    cells = valid_shape[0] * valid_shape[1]
    print(
        f"PML probe passed: id={record['id']} GT={len(points)} "
        f"D8={valid_shape[0]}x{valid_shape[1]} distance_matrix={cells}x{len(points)} "
        f"prediction={float(predicted.item()):.3f} "
        f"loss={float(total.item()):.4f} "
        f"point={float(terms[4].item()):.4f} "
        f"transport={float(terms[5].item()):.4f} "
        f"grad_L1={gradient_norm:.3f} peak_allocated={peak_gib:.2f}GiB "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )
    if measured:
        print("new_module_grad_L1=" + ", ".join(f"{key}:{value:.4g}"
                                                   for key, value in measured.items()), flush=True)


if __name__ == "__main__":
    main()
