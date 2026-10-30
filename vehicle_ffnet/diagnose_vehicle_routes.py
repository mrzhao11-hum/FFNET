"""Inspect spatial variation of point-trained FFNet vehicle modules.

This uses only image/count/point manifests.  Route maps are diagnostic latent
features; their values are not occlusion or vehicle-size labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import VehicleDataset
from .model import build_model


def predict_valid_d8(model, image: torch.Tensor) -> torch.Tensor:
    height, width = image.shape[-2:]
    stride = getattr(model, "output_stride", 8)
    padded_height = (height + 31) // 32 * 32
    padded_width = (width + 31) // 32 * 32
    padded = F.pad(image, (0, padded_width - width, 0, padded_height - height),
                   mode="replicate")
    density, _ = model(padded)
    expected = (padded_height // stride, padded_width // stride)
    if density.shape[-2:] != expected:
        raise ValueError(f"unexpected D8 output: {tuple(density.shape)}")
    return density[0, 0, :(height + stride - 1) // stride, :(width + stride - 1) // stride]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only-ids", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = build_model(
        pretrained=False, attention_norm=config.get("attention_norm", "batch"),
        fusion_mode=config.get("fusion_mode", "original"),
        vehicle_variant=config.get("vehicle_variant", "baseline"),
        density_activation=config.get("density_activation", "relu"),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()

    logits: dict[str, torch.Tensor] = {}
    handles = []
    def capture(name: str):
        def hook(_module, _inputs, output):
            logits[name] = output.detach()
        return hook

    if getattr(model, "lde", None) is not None:
        handles.append(model.lde.gate[-1].register_forward_hook(capture("f4_gate")))
    if hasattr(model.ccsm1, "route"):
        handles.append(model.ccsm1.route[-1].register_forward_hook(capture("f8_experts")))
    if hasattr(model.fusion, "route"):
        handles.append(model.fusion.route[-1].register_forward_hook(capture("scale_route")))
    late = getattr(model, "late_detail", None)
    if late is not None and hasattr(late, "gate"):
        handles.append(late.gate[-1].register_forward_hook(capture("late_f4_gate")))
    if model.output_stride == 4:
        handles.append(model.fusion.gate[-1].register_forward_hook(capture("late_f4_gate")))
    if getattr(model.fusion, "mode", "") == "c3_semantic":
        handles.append(model.fusion.gate[-1].register_forward_hook(capture("fusion_context_gate")))
    if not handles:
        parser.error("checkpoint architecture has no vehicle routes")

    dataset = VehicleDataset(
        config["data_root"], Path(config["manifest_dir"]) / f"{args.split}.json",
        crop_size=config["crop_size"], train=False,
    )
    selected = set(args.only_ids)
    known = {record["id"] for record in dataset.records}
    if selected - known:
        parser.error(f"IDs outside {args.split}: {sorted(selected - known)}")
    totals: dict[str, dict[str, list[float]]] = {}
    count_errors = []
    try:
        with torch.inference_mode():
            for index, record in enumerate(dataset.records):
                if selected and record["id"] not in selected:
                    continue
                image, count, _ = dataset[index]
                logits.clear()
                density = predict_valid_d8(model, image.unsqueeze(0).to(device))
                count_errors.append(float(density.sum().item()) - float(count))
                for name, raw in logits.items():
                    stride = 4 if name == "late_f4_gate" else 8
                    h,w = image.shape[-2:]
                    valid = raw[:, :, :(h+stride-1)//stride, :(w+stride-1)//stride]
                    if name in {"late_f4_gate", "fusion_context_gate"}:
                        weight = .1 + .9 * torch.sigmoid(valid)
                    else:
                        weight = (torch.sigmoid(valid) if name == "f4_gate"
                                  else torch.softmax(valid, dim=1))
                    means = weight.flatten(2).mean(dim=2)[0].cpu().tolist()
                    stds = weight.flatten(2).std(dim=2, unbiased=False)[0].cpu().tolist()
                    bucket = totals.setdefault(name, {"mean": [], "spatial_std": []})
                    bucket["mean"].append(means)
                    bucket["spatial_std"].append(stds)
    finally:
        for handle in handles:
            handle.remove()

    if not count_errors:
        raise ValueError("selected no images")
    report = {
        "split": args.split,
        "vehicle_variant": config.get("vehicle_variant", "baseline"),
        "images": len(count_errors),
        "mae": sum(abs(error) for error in count_errors) / len(count_errors),
        "bias": sum(count_errors) / len(count_errors),
        "routes": {},
    }
    for name, bucket in totals.items():
        report["routes"][name] = {
            key: [sum(image[channel] for image in values) / len(values)
                  for channel in range(len(values[0]))]
            for key, values in bucket.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{args.split}: n={report['images']} MAE={report['mae']:.3f} "
          f"bias={report['bias']:+.3f}")
    for name, values in report["routes"].items():
        print(f"{name}: mean={values['mean']} spatial_std={values['spatial_std']}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
