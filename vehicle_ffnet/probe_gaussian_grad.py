"""Measure Gaussian-vs-global-count gradients at FFNet's final density layer."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .data import box_centers, normalize_image
from .gaussian_targets import gaussian_density_d8
from .model import build_model
from .prepare import load_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--require-live", action="store_true",
                        help="fail if a nonempty sample predicts almost zero or both head gradients vanish")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("supervision") != "gaussian":
        parser.error("config must use Gaussian supervision")
    device = torch.device(args.device)
    model = build_model(
        pretrained=False, attention_norm=config["attention_norm"],
        fusion_mode=config.get("fusion_mode", "original"),
        vehicle_variant=config.get("vehicle_variant", "baseline"),
        density_activation=config.get("density_activation", "relu"),
    ).to(device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved.get("model_state_dict", saved), strict=True)
    model.eval()
    stride = model.output_stride
    head = model.fusion.final[0].weight
    root = Path(config["data_root"])
    records = json.loads(
        (Path(config["manifest_dir"]) / f"{args.split}.json").read_text(encoding="utf-8")
    )["samples"]
    ratios = []
    failed = []
    print(f"activation={type(model.fusion.final[1]).__name__} "
          f"head_bias={float(model.fusion.final[0].bias.detach().mean()):.6f}")
    for record in records[:args.samples]:
        with Image.open(root / record["image"]) as source:
            image = source.convert("RGB")
        height, width = image.height, image.width
        boxes = load_boxes(root / record["annotation"])
        points = box_centers(boxes).numpy()
        if len(points) != record["count"]:
            raise ValueError(f"count mismatch for {record['id']}")
        if config.get("gaussian_target_source") == "h5":
            from .prepare_gaussian_h5 import load_gaussian_h5
            from .gaussian_targets import block_sum_density
            full_gt = load_gaussian_h5(
                Path(config["gaussian_target_root"]) / f"{record['id']}.h5", points,
                height, width, config.get("gaussian_sigma", 8.0),
                config.get("gaussian_truncate", 3.0))
            target = block_sum_density(full_gt, stride)
        else:
            target = gaussian_density_d8(points, height, width,
                                         sigma=config.get("gaussian_sigma", 8.0), stride=stride)
        gt = torch.as_tensor(target, device=device)[None, None]
        x = normalize_image(image).unsqueeze(0).to(device)
        padded_height = (height + 31) // 32 * 32
        padded_width = (width + 31) // 32 * 32
        x = F.pad(x, (0, padded_width - width, 0, padded_height - height), mode="replicate")
        prediction, _ = model(x)
        prediction = prediction[:, :, :gt.shape[-2], :gt.shape[-1]]
        density_loss = config.get("gaussian_density_weight", 1.0) * (prediction - gt).square().sum()
        count_loss = config.get("gaussian_count_weight", 0.002) * (
            prediction.sum() - len(points)
        ).abs()
        density_grad = torch.autograd.grad(density_loss, head, retain_graph=True)[0]
        count_grad = torch.autograd.grad(count_loss, head)[0]
        if not (torch.isfinite(density_grad).all() and torch.isfinite(count_grad).all()):
            raise FloatingPointError("nonfinite density/count gradient")
        density_norm = float(density_grad.norm().item())
        count_norm = float(count_grad.norm().item())
        ratio = count_norm / max(density_norm, 1e-12)
        ratios.append(ratio)
        predicted_count = float(prediction.detach().sum())
        active_fraction = float((prediction.detach() > 0).float().mean())
        zero_sse = float(gt.square().sum())
        if len(points) and (predicted_count < 0.01 or density_norm + count_norm == 0):
            failed.append(record['id'])
        print(f"{record['id']} GT={len(points)} prediction={float(prediction.sum()):.2f} "
              f"density={float(density_loss):.4f} weighted_count={float(count_loss):.4f} "
              f"head_grad_count/density={ratio:.3f} "
              f"active_fraction={active_fraction:.6f} zero_prediction_SSE={zero_sse:.4f} "
              f"density_grad_norm={density_norm:.6g} count_grad_norm={count_norm:.6g}")
    print(f"median head gradient ratio: {statistics.median(ratios):.3f}")
    if args.require_live and failed:
        raise RuntimeError(f"near-zero output or dead head gradients on: {failed}")


if __name__ == "__main__":
    main()
