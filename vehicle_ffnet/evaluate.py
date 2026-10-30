"""Evaluate a saved FFNet checkpoint on fixed vehicle manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import VehicleDataset
from .inference import predict_density, predict_density_full
from .model import build_model


@contextmanager
def use_current_image_bn_statistics(model):
    """Diagnostic: use batch statistics without changing saved BN buffers."""
    saved = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            saved.append((module, module.training, module.track_running_stats))
            module.track_running_stats = False
            module.train()
    try:
        yield len(saved)
    finally:
        for module, training, tracking in saved:
            module.track_running_stats = tracking
            module.train(training)


def evaluate_model(
    model,
    data_root: Path,
    manifest: Path,
    crop_size: int,
    patch_batch_size: int,
    device: torch.device,
    workers: int = 0,
    limit: int | None = None,
    inference_mode: str = "tiles",
    bn_batch_stats: bool = False,
) -> tuple[list[dict], dict]:
    if inference_mode not in {"tiles", "full"}:
        raise ValueError(f"unsupported inference mode: {inference_mode}")
    dataset = VehicleDataset(data_root, manifest, crop_size, train=False, limit=limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)
    model.eval()
    rows = []
    context = use_current_image_bn_statistics(model) if bn_batch_stats else nullcontext()
    with context:
        for image, count, image_id in loader:
            image = image.to(device)
            density = (
                predict_density_full(model, image)
                if inference_mode == "full"
                else predict_density(model, image, crop_size, patch_batch_size)
            )
            prediction = float(density.sum().item())
            target = float(count.item())
            rows.append({"id": image_id[0], "gt": target, "prediction": prediction, "error": prediction - target})
    errors = [row["error"] for row in rows]
    summary = {
        "images": len(rows),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "bias": sum(errors) / len(errors),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patch-batch-size", type=int)
    parser.add_argument("--inference-mode", choices=("tiles", "full"), default="tiles")
    parser.add_argument("--bn-batch-stats", action="store_true",
                        help="diagnostic: use current-image BN statistics without updating buffers")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = args.data_root or Path(config["data_root"])
    manifest_dir = args.manifest_dir or Path(config["manifest_dir"])
    device = torch.device(args.device)
    model = build_model(
        pretrained=False, attention_norm=config.get("attention_norm", "batch"),
        fusion_mode=config.get("fusion_mode", "original"),
        vehicle_variant=config.get("vehicle_variant", "baseline"),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    rows, summary = evaluate_model(
        model, data_root, manifest_dir / f"{args.split}.json", config["crop_size"],
        args.patch_batch_size or config["patch_batch_size"], device,
        workers=args.workers, limit=args.limit, inference_mode=args.inference_mode,
        bn_batch_stats=args.bn_batch_stats,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "gt", "prediction", "error"))
        writer.writeheader()
        writer.writerows(rows)
    summary.update({"dataset": config["dataset"], "split": args.split,
                    "attention_norm": config.get("attention_norm", "batch"),
                    "vehicle_variant": config.get("vehicle_variant", "baseline"),
                    "inference_mode": args.inference_mode,
                    "bn_batch_stats": args.bn_batch_stats,
                    "checkpoint": str(args.checkpoint)})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{config['dataset']} {args.split}: n={summary['images']} MAE={summary['mae']:.3f} "
          f"RMSE={summary['rmse']:.3f} bias={summary['bias']:+.3f}")


if __name__ == "__main__":
    main()
