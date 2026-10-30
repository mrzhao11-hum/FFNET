"""Measure where full-image FFNet training time is spent without saving weights."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import VehicleDataset, full_image_collate
from .evaluate import evaluate_model
from .model import build_model
from .rectangular_ot import RectangularOTLoss
from .train import set_seed


STAGES = ("data_wait", "transfer", "forward", "ot", "count_tv", "backward_step", "logging", "total")


def clock(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def summarize(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    ordered = sorted(values)
    return {
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "p90_seconds": ordered[math.ceil(0.9 * len(ordered)) - 1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--val-samples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1 or args.val_samples < 0:
        parser.error("warmup must be nonnegative, steps positive, and val-samples nonnegative")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("supervision", "ot") != "ot" or config.get("output_stride", 8) != 8:
        parser.error("profile_full measures the original D8 OT path only; use probe_pml for PML memory/backward checks")
    if config.get("train_mode") != "full" or config["batch_size"] != 1:
        parser.error("this profiler requires a full-image, batch-one config")
    device = torch.device(args.device)
    set_seed(config["seed"])
    root = Path(config["data_root"])
    manifests = Path(config["manifest_dir"])
    workers = args.workers if args.workers is not None else config["workers"]
    dataset = VehicleDataset(
        root, manifests / "train.json", crop_size=config["crop_size"],
        train=True, flip_probability=0.0, full_image=True,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=workers,
        collate_fn=full_image_collate, pin_memory=device.type == "cuda",
    )
    if args.warmup + args.steps > len(loader):
        parser.error("requested more images than one training epoch contains")

    model = build_model(
        pretrained=False, attention_norm=config["attention_norm"],
        fusion_mode=config.get("fusion_mode", "original"),
        vehicle_variant=config.get("vehicle_variant", "baseline"),
        density_activation=config.get("density_activation", "relu"),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    ot_loss = RectangularOTLoss(
        8, device, config["ot_iterations"], config["ot_regularization"],
        config.get("ot_reference_size", 256.0),
    )
    sinkhorn = ot_loss.sinkhorn
    error_checks: list[int] = []

    def counted_sinkhorn(*call_args, **call_kwargs):
        result = sinkhorn(*call_args, **call_kwargs)
        error_checks.append(len(result[1].get("err", [])))
        return result

    ot_loss.sinkhorn = counted_sinkhorn
    batches = iter(loader)
    rows = []
    for step in range(args.warmup + args.steps):
        t0 = clock(device)
        images, points, discrete, valid_shape = next(batches)
        t1 = clock(device)
        images = images.to(device, non_blocking=True)
        points = [point.to(device, non_blocking=True) for point in points]
        discrete = discrete.to(device, non_blocking=True)
        t2 = clock(device)

        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(images)
        prediction = prediction[:, :, :valid_shape[0], :valid_shape[1]]
        if prediction.shape != discrete.shape:
            raise ValueError("valid prediction and target shapes differ")
        source_count = prediction.flatten(1).sum(1)
        normalized = prediction / (source_count[:, None, None, None] + 1e-6)
        t3 = clock(device)

        error_checks.clear()
        ot_raw, _, _ = ot_loss(normalized, prediction, points)
        t4 = clock(device)

        target_count = torch.tensor([len(point) for point in points], device=device, dtype=torch.float32)
        count_loss = F.l1_loss(source_count, target_count)
        target_normalized = discrete / (target_count[:, None, None, None] + 1e-6)
        tv_loss = (torch.abs(normalized - target_normalized).flatten(1).sum(1) * target_count).mean()
        total = count_loss + config["wot"] * ot_raw.squeeze() + config["wtv"] * tv_loss
        if not torch.isfinite(total):
            raise FloatingPointError("nonfinite profiling loss")
        t5 = clock(device)

        total.backward()
        optimizer.step()
        t6 = clock(device)
        # Match the training loop's scalar extraction and its synchronization cost.
        _ = tuple(float(value.detach().item()) for value in
                  (total, config["wot"] * ot_raw.squeeze(), count_loss, config["wtv"] * tv_loss))
        _ = float(torch.abs(source_count.detach() - target_count).sum().item())
        t7 = clock(device)

        if step >= args.warmup:
            rows.append({
                "points": len(points[0]),
                "density_cells": valid_shape[0] * valid_shape[1],
                "ot_matrix_elements": len(points[0]) * valid_shape[0] * valid_shape[1],
                "sinkhorn_error_checks": sum(error_checks),
                **dict(zip(STAGES, (
                    t1 - t0, t2 - t1, t3 - t2, t4 - t3,
                    t5 - t4, t6 - t5, t7 - t6, t7 - t0,
                ))),
            })

    report = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "warmup": args.warmup,
        "measured_steps": len(rows),
        "mean_points": statistics.mean(row["points"] for row in rows),
        "mean_ot_matrix_elements": statistics.mean(row["ot_matrix_elements"] for row in rows),
        "mean_sinkhorn_error_checks": statistics.mean(row["sinkhorn_error_checks"] for row in rows),
        "stages": {key: summarize(rows, key) for key in STAGES},
    }
    if args.val_samples:
        model.eval()
        t0 = clock(device)
        _, validation = evaluate_model(
            model, root, manifests / "val.json", config["crop_size"],
            config["patch_batch_size"], device, workers=workers,
            limit=args.val_samples, inference_mode="full",
        )
        elapsed = clock(device) - t0
        report["validation_timing"] = {
            "images": validation["images"],
            "total_seconds": elapsed,
            "seconds_per_image": elapsed / validation["images"],
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"measured={len(rows)} mean_points={report['mean_points']:.1f} "
          f"mean_OT_elements={report['mean_ot_matrix_elements']:.0f} "
          f"Sinkhorn_error_checks={report['mean_sinkhorn_error_checks']:.1f}")
    for key in STAGES:
        print(f"{key:14s} mean={report['stages'][key]['mean_seconds']:.4f}s "
              f"p90={report['stages'][key]['p90_seconds']:.4f}s")
    if args.val_samples:
        print(f"validation={report['validation_timing']['seconds_per_image']:.4f}s/image")
    if args.output:
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
