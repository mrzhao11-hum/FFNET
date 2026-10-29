"""Compare FFNet patch predictions with CARPK box-center counts on train or val."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch

from .data import VehicleDataset, box_centers
from .model import build_model
from .prepare import load_boxes


def windows(height: int, width: int, size: int):
    """Use exactly the same window coordinates as overlapping-patch inference."""
    for i in range(0, height, size):
        top = max(min(height - size, i), 0)
        for j in range(0, width, size):
            left = max(min(width - size, j), 0)
            yield top, left


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"patches": 0}
    n = len(rows)
    return {
        "patches": n,
        "mean_gt": sum(row["gt"] for row in rows) / n,
        "mean_pred": sum(row["prediction"] for row in rows) / n,
        "bias": sum(row["error"] for row in rows) / n,
        "mae": sum(abs(row["error"]) for row in rows) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample", type=int, default=30,
                        help="randomly selected images; 0 means the full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.sample < 0:
        raise ValueError("--sample must be nonnegative")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["data_root"])
    manifest = Path(config["manifest_dir"]) / f"{args.split}.json"
    dataset = VehicleDataset(root, manifest, crop_size=config["crop_size"], train=False)
    device = torch.device(args.device)
    model = build_model(
        pretrained=False, attention_norm=config.get("attention_norm", "batch"),
        fusion_mode=config.get("fusion_mode", "original"),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()

    indices = list(range(len(dataset)))
    if args.sample:
        indices = sorted(random.Random(args.seed).sample(indices, min(args.sample, len(indices))))
    rows = []
    with torch.inference_mode():
        for index in indices:
            image, _, image_id = dataset[index]
            record = dataset.records[index]
            points = box_centers(load_boxes(root / record["annotation"]))
            _, height, width = image.shape
            image = image.unsqueeze(0).to(device)
            for top, left in windows(height, width, config["crop_size"]):
                patch = image[:, :, top : top + config["crop_size"],
                              left : left + config["crop_size"]]
                prediction = float(model(patch)[0].sum().item())
                inside = (
                    (points[:, 0] >= left) & (points[:, 0] < left + config["crop_size"])
                    & (points[:, 1] >= top) & (points[:, 1] < top + config["crop_size"])
                )
                gt = int(inside.sum().item())
                rows.append({
                    "id": image_id, "top": top, "left": left,
                    "gt": gt, "prediction": prediction, "error": prediction - gt,
                })

    summary = {
        "split": args.split,
        "images": len(indices),
        "crop_size": config["crop_size"],
        "attention_norm": config.get("attention_norm", "batch"),
        "all": summarize(rows),
        "empty": summarize([row for row in rows if row["gt"] == 0]),
        "one_to_five": summarize([row for row in rows if 1 <= row["gt"] <= 5]),
        "six_to_nine": summarize([row for row in rows if 6 <= row["gt"] <= 9]),
        "ten_plus": summarize([row for row in rows if row["gt"] >= 10]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_patch.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "top", "left", "gt", "prediction", "error"))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for group in ("all", "empty", "one_to_five", "six_to_nine", "ten_plus"):
        stats = summary[group]
        print(f"{args.split} {group}: n={stats['patches']} "
              f"GT={stats.get('mean_gt', 0):.2f} pred={stats.get('mean_pred', 0):.2f} "
              f"bias={stats.get('bias', 0):+.2f} MAE={stats.get('mae', 0):.2f}")
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
