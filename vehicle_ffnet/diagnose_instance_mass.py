"""Diagnose missing vehicle mass in full-image FFNet predictions.

Each D8 cell is assigned to its nearest annotated center, as in PML.
The optional 24 px radius is a *diagnostic* neighborhood, not a GT mask:
mass outside it is not automatically a false positive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F

from .data import VehicleDataset, box_centers
from .model import build_model
from .prepare import load_boxes


INSTANCE_FIELDS = (
    "id", "instance_index", "x", "y", "box_width", "box_height",
    "edge_distance_px", "nearest_neighbor_px", "assigned_mass",
    "local_mass", "assigned_far_mass",
)
IMAGE_FIELDS = (
    "id", "gt", "prediction", "error", "assigned_mass_sum", "local_mass_sum",
    "far_mass", "median_assigned_mass", "mean_assigned_mass",
    "assigned_lt_0p5", "assigned_lt_0p8", "local_lt_0p5",
)


def predict_valid_d8(model, image: torch.Tensor) -> torch.Tensor:
    """Return the valid D8 output used by full-image evaluation before upsampling."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("expected one normalized image [1,3,H,W]")
    height, width = image.shape[-2:]
    stride = getattr(model, "output_stride", 8)
    padded_height = (height + 31) // 32 * 32
    padded_width = (width + 31) // 32 * 32
    padded = F.pad(
        image, (0, padded_width - width, 0, padded_height - height),
        mode="replicate",
    )
    prediction, _ = model(padded)
    if prediction.shape != (1, 1, padded_height // stride, padded_width // stride):
        raise ValueError(f"unexpected FFNet output shape: {tuple(prediction.shape)}")
    return prediction[0, 0, :(height + stride - 1) // stride, :(width + stride - 1) // stride]


def assign_mass(
    density: torch.Tensor, points: torch.Tensor, radius_px: float = 24.0,
    stride: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return nearest-point mass, local mass, and mass outside local neighborhoods.

    D8 cell coordinates exactly match the PML loss: (index + 7/16)*8.
    Each cell has one owner, so sum(assigned) == sum(density).
    """
    if density.ndim != 2 or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("expected density [H,W] and original-image centers [K,2]")
    if radius_px <= 0 or stride <= 0:
        raise ValueError("radius and stride must be positive")
    if not bool(torch.isfinite(density).all()) or not bool(torch.isfinite(points).all()):
        raise ValueError("non-finite density or centers")
    flat = density.reshape(-1)
    if len(points) == 0:
        empty = flat.new_zeros((0,))
        return empty, empty, flat.sum()
    rows, cols = density.shape
    cell_offset = (stride - 1) / (2 * stride)
    ys = torch.arange(rows, device=density.device, dtype=torch.float32) + cell_offset
    xs = torch.arange(cols, device=density.device, dtype=torch.float32) + cell_offset
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid_xy = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
    center_xy = points.to(device=density.device, dtype=torch.float32) / stride
    distances = torch.cdist(grid_xy, center_xy)
    nearest, owner = distances.min(dim=1)
    assigned = flat.new_zeros((len(points),)).scatter_add_(0, owner, flat)
    local = flat.new_zeros((len(points),)).scatter_add_(
        0, owner, flat * (nearest * stride <= radius_px).to(flat.dtype)
    )
    far_mass = flat.sum() - local.sum()
    return assigned, local, far_mass


def summarize_bucket(image_rows: list[dict], instance_rows: list[dict]) -> dict:
    if not image_rows:
        return {"images": 0, "vehicles": 0}
    errors = [row["error"] for row in image_rows]
    assigned = [row["assigned_mass"] for row in instance_rows]
    local = [row["local_mass"] for row in instance_rows]
    return {
        "images": len(image_rows),
        "vehicles": len(instance_rows),
        "mae": sum(abs(e) for e in errors) / len(errors),
        "bias": sum(errors) / len(errors),
        "mean_gt": sum(row["gt"] for row in image_rows) / len(image_rows),
        "mean_prediction": sum(row["prediction"] for row in image_rows) / len(image_rows),
        "median_assigned_mass": median(assigned) if assigned else None,
        "mean_assigned_mass": sum(assigned) / len(assigned) if assigned else None,
        "assigned_lt_0p5_fraction": sum(x < 0.5 for x in assigned) / len(assigned) if assigned else None,
        "assigned_lt_0p8_fraction": sum(x < 0.8 for x in assigned) / len(assigned) if assigned else None,
        "local_lt_0p5_fraction": sum(x < 0.5 for x in local) / len(local) if local else None,
        "mean_far_mass": sum(row["far_mass"] for row in image_rows) / len(image_rows),
    }


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only-ids", nargs="*", default=[])
    parser.add_argument("--radius-px", type=float, default=24.0)
    args = parser.parse_args()
    if args.radius_px <= 0:
        parser.error("--radius-px must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    manifest = Path(config["manifest_dir"]) / f"{args.split}.json"
    dataset = VehicleDataset(data_root, manifest, config["crop_size"], train=False)
    selected = set(args.only_ids)
    known = {record["id"] for record in dataset.records}
    if selected - known:
        raise ValueError(f"IDs not in {args.split} manifest: {sorted(selected - known)}")

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

    image_rows: list[dict] = []
    instance_rows: list[dict] = []
    with torch.inference_mode():
        for index, record in enumerate(dataset.records):
            if selected and record["id"] not in selected:
                continue
            image, count, image_id = dataset[index]
            boxes = load_boxes(data_root / record["annotation"])
            points = box_centers(boxes)
            if len(points) != count or image_id != record["id"]:
                raise ValueError(f"manifest and annotations disagree for {image_id}")
            density = predict_valid_d8(model, image.unsqueeze(0).to(device))
            assigned, local, far = assign_mass(density, points, args.radius_px, stride=model.output_stride)
            masses = assigned.cpu().tolist()
            local_masses = local.cpu().tolist()
            prediction = float(density.sum().item())
            assigned_sum = sum(masses)
            if not math.isclose(prediction, assigned_sum, rel_tol=1e-5, abs_tol=1e-3):
                raise RuntimeError(f"nearest-point mass is not conserved for {image_id}")
            image_rows.append({
                "id": image_id, "gt": int(count), "prediction": prediction,
                "error": prediction - count, "assigned_mass_sum": assigned_sum,
                "local_mass_sum": sum(local_masses), "far_mass": float(far.item()),
                "median_assigned_mass": median(masses) if masses else None,
                "mean_assigned_mass": assigned_sum / len(masses) if masses else None,
                "assigned_lt_0p5": sum(x < 0.5 for x in masses),
                "assigned_lt_0p8": sum(x < 0.8 for x in masses),
                "local_lt_0p5": sum(x < 0.5 for x in local_masses),
            })
            neighbor = torch.cdist(points, points) if len(points) > 1 else None
            if neighbor is not None:
                neighbor.fill_diagonal_(float("inf"))
                nearest_neighbor = neighbor.min(dim=1).values.tolist()
            else:
                nearest_neighbor = [float("nan")] * len(points)
            for object_index, ((x1, y1, x2, y2), (x, y)) in enumerate(zip(boxes, points.tolist())):
                instance_rows.append({
                    "id": image_id, "instance_index": object_index,
                    "x": x, "y": y, "box_width": x2 - x1, "box_height": y2 - y1,
                    "edge_distance_px": min(x, y, record["width"] - x, record["height"] - y),
                    "nearest_neighbor_px": nearest_neighbor[object_index],
                    "assigned_mass": masses[object_index],
                    "local_mass": local_masses[object_index],
                    "assigned_far_mass": masses[object_index] - local_masses[object_index],
                })
            if len(image_rows) % 50 == 0:
                print(f"{args.split}: {len(image_rows)} images", flush=True)

    if not image_rows:
        raise ValueError("no images selected")
    buckets = {"all": (0, float("inf")), "gt_lt_50": (0, 50),
               "gt_50_99": (50, 100), "gt_100_149": (100, 150),
               "gt_ge_150": (150, float("inf"))}
    by_id = {row["id"]: row["gt"] for row in image_rows}
    summary = {
        "config": str(args.config), "checkpoint": str(args.checkpoint),
        "split": args.split, "radius_px": args.radius_px,
        "notes": "Nearest-point mass conserves total D8 prediction. Local mass is within the diagnostic radius; far mass is not automatically a false positive. Neither metric labels occlusion.",
        "buckets": {
            name: summarize_bucket(
                [r for r in image_rows if lo <= r["gt"] < hi],
                [r for r in instance_rows if lo <= by_id[r["id"]] < hi],
            ) for name, (lo, hi) in buckets.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_image.csv", IMAGE_FIELDS, image_rows)
    write_csv(args.output_dir / "per_instance.csv", INSTANCE_FIELDS, instance_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, bucket in summary["buckets"].items():
        if bucket["images"]:
            print(f"{name}: n={bucket['images']} MAE={bucket['mae']:.3f} "
                  f"bias={bucket['bias']:+.3f} median_mass={bucket['median_assigned_mass']:.3f} "
                  f"under0.5={bucket['assigned_lt_0p5_fraction']:.1%}")
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
