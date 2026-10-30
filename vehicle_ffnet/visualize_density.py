"""Compare CARPK images, point-reference density, and FFNet predictions.

The Gaussian reference is for display only. FFNet PML training uses the
original annotation points and does not use this reference as a target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


RUNS = {
    "e0_ot": ("configs/carpk_gf2_val_group_all_full.json",
              "outputs/carpk_ffnet_gf2_val_group_all_full_seed42/best_mae.pth"),
    "e1_pml": ("configs/carpk_gf2_val_group_all_full_pml.json",
               "outputs/carpk_ffnet_gf2_pml_full_seed42/best_mae.pth"),
    "e3_gate_ot": ("configs/carpk_gf2_val_group_all_full_gate_ot.json",
                   "outputs/carpk_ffnet_gf2_gate_ot_full_seed42/best_mae.pth"),
}
DEFAULT_IDS = ("20161225_TPZ_00241", "20161225_TPZ_00415", "20161225_TPZ_00244")


def reference_density(points: np.ndarray, height: int, width: int,
                      sigma: float = 8.0) -> np.ndarray:
    """Place one image-clipped, normalized Gaussian at each annotation point."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    result = np.zeros((height, width), dtype=np.float32)
    radius = math.ceil(3 * sigma)
    for x, y in points:
        left = max(0, math.floor(float(x)) - radius)
        right = min(width, math.floor(float(x)) + radius + 1)
        top = max(0, math.floor(float(y)) - radius)
        bottom = min(height, math.floor(float(y)) + radius + 1)
        if left >= right or top >= bottom:
            raise ValueError(f"annotation point outside image: {(x, y)}")
        xs = np.arange(left, right, dtype=np.float32)[None, :]
        ys = np.arange(top, bottom, dtype=np.float32)[:, None]
        kernel = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        result[top:bottom, left:right] += kernel.astype(np.float32)
    return result


def density_colors(density: np.ndarray, vmax: float) -> np.ndarray:
    """Use one fixed color scale for every prediction of an image."""
    t = np.clip(np.asarray(density, dtype=np.float32) / max(vmax, 1e-12), 0, 1)
    positions = (0.0, 0.20, 0.45, 0.70, 1.0)
    colors = ((0, 8, 70), (0, 55, 160), (0, 175, 225),
              (250, 205, 45), (240, 45, 25))
    return np.stack(
        [np.interp(t, positions, [rgb[channel] for rgb in colors])
         for channel in range(3)], axis=-1
    ).astype(np.uint8)


def difference_colors(change: np.ndarray, limit: float) -> np.ndarray:
    """Red means added E1 mass; blue means removed E1 mass."""
    t = np.clip(np.asarray(change, dtype=np.float32) / max(limit, 1e-12), -1, 1)
    neutral = np.array((239, 239, 239), dtype=np.float32)
    positive = np.array((205, 45, 35), dtype=np.float32)
    negative = np.array((38, 92, 175), dtype=np.float32)
    weight = np.abs(t)[..., None]
    target = np.where(t[..., None] >= 0, positive, negative)
    return np.clip(neutral * (1 - weight) + target * weight, 0, 255).astype(np.uint8)


def _font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_panel(image: Image.Image, reference: np.ndarray,
               predictions: dict[str, np.ndarray], target: int,
               path: Path, panel_width: int = 640,
               prediction_scale: str = "shared") -> None:
    if panel_width < 200:
        raise ValueError("panel width must be at least 200")
    if prediction_scale not in {"shared", "individual"}:
        raise ValueError("prediction scale must be shared or individual")
    width, height = image.size
    panel_height = round(height * panel_width / width)
    title_height = 42
    note_height = 32
    figure = Image.new(
        "RGB", (3 * panel_width, 2 * (panel_height + title_height) + note_height),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(figure)
    font = _font(19)
    small = _font(15)
    values = np.stack(tuple(predictions.values()))
    vmax = max(float(np.quantile(values, 0.999)), 1e-8)
    def scale_for(name: str) -> float:
        if prediction_scale == "shared":
            return vmax
        return max(float(np.quantile(predictions[name], 0.999)), 1e-8)

    reference_max = max(float(np.quantile(reference, 0.999)), 1e-8)
    change = predictions["e1_pml"] - predictions["e0_ot"]
    change_limit = max(float(np.quantile(np.abs(change), 0.999)), 1e-8)
    tiles = (
        (f"Original image | GT={target}", image),
        ("GT Gaussian reference | display only", Image.fromarray(density_colors(reference, reference_max))),
        (f"E0 OT | predicted={predictions['e0_ot'].sum():.2f}",
         Image.fromarray(density_colors(predictions["e0_ot"], scale_for("e0_ot")))),
        (f"E1 PML | predicted={predictions['e1_pml'].sum():.2f}",
         Image.fromarray(density_colors(predictions["e1_pml"], scale_for("e1_pml")))),
        (f"E3 gate | predicted={predictions['e3_gate_ot'].sum():.2f}",
         Image.fromarray(density_colors(predictions["e3_gate_ot"], scale_for("e3_gate_ot")))),
        (f"E1 minus E0 | net={change.sum():+.2f}",
         Image.fromarray(difference_colors(change, change_limit))),
    )
    for index, (title, tile) in enumerate(tiles):
        col, row = index % 3, index // 3
        left = col * panel_width
        top = row * (panel_height + title_height)
        draw.text((left + 10, top + 9), title, fill=(20, 20, 20), font=font)
        figure.paste(tile.convert("RGB").resize((panel_width, panel_height), Image.Resampling.BILINEAR),
                     (left, top + title_height))
    note = (
        "Predictions share one scale; GT uses its own. Difference: red=added, blue=removed."
        if prediction_scale == "shared" else
        "Each prediction uses its own scale. Compare locations, not brightness across models."
    )
    draw.text((10, figure.height - note_height + 5), note,
              fill=(40, 40, 40), font=small)
    figure.save(path)


def save_overlay(image: Image.Image, density: np.ndarray,
                 points: np.ndarray, path: Path) -> None:
    vmax = max(float(np.quantile(density, 0.999)), 1e-8)
    strength = np.sqrt(np.clip(density / vmax, 0, 1))[..., None] * 0.72
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    colored = density_colors(density, vmax).astype(np.float32)
    mixed = np.clip(original * (1 - strength) + colored * strength, 0, 255)
    overlay = Image.fromarray(mixed.astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(30, 255, 100), width=1)
    overlay.save(path)


def _saved_counts(analysis_root: Path, name: str) -> dict[str, float]:
    csv_path = analysis_root / name / "per_image.csv"
    if not csv_path.is_file():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        return {row["id"]: float(row["prediction"]) for row in csv.DictReader(stream)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--ids", nargs="*", default=None,
                        help="image IDs in the chosen split; defaults to three difficult test images")
    parser.add_argument("--top-k", type=int, default=0,
                        help="also include the K most undercounted E1 images from per_image.csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sigma", type=float, default=8.0,
                        help="GT visualization width only; no effect on inference")
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    import torch
    from .data import box_centers, normalize_image
    from .inference import predict_density_full
    from .model import build_model
    from .prepare import load_boxes

    analysis_root = Path(f"outputs/analysis/ffnet_three_gpu_seed42_{args.split}")
    output_dir = args.output_dir or analysis_root / "visuals"
    base_config = json.loads(Path(RUNS["e0_ot"][0]).read_text(encoding="utf-8"))
    data_root = Path(base_config["data_root"])
    records = json.loads(
        (Path(base_config["manifest_dir"]) / f"{args.split}.json").read_text(encoding="utf-8")
    )["samples"]
    by_id = {record["id"]: record for record in records}
    if args.top_k < 0:
        parser.error("top-k must be nonnegative")
    requested = list(args.ids if args.ids is not None else (DEFAULT_IDS if args.split == "test" else ()))
    if args.top_k:
        csv_path = analysis_root / "e1_pml" / "per_image.csv"
        with csv_path.open(newline="", encoding="utf-8") as stream:
            ordered = sorted(csv.DictReader(stream), key=lambda row: float(row["error"]))
        requested.extend(row["id"] for row in ordered[:args.top_k])
    requested = list(dict.fromkeys(requested))
    if not requested:
        parser.error("supply --ids or --top-k")
    missing = [image_id for image_id in requested if image_id not in by_id]
    if missing:
        parser.error(f"IDs not found in {args.split}: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    models = {}
    saved_counts = {}
    for name, (config_path, checkpoint_path) in RUNS.items():
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if config["data_root"] != base_config["data_root"] or \
           config["manifest_dir"] != base_config["manifest_dir"]:
            raise ValueError(f"different data protocol for {name}")
        model = build_model(
            pretrained=False, attention_norm=config["attention_norm"],
            fusion_mode=config.get("fusion_mode", "original"),
            vehicle_variant=config.get("vehicle_variant", "baseline"),
            density_activation=config.get("density_activation", "relu"),
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state, strict=True)
        models[name] = model.eval()
        saved_counts[name] = _saved_counts(analysis_root, name)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ("id", "gt", "e0_count", "e1_count", "e3_count", "e1_minus_e0")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image_id in requested:
            record = by_id[image_id]
            with Image.open(data_root / record["image"]) as source:
                image = source.convert("RGB")
            boxes = load_boxes(data_root / record["annotation"])
            if len(boxes) != record["count"] or image.size != (record["width"], record["height"]):
                raise ValueError(f"image or annotation mismatch: {image_id}")
            points = box_centers(boxes).numpy()
            reference = reference_density(points, image.height, image.width, args.sigma)
            tensor = normalize_image(image).unsqueeze(0).to(device)
            predictions = {}
            for name, model in models.items():
                density = predict_density_full(model, tensor).squeeze().cpu().numpy()
                if not np.isfinite(density).all() or np.any(density < 0):
                    raise ValueError(f"invalid prediction for {name} {image_id}")
                count = float(density.sum(dtype=np.float64))
                saved = saved_counts[name].get(image_id)
                if saved is not None and abs(count - saved) > 0.05:
                    raise ValueError(
                        f"count mismatch for {name} {image_id}: visualization={count}, evaluation={saved}"
                    )
                predictions[name] = density
            save_panel(image, reference, predictions, record["count"],
                       output_dir / f"{image_id}_panel.png", args.panel_width)
            save_panel(image, reference, predictions, record["count"],
                       output_dir / f"{image_id}_panel_own_scale.png",
                       args.panel_width, prediction_scale="individual")
            save_overlay(image, predictions["e1_pml"], points,
                         output_dir / f"{image_id}_e1_overlay.png")
            counts = {name: float(density.sum(dtype=np.float64)) for name, density in predictions.items()}
            writer.writerow({
                "id": image_id, "gt": record["count"],
                "e0_count": counts["e0_ot"], "e1_count": counts["e1_pml"],
                "e3_count": counts["e3_gate_ot"],
                "e1_minus_e0": counts["e1_pml"] - counts["e0_ot"],
            })
            stream.flush()
            print(f"{image_id}: GT={record['count']} E0={counts['e0_ot']:.2f} "
                  f"E1={counts['e1_pml']:.2f} E3={counts['e3_gate_ot']:.2f}", flush=True)
    print(f"Saved {len(requested)} panels and overlays to {output_dir}")


if __name__ == "__main__":
    main()
