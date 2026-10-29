"""Create portable vehicle manifests without copying source images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLITS = ("train", "val", "test", "train_full")


def load_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"{path}:{number}: expected at least four box coordinates")
        x1, y1, x2, y2 = map(float, fields[:4])
        if x2 < x1 or y2 < y1:
            raise ValueError(f"{path}:{number}: inverted box")
        boxes.append((x1, y1, x2, y2))
    return boxes


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate image IDs in {path}")
    return ids


def partition_pucpr_train(ids: list[str], val_first_index: int = 25) -> tuple[list[str], list[str]]:
    """Keep all weather variants of a scene index in the same split."""
    train, val = [], []
    for image_id in ids:
        prefix, separator, _ = image_id.partition("_")
        if not separator or not prefix.isdigit():
            raise ValueError(f"unexpected PUCPR+ image ID: {image_id}")
        (train if int(prefix) < val_first_index else val).append(image_id)
    if not train or not val:
        raise ValueError("PUCPR+ grouped split must have nonempty train and val sets")
    return train, val


def make_record(data_root: Path, image_id: str, extension: str) -> dict:
    from PIL import Image

    image_rel = f"Images/{image_id}{extension}"
    annotation_rel = f"Annotations/{image_id}.txt"
    image_path = data_root / image_rel
    annotation_path = data_root / annotation_rel
    if not image_path.is_file() or not annotation_path.is_file():
        raise FileNotFoundError(f"missing image or annotation for {image_id}")
    with Image.open(image_path) as image:
        width, height = image.size
    boxes = load_boxes(annotation_path)
    for x1, y1, x2, y2 in boxes:
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        if not (0 <= center_x < width and 0 <= center_y < height):
            raise ValueError(f"box center outside image: {image_id}")
    return {
        "id": image_id,
        "image": image_rel,
        "annotation": annotation_rel,
        "width": width,
        "height": height,
        "count": len(boxes),
    }


def build_carpk(legacy_dir: Path, data_root: Path) -> dict[str, list[dict]]:
    groups = {}
    for split in SPLITS:
        payload = json.loads((legacy_dir / f"{split}.json").read_text(encoding="utf-8"))
        records = payload["samples"]
        groups[split] = []
        for old in records:
            record = make_record(data_root, old["id"], ".png")
            if record["count"] != old["count"]:
                raise ValueError(f"CARPK annotation count changed for {old['id']}")
            groups[split].append(record)
    if [len(groups[name]) for name in SPLITS] != [827, 162, 459, 989]:
        raise ValueError("CARPK manifests do not match the fixed 827/162/459 protocol")
    return groups


def build_pucpr(data_root: Path, val_first_index: int = 25) -> dict[str, list[dict]]:
    image_sets = data_root / "ImageSets"
    official_train = read_ids(image_sets / "train.txt")
    official_test = read_ids(image_sets / "test.txt")
    if set(official_train) & set(official_test):
        raise ValueError("PUCPR+ official train/test overlap")
    if len(official_train) != 100 or len(official_test) != 25:
        raise ValueError("PUCPR+ official split must contain 100 train and 25 test images")
    train_ids, val_ids = partition_pucpr_train(official_train, val_first_index)
    all_ids = set(official_train + official_test)
    records = {image_id: make_record(data_root, image_id, ".jpg") for image_id in all_ids}
    return {
        "train": [records[name] for name in train_ids],
        "val": [records[name] for name in val_ids],
        "test": [records[name] for name in official_test],
        "train_full": [records[name] for name in official_train],
    }


def save_manifests(output_dir: Path, dataset: str, groups: dict[str, list[dict]]) -> dict:
    destination = output_dir / dataset
    destination.mkdir(parents=True, exist_ok=True)
    train_ids = {row["id"] for row in groups["train"]}
    val_ids = {row["id"] for row in groups["val"]}
    test_ids = {row["id"] for row in groups["test"]}
    full_ids = {row["id"] for row in groups["train_full"]}
    if train_ids & val_ids or (train_ids | val_ids) != full_ids or test_ids & full_ids:
        raise ValueError(f"{dataset}: overlapping or incomplete split")
    summary = {}
    for split, records in groups.items():
        document = {"schema_version": 1, "dataset": dataset, "split": split, "samples": records}
        (destination / f"{split}.json").write_text(
            json.dumps(document, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        summary[split] = {"images": len(records), "vehicles": sum(row["count"] for row in records)}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carpk-root", type=Path, required=True)
    parser.add_argument("--carpk-legacy-manifests", type=Path, required=True)
    parser.add_argument("--pucpr-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    args = parser.parse_args()
    carpk = build_carpk(args.carpk_legacy_manifests, args.carpk_root)
    pucpr = build_pucpr(args.pucpr_root)
    for name, groups in (("CARPK", carpk), ("PUCPR+", pucpr)):
        print(name, save_manifests(args.output_dir, name, groups))


if __name__ == "__main__":
    main()
