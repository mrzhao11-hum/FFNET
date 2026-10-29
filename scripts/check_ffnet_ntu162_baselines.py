"""Fail fast if the paired OT/PML CARPK baseline protocol has drifted."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicle_ffnet.prepare import load_boxes

CONFIGS = {
    name: ROOT / "configs" / f"carpk_ntu162_ffnet_{name}_full_seed42.json"
    for name in ("ot", "pml")
}


def main() -> None:
    configs = {name: json.loads(path.read_text(encoding="utf-8"))
               for name, path in CONFIGS.items()}
    common = (
        "dataset", "validation_protocol", "data_root", "manifest_dir", "seed",
        "pretrained", "attention_norm", "fusion_mode", "train_mode", "crop_size",
        "batch_size", "patch_batch_size", "workers", "epochs", "val_interval", "patience_validations",
        "lr", "weight_decay",
    )
    for key in common:
        if configs["ot"].get(key) != configs["pml"].get(key):
            raise ValueError(f"paired baseline settings differ: {key}")
    ot, pml = configs["ot"], configs["pml"]
    if (ot["supervision"], pml["supervision"]) != ("ot", "pml"):
        raise ValueError("baseline supervision labels are wrong")
    if (ot["wot"], ot["wtv"], pml["pml_density_factor"],
            pml["pml_count_weight"]) != (0.1, 0.01, 1.0, 0.0):
        raise ValueError("unexpected loss weights or PML mass target")
    if (ot["epochs"], ot["batch_size"], ot["train_mode"], ot["attention_norm"],
            ot["weight_decay"]) != (200, 1, "full", "group_all", 0.0):
        raise ValueError("unexpected baseline training protocol")

    manifest_dir = ROOT / ot["manifest_dir"]
    data_root = ROOT / ot["data_root"]
    split_ids = {}
    for split, expected in (("train", 827), ("val", 162), ("test", 459)):
        records = json.loads((manifest_dir / f"{split}.json").read_text(encoding="utf-8"))["samples"]
        ids = {record["id"] for record in records}
        if len(records) != expected or len(ids) != expected:
            raise ValueError(f"{split}: expected {expected} unique images")
        split_ids[split] = ids
        for record in records:
            width, height = record["width"], record["height"]
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid image size: {record['id']}")
            if not (data_root / record["image"]).is_file():
                raise FileNotFoundError(data_root / record["image"])
            boxes = load_boxes(data_root / record["annotation"])
            if len(boxes) != record["count"]:
                raise ValueError(f"annotation count changed: {record['id']}")
            rows, cols = math.ceil(height / 8), math.ceil(width / 8)
            for x1, y1, x2, y2 in boxes:
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                # PML uses continuous cell coordinates (x/8, y/8); the
                # discrete map and count each preserve one unit per point.
                if not (0 <= x / 8 < cols and 0 <= y / 8 < rows):
                    raise ValueError(f"point falls outside valid D8 map: {record['id']}")
        print(f"{split}: {len(records)} images, {sum(r['count'] for r in records)} vehicles")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_ids[left] & split_ids[right]:
            raise ValueError(f"split leakage: {left}/{right}")
    if {image_id.rsplit("_", 1)[0] for image_id in split_ids["val"]} != {"20161029_NTU"}:
        raise ValueError("validation sequence is not the expected NTU holdout")
    print("OT/PML protocol and D8 point coordinates verified")


if __name__ == "__main__":
    main()
