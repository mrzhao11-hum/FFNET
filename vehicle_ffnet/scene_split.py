"""Build an alternative CARPK validation fold from one official train sequence.

The official test list is copied verbatim. This never edits the existing
CARPK/NTU manifests and never moves or duplicates image files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prepare import read_ids


def scene_of(image_id: str) -> str:
    parts = image_id.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(f"unexpected CARPK image ID: {image_id}")
    return parts[0]


def load_records(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    records = document["samples"]
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate image ID in {path}")
    return records


def build_scene_holdout(
    data_root: Path, source_dir: Path, validation_scene: str
) -> dict[str, list[dict]]:
    full = load_records(source_dir / "train_full.json")
    test = load_records(source_dir / "test.json")
    official_sets = data_root / "ImageSets"
    official_train = set(read_ids(official_sets / "train.txt"))
    official_test = set(read_ids(official_sets / "test.txt"))
    if {row["id"] for row in full} != official_train:
        raise ValueError("train_full manifest differs from official CARPK train.txt")
    if {row["id"] for row in test} != official_test:
        raise ValueError("test manifest differs from official CARPK test.txt")
    if official_train & official_test:
        raise ValueError("official train/test overlap")
    train = [row for row in full if scene_of(row["id"]) != validation_scene]
    val = [row for row in full if scene_of(row["id"]) == validation_scene]
    if not train or not val:
        raise ValueError(f"validation scene has no official training samples: {validation_scene}")
    return {"train": train, "val": val, "test": test, "train_full": full}


def write_holdout(output_dir: Path, groups: dict[str, list[dict]], scene: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"protocol": "whole_video_sequence_holdout", "validation_scene": scene}
    for split, records in groups.items():
        payload = {
            "schema_version": 1,
            "dataset": "CARPK",
            "split": split,
            "validation_scene": scene,
            "samples": records,
        }
        (output_dir / f"{split}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        summary[split] = {
            "images": len(records),
            "vehicles": sum(row["count"] for row in records),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/raw/CARPK/CARPK_devkit/data"))
    parser.add_argument("--source-dir", type=Path,
                        default=Path("data/manifests/CARPK"))
    parser.add_argument("--scene", default="20161030_GF2")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/manifests/CARPK_GF2_val"))
    args = parser.parse_args()
    groups = build_scene_holdout(args.data_root, args.source_dir, args.scene)
    print(write_holdout(args.output_dir, groups, args.scene))


if __name__ == "__main__":
    main()
