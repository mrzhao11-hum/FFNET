"""Precompute original-resolution Gaussian GT; no model or GPU is required."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .gaussian_targets import gaussian_density_image
from .prepare import load_boxes


def load_gaussian_h5(path, points, height, width, sigma=8.0, truncate=3.0):
    """Reject stale annotations, parameters, malformed maps and lost mass."""
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema", "") != "ffnet_gaussian_full_v1":
            raise ValueError(f"unsupported H5 schema: {path}")
        for key, expected in (("height", height), ("width", width),
                              ("sigma", sigma), ("truncate", truncate)):
            if key not in handle.attrs or not np.isclose(handle.attrs[key], expected):
                raise ValueError(f"H5 {key} mismatch: {path}")
        density = handle["density"][:]
        stored_points = handle["points"][:]
        count = int(handle["count"][()])
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if (count != len(points) or stored_points.shape != points.shape
            or not np.allclose(stored_points, points, rtol=0, atol=1e-5)):
        raise ValueError(f"H5 annotation mismatch: {path}")
    if (density.shape != (height, width) or not np.isfinite(density).all()
            or (density < 0).any()
            or not np.isclose(density.sum(dtype=np.float64), count, rtol=0, atol=1e-3)):
        raise ValueError(f"H5 density shape/value/mass mismatch: {path}")
    return density.astype(np.float32, copy=False)


def prepare_record(record, data_root, target_root, sigma=8.0, truncate=3.0,
                   overwrite=False, check_only=False):
    boxes = np.asarray(load_boxes(Path(data_root) / record["annotation"]),
                       dtype=np.float32).reshape(-1, 4)
    points = (boxes[:, :2] + boxes[:, 2:]) / 2
    height, width = record["height"], record["width"]
    with Image.open(Path(data_root) / record["image"]) as image:
        if image.size != (width, height) or len(points) != record["count"]:
            raise ValueError(f"manifest/image/annotation mismatch: {record['id']}")
    path = Path(target_root) / f"{record['id']}.h5"
    if check_only or (path.exists() and not overwrite):
        load_gaussian_h5(path, points, height, width, sigma, truncate)
        return "checked"
    density = gaussian_density_image(points, height, width, sigma, truncate)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.h5")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(schema="ffnet_gaussian_full_v1", height=height,
                                width=width, sigma=sigma, truncate=truncate)
            handle.create_dataset("density", data=density, compression="gzip",
                                  compression_opts=1, shuffle=True)
            handle.create_dataset("points", data=points)
            handle.create_dataset("boxes", data=boxes)
            handle.create_dataset("count", data=np.int32(len(points)))
        load_gaussian_h5(temporary, points, height, width, sigma, truncate)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "generated"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test", "train_full"),
                        default=["train", "val"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("gaussian_target_source") != "h5":
        parser.error("config must select gaussian_target_source=h5")
    seen = set()
    total = 0
    for split in args.splits:
        rows = json.loads((Path(config["manifest_dir"]) / f"{split}.json").read_text(
            encoding="utf-8"))["samples"]
        if args.limit is not None:
            rows = rows[:args.limit]
        generated = 0
        for index, record in enumerate(rows, 1):
            if record["id"] in seen:
                raise ValueError(f"duplicate image across selected splits: {record['id']}")
            seen.add(record["id"])
            status = prepare_record(record, config["data_root"], config["gaussian_target_root"],
                                    config.get("gaussian_sigma", 8.0),
                                    config.get("gaussian_truncate", 3.0),
                                    args.overwrite, args.check_only)
            generated += status == "generated"
            if index % 100 == 0:
                print(f"{split}: {index}/{len(rows)}", flush=True)
        total += len(rows)
        print(f"{split}: images={len(rows)} generated={generated} checked={len(rows)-generated}",
              flush=True)
    print(f"H5 ready: {total} images, target_root={config['gaussian_target_root']}")


if __name__ == "__main__":
    main()
