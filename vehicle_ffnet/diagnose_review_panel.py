"""Reuse the exploratory CARPK visibility panel with native-output diagnostics.

Labels remain AI-assisted and not independently human-confirmed. Neighborhood
mass may include a neighbor; feature RMS is not a vehicle-confidence score.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from .data import VehicleDataset, box_centers
from .diagnose_instance_mass import assign_mass, predict_valid_d8
from .model import build_model
from .prepare import load_boxes

CLASSES = {"complete", "interior_occluded_visible", "boundary_visible"}
PROTOCOLS = {"assistant_visual_blind", "assistant_original_image_review"}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def checked_panel(review_dir, records, data_root):
    records = {r["id"]: r for r in records}
    selected = []
    seen = set()
    boxes_cache = {}
    all_rows = read_rows(Path(review_dir) / "review_labels.csv")
    for row in all_rows:
        image_id, index = row["id"], int(row["object_index"])
        key = image_id, index
        if key in seen:
            raise ValueError(f"duplicate panel instance: {key}")
        seen.add(key)
        if image_id not in records:
            raise ValueError(f"panel image outside selected manifest: {image_id}")
        record = records[image_id]
        if image_id not in boxes_cache:
            boxes_cache[image_id] = load_boxes(Path(data_root) / record["annotation"])
        boxes = boxes_cache[image_id]
        expected = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
        if (index >= len(boxes) or index < 0 or tuple(boxes[index]) != expected
                or int(row["image_width"]) != record["width"]
                or int(row["image_height"]) != record["height"]):
            raise ValueError(f"panel box/index/image-size mismatch: {key}")
        if row["visibility"] in CLASSES and row["review_protocol"] in PROTOCOLS:
            selected.append(row)
    return selected, len(all_rows)


def native_grid(height, width, stride, device):
    ys = torch.arange(height, device=device) * stride + (stride - 1) / 2
    xs = torch.arange(width, device=device) * stride + (stride - 1) / 2
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def spatial_rms(feature, x, y, radius, stride, height, width):
    f = feature[0, :, :(height + stride - 1)//stride, :(width + stride - 1)//stride]
    xx, yy = native_grid(*f.shape[-2:], stride, f.device)
    mask = (xx - x).square() + (yy - y).square() <= radius**2
    return float(f[:, mask].float().square().mean().sqrt()) if mask.any() else None


def peak_metrics(density, distances, owner, index, radius=24):
    mask = distances <= radius
    if not mask.any() or float(density[mask].max()) <= 1e-8:
        return {"peak_distance_px": None, "peak_value": 0., "peak_owner_index": None,
                "peak_owned_by_this_vehicle": False}
    flat_index = int(density.masked_fill(~mask, -float("inf")).flatten().argmax())
    return {"peak_distance_px": float(distances.flatten()[flat_index]),
            "peak_value": float(density.flatten()[flat_index]),
            "peak_owner_index": int(owner.flatten()[flat_index]),
            "peak_owned_by_this_vehicle": int(owner.flatten()[flat_index]) == index}


def roi_mass(density, box, height, width, stride=8):
    """Allocate cell mass by rectangle overlap, assuming uniform mass in a cell."""
    x1, y1, x2, y2 = box
    xs = torch.arange(density.shape[1], device=density.device) * stride
    ys = torch.arange(density.shape[0], device=density.device) * stride
    right = (xs + stride).clamp(max=width)
    bottom = (ys + stride).clamp(max=height)
    wx = (right.clamp(max=x2) - xs.clamp(min=x1)).clamp(min=0) / (right-xs)
    wy = (bottom.clamp(max=y2) - ys.clamp(min=y1)).clamp(min=0) / (bottom-ys)
    return float((density * wy[:, None] * wx[None, :]).sum())


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--review-dir", type=Path, default=Path("data/review_panels/carpk_ntu162_seed43"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit-images", type=int)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    ds = VehicleDataset(cfg["data_root"], Path(cfg["manifest_dir"])/"val.json", train=False,
                        crop_size=cfg["crop_size"])
    labels, total = checked_panel(args.review_dir, ds.records, cfg["data_root"])
    by_image = defaultdict(list)
    for row in labels:
        by_image[row["id"]].append(row)
    rois_by_image = defaultdict(list)
    for row in read_rows(args.review_dir/"noncar_rois.csv"):
        if row["id"] not in {r["id"] for r in ds.records}:
            raise ValueError(f"ROI image outside validation manifest: {row['id']}")
        if row["category"] != "uncertain" and row["review_protocol"] in PROTOCOLS:
            rois_by_image[row["id"]].append(row)
    device = torch.device(args.device)
    model = build_model(pretrained=False, attention_norm=cfg["attention_norm"],
                        vehicle_variant=cfg.get("vehicle_variant", "baseline"),
                        fusion_mode=cfg.get("fusion_mode", "original"),
                        density_activation=cfg.get("density_activation", "relu")).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    captured = {}
    def capture(name):
        def hook(module, inputs, output):
            captured[name] = output.detach()
        return hook
    ftm = getattr(model.ccsm1, "base", model.ccsm1)
    hooks = [module.register_forward_hook(capture(name)) for name, module in (
        ("f4", model.backbone.stem), ("f8", model.backbone.stage1),
        ("before_attention", ftm.conv4), ("attention", ftm.sa_s),
        ("after_attention", model.ccsm1), ("logits", model.fusion.final[0]))]
    objects, images, roi_rows = [], [], []
    try:
        with torch.inference_mode():
            for n, record in enumerate(ds.records):
                image_id = record["id"]
                if image_id not in by_image and image_id not in rois_by_image:
                    continue
                if args.limit_images is not None and len(images) >= args.limit_images:
                    break
                image, count, _ = ds[n]
                h, w = image.shape[-2:]
                boxes = load_boxes(Path(cfg["data_root"])/record["annotation"])
                points = box_centers(boxes).to(device)
                captured.clear()
                density = predict_valid_d8(model, image[None].to(device))
                assigned, owned24, _ = assign_mass(density, points, stride=model.output_stride)
                if len(points) and not torch.allclose(assigned.sum(), density.sum(), atol=1e-3, rtol=1e-5):
                    raise ValueError(f"partition lost mass: {image_id}")
                xx, yy = native_grid(*density.shape, model.output_stride, device)
                grid = torch.stack((xx.flatten(), yy.flatten()), dim=1)
                owner = (torch.cdist(grid, points).argmin(dim=1).reshape(density.shape)
                         if len(points) else torch.full_like(density,-1,dtype=torch.long))
                logits = captured["logits"][0, 0, :density.shape[0], :density.shape[1]]
                attention = captured["attention"]
                if model.output_stride == 4:
                    attention = torch.nn.functional.interpolate(attention, scale_factor=2, mode="nearest")
                attention = attention[0, 0, :density.shape[0], :density.shape[1]]
                images.append(dict(id=image_id, gt=count, prediction=float(density.sum()),
                                   reviewed_vehicles=len(by_image[image_id])))
                for label in by_image[image_id]:
                    i = int(label["object_index"])
                    x, y = points[i].tolist()
                    distances = ((xx-x).square() + (yy-y).square()).sqrt()
                    local = distances <= 24
                    owned = owner == i
                    row = dict(id=image_id, instance_index=i, visibility=label["visibility"],
                               review_protocol=label["review_protocol"], x=x, y=y,
                               assigned_mass=float(assigned[i]), owned_mass_r24=float(owned24[i]))
                    row.update({f"mass_r{r}": float(density[distances <= r].sum()) for r in (8,16,24)})
                    row.update(peak_metrics(density, distances, owner, i))
                    row.update(logit_mean_local=float(logits[local].mean()),
                               logit_positive_fraction_local=float((logits[local]>0).float().mean()),
                               output_positive_fraction_owned=float((density[owned]>0).float().mean())
                               if owned.any() else None,
                               attention_mean_local=float(attention[local].mean()))
                    for name, stride in (("f4",4),("f8",8),("before_attention",8),("after_attention",8)):
                        row[f"{name}_rms_local"] = spatial_rms(captured[name],x,y,24,stride,h,w)
                    objects.append(row)
                for label in rois_by_image[image_id]:
                    box = tuple(float(label[k]) for k in ("x1","y1","x2","y2"))
                    x1,y1,x2,y2 = box
                    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                        raise ValueError(f"ROI outside original image: {image_id}")
                    overlap = sum(max(0,min(x2,b[2])-max(x1,b[0])) *
                                  max(0,min(y2,b[3])-max(y1,b[1])) for b in boxes)
                    roi_rows.append(dict(id=image_id,roi_index=label["roi_index"],
                        category=label["category"],mass=roi_mass(density,box,h,w,stride=model.output_stride),
                        gt_box_overlap_px2=overlap,review_protocol=label["review_protocol"]))
                if len(images)%25 == 0:
                    print(f"review panel: {len(images)} images",flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    if not objects:
        raise ValueError("no eligible reviewed vehicles")
    groups = {}
    for category in sorted(CLASSES):
        rows = [r for r in objects if r["visibility"] == category]
        if rows:
            groups[category] = dict(vehicles=len(rows),mean_mass=mean(r["assigned_mass"] for r in rows),
                mass_lt_0p1_fraction=mean(r["assigned_mass"]<.1 for r in rows),
                mass_lt_0p5_fraction=mean(r["assigned_mass"]<.5 for r in rows),
                mean_deficit=mean(max(1-r["assigned_mass"],0) for r in rows),
                mean_excess=mean(max(r["assigned_mass"]-1,0) for r in rows))
    report = dict(checkpoint=str(args.checkpoint),config=str(args.config),split="val",
        source_review=json.loads((args.review_dir/"summary.json").read_text(encoding="utf-8")),
        panel_rows=total,eligible_rows=len(labels),evaluated_images=len(images),
        evaluated_vehicles=len(objects),groups=groups,
        notes=["Exploratory AI-assisted labels, not independently human-confirmed.",
               "Panel image counts are not full-validation metrics.",
               "Radius masses overlap and may borrow neighboring vehicle responses.",
               "A local peak belonging to another GT is not evidence of this vehicle's localization.",
               "Feature RMS/attention correlations do not establish semantic content or causation.",
               "ROI mass assumes uniform mass within each native output cell; report GT-overlap flags."])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(args.output_dir/"reviewed_instances.csv",objects)
    write_csv(args.output_dir/"per_image.csv",images)
    write_csv(args.output_dir/"noncar_rois.csv",roi_rows)
    (args.output_dir/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(groups,indent=2))
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
