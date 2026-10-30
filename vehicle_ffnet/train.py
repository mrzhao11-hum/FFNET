"""Train FFNet vehicle-density controls with OT or PML point supervision."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import VehicleDataset, full_image_collate, train_collate
from .evaluate import evaluate_model
from .model import build_model, upstream_modules
from .rectangular_ot import RectangularOTLoss


HISTORY_COLUMNS = (
    "epoch", "train_loss", "train_ot_weighted", "train_count_l1", "train_tv_weighted",
    "train_mae", "val_mae", "val_rmse", "val_bias", "learning_rate",
)
PML_HISTORY_COLUMNS = HISTORY_COLUMNS + (
    "train_pml_loss", "train_pml_point_loss", "train_pml_transport_loss",
)
GAUSSIAN_HISTORY_COLUMNS = HISTORY_COLUMNS + ("train_gaussian_density_weighted",)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(map_location=device) also moves saved CPU RNG tensors to CUDA.
    # Both RNG restoration APIs require CPU ByteTensors.
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda"]])


def compute_loss(model, criterion, images, points, gt_discrete, wot: float, wtv: float,
                 valid_shape: tuple[int, int] | None = None, supervision: str = "ot",
                 pml_count_weight: float = 0.0, gaussian_sigma: float = 8.0,
                 gaussian_density_weight: float = 1.0,
                 gaussian_count_weight: float = 0.002,
                 gaussian_target_source: str = "online"):
    prediction, normalized = model(images)
    if valid_shape is not None:
        prediction = prediction[:, :, :valid_shape[0], :valid_shape[1]]
        normalized = prediction / (prediction.flatten(1).sum(1)[:, None, None, None] + 1e-6)
    if prediction.shape != gt_discrete.shape:
        raise ValueError(f"prediction and point map shapes differ: {prediction.shape} vs {gt_discrete.shape}")
    target_counts = torch.tensor([len(item) for item in points], dtype=torch.float32, device=images.device)
    predicted_counts = prediction.flatten(1).sum(1)
    count_loss = F.l1_loss(predicted_counts, target_counts)
    if supervision == "gaussian":
        if gaussian_target_source == "h5":
            target_density = gt_discrete.to(dtype=prediction.dtype)
            if (not torch.isfinite(target_density).all() or (target_density < 0).any()
                    or not torch.allclose(target_density.flatten(1).sum(1), target_counts,
                                          rtol=0, atol=1e-3)):
                raise ValueError("H5 target lost mass or contains invalid density")
        elif gaussian_target_source == "online":
            from .gaussian_targets import gaussian_density_d8

            rows, cols = prediction.shape[-2:]
            target_maps = np.stack([
                gaussian_density_d8(item.detach().cpu().numpy(), rows * 8, cols * 8,
                                    sigma=gaussian_sigma)
                for item in points
            ])
            target_density = torch.as_tensor(
                target_maps[:, None], device=prediction.device, dtype=prediction.dtype
            )
        else:
            raise ValueError(f"unknown Gaussian target source: {gaussian_target_source}")
        density_sse = (prediction - target_density).square().flatten(1).sum(1).mean()
        weighted_density = gaussian_density_weight * density_sse
        total = weighted_density + gaussian_count_weight * count_loss
        zero = total.new_zeros(())
        return total, (zero, count_loss, zero, zero, zero, zero, weighted_density), predicted_counts.detach(), target_counts
    if supervision == "pml":
        pml_loss, pml_point, pml_transport = criterion(
            prediction, points, return_components=True
        )
        zero = pml_loss.new_zeros(())
        total = pml_loss + pml_count_weight * count_loss
        return total, (zero, count_loss, zero, pml_loss, pml_point, pml_transport, zero), predicted_counts.detach(), target_counts
    if supervision != "ot":
        raise ValueError(f"unsupported supervision: {supervision}")
    ot_raw, _, _ = criterion(normalized, prediction, points)
    target_normalized = gt_discrete / (target_counts[:, None, None, None] + 1e-6)
    tv_raw = (torch.abs(normalized - target_normalized).flatten(1).sum(1) * target_counts).mean()
    ot_weighted = wot * ot_raw.squeeze()
    tv_weighted = wtv * tv_raw
    total = ot_weighted + count_loss + tv_weighted
    zero = total.new_zeros(())
    return total, (ot_weighted, count_loss, tv_weighted, zero, zero, zero, zero), predicted_counts.detach(), target_counts


def train_epoch(model, loader, optimizer, criterion, config: dict, device: torch.device) -> dict:
    model.train()
    sums = {"loss": 0.0, "ot": 0.0, "count": 0.0, "tv": 0.0,
            "pml": 0.0, "pml_point": 0.0, "pml_transport": 0.0,
            "gaussian": 0.0, "mae": 0.0}
    samples = 0
    for batch in loader:
        if len(batch) == 4:
            images, points, gt_discrete, valid_shape = batch
        else:
            images, points, gt_discrete = batch
            valid_shape = None
        images = images.to(device)
        points = [item.to(device) for item in points]
        gt_discrete = gt_discrete.to(device)
        optimizer.zero_grad(set_to_none=True)
        total, terms, predicted, target = compute_loss(
            model, criterion, images, points, gt_discrete, config["wot"], config["wtv"],
            valid_shape=valid_shape,
            supervision=config.get("supervision", "ot"),
            pml_count_weight=config.get("pml_count_weight", 0.0),
            gaussian_sigma=config.get("gaussian_sigma", 8.0),
            gaussian_density_weight=config.get("gaussian_density_weight", 1.0),
            gaussian_count_weight=config.get("gaussian_count_weight", 0.002),
            gaussian_target_source=config.get("gaussian_target_source", "online"),
        )
        if not torch.isfinite(total):
            raise FloatingPointError("nonfinite FFNet training loss")
        total.backward()
        optimizer.step()
        n = len(images)
        samples += n
        for key, value in zip(
            ("loss", "ot", "count", "tv", "pml", "pml_point", "pml_transport", "gaussian"),
            (total, *terms),
        ):
            sums[key] += float(value.detach().item()) * n
        sums["mae"] += float(torch.abs(predicted - target).sum().item())
    return {key: value / samples for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--crop-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-pretrained", action="store_true", help="offline smoke only")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-early-stop", action="store_true",
                        help="train to the total --epochs limit; still save best validation MAE")
    parser.add_argument("--early-stop-patience", type=int,
                        help="stop after this many validation epochs without a new best MAE; 0 disables it")
    parser.add_argument("--train-split", choices=("train", "train_full"), default="train")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("data_root", "manifest_dir", "output_dir", "workers", "batch_size", "crop_size", "epochs", "seed"):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    if args.early_stop_patience is not None:
        config["patience_validations"] = args.early_stop_patience
    if config["patience_validations"] < 0:
        raise ValueError("early-stop patience must be nonnegative")
    config["pretrained"] = bool(config["pretrained"] and not args.no_pretrained)
    config["train_split"] = args.train_split
    config["run_to_max_epochs"] = args.no_early_stop
    supervision = config.get("supervision", "ot")
    fusion_mode = config.get("fusion_mode", "original")
    if supervision not in {"ot", "pml", "gaussian"}:
        raise ValueError(f"unsupported supervision: {supervision}")
    if config.get("pml_count_weight", 0.0) < 0:
        raise ValueError("pml_count_weight cannot be negative")
    if (config.get("gaussian_sigma", 8.0) <= 0
            or config.get("gaussian_density_weight", 1.0) <= 0
            or config.get("gaussian_count_weight", 0.002) < 0):
        raise ValueError("Gaussian sigma/density weight must be positive; count weight nonnegative")
    train_mode = config.get("train_mode", "crop")
    gaussian_source = config.get("gaussian_target_source", "online")
    if gaussian_source not in {"online", "h5"}:
        raise ValueError("gaussian_target_source must be online or h5")
    if gaussian_source == "h5" and (supervision != "gaussian" or train_mode != "full"
                                     or not config.get("gaussian_target_root")):
        raise ValueError("H5 targets require Gaussian full-image training and gaussian_target_root")
    if train_mode not in {"crop", "full"}:
        raise ValueError(f"unsupported train_mode: {train_mode}")
    attention_norm = config.get("attention_norm", "batch")
    minimum_batch = 2 if attention_norm == "batch" else 1
    if config["batch_size"] < minimum_batch or config["crop_size"] % 32:
        raise ValueError(
            f"batch size must be >={minimum_batch} with attention_norm={attention_norm}, "
            "and crop size must be divisible by 32"
        )
    if train_mode == "full" and (config["batch_size"] != 1 or attention_norm not in {"group", "group_all"}):
        raise ValueError("full-image training requires batch size 1 and attention_norm=group or group_all")
    set_seed(config["seed"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = Path(config["output_dir"])
    if (output_dir / "last.pth").exists() and not args.resume:
        raise FileExistsError(f"existing run at {output_dir}; use --resume or a new output directory")
    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
        previous_config = output_dir / "resolved_config.json"
        if previous_config.is_file():
            previous = json.loads(previous_config.read_text(encoding="utf-8"))
            fixed_keys = (
                "dataset", "validation_protocol", "data_root", "manifest_dir", "crop_size", "batch_size",
                "seed", "lr", "weight_decay", "wot", "wtv", "ot_regularization",
                "ot_iterations", "train_split", "val_interval", "patience_validations",
                "attention_norm",
                "train_mode", "ot_reference_size",
                "supervision", "fusion_mode", "vehicle_variant", "pml_radius", "pml_density_factor",
                "pml_count_weight", "gaussian_sigma", "gaussian_density_weight",
                "gaussian_count_weight",
                "gaussian_target_source", "gaussian_target_root", "gaussian_truncate",
            )
            defaults = {"supervision": "ot", "fusion_mode": "original",
                        "vehicle_variant": "baseline",
                        "pml_radius": 32.0, "pml_density_factor": 1.0,
                        "pml_count_weight": 0.0, "gaussian_sigma": 8.0,
                        "gaussian_density_weight": 1.0, "gaussian_count_weight": 0.002,
                        "gaussian_target_source": "online", "gaussian_truncate": 3.0}
            changed = [key for key in fixed_keys
                       if previous.get(key, defaults.get(key)) != config.get(key, defaults.get(key))]
            if changed:
                raise ValueError(f"resume config changed immutable settings: {', '.join(changed)}")
            if previous.get("run_to_max_epochs", False) and not args.no_early_stop:
                raise ValueError("this run used --no-early-stop; keep it when resuming")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    manifest_dir = Path(config["manifest_dir"])
    data_root = Path(config["data_root"])
    train_set = VehicleDataset(
        data_root, manifest_dir / f"{args.train_split}.json", crop_size=config["crop_size"], train=True,
        limit=args.limit, full_image=train_mode == "full",
        gaussian_target_root=config.get("gaussian_target_root") if gaussian_source == "h5" else None,
        gaussian_sigma=config.get("gaussian_sigma", 8.0),
        gaussian_truncate=config.get("gaussian_truncate", 3.0),
    )
    val_manifest = manifest_dir / "val.json"
    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True, drop_last=True,
        num_workers=config["workers"],
        collate_fn=full_image_collate if train_mode == "full" else train_collate,
        pin_memory=device.type == "cuda",
    )
    if len(train_loader) == 0:
        raise ValueError("not enough training samples for one full batch")
    model = build_model(
        pretrained=config["pretrained"] and args.resume is None,
        attention_norm=config.get("attention_norm", "batch"),
        fusion_mode=fusion_mode,
        vehicle_variant=config.get("vehicle_variant", "baseline"),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if supervision == "pml":
        from .pml_loss import ProximalMappingLoss
        criterion = ProximalMappingLoss(
            stride=8, radius_px=config.get("pml_radius", 32.0),
            density_factor=config.get("pml_density_factor", 1.0),
        )
    elif supervision == "gaussian":
        criterion = None
    elif train_mode == "full":
        criterion = RectangularOTLoss(
            8, device, config["ot_iterations"], config["ot_regularization"],
            config.get("ot_reference_size", 256.0),
        )
    else:
        _, official_ot = upstream_modules()
        criterion = official_ot(
            config["crop_size"], 8, 0, device, config["ot_iterations"], config["ot_regularization"]
        )
    start_epoch = 1
    best_mae = float("inf")
    stale_validations = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_mae = checkpoint["best_mae"]
        stale_validations = checkpoint.get("stale_validations", 0)
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
        else:
            print("Resume checkpoint lacks RNG state; this continuation is not bitwise identical "
                  "to an uninterrupted run.", flush=True)

    history_path = output_dir / "history.csv"
    append = args.resume is not None and history_path.exists()
    val_size = 0 if args.train_split == "train_full" else len(
        VehicleDataset(data_root, val_manifest, config["crop_size"], limit=args.limit)
    )
    print(f"dataset={config['dataset']} train={len(train_set)} val={val_size} "
          f"input={'full' if train_mode == 'full' else config['crop_size']} "
          f"batch={config['batch_size']} seed={config['seed']} "
          f"pretrained={config['pretrained']} attention_norm={config.get('attention_norm', 'batch')} "
          f"train_mode={train_mode} fusion={fusion_mode} "
          f"vehicle_variant={config.get('vehicle_variant', 'baseline')} supervision={supervision} "
          f"early_stop={not args.no_early_stop and config['patience_validations'] > 0} "
          f"patience={config['patience_validations']} "
          f"weight_decay={config['weight_decay']} device={device}", flush=True)
    if supervision == "gaussian":
        print(f"Gaussian GT={gaussian_source} root={config.get('gaussian_target_root', 'memory')} "
              f"sigma={config.get('gaussian_sigma', 8.0)} stride=8 "
              f"density_reduction=batch_mean_sum "
              f"density_weight={config.get('gaussian_density_weight', 1.0)} "
              f"count_L1_weight={config.get('gaussian_count_weight', 0.002)}", flush=True)
    with history_path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=(
                PML_HISTORY_COLUMNS if supervision == "pml"
                else GAUSSIAN_HISTORY_COLUMNS if supervision == "gaussian"
                else HISTORY_COLUMNS
            )
        )
        if not append:
            writer.writeheader()
        for epoch in range(start_epoch, config["epochs"] + 1):
            train = train_epoch(model, train_loader, optimizer, criterion, config, device)
            row = {
                "epoch": epoch, "train_loss": train["loss"], "train_ot_weighted": train["ot"],
                "train_count_l1": train["count"], "train_tv_weighted": train["tv"],
                "train_mae": train["mae"], "val_mae": "", "val_rmse": "", "val_bias": "",
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if supervision == "pml":
                row["train_pml_loss"] = train["pml"]
                row["train_pml_point_loss"] = train["pml_point"]
                row["train_pml_transport_loss"] = train["pml_transport"]
            elif supervision == "gaussian":
                row["train_gaussian_density_weighted"] = train["gaussian"]
            validate = args.train_split == "train" and (epoch == 1 or epoch % config["val_interval"] == 0)
            if validate:
                _, val = evaluate_model(
                    model, data_root, val_manifest, config["crop_size"],
                    config["patch_batch_size"], device, workers=config["workers"], limit=args.limit,
                    inference_mode="full" if train_mode == "full" else "tiles",
                )
                row.update({"val_mae": val["mae"], "val_rmse": val["rmse"], "val_bias": val["bias"]})
                if val["mae"] < best_mae:
                    best_mae = val["mae"]
                    stale_validations = 0
                    torch.save(model.state_dict(), output_dir / "best_mae.pth")
                else:
                    stale_validations += 1
            writer.writerow(row)
            stream.flush()
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "best_mae": best_mae,
                 "stale_validations": stale_validations,
                 "rng_state": capture_rng_state()}, output_dir / "last.pth",
            )
            if supervision == "pml":
                loss_details = (f"pml_point={train['pml_point']:.4f} "
                                f"pml_transport={train['pml_transport']:.4f} "
                                f"monitor_count_L1={train['count']:.4f} "
                                f"count_weight={config.get('pml_count_weight', 0.0):.4f}")
            elif supervision == "gaussian":
                loss_details = (f"density_weighted={train['gaussian']:.4f} "
                                f"count_L1={train['count']:.4f} "
                                f"count_weight={config.get('gaussian_count_weight', 0.002):.4f}")
            else:
                loss_details = f"count_L1={train['count']:.4f}"
            print(f"epoch={epoch:03d} train={train['loss']:.4f} {loss_details} "
                  f"val_MAE={row['val_mae']} best={best_mae if validate else 'n/a'}", flush=True)
            if (validate and not args.no_early_stop and config["patience_validations"]
                    and stale_validations >= config["patience_validations"]):
                print("early stopping on validation MAE", flush=True)
                break


if __name__ == "__main__":
    main()
