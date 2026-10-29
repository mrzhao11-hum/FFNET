"""Verify that an FFNet run can resume its model, optimizer, and epoch."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    checkpoint_path = args.run_dir / "last.pth"
    history_path = args.run_dir / "history.csv"
    resolved_path = args.run_dir / "resolved_config.json"
    for path in (checkpoint_path, history_path, resolved_path, args.config):
        if not path.is_file():
            raise FileNotFoundError(f"resume prerequisite missing: {path}")

    current = json.loads(args.config.read_text(encoding="utf-8"))
    previous = json.loads(resolved_path.read_text(encoding="utf-8"))
    fixed_keys = (
        "dataset", "validation_protocol", "data_root", "manifest_dir", "output_dir",
        "seed", "attention_norm", "train_mode", "batch_size", "lr",
        "weight_decay", "wot", "wtv", "ot_regularization", "ot_iterations",
        "ot_reference_size",
    )
    changed = [key for key in fixed_keys if previous.get(key) != current.get(key)]
    if changed:
        raise ValueError(f"baseline config differs from saved run: {', '.join(changed)}")

    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty history: {history_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = ("epoch", "model_state_dict", "optimizer_state_dict", "best_mae")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint missing resume state: {', '.join(missing)}")
    epoch = checkpoint["epoch"]
    if not isinstance(epoch, int) or epoch < 1 or epoch > args.epochs:
        raise ValueError(f"checkpoint epoch outside 1..{args.epochs}: {epoch}")
    if int(rows[-1]["epoch"]) != epoch:
        raise ValueError(
            f"history last epoch {rows[-1]['epoch']} differs from checkpoint epoch {epoch}"
        )
    if not math.isfinite(float(checkpoint["best_mae"])):
        raise ValueError("checkpoint best_mae is not finite")
    if not checkpoint["model_state_dict"]:
        raise ValueError("checkpoint has no model weights")
    optimizer = checkpoint["optimizer_state_dict"]
    if not optimizer.get("state") or not optimizer.get("param_groups"):
        raise ValueError("checkpoint has no trained optimizer state")

    print(
        f"Resume verified: completed_epoch={epoch}, next_epoch={epoch + 1}, "
        f"target_epoch={args.epochs}, optimizer_tensors={len(optimizer['state'])}, "
        f"best_val_MAE={float(checkpoint['best_mae']):.4f}, "
        f"rng_state={'present' if 'rng_state' in checkpoint else 'absent_in_old_checkpoint'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
