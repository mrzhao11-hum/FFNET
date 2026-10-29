#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Vehicle Counting"

RUN_DIR="outputs/carpk_ffnet_gf2_val_group_all_full_seed42"
LOG_DIR="$RUN_DIR/logs"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ffnet310/bin/python}"
CONFIG="configs/carpk_gf2_val_group_all_full.json"
GPU_ID="${GPU_ID:-3}"

mkdir -p "$LOG_DIR"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
"$PYTHON_BIN" scripts/check_ffnet_resume.py \
  --run-dir "$RUN_DIR" --config "$CONFIG" --epochs 100

if pgrep -u "$(id -u)" -f '[p]ython.*-m vehicle_ffnet\.train.*carpk_gf2_val_group_all_full\.json' >/dev/null; then
  echo "The FFNet GF2 full-image run is already active; no second process was started." >&2
  exit 1
fi

RESUME_ARGS=(--resume "$RUN_DIR/last.pth")
echo "Resuming from $RUN_DIR/last.pth"

LOG_FILE="$LOG_DIR/train.log"
nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" \
  "$PYTHON_BIN" -u -m vehicle_ffnet.train \
  --config "$CONFIG" \
  --device cuda:0 \
  --workers 2 \
  --epochs 100 \
  --no-early-stop \
  "${RESUME_ARGS[@]}" \
  >> "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
printf '%s\n' "$PID" > "$LOG_DIR/train.pid"
echo "Started PID=$PID on physical GPU $GPU_ID (cuda:0 inside the process)"
echo "Log: $LOG_FILE"
