#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
gpu="${GPU:-1}"
epochs="${EPOCHS:-200}"
patience="${PATIENCE:-30}"
config=configs/carpk_ntu162_vehicle_a3_gaussian_h5_softplus_seed42.json
root=outputs/ffnet_vehicle_supervision_ntu162_seed42
run="$root/a3_gaussian_h5_softplus"
mkdir -p "$root/logs"
pidfile="$root/logs/a3_gaussian_h5_softplus.pid"
if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
  echo "PID $(cat "$pidfile") is still active; inspect it before relaunching." >&2
  exit 1
fi
"$python_bin" -m vehicle_ffnet.prepare_gaussian_h5 --config "$config" --check-only
resume=()
if [[ -f "$run/last.pth" ]]; then
  resume=(--resume "$run/last.pth")
elif [[ -f "$run/history.csv" ]]; then
  echo "History exists without last.pth: $run" >&2
  exit 1
fi
log="$root/logs/a3_gaussian_h5_softplus.log"
printf '\n=== %s GPU=%s epochs=%s patience=%s ===\n' "$(date -Is)" "$gpu" "$epochs" "$patience" >> "$log"
nohup env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m vehicle_ffnet.train \
  --config "$config" --device cuda:0 --workers 2 --epochs "$epochs" \
  --early-stop-patience "$patience" "${resume[@]}" >> "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$pidfile"
echo "Started PID=$pid physical_GPU=$gpu"
echo "Log: $log"
