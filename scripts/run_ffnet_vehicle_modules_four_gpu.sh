#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
root="outputs/ffnet_vehicle_modules_ntu162_seed42"
arms=(a1_lde a2_vftm a3_lde_vftm a4_lde_vftm_gate)

if [[ -z "${FFNET_VEHICLE_GPUS:-}" ]]; then
  echo 'Set FFNET_VEHICLE_GPUS to four available physical GPU IDs, e.g. "0 1 2 3"' >&2
  exit 2
fi
read -r -a gpus <<< "$FFNET_VEHICLE_GPUS"
if [[ "${#gpus[@]}" -ne 4 ]]; then
  echo 'FFNET_VEHICLE_GPUS must contain exactly four GPU IDs' >&2
  exit 2
fi
for i in 0 1 2 3; do
  if ! [[ "${gpus[$i]}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU ID: ${gpus[$i]}" >&2
    exit 2
  fi
  for ((j=i+1; j<4; j++)); do
    if [[ "${gpus[$i]}" == "${gpus[$j]}" ]]; then
      echo 'Each arm needs a different physical GPU' >&2
      exit 2
    fi
  done
done

"$python_bin" scripts/check_ffnet_vehicle_configs.py

# Validate all runs before starting any one of them.
for arm in "${arms[@]}"; do
  config="configs/carpk_ntu162_vehicle_${arm}_seed42.json"
  run="$root/$arm"
  if [[ ! -f "$config" ]]; then
    echo "Missing config: $config" >&2
    exit 1
  fi
  if pgrep -u "$(id -u)" -af '[p]ython.*-m vehicle_ffnet.train' | grep -F -- "$config" >/dev/null; then
    echo "Already running: $config" >&2
    exit 1
  fi
  if [[ "${RESUME:-0}" == 1 ]]; then
    if [[ ! -f "$run/last.pth" ]]; then
      echo "Cannot resume $arm: missing $run/last.pth" >&2
      exit 1
    fi
  elif [[ -e "$run/last.pth" || -e "$run/history.csv" ]]; then
    echo "Existing run at $run; set RESUME=1 or inspect it" >&2
    exit 1
  fi
done

mkdir -p "$root/logs"
for i in 0 1 2 3; do
  arm="${arms[$i]}"
  gpu="${gpus[$i]}"
  config="configs/carpk_ntu162_vehicle_${arm}_seed42.json"
  run="$root/$arm"
  log="$root/logs/$arm.log"
  mkdir -p "$run"
  resume_args=()
  if [[ "${RESUME:-0}" == 1 ]]; then
    resume_args=(--resume "$run/last.pth")
  fi
  printf '\n=== %s arm=%s physical_gpu=%s resume=%s ===\n' \
    "$(date -Is)" "$arm" "$gpu" "${RESUME:-0}" >> "$log"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m vehicle_ffnet.train \
    --config "$config" --device cuda:0 --workers 2 --epochs 200 \
    --early-stop-patience 30 "${resume_args[@]}" >> "$log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$run/train.pid"
  echo "Started $arm on physical GPU $gpu (PID $pid); log: $log"
done
