#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
arm="${1:-both}"
case "$arm" in ot|pml|both) ;; *) echo 'usage: run_ffnet_ntu162_baselines.sh [ot|pml|both]' >&2; exit 2 ;; esac

"$python_bin" scripts/check_ffnet_ntu162_baselines.py
"$python_bin" -m unittest tests.test_ffnet_three_gpu
root=outputs/ffnet_baselines_ntu162_seed42
mkdir -p "$root/logs"
patience="${EARLY_STOP_PATIENCE:-30}"
if ! [[ "$patience" =~ ^[0-9]+$ ]]; then
  echo 'EARLY_STOP_PATIENCE must be a nonnegative integer' >&2
  exit 2
fi
extra=(--early-stop-patience "$patience")
if [[ "${RUN_FULL_200:-0}" == 1 ]]; then
  extra+=(--no-early-stop)
fi
ot_gpu="${OT_GPU:-1}"
pml_gpu="${PML_GPU:-2}"
if [[ "$arm" == both && "$ot_gpu" == "$pml_gpu" ]]; then
  echo 'OT_GPU and PML_GPU must differ for concurrent training' >&2
  exit 2
fi

precheck_arm() {
  local name="$1"
  local config="configs/carpk_ntu162_ffnet_${name}_full_seed42.json"
  local run="$root/$name"
  if pgrep -u "$(id -u)" -af '[p]ython.*-m vehicle_ffnet.train' | grep -F -- "$config" >/dev/null; then
    echo "Already running: $config" >&2
    return 1
  fi
  if [[ -e "$run/last.pth" || -e "$run/history.csv" ]]; then
    if [[ "${RESUME:-0}" != 1 || ! -f "$run/last.pth" ]]; then
      echo "Existing run at $run. Set RESUME=1 with last.pth to continue, or inspect it." >&2
      return 1
    fi
  fi
}

start_arm() {
  local name="$1" gpu="$2"
  local config="configs/carpk_ntu162_ffnet_${name}_full_seed42.json"
  local run="$root/$name"
  local log="$root/logs/$name.log"
  mkdir -p "$run"
  local resume=()
  if [[ -f "$run/last.pth" ]]; then
    resume=(--resume "$run/last.pth")
  fi
  printf '\n=== %s arm=%s physical_gpu=%s max_epochs=200 patience=%s ===\n' \
    "$(date -Is)" "$name" "$gpu" "$patience" >> "$log"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m vehicle_ffnet.train \
    --config "$config" --device cuda:0 --workers 2 --epochs 200 \
    "${extra[@]}" "${resume[@]}" >> "$log" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$run/train.pid"
  echo "Started $name on physical GPU $gpu, PID $pid; log: $log"
}

case "$arm" in
  ot) precheck_arm ot ;;
  pml) precheck_arm pml ;;
  both) precheck_arm ot; precheck_arm pml ;;
esac

case "$arm" in
  ot) start_arm ot "$ot_gpu" ;;
  pml) start_arm pml "$pml_gpu" ;;
  both) start_arm ot "$ot_gpu"; start_arm pml "$pml_gpu" ;;
esac
