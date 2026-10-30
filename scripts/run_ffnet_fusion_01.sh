#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
root="outputs/ffnet_fusion_ntu162_seed42"
arms=(c1_channel c2_spatial c3_semantic)
# Repeated IDs intentionally mean concurrent jobs on the same physical GPU.
read -r -a gpus <<< "${FUSION_GPUS:-0 0 1}"
[[ ${#gpus[@]} -eq 3 ]] || { echo 'FUSION_GPUS needs three IDs, e.g. "0 0 1"' >&2; exit 2; }
mkdir -p "$root/logs"
mkdir "$root/.launch_lock" || { echo 'Another launcher holds .launch_lock; inspect it first.' >&2; exit 1; }
trap 'rmdir "$root/.launch_lock"' EXIT
"$python_bin" scripts/check_ffnet_fusion_configs.py
for gpu in "${gpus[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: $gpu" >&2; exit 2; }
done
for gpu in $(printf '%s\n' "${gpus[@]}" | sort -u); do
  echo "Physical GPU $gpu: free / total memory (MiB); occupancy is informational"
  nvidia-smi -i "$gpu" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits
done
# Check every run before starting any job. No existing GPU process is killed.
for arm in "${arms[@]}"; do
  run="$root/$arm"
  if [[ -f "$run/train.pid" ]] && kill -0 "$(cat "$run/train.pid")" 2>/dev/null; then
    echo "Already running: $arm" >&2; exit 1
  fi
  if [[ "${RESUME:-0}" == 1 ]]; then
    [[ -f "$run/last.pth" ]] || { echo "Missing checkpoint: $run/last.pth" >&2; exit 1; }
  elif [[ -e "$run/history.csv" || -e "$run/last.pth" || -e "$run/best_mae.pth" ]]; then
    echo "Existing run $run; use RESUME=1 or inspect it first." >&2; exit 1
  fi
done
for i in 0 1 2; do
  arm="${arms[$i]}"; gpu="${gpus[$i]}"
  run="$root/$arm"; log="$root/logs/$arm.log"
  mkdir -p "$run"
  resume_args=()
  [[ "${RESUME:-0}" != 1 ]] || resume_args=(--resume "$run/last.pth")
  printf '\n=== %s arm=%s physical_gpu=%s resume=%s ===\n' "$(date -Is)" "$arm" "$gpu" "${RESUME:-0}" >> "$log"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m vehicle_ffnet.train \
    --config "configs/carpk_ntu162_fusion_${arm}_seed42.json" --device cuda:0 \
    --workers "${WORKERS:-2}" --epochs "${EPOCHS:-200}" \
    --early-stop-patience "${PATIENCE:-30}" "${resume_args[@]}" >> "$log" 2>&1 < /dev/null &
  pid=$!; printf '%s\n' "$pid" > "$run/train.pid"
  echo "Started $arm on physical GPU $gpu, PID=$pid; log=$log"
done
