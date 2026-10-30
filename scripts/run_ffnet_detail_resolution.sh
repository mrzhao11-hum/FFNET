#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
root="outputs/ffnet_detail_resolution_ntu162_seed42"
if [[ "${1:-}" == --worker ]]; then
  gpu="$2"; shift 2
  for arm in "$@"; do
    run="$root/$arm"; log="$root/logs/$arm.log"
    mkdir -p "$run"
    resume_args=()
    [[ "${RESUME:-0}" != 1 ]] || resume_args=(--resume "$run/last.pth")
    printf '\n=== %s arm=%s physical_gpu=%s resume=%s ===\n' "$(date -Is)" "$arm" "$gpu" "${RESUME:-0}" >> "$log"
    env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m vehicle_ffnet.train \
      --config "configs/carpk_ntu162_${arm}_seed42.json" --device cuda:0 \
      --workers "${WORKERS:-2}" --epochs "${EPOCHS:-200}" \
      --early-stop-patience "${PATIENCE:-30}" "${resume_args[@]}" >> "$log" 2>&1 < /dev/null &
    pid=$!; printf '%s\n' "$pid" > "$run/train.pid"
    wait "$pid"
  done
  exit
fi
if [[ -z "${DETAIL_GPUS:-}" ]]; then
  echo 'Set DETAIL_GPUS to available physical GPU IDs. Four IDs run concurrently; fewer IDs queue the remaining arms. GPU 0 is never selected automatically.' >&2
  exit 2
fi
read -r -a gpus <<< "$DETAIL_GPUS"
read -r -a arms <<< "${DETAIL_ARMS:-b1_detail b2_self_gate b3_semantic_gate d4_detail}"
[[ ${#gpus[@]} -ge 1 && ${#gpus[@]} -le 4 ]] || exit 2
[[ ${#arms[@]} -ge 1 && ${#arms[@]} -le 4 ]] || exit 2
[[ $(printf '%s\n' "${gpus[@]}" | sort -u | wc -l) -eq ${#gpus[@]} ]] || { echo 'Duplicate GPU ID' >&2; exit 2; }
[[ $(printf '%s\n' "${arms[@]}" | sort -u | wc -l) -eq ${#arms[@]} ]] || exit 2
mkdir -p "$root/logs"
# Prevent two launchers from validating/starting the same experiment concurrently.
mkdir "$root/.launch_lock" || { echo 'Another launcher holds .launch_lock; inspect running launchers.' >&2; exit 1; }
trap 'rmdir "$root/.launch_lock"' EXIT
"$python_bin" scripts/check_ffnet_detail_configs.py
for gpu in "${gpus[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
  # Show capacity without rejecting GPUs that already have other tasks.
  echo "Physical GPU $gpu: free / total memory (MiB)"
  nvidia-smi -i "$gpu" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits
  worker_pid="$root/logs/worker_gpu${gpu}.pid"
  if [[ -f "$worker_pid" ]] && kill -0 "$(cat "$worker_pid")" 2>/dev/null; then echo "Worker already active on GPU $gpu" >&2; exit 1; fi
done
for arm in "${arms[@]}"; do
  case "$arm" in b1_detail|b2_self_gate|b3_semantic_gate|d4_detail) ;; *) echo "Unknown arm $arm" >&2; exit 2;; esac
  run="$root/$arm"
  if [[ -f "$run/train.pid" ]] && kill -0 "$(cat "$run/train.pid")" 2>/dev/null; then echo "Still running: $arm" >&2; exit 1; fi
  if [[ "${RESUME:-0}" == 1 ]]; then
    [[ -f "$run/last.pth" ]] || { echo "Missing resume checkpoint: $run" >&2; exit 1; }
  elif [[ -e "$run/last.pth" || -e "$run/history.csv" ]]; then
    echo "Existing run $run; use RESUME=1 to continue." >&2; exit 1
  fi
done
for ((slot=0; slot<${#gpus[@]}; slot++)); do
  assigned=()
  for ((i=slot; i<${#arms[@]}; i+=${#gpus[@]})); do assigned+=("${arms[$i]}"); done
  [[ ${#assigned[@]} -gt 0 ]] || continue
  gpu="${gpus[$slot]}"
  worker_pid="$root/logs/worker_gpu${gpu}.pid"
  nohup bash "$0" --worker "$gpu" "${assigned[@]}" >> "$root/logs/worker_gpu${gpu}.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$worker_pid"
  echo "Physical GPU $gpu: ${assigned[*]} (worker PID $!); logs: $root/logs/"
done
