#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Vehicle Counting"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ffnet310/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
for required in \
  configs/carpk_gf2_val_group_all_full.json \
  configs/carpk_gf2_val_group_all_full_pml.json \
  configs/carpk_gf2_val_group_all_full_gate_ot.json \
  vehicle_ffnet/pml_loss.py \
  vehicle_ffnet/spatial_gate.py \
  vehicle_ffnet/probe_pml.py \
  scripts/check_ffnet_resume.py \
  data/manifests/CARPK_GF2_val/train.json \
  data/manifests/CARPK_GF2_val/val.json \
  outputs/carpk_ffnet_gf2_val_group_all_full_seed42/last.pth; do
  if [[ ! -f "$required" ]]; then
    echo "Required file missing: $required" >&2
    exit 1
  fi
done
if ! grep -q -- '--no-early-stop' vehicle_ffnet/train.py || \
   ! grep -q 'fusion_mode' vehicle_ffnet/model.py; then
  echo "Incomplete FFNet sync: train.py or model.py is an older version" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/check_ffnet_resume.py \
  --run-dir outputs/carpk_ffnet_gf2_val_group_all_full_seed42 \
  --config configs/carpk_gf2_val_group_all_full.json --epochs 100

start_arm() {
  local gpu="$1" config="$2" run_dir="$3"
  local log_dir="$run_dir/logs"
  mkdir -p "$log_dir"

  if pgrep -u "$(id -u)" -af '[p]ython.*-m vehicle_ffnet.train' | grep -F -- "$config" >/dev/null; then
    echo "Already running: $config"
    return 0
  fi
  if [[ -f "$run_dir/history.csv" && ! -f "$run_dir/last.pth" ]]; then
    echo "History exists without last.pth at $run_dir; inspect this run before restarting." >&2
    return 1
  fi

  local -a resume_args=()
  if [[ -f "$run_dir/last.pth" ]]; then
    resume_args=(--resume "$run_dir/last.pth")
  fi
  local log_file="$log_dir/train.log"
  printf '\n=== %s GPU=%s config=%s ===\n' "$(date -Is)" "$gpu" "$config" >> "$log_file"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" -u -m vehicle_ffnet.train \
    --config "$config" --device cuda:0 --workers 2 --epochs 100 \
    --no-early-stop "${resume_args[@]}" \
    >> "$log_file" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$log_dir/train.pid"
  echo "Started GPU $gpu, PID $pid: $config"
  echo "  log: $log_file"
}

# GPU 0 belongs to another user and is deliberately not used.
start_arm 3 configs/carpk_gf2_val_group_all_full.json \
  outputs/carpk_ffnet_gf2_val_group_all_full_seed42
start_arm 1 configs/carpk_gf2_val_group_all_full_pml.json \
  outputs/carpk_ffnet_gf2_pml_full_seed42
start_arm 2 configs/carpk_gf2_val_group_all_full_gate_ot.json \
  outputs/carpk_ffnet_gf2_gate_ot_full_seed42
