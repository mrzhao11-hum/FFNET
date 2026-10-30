#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
split="${1:-val}"
if [[ "$split" != val && "$split" != test ]]; then
  echo 'usage: evaluate_ffnet_vehicle_modules.sh [val|test]' >&2
  exit 2
fi
python_bin="${PYTHON_BIN:-python}"
gpu="${EVAL_GPU:-0}"
root="outputs/ffnet_vehicle_modules_ntu162_seed42"
analysis="outputs/analysis/ffnet_vehicle_modules_ntu162_seed42_${split}"
arms=(a1_lde a2_vftm a3_lde_vftm a4_lde_vftm_gate)
for arm in "${arms[@]}"; do
  run="$root/$arm"
  if [[ ! -f "$run/resolved_config.json" || ! -f "$run/best_mae.pth" ]]; then
    echo "Missing config or best checkpoint for $arm" >&2
    exit 1
  fi
done
for arm in "${arms[@]}"; do
  run="$root/$arm"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m vehicle_ffnet.evaluate \
    --config "$run/resolved_config.json" --checkpoint "$run/best_mae.pth" \
    --split "$split" --inference-mode full --device cuda:0 --workers 2 \
    --output-dir "$analysis/$arm"
done
