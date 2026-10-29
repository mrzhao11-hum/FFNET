#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
split="${1:-val}"
case "$split" in val|test) ;; *) echo 'usage: evaluate_ffnet_ntu162_baselines.sh [val|test]' >&2; exit 2 ;; esac
python_bin="${PYTHON_BIN:-python}"
gpu="${EVAL_GPU:-2}"
root=outputs/ffnet_baselines_ntu162_seed42
for name in ot pml; do
  run="$root/$name"
  [[ -f "$run/best_mae.pth" && -f "$run/resolved_config.json" ]] || {
    echo "Missing best checkpoint or config for $name at $run" >&2; exit 2;
  }
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m vehicle_ffnet.evaluate \
    --config "$run/resolved_config.json" --checkpoint "$run/best_mae.pth" \
    --split "$split" --inference-mode full --device cuda:0 --workers 2 \
    --output-dir "outputs/analysis/ffnet_baselines_ntu162_seed42_${split}/$name"
done
