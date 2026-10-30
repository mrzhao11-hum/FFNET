#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
gpu="${SMOKE_GPU:-2}"
config=configs/carpk_ntu162_vehicle_a3_gaussian_h5_seed42.json
"$python_bin" -m unittest tests.test_ffnet_gaussian_h5 tests.test_ffnet_gaussian_targets \
  tests.test_ffnet_gaussian_loss tests.test_ffnet_vehicle_modules
"$python_bin" -m vehicle_ffnet.prepare_gaussian_h5 --config "$config" --limit 2
run="outputs/ffnet_vehicle_supervision_ntu162_seed42/smoke_h5_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m vehicle_ffnet.train \
  --config "$config" --device cuda:0 --workers 0 --limit 2 --epochs 1 \
  --no-pretrained --output-dir "$run"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m vehicle_ffnet.probe_gaussian_grad \
  --config "$config" --checkpoint "$run/best_mae.pth" --device cuda:0 --samples 2
echo "Smoke passed: $run"
