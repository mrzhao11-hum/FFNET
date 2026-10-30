#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
"$python_bin" scripts/check_ffnet_detail_configs.py
"$python_bin" -m unittest tests.test_ffnet_detail_resolution tests.test_ffnet_vehicle_modules tests.test_ffnet_instance_mass_diagnosis
root="outputs/ffnet_detail_resolution_ntu162_seed42"
read -r -a arms <<< "${DETAIL_ARMS:-b1_detail b2_self_gate b3_semantic_gate d4_detail}"
pretrained_args=()
[[ "${SMOKE_NO_PRETRAINED:-0}" != 1 ]] || pretrained_args=(--no-pretrained)
for arm in "${arms[@]}"; do
  config="configs/carpk_ntu162_${arm}_seed42.json"
  CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-2}" "$python_bin" -u -m vehicle_ffnet.probe_pml --config "$config" --device cuda:0 "${pretrained_args[@]}"
  CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-2}" "$python_bin" -u -m vehicle_ffnet.train \
    --config "$config" --device cuda:0 --workers 0 --limit 2 --epochs 1 "${pretrained_args[@]}" \
    --output-dir "$root/smoke_${arm}_$(date +%Y%m%d_%H%M%S)"
done
