#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
"$python_bin" scripts/check_ffnet_fusion_configs.py
# All temporary test/checkpoint files stay under the project and are removed.
root="outputs/ffnet_fusion_ntu162_seed42"
mkdir -p "$root"
smoke_root=$(mktemp -d "$root/.smoke_XXXXXXXX")
trap 'rm -rf -- "$smoke_root"' EXIT
mkdir -p "$smoke_root/tmp"
export TMPDIR="$PWD/$smoke_root/tmp"
export PYTHONDONTWRITEBYTECODE=1
"$python_bin" -m unittest tests.test_ffnet_fusion_modules tests.test_ffnet_vehicle_modules tests.test_ffnet_detail_resolution
pretrained_args=()
[[ "${SMOKE_NO_PRETRAINED:-0}" != 1 ]] || pretrained_args=(--no-pretrained)
for arm in c1_channel c2_spatial c3_semantic; do
  config="configs/carpk_ntu162_fusion_${arm}_seed42.json"
  CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-1}" "$python_bin" -u -m vehicle_ffnet.probe_fusion \
    --config "$config" --device cuda:0 "${pretrained_args[@]}"
  CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-1}" "$python_bin" -u -m vehicle_ffnet.train \
    --config "$config" --device cuda:0 --workers 0 --limit 2 --epochs 1 \
    --output-dir "$smoke_root/$arm" "${pretrained_args[@]}"
done
