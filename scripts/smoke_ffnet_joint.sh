#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
"$python_bin" scripts/check_ffnet_joint_configs.py
# All temporary test/checkpoint files stay under the project and are removed.
root="outputs/ffnet_joint_ntu162_seed42"
mkdir -p "$root"
smoke_root=$(mktemp -d "$root/.smoke_XXXXXXXX")
trap 'rm -rf -- "$smoke_root"' EXIT
mkdir -p "$smoke_root/tmp"
export TMPDIR="$PWD/$smoke_root/tmp"
export PYTHONDONTWRITEBYTECODE=1
"$python_bin" -m unittest tests.test_ffnet_joint tests.test_ffnet_fusion_modules tests.test_ffnet_detail_resolution
pretrained_args=()
[[ "${SMOKE_NO_PRETRAINED:-0}" != 1 ]] || pretrained_args=(--no-pretrained)
for arm in j1_b1_c1 j2_b1_c2 j3_b3_c1 j4_b3_c2; do
  config="configs/carpk_ntu162_joint_${arm}_seed42.json"
  CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-1}" "$python_bin" -u -m vehicle_ffnet.probe_fusion \
    --config "$config" --device cuda:0 "${pretrained_args[@]}"
  CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-1}" "$python_bin" -u -m vehicle_ffnet.train \
    --config "$config" --device cuda:0 --workers 0 --limit 2 --epochs 1 \
    --output-dir "$smoke_root/$arm" "${pretrained_args[@]}"
done
