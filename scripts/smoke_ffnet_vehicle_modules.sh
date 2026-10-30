#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
gpu="${SMOKE_GPU:-0}"
smoke_root="${SMOKE_RUN_ROOT:-outputs/ffnet_vehicle_modules_smoke_seed42}"
"$python_bin" scripts/check_ffnet_vehicle_configs.py
"$python_bin" -m unittest tests.test_ffnet_vehicle_modules
for arm in a1_lde a2_vftm a3_lde_vftm a4_lde_vftm_gate; do
  if [[ -e "$smoke_root/$arm/last.pth" || -e "$smoke_root/$arm/history.csv" ]]; then
    echo "Existing smoke output at $smoke_root/$arm; choose a new SMOKE_RUN_ROOT" >&2
    exit 1
  fi
done
for arm in a1_lde a2_vftm a3_lde_vftm a4_lde_vftm_gate; do
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m vehicle_ffnet.probe_pml \
    --config "configs/carpk_ntu162_vehicle_${arm}_seed42.json" --device cuda:0
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m vehicle_ffnet.train \
    --config "configs/carpk_ntu162_vehicle_${arm}_seed42.json" \
    --device cuda:0 --workers 0 --limit 2 --epochs 1 --no-pretrained \
    --output-dir "$smoke_root/$arm"
done
