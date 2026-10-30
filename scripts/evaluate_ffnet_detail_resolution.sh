#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
split="${1:-val}"
[[ "$split" == val || "$split" == test ]] || exit 2
python_bin="${PYTHON_BIN:-python}"
root="outputs/ffnet_detail_resolution_ntu162_seed42"
read -r -a arms <<< "${DETAIL_ARMS:-b1_detail b2_self_gate b3_semantic_gate d4_detail}"
for arm in "${arms[@]}"; do
  run="$root/$arm"
  [[ -f "$run/best_mae.pth" && -f "$run/resolved_config.json" ]] || { echo "Missing trained run: $arm" >&2; exit 1; }
done
for arm in "${arms[@]}"; do
  run="$root/$arm"; out="outputs/analysis/ffnet_detail_resolution_ntu162_seed42_${split}/$arm"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-2}" "$python_bin" -m vehicle_ffnet.evaluate \
    --config "$run/resolved_config.json" --checkpoint "$run/best_mae.pth" \
    --split "$split" --inference-mode full --device cuda:0 --workers 2 --output-dir "$out"
  if [[ "${MASS_DIAG:-0}" == 1 ]]; then
    CUDA_VISIBLE_DEVICES="${EVAL_GPU:-2}" "$python_bin" -m vehicle_ffnet.diagnose_instance_mass \
      --config "$run/resolved_config.json" --checkpoint "$run/best_mae.pth" \
      --split "$split" --device cuda:0 --output-dir "$out/mass"
  fi
done
