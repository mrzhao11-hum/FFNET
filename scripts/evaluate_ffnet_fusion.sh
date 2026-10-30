#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
split="${1:-val}"
[[ "$split" == val || "$split" == test ]] || { echo 'usage: evaluate_ffnet_fusion.sh [val|test]' >&2; exit 2; }
python_bin="${PYTHON_BIN:-python}"
root="outputs/ffnet_fusion_ntu162_seed42"
read -r -a arms <<< "${FUSION_ARMS:-c1_channel c2_spatial c3_semantic}"
for arm in "${arms[@]}"; do
  case "$arm" in c1_channel|c2_spatial|c3_semantic) ;; *) echo "Unknown arm: $arm" >&2; exit 2;; esac
  [[ -f "$root/$arm/resolved_config.json" && -f "$root/$arm/best_mae.pth" ]] || { echo "Missing trained run: $arm" >&2; exit 1; }
done
for arm in "${arms[@]}"; do
  run="$root/$arm"; out="outputs/analysis/ffnet_fusion_ntu162_seed42_${split}/$arm"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-1}" "$python_bin" -m vehicle_ffnet.evaluate \
    --config "$run/resolved_config.json" --checkpoint "$run/best_mae.pth" \
    --split "$split" --inference-mode full --device cuda:0 --workers "${WORKERS:-2}" --output-dir "$out"
  if [[ "${MASS_DIAG:-0}" == 1 ]]; then
    CUDA_VISIBLE_DEVICES="${EVAL_GPU:-1}" "$python_bin" -m vehicle_ffnet.diagnose_instance_mass \
      --config "$run/resolved_config.json" --checkpoint "$run/best_mae.pth" \
      --split "$split" --device cuda:0 --output-dir "$out/mass"
  fi
done
