#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Vehicle Counting"
SPLIT="${1:-val}"
if [[ "$SPLIT" != val && "$SPLIT" != test ]]; then
  echo "Usage: bash scripts/evaluate_ffnet_three_gpu.sh [val|test]" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ffnet310/bin/python}"
EVAL_GPU="${EVAL_GPU:-2}"
test -x "$PYTHON_BIN"

evaluate_arm() {
  local name="$1" config="$2" run_dir="$3"
  local checkpoint="$run_dir/best_mae.pth"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint missing: $checkpoint" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON_BIN" -m vehicle_ffnet.evaluate \
    --config "$config" --checkpoint "$checkpoint" \
    --split "$SPLIT" --inference-mode full --device cuda:0 --workers 2 \
    --output-dir "outputs/analysis/ffnet_three_gpu_seed42_${SPLIT}/$name"
}

evaluate_arm e0_ot configs/carpk_gf2_val_group_all_full.json \
  outputs/carpk_ffnet_gf2_val_group_all_full_seed42
evaluate_arm e1_pml configs/carpk_gf2_val_group_all_full_pml.json \
  outputs/carpk_ffnet_gf2_pml_full_seed42
evaluate_arm e3_gate_ot configs/carpk_gf2_val_group_all_full_gate_ot.json \
  outputs/carpk_ffnet_gf2_gate_ot_full_seed42
