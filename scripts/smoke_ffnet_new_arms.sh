#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Vehicle Counting"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ffnet310/bin/python}"
test -x "$PYTHON_BIN"
for required in \
  vehicle_ffnet/pml_loss.py \
  vehicle_ffnet/spatial_gate.py \
  vehicle_ffnet/probe_pml.py \
  configs/carpk_gf2_val_group_all_full_pml.json \
  configs/carpk_gf2_val_group_all_full_gate_ot.json \
  tests/test_ffnet_three_gpu.py; do
  if [[ ! -f "$required" ]]; then
    echo "Incomplete FFNet sync: missing $required" >&2
    exit 1
  fi
done
if ! grep -q -- '--no-early-stop' vehicle_ffnet/train.py || \
   ! grep -q 'fusion_mode' vehicle_ffnet/model.py; then
  echo "Incomplete FFNet sync: train.py or model.py is an older version" >&2
  exit 1
fi

"$PYTHON_BIN" -m unittest tests.test_ffnet_three_gpu

STAMP="$(date +%Y%m%d_%H%M%S)"
pml_gpu="${SMOKE_GPU:-${PML_GPU:-1}}"
gate_gpu="${SMOKE_GPU:-${GATE_GPU:-2}}"
for arm in pml gate_ot; do
  config="configs/carpk_gf2_val_group_all_full_${arm}.json"
  if [[ "$arm" == pml ]]; then
    gpu="$pml_gpu"
  else
    gpu="$gate_gpu"
  fi
  echo "Smoke: $arm on physical GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u -m vehicle_ffnet.train \
    --config "$config" --device cuda:0 --workers 0 --limit 2 --epochs 1 \
    --no-early-stop --output-dir "outputs/smoke_ffnet_${arm}_${STAMP}"
done

CUDA_VISIBLE_DEVICES="$pml_gpu" "$PYTHON_BIN" -u -m vehicle_ffnet.probe_pml \
  --config configs/carpk_gf2_val_group_all_full_pml.json --device cuda:0
