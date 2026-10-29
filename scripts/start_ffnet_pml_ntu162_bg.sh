#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
config=configs/carpk_ntu162_group_all_full_pml.json
run=outputs/carpk_ffnet_ntu162_pml_full_seed42
gpu="${GPU:-3}"
python_bin="${PYTHON_BIN:-python}"

for required in "$config" \
  data/manifests/CARPK/train.json \
  data/manifests/CARPK/val.json \
  data/manifests/CARPK/test.json \
  vehicle_ffnet/pml_loss.py; do
  [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done
"$python_bin" - <<'PY'
import json
from pathlib import Path
root = Path('data/manifests/CARPK')
groups = {}
for split, expected in (('train', 827), ('val', 162), ('test', 459)):
    rows = json.loads((root / f'{split}.json').read_text(encoding='utf-8'))['samples']
    ids = {row['id'] for row in rows}
    if len(rows) != expected or len(ids) != expected:
        raise SystemExit(f'{split}: expected {expected} unique images, got {len(rows)}')
    groups[split] = ids
if any(groups[a] & groups[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
    raise SystemExit('Overlapping CARPK train/val/test manifests')
print('CARPK split verified: 827 train, 162 val, 459 test')
PY

if pgrep -u "$(id -u)" -af '[p]ython.*-m vehicle_ffnet.train' | grep -F -- "$config" >/dev/null; then
  echo "Already running: $config" >&2
  exit 1
fi
if [[ -f "$run/history.csv" && ! -f "$run/last.pth" ]]; then
  echo "History exists without a resumable checkpoint at $run" >&2
  exit 1
fi
resume=()
if [[ -f "$run/last.pth" ]]; then
  resume=(--resume "$run/last.pth")
fi
mkdir -p "$run/logs"
log="$run/logs/train.log"
printf '\n=== %s GPU=%s config=%s ===\n' "$(date -Is)" "$gpu" "$config" >> "$log"
nohup env CUDA_VISIBLE_DEVICES="$gpu" \
  "$python_bin" -u -m vehicle_ffnet.train \
  --config "$config" --device cuda:0 --workers 2 --epochs 100 \
  --no-early-stop "${resume[@]}" \
  >> "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run/logs/train.pid"
echo "Started NTU-162 PML run on physical GPU $gpu, PID $pid"
echo "Log: $log"
