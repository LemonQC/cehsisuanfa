#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
CACHE="outputs/b01_b02_robot_only_cache_v120_smoke"
B01_OUT="outputs/research_runs/b01_smoke_v120_fullval_v2"
B02_OUT="outputs/research_runs/b02_smoke_v120_fullval_v2"
LOG_DIR="outputs/logs/b01_b02_fullval_smoke_v2"

if [[ ! -f "${CACHE}/manifest.json" ]]; then
  echo "ERROR: missing prepared cache: ${CACHE}/manifest.json" >&2
  exit 2
fi
for path in "$B01_OUT" "$B02_OUT" "$LOG_DIR"; do
  if [[ -e "$path" ]]; then
    echo "ERROR: refusing to overwrite existing path: $path" >&2
    exit 3
  fi
done

mkdir -p "$LOG_DIR"
echo "TRO1_B01_B02_FULLVAL_V2_START"
echo "repo=$(git rev-parse --short HEAD)"
echo "cache=$CACHE"
echo "python=$(python --version 2>&1)"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

python -m unittest discover -s tests -p 'test_robot_only_reimpl.py' -v \
  2>&1 | tee "$LOG_DIR/tests.log"

python -m agcofl.robot_only_reimpl train \
  --method b01 \
  --cache-root "$CACHE" \
  --out "$B01_OUT" \
  --steps 100 --batch 16 --allow-small-dev \
  2>&1 | tee "$LOG_DIR/b01.log"

python -m agcofl.robot_only_reimpl train \
  --method b02 \
  --cache-root "$CACHE" \
  --out "$B02_OUT" \
  --steps 100 --batch 16 --allow-small-dev \
  2>&1 | tee "$LOG_DIR/b02.log"

python - <<'PY'
import json
from pathlib import Path

runs = {
    "b01": Path("outputs/research_runs/b01_smoke_v120_fullval_v2"),
    "b02": Path("outputs/research_runs/b02_smoke_v120_fullval_v2"),
}
for name, root in runs.items():
    summary_path = root / "train_summary.json"
    checkpoint = root / "best.pt"
    if not summary_path.is_file() or not checkpoint.is_file():
        raise SystemExit(f"{name}: missing summary/checkpoint")
    data = json.loads(summary_path.read_text())
    assert data["status"] == "TRAINED_NOT_PHYSX_EVALUATED", data
    assert data["data_label"] == "SMALL_DEV_DATA_ONLY", data
    assert data["validation_protocol"] == "FULL_VALIDATION_DEMO_WINDOWS_FIXED_DDPM_NOISE", data
    assert data["validation_windows"] > 0, data
    print("TRO1_FULLVAL_SMOKE_VERIFIED", json.dumps({
        "method": name,
        "steps": data["steps"],
        "validation_windows": data["validation_windows"],
        "best_val_loss": data["best_val_loss"],
        "checkpoint_bytes": checkpoint.stat().st_size,
        "status": data["status"],
        "data_label": data["data_label"],
    }))
print("Logs:", "outputs/logs/b01_b02_fullval_smoke_v2")
PY
