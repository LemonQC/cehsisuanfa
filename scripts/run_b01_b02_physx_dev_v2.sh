#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
UPSTREAM="third_party/roboreel_models"
B01_RUN="outputs/research_runs/b01_smoke_v120_fullval_v2"
B02_RUN="outputs/research_runs/b02_smoke_v120_fullval_v2"
B01_OUT="outputs/research_runs/b01_physx_dev_fullval_v2"
B02_OUT="outputs/research_runs/b02_physx_dev_fullval_v2"
LOG_DIR="outputs/logs/b01_b02_physx_dev_v2"

for f in \
  "$UPSTREAM/src/evaluation/utils.py" \
  "$B01_RUN/best.pt" "$B01_RUN/train_summary.json" \
  "$B02_RUN/best.pt" "$B02_RUN/train_summary.json"; do
  [[ -f "$f" ]] || { echo "ERROR: required file missing: $f" >&2; exit 2; }
done
for path in "$B01_OUT" "$B02_OUT" "$LOG_DIR"; do
  [[ ! -e "$path" ]] || { echo "ERROR: refusing to overwrite: $path" >&2; exit 3; }
done

python - <<'PY'
import json
from pathlib import Path
for name in ("b01", "b02"):
    p=Path(f"outputs/research_runs/{name}_smoke_v120_fullval_v2/train_summary.json")
    d=json.loads(p.read_text())
    assert d["status"] == "TRAINED_NOT_PHYSX_EVALUATED", (name,d)
    assert d["data_label"] == "SMALL_DEV_DATA_ONLY", (name,d)
    assert d["validation_protocol"] == "FULL_VALIDATION_DEMO_WINDOWS_FIXED_DDPM_NOISE", (name,d)
    assert d["validation_windows"] > 0, (name,d)
    print("TRO1_PHYSX_DEV_CHECKPOINT_GATE", name, "PASS", d["validation_windows"])
PY

mkdir -p "$LOG_DIR"
echo "TRO1_B01_B02_PHYSX_DEV_V2_START"
echo "upstream=$UPSTREAM"
echo "device:"
python - <<'PY'
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

python -m agcofl.robot_only_reimpl eval \
  --checkpoint "$B01_RUN/best.pt" \
  --upstream "$UPSTREAM" \
  --out "$B01_OUT" \
  --mode dev --episodes 2 --max-steps 800 --save-video \
  2>&1 | tee "$LOG_DIR/b01.log"

python -m agcofl.robot_only_reimpl eval \
  --checkpoint "$B02_RUN/best.pt" \
  --upstream "$UPSTREAM" \
  --out "$B02_OUT" \
  --mode dev --episodes 2 --max-steps 800 --save-video \
  2>&1 | tee "$LOG_DIR/b02.log"

python - <<'PY'
import json
from pathlib import Path
for name in ("b01", "b02"):
    root=Path(f"outputs/research_runs/{name}_physx_dev_fullval_v2")
    p=root/"results.json"
    if not p.is_file():
        raise SystemExit(f"{name}: missing results.json")
    d=json.loads(p.read_text())
    assert d["status"] == "DEV_PHYSX_ONLY", d
    assert d["trials"] == 2, d
    videos=sorted(root.glob("rollout_*.mp4"))
    if len(videos) != 2:
        raise SystemExit(f"{name}: expected 2 rollout videos, found {len(videos)}")
    print("TRO1_PHYSX_DEV_RESULT", json.dumps({
        "method": name,
        "successes": d["successes"],
        "trials": d["trials"],
        "env_id": d["official_env_id"],
        "videos": [v.name for v in videos],
        "status": d["status"],
    }))
print("Logs:", "outputs/logs/b01_b02_physx_dev_v2")
PY
