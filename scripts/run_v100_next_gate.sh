#!/usr/bin/env bash
# v1.0 SCIENCE-FIRST, ONE COMMAND. Does not reinstall, run new physics,
# or train another net on the official evaluation clip 0080.
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4

OUT=outputs/next_gate_v100
mkdir -p "$OUT"

python -m pytest -q tests/test_analyze_v090.py tests/test_effect_projection_gate.py

# 1. Actual MP4 mask learning: baseline vs previously trained three modes.
python -m agcofl.analyze_v090 \
  --source outputs/real_human_supervision_v090 --out "$OUT"

# 2. The existing PhysX object trace is only a source-XYZ interface sanity
# check; it is NOT a human Pick source and its contact mode is UNKNOWN.
python -m agcofl.effect_projection_gate \
  --from-saved-robot-trace outputs/roboreel_cn/fixed/episode_0000 \
  --out "$OUT/effect_gate.json"

# 3. Discover actual upstream TRAIN files, not eval human IDs 0080--0099.
# Default 6 real train front clips if present, <=300 MB. Timeout bounded.
if command -v timeout >/dev/null 2>&1; then
  if ! timeout 150s env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
    python -u -m agcofl.train_data_discovery \
      --out "$OUT/train_data" --download 6 --max-mb 300; then
    echo "TRAIN_DISCOVERY_TIMEOUT_OR_ERROR: data source not assumed accessible"
  fi
else
  echo "TRAIN_DISCOVERY_SKIPPED: timeout unavailable"
fi

# 4. If actual training MP4s arrived, create visual overlay + numerical
# quality audit for each. Otherwise explicit DATA_BLOCKED status.
python -u -m agcofl.train_video_audit \
  --index "$OUT/train_data/downloaded_train_index.json" --out "$OUT"

# 5. Automatically push ONLY small derived JSON/CSV research records to the
# user's GitHub. Never push MP4/ZIP/PT or human source footage.
if bash scripts/publish_v100_report.sh; then
  echo "SMALL_REPORT_PUBLISH_SUCCEEDED_OR_UNCHANGED"
else
  echo "SMALL_REPORT_PUBLISH_NOT_COMPLETE: local files retained in $OUT; inspect AUTOPUBLISH status above"
fi

echo "======================================"
echo "V100_GATE_COMPLETE"
echo "Local JSON: $OUT/learnability_gate.json"
echo "Local JSON: $OUT/effect_gate.json"
echo "Train dataset discovery: $OUT/train_data/train_discovery.json"
echo "Training mask audit: $OUT/train_video_audit.json"
echo "If accessible: https://github.com/LemonQC/cehsisuanfa/tree/main/research_reports/2026-11-09_v100"
echo "======================================"
