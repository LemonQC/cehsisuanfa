#!/usr/bin/env bash
# Zero additional downloads and zero additional Panda physics runs.
# Uses the already downloaded AUTHENTIC human MP4 and the already completed
# official RoboReel physical trace. No synthetic video or privileged labels.
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4

HUMAN=third_party/roboreel/human_demos/eval/no_distraction/push_cube/0080/front.mp4
PHYSICS=outputs/roboreel_cn/fixed/episode_0000
OUT=outputs/real_human_supervision_v090

if [ ! -s "$HUMAN" ]; then
  echo "MISSING_AUTHENTIC_HUMAN_VIDEO: $HUMAN" >&2
  exit 2
fi
if [ ! -f "$PHYSICS/physical_trace.json" ] || [ ! -f "$PHYSICS/episode_result.json" ]; then
  echo "MISSING_MEASURED_PANDA_PHYSICS_TRACE: $PHYSICS" >&2
  echo "Do not fabricate a robot video. Re-run a real physics episode before using replay." >&2
  exit 3
fi

python -m pytest -q tests/test_real_human_supervision.py tests/test_real_video_bridge.py

# Offline top-down XY replay from the SAVED real physics trace.
python -m agcofl.physics_replay --episode-dir "$PHYSICS"

# Train three true-video supervision variants, identical model/data/init/seed.
# Exactly one existing real human demo: train-only, NO held-out test yet.
python -u -m agcofl.real_human_supervision \
  --video-root third_party/roboreel/human_demos/eval/no_distraction/push_cube \
  --max-videos 1 --frames 64 --epochs 20 --batch 8 \
  --seed 42 --device auto --out "$OUT"

# Bundle everything user needs to view and send back in one file.
python -m agcofl.package_v090

echo "================= VIDEO FILES ================="
echo "HUMAN ORIGINAL: $HUMAN"
echo "ROBOT XY REPLAY (NOT RGB): $PHYSICS/physical_xy_replay.mp4"
echo "TRAINED PREDICTIONS: $OUT/three_mode_predictions.png"
echo "TRAIN REPORT: $OUT/report.json"
echo "SOURCE QUALITY AUDIT: $OUT/source_audit.json"
echo "================= RESULTS ====================="
cat "$OUT/report.json"
echo "SINGLE REVIEW ZIP: $OUT/review_bundle.zip"
