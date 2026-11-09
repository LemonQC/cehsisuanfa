#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1

python -m pytest -q tests/test_real_video_bridge.py

RR=third_party/roboreel
if [ ! -f "$RR/src/benchmark.py" ]; then
  echo "Cloning official RoboReel benchmark..." >&2
  mkdir -p third_party
  git clone --depth 1 https://github.com/gupta-anmol99/roboreel.git "$RR"
fi

python - <<'PY'
import importlib.util
required=('torch','mani_skill','gymnasium','cv2','huggingface_hub')
missing=[x for x in required if importlib.util.find_spec(x) is None]
if missing:raise SystemExit("MISSING_PACKAGES: "+repr(missing)+
  ". Install missing dependencies only; do not reinstall or upgrade ManiSkill blindly.")
PY

# Only the PushCube front camera; not the entire ~11 GB evaluation video set.
python "$RR/scripts/download_human_demos.py" \
  --distraction no --tasks push_cube --cameras front --output-dir "$RR"

VIDEO="$RR/human_demos/eval/no_distraction/push_cube/0080/front.mp4"
python -m agcofl.roboreel_bridge inspect-video \
  --video "$VIDEO" --out outputs/roboreel_real/inspect_0080

# These are DIAGNOSTIC privileged-controller runs, not an actual human-to-robot
# learner, and NOT Full/Anchor/End supervision experiments.
python -u -X faulthandler -m agcofl.roboreel_bridge evaluate \
  --roboreel-root "$RR" --episodes 0 --mode fixed --max-steps 250 \
  --out outputs/roboreel_real/fixed \
  2>&1 | tee roboreel_fixed.log

python -u -X faulthandler -m agcofl.roboreel_bridge evaluate \
  --roboreel-root "$RR" --episodes 0 --mode video_motion --max-steps 250 \
  --out outputs/roboreel_real/video_motion \
  2>&1 | tee roboreel_video_motion.log

python -m agcofl.analyze_roboreel --out outputs/roboreel_real

echo "RESULTS:"
echo " outputs/roboreel_real/inspect_0080/human_video_diagnostic.json"
echo " outputs/roboreel_real/fixed/report.json"
echo " outputs/roboreel_real/video_motion/report.json"
echo " outputs/roboreel_real/paired_diagnostic.csv"
