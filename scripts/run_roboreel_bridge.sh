#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1

python -m pytest -q tests/test_real_video_bridge.py tests/test_roboreel_cn_download.py

RR=third_party/roboreel
if [ ! -f "$RR/src/benchmark.py" ]; then
  echo "ERROR: RoboReel benchmark source is missing at $RR." >&2
  echo "Existing server previously cloned it. Download the official GitHub source separately." >&2
  exit 2
fi

python - <<'PY'
from importlib.util import find_spec
required=('torch','mani_skill','gymnasium','cv2')
missing=[x for x in required if find_spec(x) is None]
if missing: raise SystemExit("MISSING_PACKAGES: "+repr(missing))
print("LOCAL_DEPENDENCIES_OK",flush=True)
PY

VIDEO="$RR/human_demos/eval/no_distraction/push_cube/0080/front.mp4"
READY=0

if [ -f "$VIDEO" ]; then
  if python -m agcofl.download_roboreel_cn --root "$RR"; then
    READY=1
  else
    echo "Existing MP4 is incomplete; leaving untouched for inspection." >&2
  fi
fi

if [ "$READY" -eq 0 ] && [ ! -e "$VIDEO" ]; then
  echo "CHINA_MIRROR: attempting ONE real MP4 via https://hf-mirror.com; NO VPN REQUIRED"
  if command -v timeout >/dev/null 2>&1; then
    if timeout 110s env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
       python -m agcofl.download_roboreel_cn --root "$RR"; then
      READY=1
    else
      echo "VIDEO_UNAVAILABLE: mirror failed or timed out; will test official physics without human video."
    fi
  else
    echo "No coreutils timeout; skipping network to avoid indefinite waits."
  fi
fi

if [ "$READY" -eq 0 ]; then
  echo "PHYSICS_ONLY: authentic human video unavailable. No video features are used."
  python -u -X faulthandler -m agcofl.roboreel_bridge evaluate \
    --roboreel-root "$RR" --episodes 0 --mode fixed --no-video \
    --max-steps 250 --out outputs/roboreel_cn/physics_only \
    2>&1 | tee roboreel_physics_only.log
  echo "RESULT: outputs/roboreel_cn/physics_only/report.json"
  echo "This is NOT a human-to-robot experiment. Check log for ManiSkill version incompatibility."
  exit 0
fi

echo "VIDEO_READY: using authentic MP4 $VIDEO"
python -m agcofl.roboreel_bridge inspect-video \
  --video "$VIDEO" --out outputs/roboreel_cn/inspect_0080

python -u -X faulthandler -m agcofl.roboreel_bridge evaluate \
  --roboreel-root "$RR" --episodes 0 --mode fixed --max-steps 250 \
  --out outputs/roboreel_cn/fixed \
  2>&1 | tee roboreel_cn_fixed.log

if python - <<'PY'
import json,sys
from pathlib import Path
p=Path("outputs/roboreel_cn/inspect_0080/human_video_diagnostic.json")
d=json.loads(p.read_text())
print("VIDEO_MASK_ACCEPTED",d["accepted"],"COVERAGE",d["coverage"],flush=True)
sys.exit(0 if d["accepted"] else 1)
PY
then
  python -u -X faulthandler -m agcofl.roboreel_bridge evaluate \
    --roboreel-root "$RR" --episodes 0 --mode video_motion --max-steps 250 \
    --out outputs/roboreel_cn/video_motion \
    2>&1 | tee roboreel_cn_video_motion.log
  python -m agcofl.analyze_roboreel --out outputs/roboreel_cn
  echo "RESULT: outputs/roboreel_cn/paired_diagnostic.csv"
else
  echo "INVALID_HSV_MASK: inspect outputs/roboreel_cn/inspect_0080/human_mask_overlay.png"
  echo "Skipping video-conditioned diagnostic; no guessed human object tracks."
fi
echo "RESULT: outputs/roboreel_cn/fixed/report.json"
echo "RESULT: outputs/roboreel_cn/inspect_0080/human_video_diagnostic.json"
