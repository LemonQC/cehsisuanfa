#!/usr/bin/env bash
# v1.2.1: RETRY only what was blocked in v1.2. Reuse 2 validated TRAIN H5s.
# Download/resume OFFICIAL pretrained Vid2Robot & EVAL MP4 only.
# This is NOT training on or tuning with EVAL demonstration videos.
set -uo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=12
export HF_HUB_DOWNLOAD_TIMEOUT=90
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
OUT=outputs/v120_baselines
mkdir -p "$OUT"
echo "======== V121 OFFICIAL VID2ROBOT ASSET RECOVERY ========"
echo "LOCAL_SHA $(git rev-parse --short HEAD)"
echo "DEVICE $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA_NOT_AVAILABLE")')"

# Do not use another small network error as a reason to re-run GPU ViT
# training or download the already-validated robot H5 dataset.
if ! python -m pytest -q tests/test_v120_official_assets.py tests/test_v121_resumable_hf.py tests/test_v121_official_summary_parser.py; then
  echo "V121_TEST_BLOCKED: transport/tests failed before any downloads"
  exit 3
fi
echo "TRAIN_ACTION_DATA: reusing existing 2 validated REAL_ROBOREEL demos"
if [ ! -s "$OUT/real_robot_h5_audit.json" ]; then
   echo "TRAIN_ACTION_DATA_WARN: v1.2 validated H5 report missing"
fi

# requests is small and used only for bounded fallback HTTP Range transfers.
# Do not install or upgrade torch, CUDA, ManiSkill or the policy model.
if ! python -c 'import requests' >/dev/null 2>&1; then
  python -m pip install --no-cache-dir "requests>=2.31,<3" \
    || echo "REQUESTS_INSTALL_BLOCKED: native HuggingFace cache manager still attempted"
fi

# Authoritative HF file metadata sizes are checked BEFORE each download.
# The checkpoint is about 2.37 GB, and 20 eval front MP4 total ~97 MB.
# A previously complete checkpoint is detected by expected bytes, hashed
# and reused. A partial stream uses .part and Range resume.
if command -v timeout >/dev/null 2>&1; then
   timeout 2100s python -u -m agcofl.official_vid2robot_assets \
     --out "$OUT" --endpoint https://hf-mirror.com \
     --max-checkpoint-gb 8 --max-eval-mb 250 --download \
     || echo "V121_TRANSFER_BLOCKED: partial files remain for next resume"
else
   python -u -m agcofl.official_vid2robot_assets \
     --out "$OUT" --endpoint https://hf-mirror.com \
     --max-checkpoint-gb 8 --max-eval-mb 250 --download \
     || echo "V121_TRANSFER_BLOCKED: partial files remain for next resume"
fi

# The published Vid2Robot runner needs timm (ViT-B) and decord (MP4
# decoder). Add ONLY missing small model dependencies; never replace the
# validated torch, torchvision, CUDA or ManiSkill stack.
if ! python -c 'import timm, decord, einops' >/dev/null 2>&1; then
  echo "VID2ROBOT_DEPENDENCIES_MISSING: attempting small official Python imports"
  python -m pip install --no-deps --no-cache-dir \
      "timm==1.0.11" "decord==0.6.0" "einops==0.8.2" \
      || echo "VID2ROBOT_DEPENDENCY_SETUP_BLOCKED: log original Python error"
fi

# The original official policy inference code and its original ManiSkill
# physics remain unchanged. The single preselected heldout video (0080)
# is strictly a smoke/integration check, never model selection/paper score.
if command -v timeout >/dev/null 2>&1; then
  timeout 900s python -u -m agcofl.vid2robot_official_smoke \
     --official-root third_party/roboreel_models \
     --assets "$OUT" --out "$OUT/official_vid2robot_smoke" --gpu 0 \
     || echo "V121_POLICY_RUNTIME_BLOCKED: see saved original traceback"
else
  python -u -m agcofl.vid2robot_official_smoke \
     --official-root third_party/roboreel_models \
     --assets "$OUT" --out "$OUT/official_vid2robot_smoke" --gpu 0 \
     || echo "V121_POLICY_RUNTIME_BLOCKED"
fi
bash scripts/publish_v121_report.sh || echo "V121_PUBLISH_BLOCKED: reports remain local"
echo "======================="
echo "V121_COMPLETE_NOT_A_POLICY_SUCCESS_FLAG"
echo "OFFICIAL_RELEASE_REPORT: $OUT/official_vid2robot_assets.json"
echo "POLICY_DEV_ROLLOUT_REPORT: $OUT/official_vid2robot_smoke/smoke_result.json"
echo "GITHUB_REPORT: research_reports/2026-11-09_v121/"
echo "======================="
