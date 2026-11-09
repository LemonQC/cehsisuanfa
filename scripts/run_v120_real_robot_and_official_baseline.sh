#!/usr/bin/env bash
# v1.2: run REAL official robot TRAIN action audit and first released
# Vid2Robot baseline integration, NOT repeat v1.1 ViT diagnostic training.
# External downloads are limited to 2 robot demos <=300MB, one model
# <=8GiB, and 20 original heldout PushCube front videos <=250MB.
set -uo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=10
export HF_HUB_DOWNLOAD_TIMEOUT=45
export OMP_NUM_THREADS=4
OUT=outputs/v120_baselines
SRC=third_party/roboreel_models
mkdir -p "$OUT"

echo "======== v1.2 REAL ROBOT ACTIONS + OFFICIAL VID2ROBOT ========"
echo "START_LOCAL_SHA $(git rev-parse --short HEAD)"
echo "DEVICE $(python -c 'import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA_NOT_AVAILABLE")' 2>&1)"

if ! python -m pytest -q tests/test_v110_official_baselines.py tests/test_v120_official_assets.py; then
  echo "V120_UNIT_TESTS_FAILED: no data downloads, no model evaluation"
  exit 3
fi

# A) Reuse official TRAIN layout, never hand-code robot actions or guess
# official raw path names. Only complete H5+base/hand MP4 triplets qualify.
echo "STAGE_A: authentic RoboReel robot TRAIN source"
if command -v timeout >/dev/null 2>&1; then
  timeout 200s python -u -m agcofl.official_robot_data \
     --task push_cube --max-demos 2 --max-mb 300 --download \
     --out "$OUT" --data-root "$OUT/robot_train" \
     || echo "STAGE_A_BLOCKED: inspect official_robot_data output and manifest"
else
  python -u -m agcofl.official_robot_data \
     --task push_cube --max-demos 2 --max-mb 300 --download \
     --out "$OUT" --data-root "$OUT/robot_train" \
     || echo "STAGE_A_BLOCKED"
fi

if [ -s "$OUT/robot_data_manifest.json" ]; then
  python -u -m agcofl.real_robot_h5_audit \
    --manifest "$OUT/robot_data_manifest.json" \
    --data-root "$OUT/robot_train" \
    --out "$OUT/real_robot_h5_audit.json" \
    || echo "REAL_ROBOT_H5_AUDIT_FAILED"
fi

# B) Download the ONE official released Vid2Robot ND checkpoint (if model
# metadata confirms all files are available under the size limit), and only
# 20 official EVAL front videos for a fixed task. No all-model/all-task pull.
echo "STAGE_B: released Vid2Robot pretrained policy and fixed EVAL MP4s"
if command -v timeout >/dev/null 2>&1; then
  timeout 1500s python -u -m agcofl.official_vid2robot_assets \
    --out "$OUT" --max-checkpoint-gb 8 --max-eval-mb 250 --download \
    || echo "STAGE_B_DOWNLOAD_TIMEOUT_OR_ERROR: do not claim checkpoint acquired"
else
  echo "STAGE_B_SKIPPED: timeout utility unavailable"
fi

# C) Check the EXACT published official RoboReel environment source.
# The official model checkout was cloned without submodules in v1.1.
# Fetch only the RoboReel env submodule, not every incompatible dependency.
if [ -d "$SRC/.git" ] && [ ! -d "$SRC/roboreel_env/src/envs" ]; then
  echo "STAGE_C: fetch original RoboReel environment submodule only"
  if command -v timeout >/dev/null 2>&1; then
    timeout 180s git -C "$SRC" submodule update --init --depth 1 roboreel_env \
      || echo "OFFICIAL_ENV_SUBMODULE_BLOCKED: see Git message"
  fi
fi

# D) ACTUAL physics and learned policy: the same official runner with
# authors' released model. Uses one fixed 0080 as DEV/integration ONLY.
# It is expressly NOT the paper's official 20-scene success estimate.
echo "STAGE_D: actual Vid2Robot released-policy -> physics smoke"
if command -v timeout >/dev/null 2>&1; then
  timeout 600s python -u -m agcofl.vid2robot_official_smoke \
    --official-root "$SRC" --assets "$OUT" \
    --out "$OUT/official_vid2robot_smoke" --gpu 0 \
    || echo "OFFICIAL_SMOKE_BLOCKED: no success rate should be claimed"
else
  python -u -m agcofl.vid2robot_official_smoke \
    --official-root "$SRC" --assets "$OUT" \
    --out "$OUT/official_vid2robot_smoke" --gpu 0 \
    || echo "OFFICIAL_SMOKE_BLOCKED"
fi

# E) Only small summary/metadata records; no raw MP4/H5 or checkpoint.
bash scripts/publish_v120_report.sh \
   || echo "V120_AUTOPUBLISH_BLOCKED: local reports retained"
echo "============================================================"
echo "V120_COMPLETE_IS_NOT_A_SUCCESS_FLAG"
echo "ROBOT_ACTIONS: $OUT/real_robot_h5_audit.json"
echo "OFFICIAL_RELEASE: $OUT/official_vid2robot_assets.json"
echo "OFFICIAL_PHYSICS_SMOKE: $OUT/official_vid2robot_smoke/smoke_result.json"
echo "REPORT_URL: https://github.com/LemonQC/cehsisuanfa/tree/main/research_reports/2026-11-09_v120"
echo "============================================================"
