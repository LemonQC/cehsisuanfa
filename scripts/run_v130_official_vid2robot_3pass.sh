#!/usr/bin/env bash
# Official RoboReel Vid2Robot ND PushCube (clean human/clean env),
# original 20 held-out demo IDs × 3 original env seed offsets.
# No training, no demo selection, no downloaded weights (already present).
set -uo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=4
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
OUT=outputs/v130_official_vid2robot

echo "======== v1.3 FULL OFFICIAL VID2ROBOT 20 x 3 ========"
echo "GIT_SHA $(git rev-parse --short HEAD)"
echo "RTX4090 $(python -c 'import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA")')"
echo "NO_REDOWNLOAD_OF_2_37GB_CHECKPOINT_AND_20_EVAL_MP4S"
if ! python -m pytest -q tests/test_v130_official_protocol.py tests/test_v121_official_summary_parser.py; then
  echo "V130_UNIT_TESTS_FAILED: no official evaluation executed"
  exit 3
fi

echo "V130_SOURCE_AND_ASSET_PREFLIGHT"
# This command does not import datasets or modify the original official repo.
# Each completed 20-episode seed is reused only if the original summary
# validates against all 20 IDs, 800 steps, original chunk and exact conditions.
python -u -m agcofl.official_v130_protocol \
   --out "$OUT" --source third_party/roboreel_models \
   --assets outputs/v120_baselines --gpu 0 \
   --seed-offsets 0,100000,200000
RC=$?
echo "V130_PROTOCOL_PROCESS_EXIT: $RC"

# Publish real original per-episode success flags and honest partial markers.
# Never upload MP4/H5/checkpoint weights or GUI recordings.
bash scripts/publish_v130_official_eval.sh \
    || echo "V130_AUTOPUBLISH_BLOCKED: local reports retained"

echo "===================================="
echo "V130_COMPLETED_SCRIPT_NOT_AUTOMATICALLY_FULL_3PASS"
echo "CHECK_OUTPUT: $OUT/aggregated_summary.json"
echo "STATUS_OUTPUT: $OUT/run_status.json"
echo "GITHUB_REPORT: research_reports/2026-11-09_v130"
echo "===================================="
