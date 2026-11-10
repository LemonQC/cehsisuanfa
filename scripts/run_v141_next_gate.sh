#!/usr/bin/env bash
# v1.4.2, API-free next steps. No VLM keyframe annotation or v1.3 rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-gate}"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
case "$MODE" in
  gate)
    python -m pytest -q tests/test_v140_official_pi05.py tests/test_v140_custom_pick_push.py tests/test_v141_pi05_assets.py
    python -m agcofl.v141_pi05_assets --variant uniform
    ;;
  prepare-uniform)
    # Use authors' original selective ND checkpoint download and SigLIP.
    python -m agcofl.v141_pi05_assets --variant uniform --download-checkpoint --extract-features
    ;;
  evaluate-uniform)
    # Only the API-free original Uniform model and its cached SigLIP features.
    python -m agcofl.v141_pi05_assets --variant uniform
    python -m agcofl.v140_official_pi05 --variant uniform --execute
    ;;
  custom-physics-dev)
    # Actual ManiSkill/PhysX, one paired seed; privileged controller, NOT a learned method.
    python -m agcofl.v140_custom_pick_push_physics --seed 120 --max-steps 450
    ;;
  *)
    echo "Usage: bash scripts/run_v141_next_gate.sh {gate|prepare-uniform|evaluate-uniform|custom-physics-dev}" >&2
    exit 2
    ;;
esac
