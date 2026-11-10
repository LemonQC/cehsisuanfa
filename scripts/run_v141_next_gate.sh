#!/usr/bin/env bash
# v1.4.3, fail-closed API-free asset & physics gates; no Vid2Robot rerun.
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
  prepare-checkpoint)
    # Independently retry model weights after fixing the HTTPS certificate mismatch.
    python -m agcofl.v141_pi05_assets --variant uniform --download-checkpoint
    ;;
  prepare-features)
    # Local authors' SigLIP extraction of the 20 existing official EVAL videos.
    # Requires the separate JAX/OpenPI environment and possibly cached SigLIP weights.
    python -m agcofl.v141_pi05_assets --variant uniform --extract-features
    ;;
  prepare-uniform)
    # Original selective ND checkpoint then only 20 EVAL SigLIP features.
    # Either stage may be resumed independently after resolving its own dependency.
    python -m agcofl.v141_pi05_assets --variant uniform --download-checkpoint
    python -m agcofl.v141_pi05_assets --variant uniform --extract-features
    ;;
  evaluate-uniform)
    # Only the API-free original Uniform model and its cached SigLIP features.
    # Missing checkpoint or any EVAL features must stop BEFORE official evaluator.
    python -m agcofl.v141_pi05_assets --variant uniform --require-ready
    python -m agcofl.v140_official_pi05 --variant uniform --execute
    ;;
  custom-physics-dev)
    # Actual ManiSkill/PhysX, one paired seed; privileged controller, NOT a learned method.
    python -m agcofl.v140_custom_pick_push_physics --seed 120 --max-steps 450
    ;;
  *)
    echo "Usage: bash scripts/run_v141_next_gate.sh {gate|prepare-checkpoint|prepare-features|prepare-uniform|evaluate-uniform|custom-physics-dev}" >&2
    exit 2
    ;;
esac
