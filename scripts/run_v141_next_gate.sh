#!/usr/bin/env bash
# v1.5.3, check exact published SigLIP EVAL bundles before any new base weights.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-gate}"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
case "$MODE" in
  gate)
    python -m pytest -q tests/test_v140_official_pi05.py tests/test_v140_custom_pick_push.py tests/test_v141_pi05_assets.py tests/test_hf_endpoint_probe.py tests/test_pi05_mirror_resolve.py tests/test_recover_pi05_uniform_missing.py tests/test_check_pi05_runtime.py tests/test_check_pi05_siglip_cache.py tests/test_audit_official_siglip_dataset.py
    python -m agcofl.v141_pi05_assets --variant uniform
    ;;
  probe-checkpoint-mirror)
    # Metadata-only HTTPS check, no model bytes or paid model inference APIs.
    python scripts/probe_hf_endpoint.py --endpoint "${HF_ENDPOINT:-https://hf-mirror.com}"
    ;;
  probe-checkpoint-files)
    # HEAD all 25 model files; see if mirror actually hosts weights or redirects.
    python scripts/inspect_pi05_mirror_resolve.py \
      --endpoint "${HF_ENDPOINT:-https://hf-mirror.com}"
    ;;
  prepare-checkpoint-mirror)
    # Explicit opt-in public mirror. Metadata and all file redirects are
    # checked BEFORE selective ND Uniform checkpoint download.
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
    unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
    python scripts/probe_hf_endpoint.py --endpoint "$HF_ENDPOINT"
    python scripts/inspect_pi05_mirror_resolve.py --endpoint "$HF_ENDPOINT"
    python -m agcofl.v141_pi05_assets --variant uniform --download-checkpoint
    ;;
  plan-checkpoint-repair)
    # Metadata-only file inventory. No weight download and no API inference.
    python scripts/recover_pi05_uniform_missing.py
    ;;
  recover-checkpoint-proxy)
    # Only missing/invalid files, Range-resumable .part, original repo commit
    # through operator HTTPS_PROXY. Does not need a working hf-mirror blob path.
    python scripts/recover_pi05_uniform_missing.py --download-missing
    ;;
  prepare-checkpoint)
    # Independently retry original Hugging Face network after fixing routing.
    python -m agcofl.v141_pi05_assets --variant uniform --download-checkpoint
    ;;
  check-pi05-env)
    # Read-only JAX/OpenPI import and GPU diagnosis in whichever env is active.
    python scripts/check_pi05_runtime.py --mode features
    ;;
  inspect-uniform-vision)
    # No GPU/tensor restore: inspect whether the released Orbax metadata has a vision subtree.
    python scripts/inspect_pi05_uniform_vision_metadata.py
    ;;
  audit-siglip-dataset)
    # HTTPS public HF DATASET METADATA ONLY; does not download model / videos.
    python scripts/audit_official_siglip_dataset.py
    ;;
  check-siglip-cache)
    # NO GCS network: inspect author SigLIP cache path and known local locations.
    python scripts/check_pi05_siglip_cache.py
    ;;
  smoke-one-feature)
    # Author's original SigLIP, one EVAL video only. Never trains on EVAL.
    # Failures in JAX/OpenPI, model loading, ffmpeg appear before 20-demo job.
    python scripts/check_pi05_runtime.py --mode features
    if [[ "${PI05_ALLOW_GCS_DOWNLOAD:-0}" == "1" ]]; then
      python scripts/check_pi05_siglip_cache.py --permit-gcs-download
    else
      python scripts/check_pi05_siglip_cache.py
    fi
    python -m agcofl.v141_pi05_assets --variant uniform --extract-features --max-feature-demos 1
    ;;
  prepare-features)
    # Local authors' SigLIP extraction of the 20 existing official EVAL videos.
    # Requires the separate JAX/OpenPI environment and possibly cached SigLIP weights.
    python scripts/check_pi05_runtime.py --mode features
    if [[ "${PI05_ALLOW_GCS_DOWNLOAD:-0}" == "1" ]]; then
      python scripts/check_pi05_siglip_cache.py --permit-gcs-download
    else
      python scripts/check_pi05_siglip_cache.py
    fi
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
    python scripts/check_pi05_runtime.py --mode eval
    python -m agcofl.v140_official_pi05 --variant uniform --execute
    ;;
  custom-physics-dev)
    # Actual ManiSkill/PhysX, one paired seed; privileged controller, NOT a learned method.
    python -m agcofl.v140_custom_pick_push_physics --seed 120 --max-steps 450
    ;;
  *)
    echo "Usage: bash scripts/run_v141_next_gate.sh {gate|plan-checkpoint-repair|recover-checkpoint-proxy|probe-checkpoint-mirror|probe-checkpoint-files|prepare-checkpoint-mirror|prepare-checkpoint|check-pi05-env|inspect-uniform-vision|audit-siglip-dataset|check-siglip-cache|smoke-one-feature|prepare-features|prepare-uniform|evaluate-uniform|custom-physics-dev}" >&2
    exit 2
    ;;
esac
