#!/usr/bin/env bash
# V1.1: official baseline reproducibility and REAL pretrained ViT GPU sanity.
# No trained-on-eval source, no entire dataset (104 GB), no synthetic input.
set -uo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=10
export HF_HUB_DOWNLOAD_TIMEOUT=45

OUT=outputs/v110_baselines
OFFICIAL=third_party/roboreel_models
EXPECTED=79a0fb0099e73c5560114753a8706276f93aefc2
mkdir -p "$OUT"
echo "========== V110 OFFICIAL BASELINES + PRETRAINED ViT GPU =========="
echo "Required RTX4090: $(python -c 'import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA_NOT_AVAILABLE")' 2>&1)"

if ! python -m pytest -q tests/test_v110_vit_baseline.py tests/test_v110_official_baselines.py; then
  echo "V110_UNIT_TESTS_FAILED: do not interpret subsequent model outputs as valid"
  exit 3
fi

CLONED_NOW=0
if [ ! -d "$OFFICIAL/.git" ]; then
  echo "OFFICIAL_REPO_DOWNLOAD: pulling SOURCE ONLY, NO checkpoints/submodules"
  if command -v timeout >/dev/null 2>&1; then
    if timeout 100s git clone --depth 1 --single-branch --no-recurse-submodules \
      https://github.com/weiweigu1998/roboreel_models.git "$OFFICIAL"; then
      CLONED_NOW=1
    else
      echo "OFFICIAL_REPO_BLOCKED: clone failed; preflight records missing status"
    fi
  else
    echo "OFFICIAL_REPO_BLOCKED: no bounded timeout available"
  fi
fi

if [ -d "$OFFICIAL/.git" ]; then
  SHA=$(git -C "$OFFICIAL" rev-parse HEAD 2>/dev/null || true)
  if [ "$SHA" != "$EXPECTED" ]; then
    if [ "$CLONED_NOW" -eq 1 ]; then
      echo "PINNING_OFFICIAL_SOURCE: fetching exact published code commit $EXPECTED"
      if timeout 100s git -C "$OFFICIAL" fetch --depth 1 origin "$EXPECTED" &&
         git -C "$OFFICIAL" checkout --detach "$EXPECTED"; then
        echo "OFFICIAL_SOURCE_PINNED: $EXPECTED"
      else
        echo "OFFICIAL_SOURCE_PIN_BLOCKED: do not treat HEAD as pinned"
      fi
    else
      echo "OFFICIAL_REPO_COMMIT_DIFFERENT: have=$SHA expected=$EXPECTED"
      echo "Existing external checkout left untouched; audit will report unpinned."
    fi
  fi
fi
python -u -m agcofl.official_baseline_audit --repo "$OFFICIAL" \
  --dataset outputs/next_gate_v100/train_data \
  --out "$OUT/official_baseline_audit.json"

# Fetch a tiny real robot TRAIN subset for baseline action-label integration.
# No official checkpoints or 104 GB full dataset are fetched.
if command -v timeout >/dev/null 2>&1; then
  timeout 170s python -u -m agcofl.official_robot_data \
    --task push_cube --max-demos 2 --max-mb 300 --download \
    --data-root "$OUT/official_dataset" \
    --out "$OUT" \
    || echo "ROBOT_TRAIN_DOWNLOAD_BLOCKED: see robot_data_manifest or original log"
fi

# Verify whether genuine 3D hand/object pose annotations are actually
# exposed at TRAIN-source path; no guessing XYZ from 2D blue pixels.
if command -v timeout >/dev/null 2>&1; then
  timeout 65s python -u -m agcofl.human_pose_source_audit \
    --out "$OUT/human_pose_annotation_audit.json" \
    || echo "POSE_METADATA_BLOCKED: do not assume 3D data are available"
fi

# Keep the working ManiSkill/PyTorch stack; install ONLY a missing ViT
# loader and its metadata libraries, never replace torch/CUDA.
if ! python -c 'import transformers;from transformers import AutoModel,AutoImageProcessor' >/dev/null 2>&1; then
  echo "TRANSFORMERS_MISSING: trying small, pinned Hugging Face DINOv2 interface"
  python -m pip install --no-cache-dir "transformers==4.43.4" "safetensors>=0.4.5,<1" \
     || echo "TRANSFORMERS_INSTALL_BLOCKED: model pilot will record a blocker"
fi

# Unlike v1.0, THIS ACTUALLY RUNS DINOv2 PRETRAINED on CUDA, then
# trains a temporal TRANSFORMER on six real TRAIN human videos.
if python -u -m agcofl.vit_flow_pilot \
    --index outputs/next_gate_v100/train_data/downloaded_train_index.json \
    --out "$OUT/vit_pilot" --epochs 15 --frames 48 \
    --batch 16 --backbone facebook/dinov2-small --device cuda; then
  echo "VIT_GPU_TRAINING_COMPLETED"
else
  echo "VIT_GPU_TRAINING_BLOCKED: no random ViT fallback and no fake metrics"
  python - <<'PY'
import json
from pathlib import Path
p=Path("outputs/v110_baselines/vit_pilot/status.json")
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({"status":"VIT_GPU_TRAINING_BLOCKED",
  "checkpoint":"facebook/dinov2-small",
  "note":"Look above for runtime exception; no training success claimed",
  "used_official_eval_data":False},indent=2))
PY
fi

# Only human-readable derived reports are pushed to the user's GitHub.
bash scripts/publish_v110_report.sh || echo "V110_REPORT_PUSH_BLOCKED: local JSON/CSV retained"
echo "=============================================="
echo "V110_COMPLETE. Results: outputs/v110_baselines/"
echo "Official baselines: $OUT/official_baseline_audit.json"
echo "Robot train source: $OUT/robot_data_manifest.json"
echo "Human 3D pose source audit: $OUT/human_pose_annotation_audit.json"
echo "ViT GPU report: $OUT/vit_pilot/report.json"
echo "GitHub: research_reports/2026-11-09_v110/"
echo "=============================================="
