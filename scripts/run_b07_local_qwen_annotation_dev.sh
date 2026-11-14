#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
UPSTREAM="third_party/roboreel_models"
EXPECTED="79a0fb0099e73c5560114753a8706276f93aefc2"
DATA="outputs/v120_baselines/official_eval_data"
PY="outputs/v120_baselines/conda_envs/uniskill/bin/python"
HF_HOME_DIR="outputs/v120_baselines/hf_cache"
OUT="$DATA/human_video_annotation_qwen3-vl-4b-b07-stackcups-dev-20261114.json"
LOG_DIR="outputs/logs/b07_local_qwen_annotation_dev_20261114"

[[ -f "$UPSTREAM/scripts/annotate_human_videos.py" ]] || { echo "BLOCKED: pinned RoboReel annotation script missing"; exit 2; }
actual="$(git -C "$UPSTREAM" rev-parse HEAD)"
[[ "$actual" == "$EXPECTED" ]] || { echo "BLOCKED: RoboReel source revision mismatch: $actual"; exit 2; }
[[ -x "$PY" ]] || { echo "BLOCKED: project-local UniSkill Python missing: $PY"; exit 2; }
for demo in 0000 0001; do
  video="$DATA/human_demos/train/no_distraction/stack_cups/$demo/front.mp4"
  [[ -s "$video" ]] || { echo "BLOCKED: missing TRAIN-only source video: $video"; exit 2; }
done
[[ ! -e "$OUT" ]] || { echo "REFUSE_OVERWRITE_EXISTING_ANNOTATION: $OUT"; exit 3; }

free_bytes="$(df -PB1 "$DATA" | awk 'NR==2{print $4}')"
[[ "$free_bytes" -ge 12884901888 ]] || { echo "BLOCKED: need >=12 GiB free for Qwen3-VL weights/cache"; exit 2; }

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="outputs/v120_baselines/tmp"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
mkdir -p "$HF_HOME" "$TMPDIR" "$LOG_DIR"

echo "TRO1_B07_LOCAL_QWEN_ANNOTATION_DEV_START"
echo "upstream=$actual"
echo "data=$DATA"
echo "output=$OUT"
echo "python=$("$PY" --version 2>&1)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

"$PY" - <<'PY'
import torch, transformers, decord
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("B07_LOCAL_QWEN_REQUIRES_CUDA")
if int(transformers.__version__.split(".")[0]) < 5:
    raise SystemExit("B07_QWEN3_VL_REQUIRES_TRANSFORMERS_GE_5")
print("gpu", torch.cuda.get_device_name(0))
PY

"$PY" "$UPSTREAM/scripts/annotate_human_videos.py" \
  --data-root "$DATA" \
  --model "Qwen/Qwen3-VL-4B-Instruct" \
  --model-name "qwen3-vl-4b-b07-stackcups-dev-20261114" \
  --camera front \
  --video-types without_distraction \
  --tasks stack_cups \
  --num-frames 16 \
  --frame-size 640 \
  --max-new-tokens 96 \
  --limit 2 \
  --save-every 1 \
  2>&1 | tee "$LOG_DIR/annotation.log"

"$PY" - <<'PY'
import json
from pathlib import Path
p=Path("outputs/v120_baselines/official_eval_data/human_video_annotation_qwen3-vl-4b-b07-stackcups-dev-20261114.json")
d=json.loads(p.read_text())
a=d.get("annotations", {})
expected={"without_distraction/stack_cups/0000","without_distraction/stack_cups/0001"}
if set(a) != expected:
    raise SystemExit(f"B07_EXPECTED_TWO_TRAIN_ANNOTATIONS got={sorted(a)}")
if any(not isinstance(v,str) or not v.strip() for v in a.values()):
    raise SystemExit("B07_EMPTY_ANNOTATION")
print("TRO1_B07_LOCAL_QWEN_ANNOTATION_READY", json.dumps({
    "model": d.get("model"),
    "camera": d.get("camera"),
    "num_frames": d.get("num_frames"),
    "annotation_count": len(a),
    "annotations": a,
    "data_split": "TRAIN_ONLY",
    "status": "ANNOTATION_DEV_ONLY_NOT_POLICY_EVALUATION",
}))
PY

echo "Logs: $LOG_DIR/annotation.log"
