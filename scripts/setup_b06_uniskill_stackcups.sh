#!/usr/bin/env bash
set -euo pipefail

# One-command, project-local B06 UniSkill setup + bounded StackCups DEV.
# All datasets, weights, HF cache and outputs remain under cehsisuanfa/outputs.
PROJECT="/mnt/4Tdata/LemonData/cehsisuanfa"
UPSTREAM="$PROJECT/third_party/roboreel_models"
BASE="$PROJECT/outputs/v120_baselines"
DATA="$BASE/official_eval_data"
CKPT="$BASE/released_checkpoints"
ISD_DIR="$BASE/uniskill_assets"
ISD="$ISD_DIR/idm.pth"
HF_CACHE="$BASE/hf_cache"
EXPECTED_UPSTREAM="79a0fb0099e73c5560114753a8706276f93aefc2"

cd "$PROJECT"
[[ -d "$UPSTREAM/.git" || -f "$UPSTREAM/.git" ]] || { echo "BLOCKED: missing upstream clone: $UPSTREAM"; exit 2; }
ACTUAL_UPSTREAM="$(git -C "$UPSTREAM" rev-parse HEAD)"
[[ "$ACTUAL_UPSTREAM" == "$EXPECTED_UPSTREAM" ]] || {
  echo "BLOCKED: RoboReel upstream revision mismatch: $ACTUAL_UPSTREAM (expected $EXPECTED_UPSTREAM)"; exit 2;
}

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE/hub"
export ROBOREEL_DATA="$DATA"
export ROBOREEL_CKPT="$CKPT"
export UNISKILL_ISD="$ISD"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$DATA" "$CKPT" "$ISD_DIR" "$HF_HOME"

for demo in 0000 0001; do
  video="$DATA/human_demos/train/no_distraction/stack_cups/$demo/front.mp4"
  [[ -s "$video" ]] || { echo "BLOCKED: missing genuine TRAIN video $video"; exit 2; }
done

# Initialize only the two upstream submodules required by UniSkill.
git -C "$UPSTREAM" submodule update --init src/external_models/UniSkill src/external_models/UniSkill-Policy
[[ -s "$UPSTREAM/src/external_models/UniSkill/extract_skill.py" ]] || { echo "BLOCKED: UniSkill IDM source submodule missing"; exit 2; }
[[ -s "$UPSTREAM/src/external_models/UniSkill-Policy/robomimic/scripts/train.py" ]] || { echo "BLOCKED: UniSkill-Policy source submodule missing"; exit 2; }

if ! conda env list | awk 'NR>2 {print $1}' | grep -qx 'uniskill'; then
  conda create -n uniskill python=3.12 -y
fi

# Follow the upstream RoboReel ENVIRONMENTS.md UniSkill recipe; keep it isolated
# from pi05/JAX. Re-running is safe and does not change the pi05 environment.
conda run -n uniskill python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.10.0" "torchvision==0.25.0"
conda run -n uniskill python -m pip install \
  "diffusers==0.37.0" "transformers==5.3.0" "accelerate==1.13.0" \
  "safetensors==0.7.0" "huggingface-hub>=0.25"
conda run -n uniskill python -m pip install \
  "setuptools<81" "mani_skill==3.0.0b22" "gymnasium==0.29.1" \
  "trimesh==4.11.1" "sapien==3.0.2" "loguru==0.7.3"
conda run -n uniskill python -m pip install \
  "numpy==2.4.2" "scipy==1.17.0" "h5py==3.15.1" \
  "opencv-python-headless==4.13.0.92" "pillow==12.1.0" \
  "einops==0.8.2" "decord==0.6.0" "packaging>=24.0"
conda run -n uniskill python -m pip install \
  "hydra-core>=1.3" "omegaconf==2.3.0" "easydict==1.13" \
  "cloudpickle==3.1.2" "tensorboard==2.20.0" "tensorboardX==2.6" \
  "imageio==2.37.2" "imageio-ffmpeg>=0.5" "wandb==0.25.0" \
  "tqdm==4.67.3" "psutil==7.2.2" "termcolor==3.3.0" \
  "tianshou==0.4.10" "bddl==3.6.0" "gym==0.25.2" "PyOpenGL==3.1.10"
conda run -n uniskill python -m pip install --no-deps -e "$UPSTREAM/src/external_models/UniSkill-Policy"

export PYTHONPATH="$PROJECT:$UPSTREAM:$UPSTREAM/src:$UPSTREAM/src/external_models:$UPSTREAM/src/external_models/UniSkill:$UPSTREAM/src/external_models/UniSkill-Policy:$UPSTREAM/roboreel_env/src${PYTHONPATH:+:$PYTHONPATH}"

# Preflight the exact missing checkpoint bytes before downloading; retain all
# model/cache data on the project drive. Require 1.5 GiB headroom for cache/temp.
conda run -n uniskill python - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import HfApi
repo = "LogosRoboticsLab/roboreel_models"
root = Path(os.environ["ROBOREEL_CKPT"])
items = [x for x in HfApi().model_info(repo, files_metadata=True).siblings
         if x.rfilename.startswith("nd/uniskill_policy/")]
if not items:
    raise SystemExit("BLOCKED: no nd/uniskill_policy files in official checkpoint repo")
missing = sum(int(x.size or 0) for x in items
               if not (root / x.rfilename).is_file()
               or (x.size and (root / x.rfilename).stat().st_size != x.size))
free = shutil.disk_usage(root).free
reserve = int(1.5 * 1024**3)
print({"checkpoint_file_count": len(items), "missing_download_bytes": missing,
       "free_bytes": free, "required_reserve_bytes": reserve,
       "destination": str(root)}, flush=True)
if free < missing + reserve:
    raise SystemExit("BLOCKED: insufficient project-drive space; no checkpoint download started")
PY

# Official RoboReel downloader; only B06 ND policy is selected, not the ~50 GB full set.
conda run -n uniskill python "$UPSTREAM/scripts/download_checkpoints.py" \
  --method uniskill_policy --condition nd --dest "$CKPT" --workers 4

# Official UniSkill author-released IDM checkpoint (80.3 MB file); download only
# idm.pth, not the full 3.52 GB Diffusers model repository.
conda run -n uniskill python - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
folder = Path(os.environ["ROBOREEL_DATA"]).parent / "uniskill_assets"
folder.mkdir(parents=True, exist_ok=True)
target = Path(os.environ["UNISKILL_ISD"])
source = Path(hf_hub_download(repo_id="HanjungKim/UniSkill",
    filename="UniSkill_final_weight/idm.pth", local_dir=str(folder)))
if source.resolve() != target.resolve():
    if target.exists():
        raise SystemExit("BLOCKED: IDM target exists; refusing to overwrite")
    shutil.copy2(source, target)
if not target.is_file() or target.stat().st_size < 10 * 1024 * 1024:
    raise SystemExit("BLOCKED: downloaded IDM checkpoint failed size verification")
print({"idm_checkpoint": str(target), "bytes": target.stat().st_size}, flush=True)
PY

POLICY="$CKPT/nd/uniskill_policy/ckpt.pth"
[[ -s "$POLICY" ]] || { echo "BLOCKED: official UniSkill policy checkpoint missing: $POLICY"; exit 2; }
[[ -s "$CKPT/nd/uniskill_policy/train_config.json" ]] || { echo "BLOCKED: official UniSkill policy train_config.json missing"; exit 2; }

# Verify the official runner, assets, CUDA runtime and released checkpoints.
conda run -n uniskill python -m scripts.roboreel_uniskill_video_dev --preflight --gpu 0

# Extract genuine human TRAIN skills only: stack_cups IDs 0000/0001, front camera.
# Batch size 8 is deliberately conservative for a single RTX 4090 (24 GB).
conda run -n uniskill python "$UPSTREAM/scripts/extract_uniskill_human_skill.py" \
  --data-root "$DATA" --output-root "$DATA/uniskill_skills" \
  --idm-checkpoint "$ISD" --device cuda:0 --tasks stack_cups \
  --video-types without_distraction --cameras front --skill-interval 20 \
  --batch-size 8 --prefetch-workers 2 --amp --progress

# Final gate then 2 paired episodes / 4 real PhysX rollouts. Results are
# automatically published as sanitized JSON to tro1-experiment-evidence.
conda run -n uniskill python -m scripts.roboreel_uniskill_video_dev --preflight --gpu 0
conda run -n uniskill python -u -m scripts.roboreel_uniskill_video_dev \
  --run --publish --gpu 0 --episodes 2 --max-steps 800
