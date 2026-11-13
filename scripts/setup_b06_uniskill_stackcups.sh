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
CONDA_ENV="$BASE/conda_envs/uniskill"
LOCK_FILE="$BASE/b06_upstream_hf_lock_v1.json"
SUBSET="$BASE/b06_train_subset"
RESULT_DIR="$PROJECT/outputs/research_runs/b06_uniskill_stackcups_dev_v1"
EXPECTED_UPSTREAM="79a0fb0099e73c5560114753a8706276f93aefc2"

# Keep ALL large conda packages, Python wheels, temporary files and weights
# on the project drive rather than default ~/.conda, ~/.cache or /tmp.
export CONDA_PKGS_DIRS="$BASE/conda_pkgs"
export PIP_CACHE_DIR="$BASE/pip_cache"
export TMPDIR="$BASE/tmp"
export XDG_CACHE_HOME="$BASE/xdg_cache"
export TORCH_HOME="$BASE/torch_cache"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME"
STAGE="project_gate"
on_exit() {
  local rc="$1"
  trap - EXIT
  if (( rc != 0 )); then
    # Only publish sanitized stage/status; NEVER a raw pip error, token,
    # private path, config or trace. A failed GitHub write does not suppress
    # locally retained immutable evidence.
    python -m scripts.b06_failure_record --stage "$STAGE" --exit-code "$rc" --publish || true
  fi
}
trap 'on_exit "$?"' EXIT
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

# Network/upload interruption after completed PhysX? Republish from existing
# verified results, without reinstalling packages or taking another step.
if [[ -s "$RESULT_DIR/results.json" ]]; then
  STAGE="republish_completed_without_physx"
  python -m scripts.roboreel_uniskill_video_dev \
    --publish-existing --out "$RESULT_DIR"
  exit 0
fi
# A partial PhysX directory must be preserved for forensic review;
# never blindly resume or redo the same seeds.
if [[ -d "$RESULT_DIR" ]]; then
  STAGE="real_physx_bounded_dev_and_publish"
  echo "BLOCKED: partial B06 PhysX directory exists; do not rerun same seeds"
  exit 2
fi

for demo in 0000 0001; do
  video="$DATA/human_demos/train/no_distraction/stack_cups/$demo/front.mp4"
  [[ -s "$video" ]] || { echo "BLOCKED: missing genuine TRAIN video $video"; exit 2; }
done

# The official skill-extraction script scans ALL demos below --data-root.
# Give it a 2-video read-only view; never process the other 78 TRAIN videos.
STAGE="bounded_video_subset"
for demo in 0000 0001; do
  original="$DATA/human_demos/train/no_distraction/stack_cups/$demo/front.mp4"
  folder="$SUBSET/human_demos/train/no_distraction/stack_cups/$demo"
  mkdir -p "$folder"
  shortcut="$folder/front.mp4"
  if [[ -e "$shortcut" || -L "$shortcut" ]]; then
    [[ -L "$shortcut" && "$(readlink -f "$shortcut")" == "$(readlink -f "$original")" ]] || {
      echo "BLOCKED: B06 bounded TRAIN subset path unexpectedly changed"; exit 2;
    }
  else
    ln -s "$original" "$shortcut"
  fi
done

# Initialize ONLY two pinned RoboReel submodules. The forked RoboReel remotes
# and original author's remotes both expose the same exact Git commit objects.
# Do not change .gitmodules, do not reset HEAD, do not remove a partial clone.
STAGE="author_source_submodules"
b06_ensure_submodule() {
  local module_name="$1" rel_path="$2" pinned_sha="$3" author_url="$4" marker="$5"
  local target="$UPSTREAM/$rel_path"
  local current=""
  if [[ -d "$target" ]]; then
    current="$(git -C "$target" rev-parse HEAD 2>/dev/null || true)"
  fi
  if [[ "$current" == "$pinned_sha" && -s "$target/$marker" ]]; then
    printf 'B06_SUBMODULE_READY existing name=%s sha=%s\n' "$module_name" "$current"
    return 0
  fi

  printf 'B06_SUBMODULE_FETCH official name=%s\n' "$module_name"
  if ! git -C "$UPSTREAM" submodule update --init -- "$rel_path"; then
    printf 'B06_SUBMODULE_RETRY original_author name=%s\n' "$module_name"
    # The official submodule may have an unborn HEAD after an interrupted
    # clone. Changing submodule.<name>.url alone does NOT repair its cached
    # .git/modules/<path> remote or resolve that unborn checkout.
    # First attempt a pinned direct fetch in that very same submodule.
    if ! bash "$PROJECT/scripts/b06_recover_submodule_git.sh" \
       "$target" "$pinned_sha" "$author_url" "$marker"; then
      echo "B06_SUBMODULE_BLOCKED: exact-revision safe recovery failed: $module_name" >&2
      return 1
    fi
  fi
  current="$(git -C "$target" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$current" != "$pinned_sha" || ! -s "$target/$marker" ]]; then
    echo "B06_SUBMODULE_BLOCKED: incorrect commit or missing entrypoint: $module_name" >&2
    return 1
  fi
  printf 'B06_SUBMODULE_READY downloaded name=%s sha=%s\n' "$module_name" "$current"
}

b06_ensure_submodule "models/UniSkill" \
  "src/external_models/UniSkill" \
  "bd9843b3168f889951bc117c2fc70be59590f8f9" \
  "https://github.com/KimHanjung/UniSkill.git" \
  "extract_skill.py"
b06_ensure_submodule "models/UniSkill-Policy" \
  "src/external_models/UniSkill-Policy" \
  "02ab999609b9e698a197a6786a7edebdb69faac5" \
  "https://github.com/kang-jaehyun/UniSkill-Policy.git" \
  "robomimic/scripts/train.py"

STAGE="project_local_conda_env"
# Prevent a large CUDA environment from being constructed on a nearly-full disk.
if [[ ! -x "$CONDA_ENV/bin/python" ]]; then
  free_bytes="$(df -PB1 "$BASE" | awk 'NR==2{print $4}')"
  [[ "$free_bytes" -ge 7516192768 ]] || { echo "BLOCKED: need >=7 GiB free on project drive before creating CUDA env"; exit 2; }
fi
# Heavy PyTorch/CUDA env stays entirely on this project drive.
if [[ ! -x "$CONDA_ENV/bin/python" ]]; then
  if [[ -d "$CONDA_ENV" ]]; then
    echo "BLOCKED: incomplete B06 Conda prefix; inspect before repair: $CONDA_ENV"; exit 2
  fi
  conda create -p "$CONDA_ENV" python=3.12 -y
fi

# Follow the upstream RoboReel ENVIRONMENTS.md UniSkill recipe; keep it isolated
# from pi05/JAX. Preserve installed wheels and checkpoint bytes on retries.
STAGE="install_uniskill_deps"
if [[ ! -f "$BASE/b06_env_install_complete.v1" ]]; then
"$CONDA_ENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.10.0" "torchvision==0.25.0"
"$CONDA_ENV/bin/python" -m pip install \
  "diffusers==0.37.0" "transformers==5.3.0" "accelerate==1.13.0" \
  "safetensors==0.7.0" "huggingface-hub>=0.25"
"$CONDA_ENV/bin/python" -m pip install \
  "setuptools<81" "mani_skill==3.0.0b22" "gymnasium==0.29.1" \
  "trimesh==4.11.1" "sapien==3.0.2" "loguru==0.7.3"
"$CONDA_ENV/bin/python" -m pip install \
  "numpy==2.4.2" "scipy==1.17.0" "h5py==3.15.1" \
  "opencv-python-headless==4.13.0.92" "pillow==12.1.0" \
  "einops==0.8.2" "packaging>=24.0"
"$CONDA_ENV/bin/python" -m pip install \
  "hydra-core>=1.3" "omegaconf==2.3.0" "easydict==1.13" \
  "cloudpickle==3.1.2" "tensorboard==2.20.0" "tensorboardX==2.6" \
  "imageio==2.37.2" "imageio-ffmpeg>=0.5" "wandb==0.25.0" \
  "tqdm==4.67.3" "psutil==7.2.2" "termcolor==3.3.0" \
  "tianshou==0.4.10" "bddl==3.6.0" "gym==0.25.2" "PyOpenGL==3.1.10"
"$CONDA_ENV/bin/python" -m pip install --no-deps -e "$UPSTREAM/src/external_models/UniSkill-Policy"
# UniSkill-Policy's pinned pyproject metadata was authored for a different
# environment than the pinned RoboReel B06 runner. Validate only the three
# acknowledged legacy version declarations, fail closed on any other issue.
# decord is unused by the official video extractor (imageio-ffmpeg) and runner.
"$CONDA_ENV/bin/python" -m scripts.b06_dependency_gate
"$CONDA_ENV/bin/python" -c 'import torch, torchvision, transformers, diffusers, gymnasium, mani_skill, sapien, numpy, imageio.v2; print("B06_ENV_IMPORT_OK")'
touch "$BASE/b06_env_install_complete.v1"
fi

export PYTHONPATH="$PROJECT:$UPSTREAM:$UPSTREAM/src:$UPSTREAM/src/external_models:$UPSTREAM/src/external_models/UniSkill:$UPSTREAM/src/external_models/UniSkill-Policy:$UPSTREAM/roboreel_env/src"

# Resolve and freeze both HF source revisions, then check free project space.
# Never silently refresh weights from a moving main branch on a retry.
STAGE="pin_hf_revisions_and_disk_gate"
"$CONDA_ENV/bin/python" - <<'PYLOCK'
import json, os, shutil
from pathlib import Path
from huggingface_hub import HfApi
root = Path(os.environ["ROBOREEL_CKPT"])
base = root.parent
lock = base / "b06_upstream_hf_lock_v1.json"
api=HfApi()
if lock.exists():
    metadata=json.loads(lock.read_text())
else:
    robo=api.model_info("LogosRoboticsLab/roboreel_models", files_metadata=True)
    uniskill=api.model_info("HanjungKim/UniSkill", files_metadata=True)
    policy=[{"filename":x.rfilename, "bytes":int(x.size or 0)} for x in robo.siblings
            if x.rfilename.startswith("nd/uniskill_policy/")]
    matches=[x for x in uniskill.siblings if x.rfilename=="UniSkill_final_weight/idm.pth"]
    if not policy or not matches or not robo.sha or not uniskill.sha or any(x["bytes"]<=0 for x in policy) or not (matches[0].size or 0):
        raise SystemExit("BLOCKED: official B06 policy/IDM metadata unavailable")
    metadata={"schema":"B06_HF_SOURCE_LOCK_V1", "policy_repo":"LogosRoboticsLab/roboreel_models",
              "policy_revision":robo.sha,"policy_files":policy,"idm_repo":"HanjungKim/UniSkill",
              "idm_revision":uniskill.sha,"idm_file":"UniSkill_final_weight/idm.pth",
              "idm_bytes":int(matches[0].size or 0)}
    with lock.open("x",encoding="utf-8") as f:
        f.write(json.dumps(metadata,sort_keys=True,indent=2)+"\n")
missing=sum(f["bytes"] for f in metadata["policy_files"]
            if not (root/f["filename"]).is_file()
            or (root/f["filename"]).stat().st_size != f["bytes"])
idm=base/"uniskill_assets/idm.pth"
if not idm.is_file():
    missing+=metadata["idm_bytes"]
free=shutil.disk_usage(root).free
reserve=int(1.5*1024**3)
print({"policy_revision":metadata["policy_revision"],"idm_revision":metadata["idm_revision"],
       "missing_download_bytes":missing,"disk_free_bytes":free,"reserve_bytes":reserve},flush=True)
if free<missing+reserve:
    raise SystemExit("BLOCKED: insufficient project drive space before HF downloads")
PYLOCK

ROBO_REV="$("$CONDA_ENV/bin/python" -c "import json;print(json.load(open('$LOCK_FILE'))['policy_revision'])")"
IDM_REV="$("$CONDA_ENV/bin/python" -c "import json;print(json.load(open('$LOCK_FILE'))['idm_revision'])")"
export IDM_REV

# Official downloader, locked to a specific revision; ONLY B06 ND.
STAGE="download_pinned_official_uniskill_policy"
"$CONDA_ENV/bin/python" "$UPSTREAM/scripts/download_checkpoints.py" \
  --method uniskill_policy --condition nd --dest "$CKPT" \
  --revision "$ROBO_REV" --workers 4

STAGE="download_pinned_author_idm"
# Official UniSkill author-released IDM checkpoint (80.3 MB file); download only
# idm.pth, not the full 3.52 GB Diffusers model repository.
"$CONDA_ENV/bin/python" - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
folder = Path(os.environ["ROBOREEL_DATA"]).parent / "uniskill_assets"
folder.mkdir(parents=True, exist_ok=True)
target = Path(os.environ["UNISKILL_ISD"])
source = Path(hf_hub_download(repo_id="HanjungKim/UniSkill",
    filename="UniSkill_final_weight/idm.pth", revision=os.environ["IDM_REV"],
    local_dir=str(folder)))
if source.resolve() != target.resolve():
    if target.exists():
        import hashlib
        def digest(path):
            h=hashlib.sha256()
            with open(path,'rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            return h.hexdigest()
        if digest(target)!=digest(source):
            raise SystemExit("BLOCKED: IDM target has different content; refusing overwrite")
    else:
        shutil.copy2(source, target)
if not target.is_file() or target.stat().st_size < 10 * 1024 * 1024:
    raise SystemExit("BLOCKED: downloaded IDM checkpoint failed size verification")
print({"idm_checkpoint": str(target), "bytes": target.stat().st_size}, flush=True)
PY

POLICY="$CKPT/nd/uniskill_policy/ckpt.pth"
[[ -s "$POLICY" ]] || { echo "BLOCKED: official UniSkill policy checkpoint missing: $POLICY"; exit 2; }
[[ -s "$CKPT/nd/uniskill_policy/train_config.json" ]] || { echo "BLOCKED: official UniSkill policy train_config.json missing"; exit 2; }

# Verify the official runner, assets, CUDA runtime and released checkpoints.
STAGE="preflight_before_skill_extraction"
"$CONDA_ENV/bin/python" -m scripts.roboreel_uniskill_video_dev --preflight --gpu 0

# Extract ONLY the two selected TRAIN videos. The original official script
# otherwise scans all 80 TRAIN videos; the bounded source is just symlinks.
# Batch 8 is deliberately conservative for a single 24 GB RTX 4090.
STAGE="extract_two_genuine_human_skills"
"$CONDA_ENV/bin/python" "$UPSTREAM/scripts/extract_uniskill_human_skill.py" \
  --data-root "$SUBSET" --output-root "$DATA/uniskill_skills" \
  --idm-checkpoint "$ISD" --device cuda:0 --tasks stack_cups \
  --video-types without_distraction --cameras front --skill-interval 20 \
  --batch-size 8 --prefetch-workers 2 --amp --progress

# Final gate then 2 paired episodes / 4 real PhysX rollouts.
# A completed prior run is *republished only*, NEVER physically repeated.
STAGE="final_preflight"
"$CONDA_ENV/bin/python" -m scripts.roboreel_uniskill_video_dev --preflight --gpu 0
if [[ -s "$RESULT_DIR/results.json" ]]; then
  STAGE="republish_completed_without_physx"
  "$CONDA_ENV/bin/python" -m scripts.roboreel_uniskill_video_dev \
    --publish-existing --out "$RESULT_DIR"
  exit 0
fi
if [[ -d "$RESULT_DIR" ]]; then
  echo "BLOCKED: partial B06 PhysX directory exists. Inspect and preserve; no automated rerun."
  exit 2
fi
STAGE="real_physx_bounded_dev_and_publish"
"$CONDA_ENV/bin/python" -u -m scripts.roboreel_uniskill_video_dev \
  --run --publish --gpu 0 --episodes 2 --max-steps 800
