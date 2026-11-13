# TRO1: reopen genuine ObjectInDrawer with authorized PartNet-Mobility 44962 (2026-11-13)

## Authorization has changed

The operator received accepted access requests for **ShapeNet/ShapeNetCore** and **sapien-sim/PartNetMobility** from Hugging Face. This changes the earlier *unavailable PartNet data* constraint, but access still needs to be proven using **the approved account's own authenticated Hugging Face client** on the operator's server. No token is provided through ChatGPT or checked into GitHub. PartNet terms cover non-commercial research and educational use and inherit the ShapeNet terms. Respect them. Do not publish copied gated model assets.

## Exact source and layout

The original published RoboReel code (`gupta-anmol99/roboreel@afaf4a03`) hard-codes `_MODEL_ID='44962'` for `ObjectInDrawer-v1`. The original asset parser reads `mani_skill.ASSET_DIR/configs/info_*.json` and `mani_skill.ASSET_DIR/dataset/44962/{mobility_cvx.urdf,mobility_fixed.urdf,mobility.urdf}`. GitHub repository already contains the correct authored `info_drawer.json` with `scale=0.3`.

Hugging Face's official `sapien-sim/PartNetMobility` repo stores one ZIP per model ID. **Only `44962.zip`** is needed. The new installer obtains a version-pinned SHA and file byte count using Hugging Face authenticated metadata, explicitly downloads only `44962.zip`, checks size and SHA256, safely unpacks with zip-slip/symlink/duplicate/bomb rejection, verifies the original URDF, copies the pinned author's metadata, and installs to:

```
outputs/roboreel_partnet_single_44962/data/configs/info_drawer.json
outputs/roboreel_partnet_single_44962/data/dataset/44962/{mobility.urdf,...}
```

`MS_ASSET_DIR` is set by the existing official runner's asset gate before ManiSkill is imported. The original SAPIEN-token `--download` and approved local `--import-dir` are preserved as alternatives, but **Hugging Face authorization is different from a SAPIEN website access token**. The new workflow uses only HF login. No full PartNet, ShapeNet or cabinet bulk downloads are performed. Data/cache remain under project `outputs/` (gitignored).

## Server commands

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main

# Authenticate locally with the account whose gated repo access was approved.
# The mirror cannot approve a user; go through the official login endpoint.
env -u HF_ENDPOINT hf auth login
env -u HF_ENDPOINT hf auth whoami

python -m unittest discover -s tests -p 'test_download_roboreel_partnet44962_hf.py' -v
python -m unittest discover -s tests -p 'test_prepare_roboreel_drawer_partnet.py' -v

# Uses mirror first, official HTTPS second, with existing auth token.
# This is one original PartNet model, not a snapshot/full download.
python -u -m scripts.prepare_roboreel_drawer_partnet --hf-download

# Independently verify asset file layout and original metadata.
python -m scripts.prepare_roboreel_drawer_partnet

# Smoke ONLY the original authored drawer scene; no pi0.5 policy rollouts.
env -u DISPLAY -u WAYLAND_DISPLAY python -u -m scripts.check_roboreel_drawer_scene

# Confirm existing two genuine human TRAIN videos and SigLIP bundles, do not recompute.
python -m scripts.roboreel_video_dependency_dev --task toys_in_drawer \
  --out outputs/roboreel_video_dependency_dev/drawer_hf_v1
```

Only if **`ROBOREEL_HF_PARTNET44962_READY` or original `ROBOREEL_PARTNET_44962_READY`** and `ROBOREEL_DRAWER_SCENE_SMOKE` explicitly prints `DRAWER_SCENE_READY_FOR_BOUNDED_DEV`, and only if the DEV preflight says `READY_FOR_BOUNDED_DEV`, perform exactly 2 paired scenes × 2 video conditions in **new** output directory (existing failed drawer `v2`/`v3` remain frozen and unmodified):

```bash
env -u DISPLAY -u WAYLAND_DISPLAY python -u -m scripts.roboreel_video_dependency_dev \
  --task toys_in_drawer --run --episodes 2 --max-steps 800 \
  --out outputs/roboreel_video_dependency_dev/drawer_hf_v1
```

Do not blindly rerun if any real manifest or actions exist. On download failure: `ROBOREEL_PARTNET_44962_BLOCKED` includes only coarse exception class/endpoint, not token. The read token is never exposed in CLI output. If mirror fails due gated credentials, official fallback occurs automatically. If login command is unavailable on older `huggingface_hub`, the locally installed `huggingface-cli login` may be used instead, with the same approval account.

## Evidence and research integrity

- Previous `StackCups-v1` 4 real rollouts were **0/4 successful**, with no within-scene outcome reversal; do not erase the NO-GO conclusion for stage synchronization. This new authorized asset permits revisiting **ObjectInDrawer** infrastructure, not automatically claiming an original T-RO innovation.
- Frozen official PushCube π0.5 46/60, Vid2Robot 3/60, StackCups 0/4 remain untouched.
- Existing `toys_in_drawer` genuine TRAIN `0000` and `0001` features were generated before the asset blocker; **do not repeat 0.66 GB of SigLIP computation**. We have not yet demonstrated physical success or runtime scene creation with the newly approved model.
- **HF archive structure has not yet been inspected under the user's gated token**. The installer accepts a single model directory archive or a flat single-model archive and fails safely when unexpected layouts are encountered; this is more honest than assuming a downloaded ZIP is automatically compatible.
- `ShapeNetCore` authorization exists but does not supply or replace PartNet's articulated `44962` cabinet.
