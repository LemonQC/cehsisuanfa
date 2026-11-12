# TRO1 | 2026-11-12 | RoboReel ObjectInDrawer required PartNet 44962

## Actual failure and scope

The operator's `pi05` process reached **frozen pi0.5 model initialization** and `evaluation.utils.make_eval_env('toys_in_drawer',False)`, but failed in pinned RoboReel env `object_in_drawer.py` → `_load_asset()` → `get_partnet_mobility_builder()` with `Partnet Mobility dataset not found`. No rollout action executed. Original v3 manifest and logs are preserved, not reused. The two genuine TRAIN human MP4s and SigLIP bundles are complete and **must not be re-downloaded/re-encoded**.

The actual pinned submodule (gupta-anmol99/roboreel@afaf4a03...) hardcodes `_MODEL_ID='44962'`; its asset loader scans `mani_skill.ASSET_DIR/configs/info_*.json`, then `mani_skill.ASSET_DIR/dataset/44962/{mobility_cvx,mobility_fixed,mobility}.urdf`. The metadata `roboreel_env/assets/data/configs/info_drawer.json` is already in source control. It is *not* correct to download the entire `partnet_mobility_cabinet` set under the default `~/.maniskill` and assume this fixes the RoboReel-specific layout.

## Implementation

- New `scripts/prepare_roboreel_drawer_partnet.py`: **only model 44962**; no default network, no simulator; project-scoped output root `outputs/roboreel_partnet_single_44962` with `data/configs/info_drawer.json` and `data/dataset/44962/`.
- Run default inspect first. For an existing authorized 44962 folder (possibly from official gated Hugging Face), use `--import-dir /your/approved/44962`. Or use `--download` with an **authorized SAPIEN access token**. Script silently prompts on the server TTY when token is not already in `SAPIEN_ACCESS_TOKEN`; **never paste tokens into ChatGPT or shell commands**. The SAPIEN library prints URLs containing token by default; this helper suppresses its output and redacts exception details. Download size is one model; complete staging is capped at 512 MiB unpacked, but network bytes are unknown in advance.
- In our **own** DEV runner, `require_drawer_scene_asset` executes before writing a new manifest and before importing ManiSkill. It checks pinned metadata and model URDF and sets `MS_ASSET_DIR` to the project-scoped root; hence `mani_skill.ASSET_DIR` resolves to the correct `.../data` directory. No third-party source is patched.
- URDF/metadata file checks are not a physical scene-loading validation. A successful environment build with zero PhysX actions is the next validation; only then the bounded DEV rollouts can execute.

## Commands on operator RTX 4090

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_prepare_roboreel_drawer_partnet.py' -v
python -m unittest discover -s tests -p 'test_roboreel_video_dependency_dev.py' -v
python -m scripts.prepare_roboreel_drawer_partnet
```

If `NEED_OFFICIAL_PARTNET_44962`:

1. If model `44962` has already been obtained under the official SAPIEN/PartNet-Mobility license, use `python -m scripts.prepare_roboreel_drawer_partnet --import-dir /path/to/44962` (without downloading).
2. Otherwise register for access via official [SAPIEN downloads](https://sapien.ucsd.edu/downloads) or request the gated [sapien-sim/PartNetMobility on Hugging Face](https://huggingface.co/datasets/sapien-sim/PartNetMobility). With authorized SAPIEN token, run `python -m scripts.prepare_roboreel_drawer_partnet --download` **interactively on the server**; the prompt hides the token. If no authorized token, do not run download or invent another data source. If only HF access is available, obtain the single original 44962 asset with licensed HF client and point `--import-dir` to its extracted directory; the HF repo layout has not been independently verified by this tool.

When `ROBOREEL_PARTNET_44962_READY` reports `READY_FOR_SCENE_INIT`, use **new** `v4`:

```bash
env -u DISPLAY -u WAYLAND_DISPLAY python -u -m scripts.roboreel_video_dependency_dev \
  --run --episodes 2 --max-steps 800 \
  --out outputs/roboreel_video_dependency_dev/v4
```

Do not rerun v3, which contains a frozen failed manifest. Do not download all cabinet models by default. Do not claim the checkpoint is physically validated by an asset check. Zero new experimental actions so far.
