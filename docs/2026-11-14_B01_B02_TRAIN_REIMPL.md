# TRO1 — B01/B02 RoboReel robot-only train/eval adapter (2026-11-14)

## Truthful scope

- **B01**: RoboReel RGB+qpos 2-frame Transformer Behavior Cloning. This is **REIMPL_ON_ROBOREEL**, not a released ACT/Transformer-BC author's implementation.
- **B02**: same RGB+qpos and actions, compact conditional DDPM noise-prediction model. **REIMPL_ON_ROBOREEL**, not original Diffusion Policy architecture/weights.
- Both use full TRAIN demonstrations 0000–0079 only; demo-level 80/20 split, train-only action/qpos means and stds, base_camera with pd_joint_pos 8-D actions; provenance SHA256 for actual input H5 and both MP4s; joint qpos from articulation columns 13:22; actual live state from agent/qpos.
- Official RoboReel PushCube-v2 env is never modified. Formal results require frozen env bank, held-out scene bank entries 0–59 split in 20×3 passes, original success metric, nonprivileged agent observation. No human video is read by robot-only policy. **One 20-scene pass is not 60-rollout final evidence.**
- An offline integration test with synthetic 2-demo H5+MP4 was run on CPU: **4/4 tests passed**; both B01 and B02 one-step training, checkpoint load, output and backprop tested. **Real user 4090 training and PhysX runs have NOT RUN; no success rate is claimed.**
- DEV evaluation uses new reset seeds 310000+N; it **never opens official held-out bank**. Formal mode cannot use a small-dev checkpoint.

## Foreground commands on user server

Use an existing working PyTorch+ManiSkill Conda env; do not overwrite pi05 if dependency conflict. Verify `torch`, `h5py`, `numpy`, `Pillow`, `imageio`, `imageio-ffmpeg`, `mani_skill` are importable. Paths are under the existing project disk.

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git fetch origin
# After merging PR #1, or checkout its branch for the additional code:
git switch tro1-b01-b02-shared-train-data-gate-20261114
git pull --ff-only origin tro1-b01-b02-shared-train-data-gate-20261114
python -m unittest discover -s tests -p test_robot_only_reimpl.py -v

# For a quick genuine-data engineering smoke with >=2 demos, explicitly label SMALL_DEV_DATA_ONLY:
python -m agcofl.robot_only_reimpl prepare \
  --data-root outputs/v120_baselines/official_dataset/robot_trajectories/no_distraction/push_cube \
  --cache-root outputs/b01_b02_robot_only_cache_smoke --min-demos 64 --allow-small-dev

python -m agcofl.robot_only_reimpl train --method b01 \
  --cache-root outputs/b01_b02_robot_only_cache_smoke \
  --out outputs/research_runs/b01_smoke_v1 --steps 100 --batch 16 --allow-small-dev
python -m agcofl.robot_only_reimpl train --method b02 \
  --cache-root outputs/b01_b02_robot_only_cache_smoke \
  --out outputs/research_runs/b02_smoke_v1 --steps 100 --batch 16 --allow-small-dev

# Real physically executed non-heldout DEV scene, 2 episodes each; optional robot MP4:
python -m agcofl.robot_only_reimpl eval \
  --checkpoint outputs/research_runs/b01_smoke_v1/best.pt \
  --out outputs/research_runs/b01_physx_dev_v1 --mode dev --episodes 2 --save-video
python -m agcofl.robot_only_reimpl eval \
  --checkpoint outputs/research_runs/b02_smoke_v1/best.pt \
  --out outputs/research_runs/b02_physx_dev_v1 --mode dev --episodes 2 --save-video
```

If real 64+ demos are already downloaded, use a **fresh cache directory**, omit `--allow-small-dev`, train in distinct paths with more updates and monitor validation loss. The code refuses an existing cache/output to avoid overwriting experiments.

Formal 20×3, **only once** pretrained checkpoints are audited, real-data quality and action compatibility are verified and DEV is meaningful (needs official held-out bank `roboreel_env/init_states/without_distraction/push_cube.pkl`):

```bash
# Demonstrates one formal pass, not the pooled 60.
python -m agcofl.robot_only_reimpl eval \
  --checkpoint outputs/research_runs/b01_formal_train_v1/best.pt \
  --mode formal --episodes 20 --pass-index 0 \
  --out outputs/research_runs/b01_official_bank_pass0_v1 --save-video
# Repeat with --pass-index 1 and 2 into distinct output directories,
# then aggregate success k/60 and compute 95% Wilson CI.
```

## Current local-data fact

The repository's immutable `research_reports/2026-11-09_v120/robot_data_manifest.json` records that genuine official robot TRAIN triplets for PushCube IDs `0000` and `0001` were acquired under `outputs/v120_baselines/official_dataset`. The older v110 manifest records a 404 for its former dataset path. Therefore the code default and commands use the v120 path. With the two currently recorded demos, `prepare --allow-small-dev` can only create a `SMALL_DEV_DATA_ONLY` cache (one train / one validation demo); this is an engineering smoke, not a formal baseline. Do not increase the count by duplicating episodes.

## Unresolved preflight checks

1. Verify actual H5 trajectory row/MP4 decode and articulation robot key match pinned official source; any mismatch is a hard error, not silently truncated.
2. Runtime dependency and action-space behavior in user's exact physics/CUDA environment remain unverified. Do not call a checkpoint an official author reproduction.
3. No actual 64-demo training, 2× PhysX smoke, or 20×3 formal result is claimed by the code delivery alone.
4. PushCube fixed +X success is a **robot competency control**, not evidence of cross-behavior/video necessity. The G0 dual-opposing-goal source-video gate remains BLOCKED.
