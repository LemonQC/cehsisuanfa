# B09 — Point Policy method card

Status: public author code and dataset located; no author policy checkpoint or RoboReel runner verified; native reproduction requires a per-task dataset, Franka camera calibration and training.

## Source lock

- Repository: https://github.com/siddhanthaldar/Point-Policy
- Revision: `23d82ad90bf08815afba6d3e9c14347775e73c74` (main, 2025-07-21)
- Paper: arXiv:2502.20391, “Point Policy: Unifying Observations and Actions with Key Points for Robot Manipulation”.
- License: MIT.
- Submodules: Franka-Teach, CoTracker fork, DIFT fork; the repository uses Franka-specific data collection and calibration code.

## Native reproduction path

The author instructions provide training and inference commands for BC, MT-π, P3PO and Point Policy. Inference requires a trained `bc_weight`; the repository does not advertise a downloadable pretrained policy checkpoint. The official data is hosted at https://huggingface.co/datasets/siddhanthaldar/Point-Policy and contains eight task pickle files totaling about 16.3 GB. Individual task files are about 1.1–3.0 GB.

The model uses semantically meaningful human hand/object keypoints, Franka camera calibration, CoTracker/DIFT features, and robot actions converted from human demonstrations. The default config is real Franka environment and camera images at 640×480. The original paper reports eight real-world tasks.

## What is and is not portable

RoboReel's Panda is a Franka embodiment, which is closer than UR5e/Sawyer baselines. However, the Point Policy data and runner are not RoboReel: they assume their own Franka-Teach teleoperation, calibration file, two camera streams, point labels and robot action dataset. A faithful RoboReel port must reconstruct the same point representation from RoboReel camera calibration and robot observations, then map the learned action head into the pinned Panda control interface. Do not pass their real-robot action vectors directly into ManiSkill.

## Resource decision

- Do not download the full 16.3 GB dataset by default.
- First check whether any task-specific pickle already exists in the project assets.
- If not, choose one task only after checking the task has a semantically compatible RoboReel task and that a valid robot-action dataset exists.
- Since no released policy checkpoint was found in the repository instructions, native evaluation needs training from the selected task's data; do not count an untrained/random policy as a smoke result.

## Current gate

`L0_SOURCE=PUBLIC_PINNED`; `NATIVE_DATA=PUBLIC_BUT_NOT_LOCAL_VERIFIED`; `NATIVE_TRAIN/EVAL=NOT_RUN`; `ROBOREEL_PORT=NOT_STARTED`; `FORMAL=NOT_RUN`.
