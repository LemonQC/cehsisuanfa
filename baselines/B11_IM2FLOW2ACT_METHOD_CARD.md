# B11 — Im2Flow2Act method card

Status at source audit: `OFFICIAL_SOURCE_PINNED`; native assets/runtime on the user's server remain unverified; RoboReel port not started.

## Source lock

- Repository: https://github.com/real-stanford/im2Flow2Act
- Pinned revision: `f44b368b76195c30bf38074f6419fc8b9f8feb5f`
- Paper: CoRL 2024, arXiv:2407.15208
- License: MIT (verified at the pinned revision); preserve the copyright and license notice.
- Author-provided assets: `checkpoints.zip`, Stable Diffusion 1.5 pretrained weights, `simulation_evaluation.zip`, and simulated-play data. No weights/data are bundled in this project repository.

## Native reproduction path

The official end-to-end evaluation entrypoint is `scripts/controll/evaluate_from_flow_generation.py`. It loads the flow-conditioned diffusion policy, its statistics/configuration, camera intrinsic and pose calibration, then invokes the author's evaluator and online point tracker. The default evaluation config expects:

- `checkpoints/manipulation_policy/epoch=300.ckpt` plus `.hydra/config.yaml` and `stats.pickle`;
- `checkpoints/flow_generation/epoch=1800.ckpt` plus `.hydra/config.yaml`;
- Stable Diffusion 1.5 weights under `pretrain_weights/StableDiffusion1.5`;
- original `data/simulation_evaluation/{drawer_open,pickNplace,pouring}`;
- `data_local/simulation/instrinsic_5-1.pkl` and `pose_matrix_5-1.pkl`.

This is a native UR5e/MuJoCo pipeline with calibrated camera/depth, 128 tracked points, flow horizon 32 and a 7-D policy action. The original configuration uses the authors' own simulated sphere demonstrations and evaluation buffers.

## Hardware/dependency constraints

- README explicitly says at least 24 GB GPU memory is needed to run online point tracking and policy inference concurrently. One RTX 4090 has 24 GB nominal memory, so this is a hard memory-risk gate, not a guarantee of fit.
- The pinned `environment.yml` is a large Python 3.10 environment with a mixed CUDA package stack (CUDA 11.8/12.1 entries and FAISS CUDA 11.2 build); do not install it into `pi05`.
- Use an isolated project-local environment only after checking free disk and current package conflicts. Preserve the author's original dependency pins; any repair must be logged as a port.
- First run a single native evaluation episode on an existing evaluation buffer after assets pass checks. No training from scratch initially.

## RoboReel adaptation boundary

The native policy outputs 7-D UR5e actions and uses the authors' MuJoCo task stack. RoboReel uses Panda wrist-camera observations and `pd_joint_pos` 8-D actions. Therefore B11 is **not** a drop-in RoboReel baseline. A port needs a separately tested observation/calibration adapter, robot/action retargeting, control-rate mapping and success/termination equivalence. Do not feed UR5e actions into Panda or change the RoboReel success oracle.

## Current gate

`L0_SOURCE=PINNED_AT_PUBLIC_REVISION`; `L1/L2_NATIVE=NOT_RUN`; `L3_ROBOREEL_PORT=NOT_STARTED`; `L4_FORMAL=NOT_RUN`; `L5_SCIENCE=NOT_RUN`.

Next step: run `scripts/audit_b11_b12_native_assets.py`; only download missing author assets after the exact missing set and disk budget are known.
