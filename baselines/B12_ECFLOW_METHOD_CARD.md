# B12 — EC-Flow method card

Status at source audit: `OFFICIAL_SOURCE_PINNED`; native assets/runtime on the user's server remain unverified; RoboReel port not started.

## Source lock

- Repository: https://github.com/YixiangChen515/EC-Flow
- Pinned revision: `7d0a6243fd8b92500b5f864d70b3b60d6b6c853a`
- Paper: ICCV 2025, “EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Embodiment-Centric Flow”.
- License: MIT at the pinned revision.
- Author-provided assets: flow predictor and goal-image predictor checkpoints from the authors' Hugging Face repository, plus SAM2.1, GroundingDINO and CoTracker checkpoints.

## Native reproduction path

The repository bundles its Meta-World code and uses Sawyer environments. Its native evaluation entrypoint is `experiment/eval_my_flow_policy.py`, launched by `experiment/eval_policy.sh`. The paper script evaluates nine Meta-World v2 goal-observable tasks with 25 episodes per task. The policy combines a DiT flow predictor, CLIP RN50 text conditioning, point sampling/segmentation and a geometry-based controller.

Expected assets:
- `ckpt/flow.pt`;
- `ckpt/goal_img.pt`;
- `sam_and_track/checkpoints/sam2.1_hiera_large.pt`;
- `sam_and_track/gdino_checkpoints/groundingdino_swint_ogc.pth`;
- `sam_and_track/co-tracker/checkpoints/scaled_offline.pth`.

The README says full training used 8 RTX 4090 GPUs. The first project milestone is released-checkpoint inference on one native Meta-World task, not an attempted eight-GPU training reproduction.

## Evaluation caveat found in source

The pinned `experiment/eval_my_flow_policy.py` increments success when `len(policy.images) <= 500`. The underlying `experiment/policy.py` sets `done = info['success']` and collects frames, but the aggregate script uses episode length as its success proxy rather than explicitly accumulating `info['success']`. A faithful native reproduction must preserve and report the authors' published/script metric, while also logging the environment success signal independently. Do not silently replace the metric or claim the proxy is equivalent without checking task semantics.

## RoboReel adaptation boundary

Native EC-Flow uses Sawyer/Meta-World, a task-specific point/flow geometry controller and the Meta-World camera/depth/segmentation model. RoboReel uses Panda wrist-camera RGB and `pd_joint_pos` 8-D actions. This is a substantial port, not a wrapper-only adaptation. First finish native inference and inspect flow/control traces. Any RoboReel bridge must explicitly implement camera geometry, robot embodiment transform, action/control mapping and matched success checks.

## Current gate

`L0_SOURCE=PINNED_AT_PUBLIC_REVISION`; `L1/L2_NATIVE=NOT_RUN`; `L3_ROBOREEL_PORT=NOT_STARTED`; `L4_FORMAL=NOT_RUN`; `L5_SCIENCE=NOT_RUN`.

Next step: run `scripts/audit_b11_b12_native_assets.py`; then choose the first native checkpoint smoke based on which complete asset set is already present and fits the 4090 memory budget.
