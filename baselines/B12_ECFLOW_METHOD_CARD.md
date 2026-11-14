# B12 — EC-Flow method card

Status at source audit: `OFFICIAL_SOURCE_PINNED`; native assets/runtime on the user's server remain unverified; RoboReel port not started.

## Source lock

- Repository: https://github.com/YixiangChen515/EC-Flow
- Pinned revision: `7d0a6243fd8b92500b5f864d70b3b60d6b6c853a`
- Paper: ICCV 2025, “EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Embodiment-Centric Flow”.
- License: MIT at the pinned revision (verified in the source LICENSE).
- Author-provided assets: flow predictor and goal-image predictor checkpoints from the authors' Hugging Face repository, plus SAM2.1, GroundingDINO and CoTracker checkpoints.

## Native reproduction path

The repository bundles its Meta-World code and uses Sawyer environments. Its native evaluation entrypoint is `experiment/eval_my_flow_policy.py`, launched by `experiment/eval_policy.sh`. The paper script evaluates nine Meta-World v2 goal-observable tasks with 25 episodes per task. The policy combines a DiT flow predictor, CLIP RN50 text conditioning, point sampling/segmentation and a geometry-based controller.

## Asset split for the actual policy-evaluation path

A source-level trace of the pinned evaluation entrypoint narrows the assets needed for native policy evaluation:

- **Required learned weights:** `ckpt/flow.pt` and the OpenAI CLIP RN50 weight cache used by `clip.load("RN50")`.
- **Python code dependencies imported at module load:** SAM2, GroundingDINO and CoTracker packages are imported by `inference.py` through `sam_and_track.utils`; their importable code may be needed even though the evaluation path does not instantiate these models.
- **Not loaded by `experiment/eval_my_flow_policy.py`:** `ckpt/goal_img.pt`, SAM2 checkpoint, GroundingDINO checkpoint and CoTracker checkpoint. Those are used by the separate standalone `inference.py` visualization/demo path, not by `FlowPredModel` used in the Meta-World policy evaluator.

The policy evaluator constructs `FlowPredModel`; its constructor loads the `DiT-XL-NoPosEmb-Lang` flow predictor from `ckpt/flow.pt` and CLIP RN50. It does not instantiate `GoundedSam2`, `PointTracker` or the goal-image DiT. The audit now reports this distinction explicitly to avoid downloading unused large weights.

The Hugging Face model card lists `flow.pt` at about 9.46 GB and `goal_img.pt` at about 715 MB. The large flow file is a disk/host-RAM loading concern, not a direct estimate of GPU VRAM; the actual GPU peak must be measured with the official model and one forward pass. The flow checkpoint is loaded to CPU first, so available system RAM must also be checked.

The README says full training used 8 RTX 4090 GPUs. We will not reproduce that training recipe. The first project milestone is released-checkpoint inference on one native Meta-World task, subject to the single-4090 memory gate.

## Evaluation caveat found in source

The pinned `experiment/eval_my_flow_policy.py` increments success when `len(policy.images) <= 500`. The underlying `experiment/policy.py` sets `done = info['success']` and collects frames, but the aggregate script uses episode length as its success proxy rather than explicitly accumulating `info['success']`. Because `collect_video()` can also break on an exception, a short partial episode may be counted as success by the wrapper. A faithful native reproduction must preserve and report the authors' script metric, while separately logging `info['success']`, termination/truncation and exception state. Do not silently replace the metric or claim the proxy is equivalent.

## RoboReel adaptation boundary

Native EC-Flow uses Sawyer/Meta-World, a task-specific point/flow geometry controller and the Meta-World camera/depth/segmentation model. RoboReel uses Panda wrist-camera RGB and `pd_joint_pos` 8-D actions. This is a substantial port, not a wrapper-only adaptation. First finish native inference and inspect flow/control traces. Any RoboReel bridge must explicitly implement camera geometry, robot embodiment transform, action/control mapping and matched success checks.

## Current gate

`L0_SOURCE=PINNED_AT_PUBLIC_REVISION`; `L1/L2_NATIVE=NOT_RUN`; `L3_ROBOREEL_PORT=NOT_STARTED`; `L4_FORMAL=NOT_RUN`; `L5_SCIENCE=NOT_RUN`.

Next step: inspect the B12 policy-evaluation profile in the updated asset audit. Do not download `goal_img.pt` or SAM2/GroundingDINO/CoTracker weights for the policy evaluator. Only consider `flow.pt` after checking local presence, host RAM, CLIP RN50 cache and code dependencies; then run an isolated model-only GPU memory preflight before starting Meta-World.
