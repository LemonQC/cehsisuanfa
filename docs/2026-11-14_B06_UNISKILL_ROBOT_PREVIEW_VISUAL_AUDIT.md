# TRO1 · B06 UniSkill StackCups 真实 ROBOT 轨迹可视审计（2026-11-14）

## Evidence source and strict status

This is based on the genuine ROBOT-policy contact sheet, JSON and README uploaded to GitHub. The image is **not** an original human demonstration.

- [4x6 ROBOT preview](https://github.com/LemonQC/cehsisuanfa/blob/tro1-experiment-evidence/evidence/runs/b06_uniskill_stackcups_dev_v1_514e11b6d07ac005/robot_rollout_preview.jpg)
- [evidence.json](https://github.com/LemonQC/cehsisuanfa/blob/tro1-experiment-evidence/evidence/runs/b06_uniskill_stackcups_dev_v1_514e11b6d07ac005/evidence.json)
- [README.md](https://github.com/LemonQC/cehsisuanfa/blob/tro1-experiment-evidence/evidence/runs/b06_uniskill_stackcups_dev_v1_514e11b6d07ac005/README.md)

**DEV_COMPLETE_NOT_FORMAL** — 4 actual PhysX robot-policy rollouts completed; 0/4 succeeded, 800 steps each (3200 total), no runtime errors. Paired TRAIN demo IDs 0000/0001 and reset seeds 310000/310001; not the 20 held-out demos × 3 seeds.

The user's terminal excerpt says "Ran 18 tests ... OK" and independently confirms successful evidence publication. That test count does not imply 18 physical rollouts. The exact selected test suite is not specified by that excerpt.

## Visual audit of sampled frames

The montage consists of six frames per rollout around steps 0, 160, 320, 480, 640 and 800. The rows correspond to episode_00/source_0000, episode_00/source_0001, episode_01/source_0000, episode_01/source_0001.

- The end effector moves down toward the cups in every row. It is false to claim the robot never approaches them.
- Some cup configurations visibly change. In the bottom row the red cup appears tilted/displaced laterally by later frames.
- No completed stable three-cup stack is visible in the sampled frames.
- The sparse montage **does not prove** whether a secure grasp or meaningful contact was achieved, nor determine whether grasping, placement, normalization, control execution or policy conditioning is the main failure cause.
- Do NOT assign failure labels like NO_CONTACT or SUCCEEDED_PICK_FAILED_PLACE without a higher-resolution trace or instrumented evidence.

## Matched-pair evidence

Episode 0: input videos 0000 vs 0001; mean first-common-step 8D action L2 0.049234755, successes [false, false].
Episode 1: input videos 0001 vs 0000; mean first-common-step 8D action L2 0.077750869, successes [false, false].
Matched-pair success preference reversal 0/2. Changing videos changes some robot actions but does not establish correct goal semantics or causal video necessity.

## Decision and next gates

1. Freeze the original four B06 DEV results; no additional StackCups PhysX or speculative retuning; never alter success predicates, checkpoints or evaluation scenes.
2. If actionable debugging is desired, first analyze **already recorded** robot MP4 and action traces read-only for cup lifting, tipping and action saturation. Don't infer contact force from JPEG appearance.
3. Do not spend on B06 official 20×3 held-out trials unless a grounded reason for doing so is identified. No claim of 0% official UniSkill performance.
4. Continue independent B01/B02 common authentic TRAIN gate (GitHub PR #1), and audit B11/B12 source portability. Distinguish dataset adapter from model training and from official evaluation.
5. TRO1's separate G0 science gate remains BLOCKED until legitimate same-scene opposing-goal human video evidence is available.

Labels: AUTHOR_POLICY_REAL_PHYSX_DEV_EXECUTED; FORMAL_HELDOUT_EVAL_NOT_RUN; NEW_PHYSX_THIS_AUDIT=0; NEW_TRAINING_THIS_AUDIT=0.
