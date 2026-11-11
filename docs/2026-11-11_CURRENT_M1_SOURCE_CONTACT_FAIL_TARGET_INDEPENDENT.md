# 2026-11-11 — Current M1 side-grasp replacement FAILED physical contact; decouple target-executor validation

## Verified user-server result

Original M1-B3 physics setup retained: task cube [0,0,.020] m, same final goal x=.210, blocker at x=.100 with original side pocket and 50mm throat.

The user's first `m1_current_executor_swap` **source_side_grasp** smoke completed and saved a genuine 106-step PhysX H5/MP4 and trace:

```json
{
 "mode":"source_side_grasp","success":false,"steps":106,
 "grasp_steps":0,
 "error":"RuntimeError: CONTACT_NO_GO: official RRT grasp pose executed, but Panda did not physically grasp cube"
}
```

`M1_EXECUTOR_REAL_SOURCE_GATE`: `success=false`,
`actual_grasp_steps=0`, `observed_lift_m=0`, final task XY
error `0.2100003638177595` m. The gate correctly blocked the
old runner's target stage: **target_original_planner NOT_RUN**.
The source OBB/RRT solver returning success does NOT equal actual
PhysX grasp success.

This does not establish if the failure is wrong TCP pose,
missing gripper-closure command, gripper finger interference with
narrow static walls, or wrong OBB/hand orientation.
A loose interpretation that the original paper's Panda PickCube
3/3 automatically transfers to this scene is invalid.
No data supports specific contact force or collision diagnosis yet.

## Corrected implementation priorities

**(1) Reuse the existing source failed and successful data — NO
new source replay.** New small offline comparator
`scripts/m1_current_compare_grasp_contact.py` reads:

- failed `outputs/m1_current_executor_swap_smoke/source_side_grasp/{status.json,physical_trace.json}`;
- already successful M1-B2 source
  `outputs/m1b2_same_goal_dev/source/{status.json,physical_trace.json}`.

It checks original 106-step evidence identity and reference
`gate.valid=true`, reports:
actual closed-gripper command count (do NOT infer gripper
physical aperture from action), first closure TCP XYZ,
TCP/object center distance at closure and 5 steps after,
minimum TCP/object center distance,
actual per-step `is_grasping_task`, cube max Z,
and source/target goal values **separately**.
It saves SHA256 hashes for original status/trace files.
No ManiSkill import, no GPU, no simulator. It does NOT relabel
the earlier x=.150 successful source as an x=.210 same-goal source.

**(2) Target original MPLib Push: independent hardware feasibility
probe, not chained to source success.** New runner
`scripts/run_m1_current_target_only_once.sh`
uses the existing (already committed) `target_official_push`
implementation on the exact same M1-B3 geometry, seed=120.
It uses Panda `pd_joint_pos` joint-space RRTConnect + Screw
physical motion, same blocker parking position, same x=.210 task
goal, always **OPEN** gripper. It does NOT use the former
`DiagnosticPushTeacher`, does not change geometry, and
will run at most once. Existing files are immutable, partial
artifacts cause a hard stop. This is a *privileged scripted
low-level executor*, NOT an unmodified published PushCube
solution, not a learned baseline, not a strong TAMP.

Expected outputs: `M1_TARGET_OFFICIAL_MPLIB_INDEPENDENT_RESULT`,
with actual per-stage physical state/errors, original
H5, JSON, MP4, trace.

## Server commands

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05

# No simulation; compare existing failure and true successful grasp:
python -m unittest discover -s tests \
    -p 'test_m1_current_compare_grasp_contact.py' -q
python -m scripts.m1_current_compare_grasp_contact

# One BOUNDED PhysX test of target solver, regardless of source failure:
bash scripts/run_m1_current_target_only_once.sh
```

The user should provide the `M1_SOURCE_CONTACT_ROOT_DIAG`
and `M1_TARGET_OFFICIAL_MPLIB_INDEPENDENT_RESULT` messages.
If target execution fails, preserve the original error,
do NOT blindly modify action gains and repeat.
If the source had close command but never physically grabbed,
inspect actual TCP/contact and gripper-wall clearance before
choosing a different **mature** source grasp tool. If target
executes, its physical feasibility is independent of missing
same-goal source.

## Research status remains fixed

M0 original RoboReel pi05 Uniform 46/60 and Vid2Robot 3/60
are frozen, original ManiSkill Pick reference 3/3 remains
valid only in its own task. Previous M1B2 source 170-step
success ended at x=.150; current M1B3 goal x=.210.
NO matched goal source success in new scene has been
established; neither source-video information advantage
nor strong equal-information TAMP nor M2 learned method
has been run. No unrelated long-horizon pivot or B01
training/dataset download.
