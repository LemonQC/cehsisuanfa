# 2026-11-11 — Implemented replacement of CURRENT M1 source/target executors (NO research pivot)

## Request clarified

The user wants to **replace the unreliable implementation IN THE CURRENT same-goal Pick→Push blocker-clear experiment**, not replace the scientific problem, start RoboReel multi-step tasks, or download another dataset. The unrelated B01 and M1C scaffolds introduced earlier were deleted before this patch.

This uses the old and already frozen `AGCOFLM1B3Corridor-v0` geometry, robot Panda, cube start [0,0,.02], common task goal x=.210, static walls, side-pocket blocker, seed 120 and 20mm task success criterion. M0/M1-B1/B2/B3/B4 raw data are NOT edited, deleted, or rerun. No new model or dataset.

## Why a real replacement is needed

The M1B4 *genuine* frozen-source preflight measured **0 grasp steps**, source cube max z≈.02m, total horizontal travel≈4.7µm. Although the old custom source's MPLib `approach/grasp/close/lift/across` calls returned nominal success, the cube was not held. Therefore a `lower`-only planner replacement is insufficient.

The old `DiagnosticPushTeacher` physically cleared the blocker in 32 steps, but pushed the task cube into the wall front at x≈.12527, y≈−.03082 and stalled. The old `open_direct` worked 35 steps, establishing the simulated robot and original passage permit some task actions.

## Direct upstream interfaces verified

From the ManiSkill original package:
- `mani_skill.examples.motionplanning.panda.motionplanner.PandaArmMotionPlanningSolver`
- `mani_skill.examples.motionplanning.base_motionplanner.utils.compute_grasp_info_by_obb` and `get_actor_obb`
- Official reference `panda/solutions/pick_cube.py`: OBB grasp, `move_to_pose_with_screw`, close gripper, move to goal.
- Official reference `panda/solutions/push_cube.py`: upstream Panda motion planner; precontact and planned physical push.

**Important reproduction differences:** the new source approaches the cube from -X (local grasp approach +X; gripper closing direction Y) instead of the previously unsuccessful top-down pose. It tries the original MPLib RRTConnect path planning to the reach/grasp and checks the *actual* `agent.is_grasping` contact state immediately after closing. Without real contact, stop before lifting/carrying. This is a single declared alternative, not the verified official original PickCube 3/3 policy. The target adapts the upstream joint-space planner into a two-object clear-first execution but keeps the Panda gripper **OPEN**; original upstream PushCube closes it. Do not call this an unmodified official solver, a learned policy, or TAMP.

## Code actually committed

- `agcofl/m1_current_executor_swap.py`:
  - `source_side_grasp` — exact frozen corridor, original OBB + MPLib *side* grasp, actual contact/lift/cross/place/release gates and original-object ground-truth trace/H5/MP4.
  - `target_official_push` — the published Panda/MPLib `pd_joint_pos` planner used for two real object contact motions (clear blocker to old side pocket then push task to old x=.210 goal), with forced open gripper and physical object checks; no gains tuned, no new task/waypoint experiment.
  - Underlying original `m1b3_corridor_physics.make_env` still used, new output folder only.
  - `--preflight` tests the original corridor geometry without physics.
- `tests/test_m1_current_executor_swap.py` — stdlib tests for frozen scene, original planner action convention (open gripper = +1 in `pd_joint_pos`), immutable outputs, and no old-mode relaunch.
- `scripts/run_m1_current_executor_swap_once.sh` — tests/preflight, **one bounded source** attempt first. If source does not **physically** grasp and actually place same cube within 20mm of same goal, it aborts **before** target. If source succeeds, one bounded target attempt, then stop. Each subprocess max 500s, PhysX max 450 steps, same one fixed seed 120. Never overwrite old experiments.

## Execution

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05

bash scripts/run_m1_current_executor_swap_once.sh
```

Expected first markers:

```
Ran 4 tests ... OK
M1_CURRENT_EXECUTOR_SWAP_PREFLIGHT { ... }
M1_EXECUTOR_SOURCE_OFFICIAL_SIDE_GRASP_SINGLE_SMOKE
M1_CURRENT_EXECUTOR_SWAP {"mode":"source_side_grasp",...}
M1_EXECUTOR_REAL_SOURCE_GATE {"success":...,"actual_grasp_steps":...}
```

If source success is false, report the precise `error`, `actual_grasp_steps`, TCP and grasp object state, and do not rerun or claim the same source. If source is true, the script automatically runs only one `target_official_push` case. All raw per-step, H5 and RGB files under `outputs/m1_current_executor_swap_smoke/`.

## Scope, evaluation and scientific gates

This is a **local executor swap/validation**, NOT a new benchmark experiment. Two new single-seed *smoke* trajectories may be generated ONLY to validate that the substituted existing low-level solver is usable in the already locked task. If user requires zero new simulator actions, run only `python -m agcofl.m1_current_executor_swap --preflight`, but actual physical replacement cannot be claimed without targeted execution.

The successful original ManiSkill Pick 3/3 videos can provide an existing expert reference, **but are not automatically a goal-matched M1 source video**. The original M1B2 source is genuinely successful but x=.15 versus frozen M1B3 x=.21. Neither is silently relabelled. An actually successful `source_side_grasp` at x=.21 is a newly validated in-task source under the same geometry.

Even if both executor smoke results succeed, neither strong equal-information TAMP nor source video added-information value nor new learned algorithm has been evaluated. No M2 training until those original scientific gates pass. If the solver substitute fails, that is a failed executor attempt, not evidence that the entire proposed cross-behavior problem is mathematically impossible.
