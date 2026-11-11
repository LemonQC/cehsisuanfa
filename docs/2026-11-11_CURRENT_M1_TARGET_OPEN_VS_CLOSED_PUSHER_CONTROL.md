# 2026-11-11 — Current M1 target: PhysX original-Panda open/closed finger reversal audited; ONE matched controller ablation

## User-provided new REAL operator result

One independent target Panda/MPLib physical rollout on the unchanged
`AGCOFLM1B3Corridor-v0` has **actually executed**:

```json
{
 "mode":"target_official_push",
 "success":false,
 "steps":88,
 "grasp_steps":0,
 "error":"RuntimeError: PHYSICS_clear_blocker_GOAL_NOT_REACHED: error 0.0933m",
 "phases":[{
   "name":"clear_blocker", "actual_steps":82,
   "initial_xy_m":[0.09999964386224747,-0.0000006951020168344257],
   "goal_xy_m":[0.1,0.125],
   "observed_xy_m":[0.1073780357837677,0.03197958692908287],
   "actual_error_m":0.09331255360298853,
   "actual_grasp_steps":0
 }]
}
```

The 88 actual action steps comprise 6 explicit gripper commands
plus 82 reach/push steps. The blocker only moved ~32.8mm from
its true start, still 93.3mm from the side parking goal.
The subsequent `push_task` never executed. This adapter
cannot replace the previous physically successful clearance
primitive based on its observed evidence.

## Source-level root cause requiring correction

**Official upstream verified:** `mani_skill/examples/motionplanning/two_finger_gripper/motionplanner.py`
declares `OPEN=+1`, `CLOSED=-1`. The official original
`mani_skill/examples/motionplanning/panda/solutions/push_cube.py`
calls `planner.close_gripper()` **BEFORE** the reach and push.
The new failed target adaptation explicitly called
`planner.open_gripper(t=6)`: it is NOT a faithful reproduction
of even the upstream tool/contact configuration.

The original successful M1B3 clearance from
`outputs/m1b3_corridor_dev/blocked_clear` used
`action[..., -1]=-1` (CLOSED), and reached side pocket within
5.8mm in 32 actual steps. But the legacy M1B3 code declares
`gripper_never_closed = all(gripper_action <= 0)`, incorrectly
labeling the exact **closed** action as 'never closed'.
Old M1B2 also used `-1` but reversed the
`saw_gripper_close` predicate (checked `>0`).
**These legacy metadata fields must no longer be used as
evidence of open-finger execution.** The prior physical
object positions and successes remain recorded facts, but
the old descriptions of the end-effector configuration are
wrong. Preserve the original raw/status exactly, and append
an audit note rather than changing old raw video/H5 or
retroactively inventing corrected experimental data.

IMPORTANT: A mechanically CLOSED Panda gripper may be used
as a narrow rigid PUSHER **without grasping either object**.
Push-only constraints should be validated by physically
observed `is_grasping(task)==False`,
`is_grasping(blocker)==False`, object grounding and max-Z
clearance. Explicitly record whether gripper was open or
closed; do not conflate these with no object grasp.
The earlier trajectories did not log blocker grasp flag, so
they cannot claim it was never grasped throughout a trial
without a new physical check.

## One-factor controlled experiment (optional targeted smoke, not a new benchmark)

New code `agcofl/m1_current_closed_pusher_ablation.py`
and `scripts/run_m1_target_closed_pusher_once.sh`
use the **same custom M1B3 geometry/seed/goal** and exactly
the previous `target_official_push` MPLib RRTConnect
precontact + Screw push phases and original 50mm precontact
and postcontact offsets. The **sole intended change**
is `planner.open_gripper(t=6)` → `planner.close_gripper(t=6)`.
Because hand closure can affect robot initial configuration
and contact physics, the resulting realized trajectory
need not be mathematically identical. This is still a
controlled interface/contact-mode ablation, not
a pretrained learning baseline, not equal-info TAMP,
and not an end-to-end source-target pair.

The new `ContactTracer` records both Panda
`is_grasping(obj)` and `is_grasping(blocker)` at each
actual simulator action, both finger joint qpos, and
actual simulator cube/blocker/TCP positions. Successful
closed-tool Push-only requires NO grasp of either actor,
grounded task/blocker, actual side-clearance to pocket
and original 20mm task final goal.

Runner PRECONDITION checks the immutable genuine earlier
target CLOSED successful clearance first 32 actions, and
the genuine latest failed MPLib OPEN 88-step trace.
Only if both raw records and sign commands exactly match
does it authorize a single fresh 450-step / 500s bounded
target-only trial. New result folder:
`outputs/m1_current_closed_pusher_v2`.
Never overwrite the old M1B2/3/4, old side-grasp 106-step
failure or prior open MPLib 88-step data.

Run on the user's `pi05` server:

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05
bash scripts/run_m1_target_closed_pusher_once.sh
```

Expected markers:

- `M1_TARGET_GRIPPER_SEMANTICS_PREFLIGHT`
- `M1_TARGET_SIGN_ROOT_CAUSE_EVIDENCE`
- `M1_TARGET_CLOSED_PUSHER_PHYSICS`
- `M1_TARGET_CLOSED_PUSHER_RESULT`

Interpretation after that *single* test:
- **If blocker moves fully and cube reaches goal** with neither
  object grasped, target push execution can be kept as an
  instrumented physics oracle, not new method.
- **If blocker still not parked**, reject the MPLib push adapter.
  REUSE the already successful old `DiagnosticPushTeacher` for
  clearance and focus only on the task-cube passage/handoff.
  Do NOT repeat the failed side-grasp source or run a grid of
  pusher gains/poses.
- **If blocker clears but cube stalls again**, isolate task
  alignment at the original x≈0.126 wall front; retain the
  existing valid clearance and stop rewiring that phase.

## Source issue is independent and still open

The new side-grasp source trial has **106 real actions,
no physical grasp and no cube lift**. The read-only
`scripts/m1_current_compare_grasp_contact.py` already
exists to compare its true TCP and gripper commands to
the prior genuinely successful M1B2 170-step source.
This comparison has not been supplied in the current
chat, so no unsupported determination of finger collision
vs TCP misalignment should be made. Do not restart the
source trial until the actual offline contact comparison
is examined. Prior M1B2 x=.15 source is NOT the x=.21
M1B3 source pair.

No M0 rerun, no new data download, no new environment,
no Method/M2 or claimed T-RO gap. Honest scientific
readiness remains: source-target same-goal physics pair
NOT established, strong equal-info TAMP NOT_RUN,
video-specific information gain NOT_PROVEN.
