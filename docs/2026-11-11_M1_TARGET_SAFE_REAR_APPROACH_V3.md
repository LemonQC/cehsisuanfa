# 2026-11-11 — M1 current-scene target v3: prevent rear-approach from pushing task cube backward

## Real operator diagnosis that fixes ONE execution problem

Latest *read-only* user-server audit:

    M1_CLOSED_PUSHER_450_STEP_TRACE_DIAG
    no_new_physics=true
    closed_tool_clearance_success=true
    clearance_steps=82
    after_clearance_actual_steps=362
    cube_start_xyz≈[0,0,0.020]
    cube_final_xyz=[-0.05004404485,0.00167253753,0.01999992691]
    cube_net_goal_progress_m=-0.05004905309
    cube_total_xy_motion_m=0.07539328133
    cube_min_tcp_distance_m=0.03702491995
    last_15_cube_motion_m=0
    signature=CUBE_MOVED_FARTHER_FROM_GOAL_AFTER_CLEARANCE

The old code for the second-stage Panda/MPLib motion was:

    direction=(goal[:2]-cube[:2])/norm
    pre = [cube.x,cube.y,cube.z] - 0.050*direction_in_xy
    planner.move_to_pose_with_RRTConnect(pre)
    planner.move_to_pose_with_screw(goal+0.050*direction)

At this stage the original target cube starts x≈0 and its goal is
x=+0.210m. The previous **precontact TCP target** is x=−0.050m,
which numerically matches the final **physical cube** x=−0.050044m.
This supports a strong *specific* hypothesis that the **transition
from blocker clearance back to precontact may sweep against the
task cube and push it backward**, before any desired +X pushing.
The prior status does not provide the exact planner-stage boundary
inside the unfinished 362-step suffix, so this hypothesis is NOT
an experimentally proven temporal contact explanation. Do not
claim measured contact forces or direct cause solely from a
coordinate coincidence.

## v3 implementation: only current M1B3 executor hand-off changes

New code and one-shot runner:

- agcofl/m1_current_target_safe_approach_v3.py
- tests/test_m1_current_target_safe_approach_v3.py
- scripts/run_m1_target_safe_approach_once.sh

**Frozen exactly:** existing "AGCOFLM1B3Corridor-v0" custom ManiSkill
PhysX geometry with size-selective 50mm throat, dynamic blocker
at [0.100,0], side pocket at [0.100,0.125], original task cube
start [0,0,0.020], final goal [0.210,0,0.001], Panda
pd_joint_pos control, closed finger state -1, seed120, MAX_STEPS=450,
same original goal-success threshold 20mm, original physically
validated upstream \`PandaArmMotionPlanningSolver\`.

**Unchanged v2 successful phase:** close Panda fingers for 6
simulation steps, RRTConnect to blocker precontact pose, Screw
to blocker side-pocket target with identical old 50mm offsets,
required actual clearance residual <22mm and block y≥0.090m.
No edits to this previously valid low-level execution.

**Only modified target cube phase:**

1. From after-clear TCP pose, **retreat to 160mm height** at its
   existing x/y (RRTConnect). Task cube is monitored so that any
   backward displacement >6mm OR any unexpected horizontal
   displacement >10mm BEFORE deliberate push immediately throws
   a physical contact violation.
2. Travel at 160mm over the cube to x=cube.x−65mm, y=cube.y,
   still RRTConnect; same guards.
3. Descend from *behind* cube at fixed x and y to the measured
   cube center height (Screw). Physical TCP after descending
   must actually be at least 38mm behind cube center,
   within 12mm lateral and 18mm height; otherwise stop
   without asking it to push.
4. Only then run actual +X Screw pushing in three bounded
   substeps, TCP target x=.110, .180, .260; require measured
   task progress, no unexpected reverse x>6mm, no accumulated
   x setback >9mm, and y within ±10mm. At x>.095 require
   y within 6mm of central opening (task nominal halfwidth
   20mm, throat halfwidth 25mm: strict 5mm geometric band,
   6mm only as an early warning; the physical success may
   still fail at this margin). Abort immediately if
   directions/cube z/bilateral object grasp violate the
   frozen Push-only physical constraints. DO NOT change
   nominal model geometry to make a failure pass.
5. The final acceptance is the actual original PhysX
   \`evaluate()["success"]\`, measured goal error <20mm,
   both target and blocker grounded, blocker parked, both
   grasp states false, no source effect/Method/TAMP success
   inferred.

This is a **privileged-state executor repair** using already existing
upstream ManiSkill planning operations and observed cube position.
It is NOT a novel policy, TAMP, human-video editor, or paper-ready
comparison. The fixed x-substep waypoints and state guards are
engineering heuristics registered in this v3 attempt; do not claim
"unchanged algorithm" or independent learning capability.

## Real-data gate and no-replay command

\`\`\`bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05

bash scripts/run_m1_target_safe_approach_once.sh
\`\`\`

The runner first executes 5 CPU/no-simulator unit tests and
geometry preflight. It then **reads the actual original**
\`outputs/m1_current_closed_pusher_v2/target_closed_mplib\`
status+450-step trace and the old 35-step open-direct success
to independently verify x≈−0.050m physical outcome,
negative x goal progress, the 82+362 decomposition,
matching task start/goal and actual file SHA256. It
does not rely solely on hardcoded x constants. If the
real evidence does not match, **zero new PhysX**.

Only then does the runner allow **one** new, fixed,
max 450-action and 500s real PhysX attempt, in separate
\`outputs/m1_current_target_safe_approach_v3/target_safe_approach\`.
New raw H5, MP4, status and full per-action trace
are saved. Every recorded new action has an
\`execution_phase\` and the trace contains actual
TCP/object/gripper/finger states.

Markers:

- \`M1_TARGET_SAFE_APPROACH_PREFLIGHT\` — unchanged original scene
- \`M1_TARGET_SAFE_APPROACH_REAL_EVIDENCE_GATE\` — previous actual data justified the repair
- \`M1_TARGET_SAFE_APPROACH_RESULT\` — new physical result
- \`M1_TARGET_SAFE_APPROACH_GATE\` — final result extracted from actual status.json

If a motion command fails planning or the new guard detects
any negative progress, freeze resulting data and diagnose
the exact phase. DO NOT automatically increase step budget
or retune path gains. If it succeeds, retain target
feasibility but remember source real grasp still failed
(106 actions, 0 valid two-finger grasps).

## Scientific status not solved by engineering repair

Previous target clear physically succeeded either with
the original DiagnosticPushTeacher (32 actions) or
closed-tool Panda/MPLib (82 actions). Current target
reached only **negative-X**, not the goal; source Pick
at current x=.210 blocked-scene is invalid. Even a
v3 target success does not by itself establish the
required SAME GOAL source-target physical pair,
source-video information value, fair strong capability-aware
TAMP gap, learned effect-editing Method, or IEEE T-RO-level novelty.

**No new environment, no B01/M1C data/task pivot, no repeat of
original M0 46/60, Vid2Robot 3/60, Pick reference 3/3,
M1B2/B3/B4 or past source/target failure traces.**
