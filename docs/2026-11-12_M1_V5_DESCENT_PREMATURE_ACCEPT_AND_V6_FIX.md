# 2026-11-12 — M1 same original corridor: actual v5 premature descent acceptance, one v6 corrective test

## Operator REAL v5 physics, immutable raw evidence

Latest true user server result:

```json
{"success":false,"steps":54,"clearance_valid":true,
 "goal_error_m":0.2100006481501995,
 "error":"RuntimeError: ORIGINAL_PUSH_CONTACT_POSE_INVALID",
 "cube_final":[-6.547037401105626e-7,-1.1169277058797888e-6,.019999919459223747]}
```

Actual original v5 stages:
- original pd_ee_delta_pose closed-tool blocker `ActorPush`: **32 steps successful**, parking error 0.005780176577m
- `retreat_vertical`: 7 steps, TCP z=.150545m
- `hover_rear`: 9 steps, TCP x=−.045112m, z=.161158m
- `descend_rear`: 6 steps, final TCP [−.0592191070,.0001949806,.0568885580]m
- deliberate `push_cartesian_original_reference`: **0 steps**. Cube unchanged, both actor grasp flags false, both closed fingers qpos zero

The requested rear approach TCP target was approximately
[−.060,0,.038]m. Actual total XYZ difference is ≈**18.91mm**.

The v5 call in `execute()` was:
```python
motion_to(traced,"descend_rear",behind,start,tolerance=.022)
```
It stopped once the **Euclidean** 3D residual became **≤22mm**.
That prematurely treated a 56.89mm-high TCP as an accepted
38mm-high contact pose. Immediately afterward,
`push_to_goal` required `0.032 ≤ TCP_z ≤ 0.052`m.
Since actual TCP z=.05689m > .052m, the **internal contract
itself is inconsistent**. This is a definite program
acceptance bug; it is not a measurement of a wall collision,
a proven robot reachability failure, or a failure of the
original Cartesian push primitive.

## Exact single-factor v6 correction

New separate files:
- `agcofl/m1_original_scene_cartesian_pusher_v6.py`
- `tests/test_m1_original_scene_cartesian_pusher_v6.py`
- `scripts/run_m1_original_scene_cartesian_pusher_v6_once.sh`

Only change in executed logic beyond distinct outputs/identifiers:

```python
DESCEND_REAR_TOLERANCE_M = .008
motion_to(traced,"descend_rear",behind,start,
          tolerance=DESCEND_REAR_TOLERANCE_M)
```

The reference contact target **remains** TCP z=.038m.
8mm total distance tolerance prevents accepting the
measured 18.91mm residual from v5. It does not relax
`push_to_goal` physical preconditions. The controller
may still be unable to reach within 8mm; if so,
`motion_to` stops on its existing 50-action finite
limit and reports `NO_VALID_POSITION_AFTER_descend_rear`,
rather than claiming a completed contact pose.

**Everything else stays frozen:** M1B3 original Panda scene,
blocked side door and 50mm throat, blocker and cube initial
state, final cube goal x=.210m ±.020, seed120, original
`pd_ee_delta_pose`, 7D control, closed Panda gripper
`-1`, original successful `ActorPush` blocker clear,
safe vertical retreat and overhead return, original
`delta_pose_action` +X push, physical no-grasp,
grounding and lateral guards, max 450 total actions,
120 push actions and the 15-step TCP+cube stall stop.

No old source, M0, M1B1/2/3/4, v2/3/4/5 episodes rerun.
All old H5/MP4/JSONs preserved. v6 has new
`outputs/m1_original_scene_cartesian_pusher_v6/target_cartesian_contact_fixed`.

## How to run exactly once

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05
bash scripts/run_m1_original_scene_cartesian_pusher_v6_once.sh
```

The runner performs four CPU-only regression tests, frozen
scene preflight, and a **read-only actual** old v5
54-step trace gate: checks 4 real phases with action
counts 32/7/9/6, initial goal/cube, actual rear
TCP [.05689 Z], Euclidean residual accepted by old
22mm threshold but not new 8mm threshold, and original
trace file SHA256. If any mismatch, no new physics.

If evidence matches, one bounded new original-scene PhysX
target-only execution is allowed. It writes real
H5/MP4, status and per-action trace. Old data are immutable.

Final marker: `M1_V6_PHYSICAL_GATE`.
Do **not** interpret code completion or source/reference
differences as physical success.

## Decision and scientific honesty

- If actual descent reaches contact and +X push succeeds
  under unchanged goal/physical gates, the **target
  execution instrument** is workable on this one seed.
- If descent still fails, stop engineering iterations and
  inspect fingertip link/collision/IK rather than tweaking
  gain or blindly increasing episode budget.
- If target push starts but stalls, honor the 15-step early
  stopping and inspect actual grip/robot collision.
- The **source** same-goal Pick still recorded 0 valid
  physical grasp steps (106 actions). Strong equal-info
  TAMP, video-specific advantage and learned M2 remain
  NOT_RUN. Target-only success is insufficient for
  T-RO scientific viability.

This document records actual v5 operator observations
and code differences; v6 server physics is NOT_RUN.
