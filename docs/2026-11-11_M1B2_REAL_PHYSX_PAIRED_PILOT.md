# 2026-11-11 — M1-B1 3/3 official Pick success → M1-B2 SAME-GOAL physical pilot (first real candidate)

## Proven server results

M0: original published RoboReel video-conditioned π0.5 Uniform ND PushCube
**46/60 = 76.67%**, actual 60-trial SHA256 archive on GitHub
`research_reports/2026-11-11_m0_pi05_uniform/`, user commit `d5de864`.
Published Vid2Robot ND PushCube **3/60** frozen.

M1-A: 2 real RoboReel PushCube TRAIN H5+dual-camera episodes and 6
human PushCube TRAIN front MP4, current local machine confirmed.
No verified human PickCube TRAIN or matched-goal paired data.

**M1-B1 now ACTUAL PASSED**. User's local saved original ManiSkill
motionplanner `PickCube-v1` H5+JSON+RGB were re-verified **without
rerunning simulation**:

    M1B_OFFICIAL_PICK_EXISTING_VERIFIED
    requested_attempts=3
    recorded_episodes=3
    successes=3
    any_video=true
    usable_reference=true
    same_goal_pair_validated=false

This proves the official expert can grasp in original ManiSkill, NOT
same-goal Pick→Push success.

## M1-B2 now has executable code, not physical evidence

New `agcofl/m1b2_same_goal_effect_physics.py` defines a custom
`AGCOFLM1B2SameGoal-v0` PhysX scene (separate from official RoboReel
M0), based on pinned `PushCubeEnvV2`.

- Panda table, original RoboReel `PushCubeEnvV2` goal success
  (grounded task cube within XY radius).
- Frozen task cube start [0,0,0.02] and goal at +0.15 m along X,
  identical for *three independent resets using the same seed 120*.
- Movable dynamic blocker half-size 0.025 m, at X=0.077 m, Y=0.
- Two short static wall cubes half-size 0.027 m centered at
  X=0.14 m and Y=±0.083 m; tentative passage, *not a proven unique
  feasible corridor*. Blocker can attempt clearing toward Y=+0.115 m.
- Source case: available grasp, original ManiSkill
  `PandaArmMotionPlanningSolver` + OBB top-down grasp
  (`compute_grasp_info_by_obb`), then *adapted* 0.13 m lift,
  move above same XY goal, lower, release. This final placement
  and custom environment are **not** part of the original
  `solvePickCube` released policy. Fail any MPLib phase, lack
  of real grasp/lift/overflight/release or final grounded goal.
- Target direct control: Panda forced open-gripper, raw
  `DiagnosticPushTeacher` on the actual task cube.
- Target clear control: SAME privileged-state pushing
  primitive first targets the movable blocker at
  Y+0.115 m, then the original task cube at the same goal.
  Blocker must physically displace >0.075 m; task cube
  must reach goal grounded without grasp/lift.
- Source uses original MPLib `pd_joint_pos`; targets use prior
  privileged physical `pd_ee_delta_pose` push executor;
  this **mode difference is declared**. They are not learned.
- Underlying action states are recorded with real
  `RecordEpisode` H5/JSON/MP4 plus a separate time-indexed
  `physical_trace.json` for task cube, blocker, TCP, gripper
  commands and actual success state. The script explicitly flushes
  recordings because RecordEpisode(save_on_reset=False) does not
  flush merely on close.
- Fixed 450 step budget; 500s subprocess bound per case; no
  indefinite retry. Existing status/H5 are never overwritten.
  Any solver/runtime failure saves a clear invalid status.

**CAUTION:** The two short wall pieces do NOT establish that the
blocker is the only path to the goal. A target may go around,
or push the blocker and cube in series. This first fixed pilot only
checks physical success and source/target sequence contrast.
`target_clear` here is a **privileged-state scripted upper-bound**
with a fixed human-selected action order, NOT a capable
capability-aware TAMP baseline (which must actually plan its
own effect order from perception and allowed actions). Label it
honestly.

## Exactly one bounded server run

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05
    bash scripts/run_m1b2_same_goal_once.sh

Runner invokes one stdlib unittest suite, then 3 individual
independent actual PhysX episodes in the frozen scene:
`source`, `target_direct`, `target_clear`, each seed=120.
It executes none that already have `status.json` and refuses
overwriting raw H5/JSON from an interrupted process. No checkpoint
downloads, no original M0 model, no further PickCube-v1 test.

Out folder:

    outputs/m1b2_same_goal_dev/
      source/         status.json physical_trace.json source.h5 source.json *.mp4
      target_direct/  status.json physical_trace.json target_direct.h5 ...
      target_clear/   status.json physical_trace.json target_clear.h5 ...
      physical_gate_summary.json
      *-console.log

Expected terminal `M1B2_PHYSICS_GATE` — report fields:
`source_pick_place_valid`, `target_direct_push_success`,
`target_clear_then_push_success`,
`target_only_blocker_displacement_observed`,
`matched_physics_start_and_goal` and
`pilot_contrast_pass`. A source success only in original
PickCube-v1 does **not** satisfy the new source gate.

If source fails in adapted scene, record one actual error/trajectory
and do not perform blind parameter sweep. If target clear fails,
that proves only this scripted primitive did not achieve the required
contact transition; it does not prove goal infeasibility or
scientific novelty. If both targets succeed, there is no need to
claim inserted blocker clearing was necessary in this geometry.

After the single run, require a **separate** stronger topology
falsification and information-matched TAMP benchmark before M2.
No claim that the developed code or unit tests have been executed
on the user's GPU yet; physical outcomes are strictly NOT_RUN.

## M1-B3 stop/go

- B2 pilot valid if both robots actually achieve same goal, source
  grasp+lift+release, target clear causes real extra blocker motion.
- Determine if there is *any* push-around or simultaneous push
  plan using a physically constrained corridor/geometry control.
- Implement a strong capability-aware TAMP planner with equal
  perceptual state, goal, capability and primitive library, with a
  separate privileged simulator-state oracle upper bound. Compare
  Goal-only, Raw Flow, geometry replan and video-conditioned policy
  on exact same object-state goal.
- If ordinary equal-information TAMP solves the effect insertion
  and source video adds nothing beyond goal/state, stop the
  cross-behavior novelty claim rather than train an unnecessary editor.

Paper readiness: M0 complete; M1-B1 3/3 successful source *reference*
only; M1-B2 and B3 still require **actual measured results**; M2 not
started. Do not confuse robot RGB motionplanning reference with human
source video or original RoboReel cross-behavior benchmark.
