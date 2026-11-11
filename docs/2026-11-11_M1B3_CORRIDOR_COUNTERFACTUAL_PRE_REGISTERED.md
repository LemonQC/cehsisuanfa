# 2026-11-11 — M1-B3 corridor/counterfactual candidate: code prepared, NOT run

## Why this is a NEW scene

The previous M1-B2 real PhysX scene must remain frozen:
- Source Pick+Place: **SUCCESS, 170 steps**
- Target direct Push: **SUCCESS, 31 steps**
- Target manually clear blocker → Push: **SUCCESS, 51 steps**
- `pilot_contrast_pass=false`; the previous small obstacles did NOT
  make clearing necessary. Raw H5/videos remain on the user's server.
  Published M0 RoboReel π0.5 Uniform 46/60 and official ManiSkill
  PickCube original expert 3/3 are also frozen and must not be rerun.

This experiment deliberately changes the **task topology**, not any
previous source/target model weights or low-level policy gains.

## M1-B3 fixed geometry and evidence boundaries

New `AGCOFLM1B3Corridor-v0` is a *CUSTOM* ManiSkill PhysX task,
on pinned RoboReel PushCubeEnvV2 with Panda and genuine dynamic
actor contacts. It is NOT any released RoboReel eval scene.
Nominal object start [0,0,0.020] m, grounded goal [.210,0,.001] m,
success tolerance **20 mm**, blocker start [.100,0,.029] m;
blocker half-width 29 mm versus task half-width 20 mm.

- Corridor rails at y ±68 mm leave inner clear edges y ±52 mm.
  Side doors at x from 65 to 135 mm (70 mm opening);
  there are matching lower/upper side openings.
- A size-selective throat centered at x=158 mm permits an
  approximately **50 mm** opening. The 40 mm task cube fits
  with limited margin, while the 58 mm blocker **cannot**
  fit at table height.
- A fixed perimeter fence at x around −95 and 285 mm and
  y ±170 mm disallows simple end-runs around the corridor.
  Walls are 75 mm high. The grasp-capable source must lift
  the task cube over them and release it grounded at the
  *same final goal*.
- Side parking location is blocker center [.100,.125,.029] m;
  the blocked condition starts it at y=0.
- An offline 3 mm-grid square-footprint configuration-space
  BFS checks (1) BLOCKED static blocker prevents task-table
  traversal and (2) OPEN parked-blocker corridor permits
  task-table traversal. Separate width checks require task
  fits throat, blocker fails throat, blocker fits door,
  and blocker fits inside parking fence.
- **This is not an impossibility proof for arbitrary dynamics.**
  It holds only for a stationary-blocker conservative 2D
  tabletop model. PhysX may allow pushing, rotation, temporary
  tip, non-quasistatic contact or an as-yet-unmodeled plan.
  A later strong TAMP comparison is mandatory.

**Action interfaces and provenance:** the source uses the existing
adapted official ManiSkill Panda/MPLib OBB Pick+Place solver
(`pd_joint_pos`); target uses the same existing privileged-state
`DiagnosticPushTeacher` primitive (`pd_ee_delta_pose`,
gripper forced open) as a physical feasibility diagnostic.
No new learned method, fair RGB policy, strong TAMP or video-effect
editing has been trained. The manually prechosen clear-first order
is *not* a TAMP planner.

## Exactly FOUR pre-registered single-seed PhysX cases

```
blocked_source : blocked gate, source Pick/lift/overfly/place/release
blocked_direct : blocked gate, target push cube directly to goal
blocked_clear  : blocked gate, target physically move blocker into
                 side pocket, then push cube to same goal
open_direct    : parked blocker at side pocket, target directly
                 push cube to same goal
```

All four modes use seed 120 and up to 450 real robot actions.
Each case has its own immutable status/H5/JSON/MP4 and per-step
measured task/blocker/TCP position and gripper state. Separate
subprocesses avoid contaminated PhysX state; no case overwrites
the historic M1-B2 results.

Physical source success requires real held grasp, high lift,
crossing the NEW throat above the wall, grounded placement within
20 mm, *without first moving the blocker*. Physical target
success requires a grounded task cube at the exact same goal,
no grasp/no gripper close, task and blocker remain supported on
the tabletop. `blocked_clear` also requires measured blocker
displacement out to the side pocket. The `open_direct`
counterfactual fixes the goal/robot/geometry but starts the
blocker already parked. All actuator and object evidence comes
from actual PhysX stepping.

### One command (run on the user's RTX 4090 Linux server)

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05
bash scripts/run_m1b3_corridor_once.sh
```

The runner:
1. Executes a CPU-only `unittest` C-space + report-schema suite.
2. Executes `python -m agcofl.m1b3_corridor_physics --preflight-only`,
   which uses **no ManiSkill/PhysX runtime** to check geometry.
3. Only after those pass, runs the four fixed cases once, each
   bounded by 500s subprocess time and 450 PhysX actions.
4. Summarizes via `python -m scripts.m1b3_validate_corridor`.
   The marker is `M1B3_PHYSICS_GATE`.

Outputs: `outputs/m1b3_corridor_dev/`. Never upload large
H5/MP4 to GitHub automatically. No checker reports paper
readiness or proof of causal novelty; pass only indicates the
intended scripted physical contrast was observed.
All four new PhysX runs and tests were **CODE COMMITTED,
NOT YET EXECUTED** when this document was written.

## Distinguishing outcomes and non-negotiable stop gate

| Observation | Scientific conclusion | Next step |
|---|---|---|
| Source success; blocked direct fails; blocked clear succeeds; open direct succeeds | Candidate insertion contrast seen in physics; **not yet proof of universal necessity** | Strong equal-info TAMP, contact-mode and dynamic/topology falsification |
| Blocked direct succeeds | Still no demonstrated need to clear first | Freeze candidate, inspect measured trajectories; do not claim an insertion contribution |
| Blocked clear fails | This existing push primitive cannot execute side clearance; geometric BFS is not sufficient | Diagnose exactly one real failure; do not blind-tune seeds/controls |
| Source fails | Matched same-goal source is not feasible in new scene with current adapted official solver | Freeze source trace and evaluate task feasibility; no learned method |
| Open direct fails | The supposed counterfactual "no blocker" is not physically validated | Scene/low-level actuator feasibility remains blocked |

Even the ideal first row cannot justify a T-RO method. Next is an
**information-matched capability-aware TAMP**, with the same target
goal, observations, action primitives and constraints, plus
goal-only and raw-flow controls on physically successful scenes.
If ordinary TAMP can do this or human/source video adds no task
information beyond goal/state, the editor paper's claimed novelty
is not established. Keep M2 model work STOPPED.
