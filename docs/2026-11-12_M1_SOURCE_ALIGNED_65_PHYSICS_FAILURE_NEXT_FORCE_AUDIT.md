# 2026-11-12 — Original M1B3 SOURCE one geometry-aligned top-down physical attempt: 65 actions, grasp descent planning FAILED

## Exact observed server result (operator-provided, not independently executed here)

The user ran the original-scene source candidate
`bash scripts/run_m1_source_geom_aligned_topdown_once.sh`
and received:

```json
{
  "success":false,
  "steps":65,
  "physical_grasp_reached":false,
  "actual_grasp_steps":0,
  "cube_lift_m":0.0,
  "goal_error_m":0.21000093018257568,
  "blocker_xy_shift_m":7.803726335100881e-7,
  "contact_force_measured_steps":65,
  "error":"RuntimeError: SOURCE_UPSTREAM_grasp_pose_PLANNING_FAILED"
}
```

The scene remained the original `AGCOFLM1B3Corridor-v0`,
seed120, original Panda, goal cube x=.210m,
75mm rail height and dynamic blocker unchanged.
The target Push-only v6 physically succeeded in 66 actions
and is **frozen**. This source candidate DID NOT produce
a physically valid same-goal source demonstration.

### Actual phases

| Phase | Real actions | Result |
|---|---:|---|
| Static walls+blocker obstacle pointcloud | 0 | 792 surface points added and upstream `use_point_cloud=True` |
| Set intermediate opening | 6 | Real Panda finger qpos [0.02385227,0.02385227] m, close to requested q=.023 |
| `pregrasp` MPLib screw plan and execution | 59 | Planner return success, actual TCP [−.00179061,−.01405345,.07105435]m; distance to cube center **52.983mm** |
| `grasp_pose` MPLib screw plan | **0** | **Plan failed**, no physical motion toward final grasp |
| Physically close finger, lift, carry, release | **0** | NOT EXECUTED, physical grasp never established |

Pregrasp commanded target is about [0,0,.080]m
(cube center [.0,.0,.020] plus 60mm along world +Z).
The measured final pregrasp pose is around 16.8mm
away from the commanded target, predominantly
14.1mm lateral Y deviation and 8.95mm height mismatch.
A successful upstream planning return DID NOT imply exact
Cartesian tracking.

**Do not confuse two separate failure modes:**

1. Actual executed approach remained ~53mm from cube;
   this is not a grip-force or closure-time problem.
2. For the **final** grasp pose the upstream MPLib screw
   planner rejected the path before any physics action,
   under the sampled geometry pointcloud. That does not
   mean the real robot physically struck the wall at the
   target grasp pose; no attempted final descent occurred.

`wall_force_measured_steps=65` only means the wrapper
successfully recorded per-step hand/finger ↔ original
STATIC walls force values. **It does not mean there
were nonzero wall contact forces.** No per-step force
values appeared in this console output.

Prior attempts already failed original source Pick:
- Original M1B3 top-down 335 actions, closest actual
  TCP to cube **63.97mm**, zero grasp/lift.
- Side approach 106 actions, closest actual TCP
  **102.42mm**, zero grasp/lift.
- Earlier M1B2 REAL source grasp 84 valid steps with
  goal **x=.150m** is a low-level contact reference,
  not the M1B3 x=.210 source.

### Next step: one read-only local inspection, ZERO new physics

New `scripts/m1_source_65_step_force_vs_planner_audit.py`
and five standard-library tests in
`tests/test_m1_source_65_step_force_vs_planner_audit.py`
read ONLY the frozen status and 65-action trace.

The audit reports:
- all 65 measured contact-force values as their count,
  maximum, and whether any physically measured
  hand/finger/static-wall force exceeds 0.5 N;
- last measured TCP, expected pregrasp waypoint,
  residual pose-position error and actual TCP quaternion's
  closing-axis alignment with world X;
- actual finger qpos and original file SHA256;
- strict distinction between physical contact detection
  and **zero-action** screw-plan failure.

Caveats: The previous logger collected force only between
hand / left finger / right finger and original **STATIC
walls**, not robot arm links or table/blocker, and did
not log joint tracking target errors. A zero value can
therefore NEVER prove that all tool, arm and environment
collisions are absent or that the target pose is feasible.

Run the CPU-only evidence check:

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05
python -m unittest discover -s tests \
    -p 'test_m1_source_65_step_force_vs_planner_audit.py' -q
python -m scripts.m1_source_65_step_force_vs_planner_audit
```

The marker is `M1_SOURCE_65_STEP_FORCE_VS_PLANNER_DIAG`.
It creates a *new immutable* small report
`outputs/m1_source_geom_aligned_topdown_v1/real65_step_contact_audit.json`,
not a new H5 / video or old evidence rewrite.

## Binding scientific decision

**No new source grasp versions or source FULL PhysX**
until actual tool-wall contact values and orientation
are reviewed. No reattempt of existing v1 source, no
MPLib action-budget increase and no changes to original
wall size, robot, goal or task.

If real tool-wall forces occur and orientational
alignment is known, inspect the corresponding
link/pose and source feasibility; do not assume
one more random grasp orientation will fix it.
If force is absent, this still could be a
collision-aware virtual planning infeasibility,
IK/joint-limit problem, or contact from an unlogged
robot link. A collision-specific conclusion then
requires a more complete robot/obstacle geometry
analysis, not a blind simulator retry.

**High-level TRO status:** The target is solvable by a
66-action privileged scripted clear-then-push oracle
on seed120. M1B3 source remains NOT physically
solved after three distinct failures. Even with a
successful source, no demonstrated need for source
video over strong state-aware TAMP currently exists.
Do not begin learned M2/Method or claim T-RO novelty
before a valid same-goal source/target physical pair
AND fair information-matched baseline gap.

This report transcribes user-provided console output;
the 65-action raw JSON/H5/MP4 remains on the user's
server and has not been independently evaluated here.
