# 2026-11-11 — M1-B3 offline diagnosis: exact measured throat jam; one final gated M1-B4 feasibility rescue

## Actual source evidence

The operator ran `python -m scripts.m1b3_diagnose_existing`.
`Ran 3 tests ... OK`. It read the original M1-B3 four cases;
`ran_simulator=false`. The frozen original source solver phases:

```
approach=true, grasp_pose=true, close_gripper=true,
lift=true, across=true, lower=false
error="RuntimeError: official_mplib_lower_failed"
```

**Crucial interpretation:** these flags report *planner-action return
values*, NOT whether the source physically held and transported the
task cube. The original source ended ~210mm from the goal. Before
trying any lower-planning fallback, verify from the saved
`diagnosis.json` that `task_grasped_steps>0`,
`task_max_z_m > wall_height + task_half + 0.005`
and `task_total_xy_m>0.08`. Otherwise **STOP**: neither a
new lower pose nor a new controller gain would solve a missing grasp.

## Measured target evidence and geometry

`blocked_clear` phase `clear_blocker`: **32 actual PhysX steps,
success**, block moved from [0.100,0] to approximately
[0.10254,0.11981] m; error to intended side-parking goal
5.78mm; cube did not move (~0.000001m).

`push_cube` then **418 PhysX steps, failure** (no phase-budget-exhausted
flag, but globally at 450 actions). Task cube moved from x≈0 to
**[0.1252731,-0.0308216] m**, goal error **90.16mm**, last
15 measured cube steps had **0m** XY movement.
Task remained close to tabletop, max z 0.02018m.
Source of drift or actual contact forces cannot be established from
these JSON positions alone.

The M1-B3 static throat begins at x=0.158−0.012=**0.146m**;
cube half-width=0.020m. If the cube approaches from outside the
open center band, a cuboid collision at the front face occurs
around cube-center x=**0.126m**. The opening is 0.050m wide
for a 0.040m cube, giving an allowable centerline y between
**−0.005 and +0.005m** under the 2D square-footprint assumption.
Actual stall is x=0.12527m, y=−0.03082m — quantitatively
consistent with **a misaligned cube contacting the throat's
front/lower wall**. This is strong geometric localization,
NOT a direct measurement of the contact force.
No extra seeds or random tuning are justified.

The old `blocked_direct` physically pushed the cube to x≈.068m,
moved blocker by only ~17mm, and then failed.
`open_direct` achieved goal in 35 steps with ~14.3mm error.
This does not prove that a more capable dynamic planner cannot
make an alternative Push-only plan.

## ONE versioned control rescue M1-B4, NOT a research method

The new code `agcofl/m1b4_one_time_feasibility.py` uses exactly
the already-frozen M1-B3 geometry with seed=120, keeps
old `outputs/m1b3_corridor_dev/` immutable and creates only
`outputs/m1b4_one_time_feasibility/`. The old scene/controls
and official M0 policies are not edited.

### Hard stop before physics

`measured_mechanism_gate` reads the *existing* M1-B3
`diagnosis.json`, checks the observed throat stall against geometry,
and reports the existing source's actual held-grasp, cube peak-z
and measured total XY travel. If **any** source grasp/lift/travel
gate fails, the runner exits nonzero before starting any PhysX
and prints `M1B4_NO_GO_SOURCE_DID_NOT_PHYSICALLY_CARRY_CUBE_IN_FROZEN_RUN`.
This is deliberately conservative; it avoids repeating the same
grasp controller when its old measured evidence shows no carried object.

### Exactly two NEW physical cases, only if preflight passes

- `source_lower_fallback`: reuses upstream ManiSkill OBB grasp
  and the exact Panda/MPLib Screw planner, checks **actual**
  grasp/lift/translation at checkpoints; only if original
  `lower` Screw planning fails, one call to the upstream
  `move_to_pose_with_RRTConnect` is permitted. Goal remains
  grounded [.210,0], 20mm tolerance, no blocker movement.
  No hand-written robot grasp PID.
- `target_align_after_clear`: same old `ActorPush`
  primitive with block clearing; pushes to pre-throat
  x=.103,y=0, checks actual cube center, optionally uses a
  single lateral actor-push into the ±4mm target center band
  before directing the cube through the unchanged 50mm throat
  and toward [.210,0]. No new learned module or primitive gain
  tuning. Every sub-phase saves physical cube/blocker poses.
  If any stage fails or old 450-step budget is reached, stop.

Each new case is capped at **450 actual steps and 500s walltime**,
preserves H5/JSON/MP4 and per-step PhysX trace, and refuses
overwriting completed or partial files. The runner stops after
source failure, rather than wasting a target-only trial when
no same-goal physical pair is possible. Any failed M1-B4 attempt
means **stop controller repair on this scene**, do not open M1-B5
by changing waypoints again.

### Server command

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05
bash scripts/run_m1b4_one_time_feasibility.sh
```

The runner automatically runs four standard-library unit tests,
then the *no-physics* old-evidence preflight, then zero, one, or two
bounded actual PhysX cases: zero if the saved source never physically
carried the cube, one if the new source rescue fails (target is then skipped),
or two only if the rescued source succeeds.

The positive case outputs
`M1B4_FEASIBILITY_GATE` and local
`outputs/m1b4_one_time_feasibility/gate.json`; negative old
source evidence yields the `M1B4_NO_GO...` marker without
any simulation. Both are valid scientific stop decisions.
**New code has been committed, NOT executed on the user's server.**

## Paper decision — not waived by rescue

Even if two fresh physical cases both succeed,
`causal_necessity_proven=false`,
`strong_tamp_evaluated=false`,
`source_video_information_measured=false`,
`paper_ready=false`. A manually defined effect sequence is
ordinary capability-aware planning, not an innovation by
itself. After this feasibility gate, next is a genuine
**same-information TAMP** control and a test proving that the
source demonstration supplies information not already fixed
by the goal and target observation. If those controls solve
everything or source video is redundant, stop the proposed
T-RO effect-editor contribution.

Real frozen history: RoboReel π0.5 Uniform official 46/60;
Vid2Robot official 3/60; ManiSkill official Pick reference 3/3;
M1B2 same-goal 3/3 physics but no required insertion;
M1B3 four-case PhysX blocked source/direct/clear all failed
and open direct succeeded; its raw H5 and videos remain local.
