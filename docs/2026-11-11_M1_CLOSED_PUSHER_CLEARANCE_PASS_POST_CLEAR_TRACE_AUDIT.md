# 2026-11-11 — Real closed-tool Panda MPLib cleared blocker; unfinished cube stage hit 450 steps

## Grounded observation from the operator (no synthetic or unexecuted claims)

The operator reported the saved result of
`bash scripts/run_m1_target_closed_pusher_once.sh`.
The runner detected an existing **immutable**
`outputs/m1_current_closed_pusher_v2/target_closed_mplib/status.json`
and reused it. The test suite reported `Ran 4 tests ... OK`.
This reinspection did NOT launch another simulation, but the
saved 450-step PhysX trial had already been executed locally.

```json
{
 "success":false,
 "steps":450,
 "error":"RuntimeError: M1B2_STEP_BUDGET_EXCEEDED: no unbounded expert planning",
 "phases":[{
   "name":"clear_blocker",
   "actual_steps":82,
   "initial_xy_m":[0.09999964386224747,-0.0000006951020168344257],
   "goal_xy_m":[0.1,0.125],
   "observed_xy_m":[0.10489228367805481,0.12900730967521667],
   "actual_error_m":0.0063239995587955735,
   "task_ever_grasped":false,
   "blocker_ever_grasped":false
 }],
 "push_only_physical_check":{
   "valid":true,
   "ever_grasped_task":false,
   "ever_grasped_blocker":false,
   "closed_gripper_command_steps":450,
   "open_gripper_command_steps":0,
   "finger_joint_qpos_final_m":[0.0,0.0]
 },
 "original_goal_error_m":0.260049416909468,
 "source_matched":false
}
```

**Interpretation:**

- The **clear-blocker primitive actually succeeded** under
  closed-tool MPLib: error 6.32mm in 82 additional
  planning/actions, plus 6 initial close-finger actions.
  The earlier original `DiagnosticPushTeacher` achieved
  physical clearance in 32 actions. BOTH demonstrate clearance
  feasibility in this exact scene; the M1 target problem is
  **no longer the ability to push the blocker sideways**.
- The remaining 450−6−82=**362 real action steps** were
  attempted in the cube stage after clearance. That incomplete
  phase is not added to the `phases` array because
  `TracedEnv` throws at the budget when the planner attempts
  one more step and `execute` catches it. **Do not claim
  the cube did not move during those 362 steps**, nor claim
  a specific backward displacement or collision from the
  goal-error scalar. The 260.05mm error is worse than the
  nominal initial 210mm, but its cause must be read from
  the saved cube/TCP trace.
- The frozen physical Push-only interpretation is supported
  for this trial: both object grasp flags false throughout,
  finger joints closed, no object attachment. Original old
  gripper-open labeling was WRONG; the old raw files remain
  unchanged.
- No source Pick success under the x=.210 same-goal scenario.
  The source 106-step real side-grasp trial is still invalid,
  and the strong equal-info TAMP and learned method are
  NOT_RUN. This is **engineering progress, not a T-RO result**.

## Next computation: offline recover the unfinished target phase

New `scripts/m1_current_closed_pusher_trace_audit.py` performs
**zero PhysX, no GPU, no model training**. It loads the exact
raw `status.json` and `physical_trace.json` of:

1. The saved closed-tool MPLib 450-action outcome.
2. The genuine older M1B3 `open_direct` original-scene
   success, 35 steps, used only as a physical traversability
   reference. The obstacle is pre-parked and controller differs;
   this is NOT an equal-information method comparison.

It rejects non-matching original scene, initial cube and
final goal, wrong fixed seed, missing per-action cube/TCP/
blocker coordinates, corrupted action indices, missing real
two-object grasp flags or finger qpos, misrepresented final
physical state or overwritten records.

It reconstructs three actual physical segments:
`close_gripper` (6 steps), `clear_blocker` (82 steps)
and *unfinished* `post_clearance_attempt` (362 steps).
For the unfinished suffix it reports true cube initial/
final 3D position, total XY travel, start/closest/end goal
error, signed progress vs goal, lateral drift, TCP traveled
distance, closest TCP-to-cube-center distance, last 15-step
stagnation, object z, and exact stage/timeline checkpoints.
Original JSON SHA256s are archived in a NEW small report
without changing raw input bytes.

The script returns one **evidence-oriented** signature:

- cube moved FARTHER from goal after clearance;
- cube stalled misaligned at the old x≈0.126 throat front;
- no meaningful cube motion after clearance;
- or inconclusive, requiring actual TCP-to-object review.

This classification is NOT a contact-force or
universally exhaustive controller-failure diagnosis.

### One command on the user's existing pi05 server

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05
python -m unittest discover -s tests \
  -p 'test_m1_current_closed_pusher_trace_audit.py' -q
python -m scripts.m1_current_closed_pusher_trace_audit
```

Expected marker: `M1_CLOSED_PUSHER_450_STEP_TRACE_DIAG`.
Report:
`outputs/m1_current_closed_pusher_v2/closed_pusher_450_step_offline_audit.json`.

Return the marker for next precise actuator/hand-off decision.
**No M1-B5 scene, no new source-grasp attempt and no
new 450-step controller test is authorized by this
read-only analysis alone.** Keep successful blocker-clear
executor. If later a small exact hand-off fix is justified,
version it separately and keep original 450 physical actions
immutable. Do not train M2 before a physically valid
same-goal pair, baseline information-matching and
non-trivial video necessity.
