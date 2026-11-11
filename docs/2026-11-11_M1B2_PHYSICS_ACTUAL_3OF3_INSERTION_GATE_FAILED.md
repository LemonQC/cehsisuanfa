# 2026-11-11 — M1-B2 actual PhysX triad: executor feasibility PASS, causal-insertion gate FAIL

## Observed user-server terminal evidence

The user executed `bash scripts/run_m1b2_same_goal_once.sh` after the
two preflight fixes and supplied the actual output:

```json
{
  "type": "M1B2_SAME_GOAL_PHYSX_GATE",
  "status": "PILOT_INCONCLUSIVE_OR_FAILED",
  "source_pick_place_valid": true,
  "target_direct_push_success": true,
  "target_clear_then_push_success": true,
  "target_only_blocker_displacement_observed": true,
  "matched_physics_start_and_goal": true,
  "all_three_real_rollout_traces_present": true,
  "pilot_contrast_pass": false,
  "causal_necessity_proven": false,
  "strong_capability_aware_tamp_evaluated": false,
  "learned_method_evaluated": false,
  "paper_ready": false,
  "per_case": {
    "source": {"success": true, "steps": 170, "error": null},
    "target_direct": {"success": true, "steps": 31, "error": null},
    "target_clear": {"success": true, "steps": 51, "error": null}
  }
}
```

The direct `target_clear` case stdout additionally reported:
`goal_error_m=0.04562683011663283` (goal tolerance in this
environment is 0.05 m); grounded=true,
`peak_lift_m=1.21e-7`, ever_grasped=false,
blocker effect=true. This outcome is only 4.37 mm inside
the success radius; not a precise-placement robustness result.

The Gymnasium Tensor-vs-NumPy deprecation/type warnings shown by
the user were warnings, not the cause of failure. All three
cases produced actual physics traces. The 3 statuses were generated
by user-local simulator runs; **raw JSON/H5/MP4 are not yet present
in GitHub**, and the values above are transcribed from the user's
terminal evidence. Avoid claiming independent remote raw-file audit.

## Scientific interpretation

**Engineering / simulation feasibility:** PASS for one development seed.
The adapted ManiSkill expert actually Pick+Placed; the two privileged
Panda Push executors each reached the identical final goal, one
with explicitly prescribed blocker displacement.

**Central M1 hypothesis:** FAIL in this geometry. The `target_direct`
script reached the identical goal in only 31 steps without an explicit
preliminary clear step. The `target_clear` order took 51 steps.
Therefore the target-only blocker displacement is **not established
as a causally necessary action**. The scene is *non-discriminating* for
effect insertion; cannot be counted as M1 necessity proof. This does
NOT yet prove all conceivable source-video editing hypotheses false.

The current scene has only two short static wall pieces;
neither scene topology nor dynamic blockers enforce that clearing
must precede target cube motion. A preselected `clear_then_push`
sequence is a privileged scripted control, **not** TAMP or learned
behavior. No M1 learning baselines (Goal-only, Raw Flow, video-conditioned
or equal-information TAMP) were run. Do not start M2 on the premise
that this scene demonstrates a required new causal insertion.

## Locked next decision

1. Freeze `outputs/m1b2_same_goal_dev` from this single seed.
   Do NOT tune old seed120, recode the grasp solver, or rerun
   the successful M0/M1-B1 baselines.
2. **Change task topology, not controller tuning:** a verifiable
   physically constrained passage with a movable blocker that
   prevents any table-supported Push-only cube transit until
   blocker clearance; add a side parking region reachable to
   the target and physically test clearance. The source is
   grasp-capable and can lift/translate over the obstacle to
   the *same* final grounded cube goal.
3. Before any learned method, require a paired **blocked/open**
   counterfactual at the same goal and scene information budget.
   The blocked condition must show (a) no valid push-through or
   push-around trace in the allowed workspace under geometry
   constraints, (b) physically viable block relocation, and
   (c) subsequent successful cube push to goal.
   The open condition should omit any need for blocker relocation.
   Failure of the scripted `target_direct` alone is not proof
   of impossibility: validate actual reachable-space topology.
4. Benchmark strong equal-information capability-aware TAMP
   (including allowed action primitives) before any new editor.
   If TAMP succeeds or source video is redundant relative to
   fully observed goal, narrow or stop the novelty claim.
5. Archive small `status.json` and `physical_gate_summary.json`
   with source file hashes only after user machine exports them.
   Never push large H5/videos blindly.

**GO/NO-GO:** Simulator mechanics are viable; the current causal
experiment is a **NO-GO** for proving target-only effect insertion.
Freeze it as negative pilot evidence, do not promote it to a
T-RO main result.
