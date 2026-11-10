# 2026-11-10 — v1.5.6 Real same-goal Pick source failed; diagnose existing trace before changing controller

## Evidence: real ManiSkill/PhysX, seed 120

The operator supplied `outputs/v140_custom_pick_push_dev/source_pick/source_summary.json`
and `outputs/v140_custom_pick_push_dev/status.json`. These are real
runtime measurements, not hypothetical tests:

| Field | Value |
|---|---|
| steps | 281 |
| initial cube XYZ | [0.01132401, 0.01613464, 0.02000000] |
| goal XYZ | [0.16132402, 0.01613464, 0.00100000] |
| ever_grasped | true |
| peak_lift_m | 0.0000882521 m (0.088 mm) |
| final_goal_error_m | 0.1292371 m |
| wall_overflight_observed | false |
| released_on_table | true |
| env_success | false |
| end_phase | failed |
| valid_pick_source | false |
| status | SOURCE_INVALID_NO_TARGET_COMPARISON |
| target trials | zero |

The previous NumPy bool JSON serialization bug has been fixed and
the report was actually written. The strict source validity gate
correctly prevented the three target controls from running. No new
trained policy success is claimed.

## Critical code-level insight

In `agcofl/v140_custom_pick_push_physics.py`,
`ScriptedPickSource` enters `failed` ONLY if:
`self.phase == "grasp" and self.steps_in_phase > 45`
without observing `grasp` at the start of a controller step.
Once the controller observes grasp in the grasp phase, it switches
to `lift`, where no `failed` transition currently exists.

By contrast `ever_grasped` is sampled *after* EVERY environment step,
including `above`, `descend` and transient contact, and may reflect
a grasp test that was never sustained during the controller's
actual grasp phase. The result `ever_grasped=true` + `end_phase=failed`
is therefore **not** proof that the controller successfully grabbed
and then lost the cube while lifting. The source trace needs
phase-specific analysis. The cube moved sideways about 2.5cm and
remained at z≈0.02; suspect poor alignment/contact/pushing during
gripper close, but mechanism is unverified.

## Non-invasive next action — no new physics yet

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate agcofl4090
    python -m pytest -q tests/test_diagnose_custom_pick_source.py
    bash scripts/run_v141_next_gate.sh diagnose-custom-source

New `scripts/diagnose_custom_pick_source.py` reads the already saved
`source_summary.json` and `source_trace.json` and reports counts
per `above`, `descend`, `grasp`, `lift`, `failed` phase;
for each: detected-grasp samples, longest consecutive grasp span,
first/last TCP and cube XYZ, TCP displacement, peak cube z and
number of closed-gripper commands. No simulation, weights or new
network calls. This tool does not mutate files.

Next science decision must be based on actual trace:
- If grasp stays false throughout `grasp`: inspect TCP-center alignment,
  descent z and cube lateral displacement, gripper closure duration.
- If grasp becomes true but source still says failed: inspect
  before/after action logging and contact duration; add synchronized
  telemetry and test the phase transition instead of changing task goal.
- If source lifts but cannot overfly: evaluate waypoint/obstacle geometry
  and clearance only after the grasp is physically stable.

Avoid blind changes to all control thresholds, changing the success
predicate, claiming success from ever_grasped, or running many new seeds.
First establish at least one real valid source roll-out. Then proceed
to target push controls and eventually learned cross-behavior policies.

Official pi0.5 Uniform remains separately blocked by original SigLIP
base encoder absent from cache; no new pi0.5 physical results.
