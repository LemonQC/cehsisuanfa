# 2026-11-12 — M1 v3 25-step descent: compare successful direct physical TCP contact height; v4 one-factor correction

## Original evidence supplied by operator

The user's **existing saved H5/trace**, not a new run, yields:

| Observation | Actual TCP XYZ | Task cube XYZ | Provenance |
|---|---|---|---|
| Old `open_direct` first cube XY displacement >3mm, step26 | [−0.01517760,0.00032982,0.03737768]m | [0.0138031,−0.0000813,0.0200001]m | 35-step SUCCESS, Push-only, closed Panda finger commands −1, **pd_ee_delta_pose** |
| Old `open_direct` step34 | [0.12471045,−0.00091020,0.04666556]m | [0.19592996,0.00263381,0.02000012]m | same successful 35-step trace |
| v3 final descent step168 | [−0.05478160,0.00135548,0.04128367]m | [≈0,≈0,0.01999992]m | blocked current scene, closed Panda fingers, **pd_joint_pos**, 169 steps, no cube push attempted |

Old successful direct-push TCP minimum while the cube moved was
**0.0367314m**. The v3 TCP at **0.0412837m** is only about
3.91mm above the successful first-push TCP height, but the robot
has not reached close physical X proximity: its center is
~54.8mm behind the cube versus ~29mm behind the *moved*
cube at first success. Both genuine open/direct and v3 used
closed-finger command `-1`, but **different control modes**,
and the original saved traces lack finger-pose orientation and
contact forces. These numeric comparisons do NOT imply identical
fingertip/tool geometry or guarantee a contact.

### Specific v3 execution bug

The v3 controller explicitly requests lowering the TCP to
**cube center z=0.020m**, and rejects the measured v3
`z=0.0413m` because it is more than 18mm above cube center.
Even if the controller passed that check, the three subsequent
`push_*` Screw pose targets **also incorrectly requested
TCP z=0.020m**. This conflicts with successful TCP heights
around 0.037m; it may induce table interference or
undesired kinematics.

Measured v3 descent shows Z plateau near 0.0413m from
step164 onward, while X moves toward cube, with no object
displacement, no confirmed contact or successful push. Exact
reason for the plateau (contact vs joint tracking vs
orientation) remains **NOT ESTABLISHED**.

## Versioned in-place v4 correction and preflight

New code:
- `agcofl/m1_current_target_reference_height_v4.py`
- `tests/test_m1_current_target_reference_height_v4.py`
- `scripts/run_m1_target_reference_height_once.sh`

**Only one contact-height correction**, grounded in the existing
successful physical sample: `CONTACT_TCP_TARGET_Z=.038m`
for both final rear descent and all +X push pose targets.
Replace the incorrect cube-center-relative height gate with
a measured absolute `0.034 <= TCP_z <= 0.050m` envelope,
and retain the measured rear-X gap `>=28mm` and
lateral tolerance `<=12mm`.
This envelope is a diagnostic acceptance window based on
actual reference TCP behavior, **not a proof of fingertip
collision-free reachability**.

Everything else remains as the v3 original scene:
original `AGCOFLM1B3Corridor-v0` geometry, goal x=.210,
blocker parked via identical 6+82-action closed-finger
Panda/MPLib, retreat to z=.160m, overhead return to behind
cube, three +X target steps, same original +/- direction
guards, no inadvertent cube displacement before intentional
push, no-grasp and grounded-object checks, same seed120,
450-step and 500s stop budget, existing raw data untouched.

The NEW output directory `outputs/m1_current_target_reference_height_v4/`
holds only a possible new one-shot v4 H5/MP4/JSON and per-step
physical trace. The runner requires **original physical
evidence first**: original 35-step `open_direct` success,
v3 169-step physical stop before contact, same task initial
state and goal, original success movement TCP height within
the physically observed envelope. It prints original input
file SHA256s and will refuse mismatched evidence without any
new PhysX. Then it executes **at most one** unchanged-scene
target-only trial, in a separate physical run.

Run:

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05
bash scripts/run_m1_target_reference_height_once.sh
```

Result markers:
- `M1_CONTACT_HEIGHT_EXISTING_PHYSICS_GATE`: existing data
- `M1_TARGET_REFERENCE_HEIGHT_RESULT`: actual v4 PhysX
- `M1_REFERENCE_HEIGHT_GATE`: saved post-run metrics and phases

### Decision / stop gate

If the v4 probe moves cube forward and through throat,
retain corrected contact-frame executor, but remember
that the *source Pick at x=.210* still has no validated grasp.
If v4 stops before forward movement or hits any original
negative-direction/lateral/grounding/no-grasp guards,
freeze its H5/MP4 and **do not start v5 for arbitrary gain
and waypoint tuning**. The remaining bottleneck may be
end-effector orientation or finger/workspace interaction,
which cannot be fixed from TCP XYZ alone. Compare actual
contact/link orientation from existing physics or inspect
video before any further interventions. Do not invent
unobserved forces or claim a T-RO method.

The active scientific research question is unchanged:
cross-behavior same-goal physical Pick-source to Push-only
target with blocker intervention; neither successful
matching source video, equal-info strong TAMP nor learned
method has been demonstrated in M1B3. Official RoboReel
M0 original 46/60 and all historical baselines are frozen.
