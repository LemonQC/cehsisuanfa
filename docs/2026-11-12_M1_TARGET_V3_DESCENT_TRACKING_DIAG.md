# 2026-11-12 — Current M1 target v3: 25-step downward TCP tracking fails near 41mm

## Measured original-source evidence (supplied by operator)

The operator ran a read-only JSON inspection of
`outputs/m1_current_target_safe_approach_v3/target_safe_approach/physical_trace.json`,
selecting `execution_phase == "descend_behind_cube"`.
The complete phase has **25 actual simulator actions**.
Selected observations:

| Step | Measured TCP X (m) | TCP Z (m) | Task cube |
|---:|---:|---:|---|
| 144 | −0.06313399 | 0.16018207 | near (0,0,0.020) |
| 149 | −0.06302217 | 0.15251520 | unchanged |
| 154 | −0.06432302 | 0.11936384 | unchanged |
| 159 | −0.06745092 | 0.06690008 | unchanged |
| 164 | −0.06197966 | 0.04134371 | unchanged |
| 165 | −0.05925493 | 0.04125454 | unchanged |
| 166 | −0.05717580 | 0.04127955 | unchanged |
| 167 | −0.05574695 | 0.04128297 | unchanged |
| 168 | −0.05478160 | 0.04128367 | unchanged |

Both finger joints were reported at 0.0m throughout.
The task cube stayed at approximately x=0, y=0, z=.020m
through the phase. Physical blocker clearance had already
succeeded at 6.32mm parked-blocker error. No +X cube push
action was executed; v3 guard returned:
`M1_TARGET_NO_VALID_REAR_CONTACT_POSE_BEFORE_PUSH`.

Requested descending TCP target was approximately
[-0.065,0,0.020]m. Final measured TCP was approximately
[-0.054782,0.001355,0.041284]m. The final absolute TCP
height error is 21.28mm, 3.28mm beyond the coded 18mm
acceptance tolerance; X is also about 10.22mm too far
forward from requested. Between steps 164 and 168, actual
Z stopped falling (~0.06mm change) while X moved forward
~7.2mm.

## Interpretation and safeguards

This is **measured height plateau plus lateral/forward drift**,
not evidence of a physically successful contact state.
Potential explanations include gripper/table contact,
link-obstacle interference, controller tracking limitation
or a poorly defined TCP/gripper geometry transform.
The observed Cartesian trace alone does **not** identify
contact-force source, link colliding, or active joint constraints.

Very important: `agcofl/m1_current_target_safe_approach_v3.py`
uses `q=np1(b.agent.tcp.pose.q)[:4]` inherited from the
post-blocker CLEAR phase; the subsequent task cube motion
is along +X, whereas the blocker clearing direction was +Y.
Hence a fixed inherited gripper orientation may not be
the correct contact-face orientation for a new push
direction. This remains a **candidate** cause; original
v3 logs contain TCP XYZ, not per-step TCP quaternion or
contact-force measurements.

**Do not** simply relax 18mm Z guard to 22mm, call the
gripper in contact, or run an unguarded +X push.
At final TCP z=.0413m, task cube nominal top is z=.040m,
but TCP is not automatically the contact location of
two fingertip links. Source 106-step real grasp remains
unresolved; no successful same-goal source-target pair
has been proven.

## Read-only next check, no robot/PhysX reexecution

Compare existing successful `open_direct` with v3:
`outputs/m1b3_corridor_dev/open_direct/physical_trace.json`
and the v3 trace above. Inspect the successful direct
pusher's TCP z range, actual cube-motion onset,
finger actions, and relative cube/TCP positions at
first motion. Also inspect original v3 retreat/overhead/
descent TCP positions and finger qpos. Only then select
a new *versioned contact-frame* implementation if the
reference provides a grounded target orientation and
contact-height precedent. No geometry changes, no
old seed reruns, no M2 or novelty claims.

This report records the user-supplied console measurements;
raw H5/video/trace bytes remain on the operator's server.
