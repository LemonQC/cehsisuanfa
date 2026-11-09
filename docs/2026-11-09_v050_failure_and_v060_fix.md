# 2026-11-09 | v0.5 Gate A expert failure → v0.6 measured-contact patch

## Input: actual RTX 4090 rollout (seed 100)

v0.5.0 expert collection: `EXPERT_COLLECTION attempted=2 success=1`.

**Upper:** complete real-physics success, `steps=174`, `check_visited=True`.

**Lower:**
- leg 1 reached the first corner [-0.10318,-0.15733] after two attempts.
- leg 2 reached [0.06755,-0.12523] with reported `end_error_m=0.04784`, `reached=true`. This was accepted because v0.5's `cube_at_waypoint` returned true within 0.050 m even though the nominal second corner was [0.105,-0.155].
- leg 3 attempted to move toward [0.145,0] but cube remained [0.06755,-0.12523] for 95 steps; `Failed contact push towards waypoint=[0.145,0]`. The episode had `check_visited=True` but final `success=False`.

## What the data proves and does not prove

Proves: a threshold/incomplete-corner issue exists in the teacher, and the final lower push does not translate the cube. It does **not** prove that the final motion is kinematically impossible, because no actual TCP start/end/target data were logged for the failed 85-step push.

## v0.6 targeted changes

- tighten *intermediate contact stop condition* from 0.050/0.055 m to 0.030 m (final approach: 0.045 m); this aims to preserve the already-effective contact along the long straight stroke and avoid starting final turn from premature corner.
- on an incomplete push servo, record TCP target/actual and cube motion, safely retract, and try one of three pre-defined contact distances (6.0, 4.5, 7.5 cm) with the same fixed max-retries=2 budget. This is bounded engineering calibration, not an oracle change to the labels.
- after at least 24 push steps, stop when 12 consecutive *actual* object samples show less than 1.5 mm motion.
- retain the real `env.step` operations, unchanged object mask labels, scene, collision barrier, two-stage checkpoint, goal and task success requirements.
- output files go to `outputs/ee_v060_episodes`; old failed datasets remain untouched.

## Decisive validation

One seed each side, then if positive a repeated-seed robustness test. Only recorded full successes from **both** routes unlock BC training. Important signals: `PUSH_SERVO_INCOMPLETE`, `PUSH_ATTEMPT.push_telemetry`, `EXPERT_COLLECTION`.

Status when this record was written: **not yet executed on the user's 4090**, no assertion of new physical success.
