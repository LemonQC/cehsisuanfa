# 2026-11-10 — v1.5.7 Fix 218-step source descent tracking instead of blind grasp retries

## Measured evidence from EXISTING seed120 PhysX source trace

Offline diagnostic tests 2 passed. Actual measured source phase sequence:
above 16, descend 218, grasp 46, failed 1. During descend (open gripper)
the cube moved 0.0336668m laterally. During grasp all 45 close commands
produced ZERO pre-action verified grasp detections and max lift was
0.00008825m; source goal error=0.1292371m and no wall overflight.
No Push-only controls were executed (source invalid).

The old source controller:
- aimed at frozen source initial [x,y,z] for BOTH descend and grasp;
- commanded TCP z to the cube center z=0.02, which can cause undesirable
  contact in the approach;
- had no early descend timeout, allowing 218 steps despite the cube
  being displaced.
These observations support a stale-target/collision mechanism but do
NOT by themselves establish exact finger contact normals or how to
tune gripper orientation.

## Conservative v1.5.7 source-only code change

agcofl/v140_custom_pick_push_physics.py:
- Always use observed current cube XY during above/descend.
- Use source_pregrasp_xyz: cube Z + 0.025m (configurable single
  diagnostic constant, not a learned grasp policy).
- Track cube XY position from start of approach. If lateral drift
  exceeds 0.010m, open gripper and reapproach from above; maximum
  two retries, then explicitly fail. Never close onto a known shifted
  target.
- Abort descend after 70 steps rather than 218; abort grasp if the cube
  shifts or the grasp-phase >45 timeout is met.
- Preserve source_valid: grasp verified, actual lift >=0.055m,
  physical overflight, release onto table, final goal success.
- Persist explicit failure_reason, commanded TCP waypoint,
  lateral movement and reapproach_count per source trace.
- No changes to custom wall geometry, task goal, PhysX, target
  push controllers, user seed, or strict evaluation predicates.

This is a *diagnostic*, physics-teacher-only modification, not a
novel learned T-RO method or evidence of success. The choice of
clearance and restart threshold is a hypothesis to test once, not
a tuned result. If the next run fails, inspect measured trajectory
and gripper orientation/contact rather than blindly sweeping gains.

## One controlled retest; avoid unnecessary experiments

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate agcofl4090
    python -m pytest -q tests/test_v140_custom_pick_push.py
    bash scripts/run_v141_next_gate.sh custom-physics-dev
    cat outputs/v140_custom_pick_push_dev/source_pick/source_summary.json
    cat outputs/v140_custom_pick_push_dev/status.json

Test only seed 120 once. New physical execution has not occurred at
commit time. The script keeps 'SOURCE_INVALID_NO_TARGET_COMPARISON'
until all source validity checks pass. The previous recorded source
trace should be preserved elsewhere if historical debugging is needed,
since the default output path is reused by the run.

If reapproach still fails, investigate whether gripper orientation
(relative to the cube), z=0.045m pregrasp, and Panda contact
mode permit grasp. The diagnostic alone cannot prove those. Consider
reference to the original simulated PickCube teacher trajectory, or
video capture, before trying further threshold edits.

Official Pi0.5 Uniform remains separately blocked by original SigLIP
vision base (11.6GiB GCS download blocked), and Vid2Robot 3/60 frozen.
