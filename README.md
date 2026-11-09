# AG-COFL Partial Object Flow — ManiSkill 3 Gate A Pilot

**Version v0.5.0 (2026-11-09): preserve physical contact across collinear push segments.**

## Verified server observations (v0.4.0)

The user's RTX 4090 reported the following **real physics**, not theoretical trajectories:

- Lower path: cube progressed through first corner (-0.103,-0.157) and center checkpoint (~0.0005,-0.152); `check_visited=True`.
- Upper path: cube progressed through first corner (-0.117,0.162) and center checkpoint (~-0.0027,0.162); `check_visited=True`.
- Both then failed with `Failed TCP descend` around TCP z=0.065 m while requesting z=0.028 m for the **next, redundant, same-direction contact**. Both episodes ended with `success=False`.

## Root cause addressed in v0.5.0

The original five-point route is:

```
START → corner A (-0.105, ±0.155)
      → middle (0, ±0.155)
      → corner B (+0.105, ±0.155)
      → GOAL
```

The two physical pushes from corner A to middle and middle to corner B are collinear and point in the **same +x direction**. v0.4.0 made the Panda retract and re-descend at the middle waypoint — exactly where it failed for both sides.

**v0.5.0 merges them into one uninterrupted physical stroke**:

```
START → corner A → corner B → GOAL
```

Crucially, `route_points()`, the **future binary-mask references**, the middle **check zone**, obstacle geometry, and the simulator's **success definition** are unchanged. The cube must still physically pass through the checkpoint on the way to corner B, which the simulator evaluates at every physics step. This is NOT a teleport, hidden waypoint bypass, success relabeling, nor easier task.

Additional v0.4.0 measured re-contact retries remain for genuinely missed *corner/goal* waypoints. New unit tests check straight-line merging and that the original middle waypoint lies on the merged line. Real success of the new long push **has not yet been tested on the user's server**.

## Single paired validation run (4090, no reinstall)

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull origin main
conda activate agcofl4090
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1

python -m pytest -q tests
python -m agcofl.cli test-loss

python -u -X faulthandler -m agcofl.cli collect \
  --episodes 1 --seed 100 --max-retries 2 \
  --out outputs/ee_v050_episodes \
  2>&1 | tee collect_v050.log

cat outputs/ee_v050_episodes/collection_summary.json
cat outputs/ee_v050_episodes/route_lower_000100.json
cat outputs/ee_v050_episodes/route_upper_000100.json
```

Expected marker `EXPERT_CONTACT_STROKES` should show **3** contact endpoints per route (first corner, second corner, goal). `PUSH_ATTEMPT` reports the measured cube displacement, progress, number of simulator steps and error for each actual physical stroke.

When reviewing progress, check that `check_visited` remains true and the cube reaches both the far corner and the final goal. If the new *long* middle stroke fails because the TCP cannot reach or contact is lost, the full failed episode and its diagnostic JSON are still recorded. We must use that data to diagnose the next geometric/workspace limit; **do not increase retry limits blindly**.

## Next stages (only after **both** teacher routes succeed repeatedly)

```bash
python -m agcofl.cli collect --episodes 15 --seed 200 \
  --out outputs/ee_v050_episodes

python -m agcofl.cli train-bc \
  --data outputs/ee_v050_episodes --epochs 100 \
  --ckpt outputs/models/flow_bc_ee_v050.pt

python -m agcofl.cli gate-a \
  --ckpt outputs/models/flow_bc_ee_v050.pt \
  --seed 9000 --episodes 5 --max-steps 500
```

Gate A: test whether a learned `pd_ee_delta_pos` Panda policy selects the upper/lower route only in response to a geometric future binary-mask reference, with identical initial robot and cube state. **Not yet passed.** Until then, high-level Full / Anchor / End supervision training remains gated.

## Experiment integrity

- ManiSkill 3, CPU PhysX, Panda Push-only, `pd_ee_delta_pos` (4D), render backend `none`.
- No native MPLib planner and no GPU camera renderer. `--video` generates **offline 2D cube XY plot movies**, not RGB.
- The mask references are synthetic geometric future occupancies, NOT human demonstrations and NOT dense optical flow.
- Training loader rejects incompatible legacy 9D joint-control demonstrations and ignores failed episodes.
- The step cap remains in place (530 recorded control steps); the episode success constraints have not been loosened.
- Model/performance claims require real server rollouts. The authoring container cannot execute ManiSkill/SAPIEN.

See [HEADLESS_GUIDE.md](HEADLESS_GUIDE.md) for previous crash diagnostics.
