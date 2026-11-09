# AG-COFL / Partial Object-Flow Supervision — 4090 ManiSkill pilot

**Current version: v0.6.0 (2026-11-09).** This repo implements a *physical* Panda push-teacher for the Gate A oracle-flow experiment, with separate Full / Anchor / End partial-supervision loss tests. Gate A is NOT yet passed.

## Experimental record (actual RTX 4090 output)

**v0.5.0 seed 100: 1/2 full teacher successes.**

- Upper: `success=true`, `steps=174`, `check_visited=true`. First corner 2 pushes, long straight segment 1 push, final target 3 pushes.
- Lower: `success=false`, `check_visited=true`. First corner reached in 2 pushes. Long straight stroke was stopped at cube `[0.06755,-0.12523]` because the old contact policy accepted a **0.04784 m** corner error; nominal far corner is `[0.105,-0.155]`. The next goal push moved the cube less than a millimeter and failed after 95 steps with `Failed contact push`.
- The lower-side cause is not proven to be an IK singularity or collision; v0.5.0 never logged actual TCP location at the failure. v0.6.0 adds TCP telemetry.

## Changes in v0.6.0

1. **Consistent waypoint arrival tolerance:** 0.030 m for the two turn endpoints during the physical teacher, 0.045 m for the final physical push. The simulator's original task success limits (check zone, final goal and cube on table) remain unchanged. The long straight push should no longer stop at 0.04784 m error; it continues while contact exists.
2. **Do not throw immediately on a failed final TCP push:** retract if safe, check *measured* cube position, then make bounded new contact attempts with final-turn gaps **0.060 → 0.045 → 0.075 m**. Retries are still bounded by `--max-retries`; no teleportation or hidden robot interventions.
3. **Early stall detection:** after at least 24 contact-push steps, if the most recent 12 cube positions vary less than 1.5 mm, stop the unproductive push instead of always consuming 85 steps. This is a conservative engineering heuristic, not an academic contribution.
4. **Diagnostic logs:** `PUSH_SERVO_INCOMPLETE` records TCP contact/target/end, target error and cube push displacement. Each `PUSH_ATTEMPT` now includes TCP before/after and push telemetry; errors preserve these details.
5. The original source flow waypoint sequence, side-specific checkpoint, barrier geometry and task success calculation are **unchanged**. The teacher still merges collinear physical pushes through the checkpoint.
6. All episodes are saved separately under `outputs/ee_v060_episodes` by default. Existing successful v0.5 examples should not be mixed blindly with later training until controller compatibility and the final collection protocol are frozen.

**v0.6.0 is a candidate fix, not a verified physical success.** The unit tests are logic tests with mocked physics, not proof of successful 4090 rollouts.

## Next steps on RTX 4090 (no reinstall)

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
  --out outputs/ee_v060_episodes \
  2>&1 | tee collect_v060.log

cat outputs/ee_v060_episodes/collection_summary.json
```

For exact failure diagnosis inspect `outputs/ee_v060_episodes/route_lower_000100.json`, particularly `push_attempts[].push_telemetry` with `tcp_target_error_m`, `cube_push_displacement_m`, `tcp_after_push`, `contact_gap_m` and `waypoint_tolerance_m`.

**Only after both routes have repeatedly successful physical demonstrations**, collect a balanced demonstration set and train the Flow-conditioned BC. Do not train on failed trajectories. The initial masks are controlled synthetic geometric occupancy references, NOT human RGB videos or learned object flow. The `--video` flag creates an offline 2D measured cube trajectory, not RGB robot footage.

## Later commands (not until physical teacher is validated)

```bash
python -m agcofl.cli collect --episodes 15 --seed 200 --out outputs/ee_v060_episodes
python -m agcofl.cli train-bc --data outputs/ee_v060_episodes --epochs 100 \
   --ckpt outputs/models/flow_bc_ee_v060.pt
python -m agcofl.cli gate-a --ckpt outputs/models/flow_bc_ee_v060.pt \
   --episodes 5 --seed 9000 --max-steps 500
```

## Limits

- Physics simulation: ManiSkill3 CPU PhysX with `pd_ee_delta_pos`; no MPLib, `render_backend=none`.
- The current authoring container cannot reproduce ManiSkill physics, so **do not interpret a passing unit test as a success rate**.
- No cross-behavior transfer, human demonstration benefit, high-level Full/Anchor/End training comparison, or T-RO-ready method has been established by this pilot.

See [diagnosis and rationale](docs/2026-11-09_v050_failure_and_v060_fix.md).
