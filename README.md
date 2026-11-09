# AG-COFL Partial Object Flow — ManiSkill 3 Gate A Pilot

**Version: v0.4.0 (2026-11-09)** — physical pushing teacher now measures cube progress and **re-contacts the cube from its actual position** when an intermediate waypoint is missed. No MPLib, no camera/Vulkan rendering. Full/Anchor/End losses are implemented but high-level training is **not yet unlocked**.

## Status / what is verified

- The user's RTX 4090 server successfully ran real ManiSkill `env.step()` calls with the v0.3.0 controller. Both first-leg pushes moved the cube to approximately y=±0.08 m, but **failed** the waypoint at y=±0.155 m. This is evidence of physical motion, NOT teacher success.
- v0.4.0 changes **only the expert rollout logic** (re-contact retries and diagnostics). The revised pushes **must be verified on the server**. Do not infer expert success or train on failures.
- The `--video` flag makes an offline 2D trace from recorded cube positions; **not RGB robot footage**.

## Run directly on the same 4090 server

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull origin main
conda activate agcofl4090
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1

python -m pytest -q tests
python -m agcofl.cli test-loss
python -u -X faulthandler -m agcofl.cli doctor 2>&1 | tee doctor_v040.log

# Initial run: one paired seed; 3 total measured pushes per missed waypoint:
python -u -X faulthandler -m agcofl.cli collect \
  --episodes 1 --seed 100 --max-retries 2 \
  --out outputs/ee_v040_episodes 2>&1 | tee collect_v040.log

cat outputs/ee_v040_episodes/collection_summary.json
cat outputs/ee_v040_episodes/route_lower_000100.json
cat outputs/ee_v040_episodes/route_upper_000100.json
```

**Expected diagnostic markers** include `EXPERT_CONTROLLER pd_ee_delta_pos, no MPLib` and one or more `PUSH_ATTEMPT {...}` messages with `leg`, `attempt`, `before`, `after`, `start_error_m`, `end_error_m`, `progress_m`, `steps`, `reached`.

**Do not train if `EXPERT_COLLECTION.success == 0` or either route lacks successful episodes.** The CLI intentionally returns an error if every physical expert rollout failed; the per-episode JSON and progress log are preserved for diagnosis.

If the second or third contact fails, look at the measured `progress_m`:
- `progress_m > 0` but `reached: false`: cube moved toward goal; movement or execution budget insufficient.
- `progress_m <= 0`: contact direction or collision/workspace geometry is likely wrong; do not just keep increasing retry count.
- `Reached episode step budget`: rework low-level push efficiency before training.

The geometric checkpoint, goal and collision conditions are unchanged. v0.4.0 has not relaxed evaluation to force a pass.

## Once BOTH expert routes have actual successes

```bash
python -m agcofl.cli collect --episodes 15 --seed 200 --max-retries 2 --out outputs/ee_v040_episodes
python -m agcofl.cli train-bc --data outputs/ee_v040_episodes --epochs 100 --ckpt outputs/models/flow_bc_ee_v040.pt
python -m agcofl.cli gate-a --ckpt outputs/models/flow_bc_ee_v040.pt --seed 9000 --episodes 5 --max-steps 500
```

Gate A: hold the same robot/object initial state and scene fixed, change only the synthetic upper/lower future object-mask reference; test whether the **learned** low-level Panda policy physically follows the desired side before proceeding to high-level Full/Anchor/End experiments.

## Experiment integrity

- Task: Panda in `AGCOFL-TwoRoute-v1`, Push-only, two passages around a central obstacle.
- Simulator: ManiSkill 3, `sim_backend=physx_cpu`, `render_backend=none`.
- Controller: `pd_ee_delta_pos` in demonstrations, dataset and policy, never mix with former 9D `pd_joint_pos` data.
- Future object masks are **synthetic geometric references**, not actual human RGB/segmentation, and not dense optical flow.
- The current teacher is an engineering data collection baseline; it is not the final scientific method.
- No physical success, Gate A performance, or T-RO readiness is claimed without simulation results.
