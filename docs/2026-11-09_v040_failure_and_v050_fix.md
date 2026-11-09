# 2026-11-09: Push teacher checkpoint → TCP descent failure (v0.4.0)

## Measured results from the RTX 4090 server

Run: `python -m agcofl.cli collect --episodes 1 --seed 100 --max-retries 2`.

- `EXPERT_COLLECTION: {"attempted": 2, "success": 0}`.
- **Lower side (-1):** first corner [-0.10318, -0.15733] achieved after two measured attempts; second (center) checkpoint [0.00047, -0.15183] achieved on first attempt, `check_visited=True`. Third contact descends toward TCP target [-0.05950,-0.15002,0.028], but actual TCP stops at [-0.04098,-0.15485,0.06544]. No cube movement on failed third leg.
- **Upper side (+1):** first corner [-0.11727,0.16159] achieved after two attempts; second checkpoint [-0.00268,0.16242] achieved on first attempt, `check_visited=True`. Third descent toward target [-0.06254,0.16654,0.028] stops at actual TCP [-0.04291,0.17098,0.06469]. No cube movement on failed third leg.

## Diagnosis

The original path's consecutive segments from [-0.105,±0.155] to [0,±0.155] and from [0,±0.155] to [+0.105,±0.155] are perfectly collinear. The old teacher retracted and attempted a new deep contact at the *center of a straight physical push*, which was unnecessary and caused the observed paired failure. The joint-space or physical cause of the unattainable deep contact has **not** been independently verified; it may involve IK workspace or collision. Do not claim a particular internal IK failure mechanism.

## v0.5.0 fix

- Add `contact_waypoints(side)`: preserve turns and final target, prune forward collinear *contact breakpoints*.
- Keep `route_points(side)` unchanged for future masks, simulator's check region and task success.
- Physically push from first corner across center directly to far corner, monitoring measured cube position on each `env.step()`.
- Preserve bounded re-contact attempts for truly missed corner/goal, all failure logs and exact success constraints.
- Save each new experiment to its own `outputs/ee_v050_episodes` dataset; never mix earlier failed experiments into training.

## Next empirical gate

Run one paired seed using v0.5.0. Both routes must produce real physical task successes, not just `check_visited=True`. Failure outcomes should be recorded and traced by `EXPERT_CONTACT_STROKES` and `PUSH_ATTEMPT` logs. **No new physical success is yet claimed.**
