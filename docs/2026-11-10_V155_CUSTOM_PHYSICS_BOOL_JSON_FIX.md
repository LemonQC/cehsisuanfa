# 2026-11-10 — v1.5.5 Custom Pick→Push NumPy JSON failure after genuine PhysX rollout

## Operator observation

While running `agcofl.v140_custom_pick_push_physics` on the 4090, the
ManiSkill simulator progressed through the scripted source rollout and
crashed at `json.dumps(summary)` before the source report could be
persisted. Exact error:

    TypeError: Object of type bool_ is not JSON serializable

It arose in `run_source` when serializing a `numpy.bool_` scalar,
most directly `released_on_table=(not grasp_now and obj[2] < ...)`.
The source PhysX rollout DID execute up to summary generation, but no
`valid_pick_source` or source goal success has been verified and the
target-only push controls did NOT execute in that invocation. We cannot
declare physical feasibility or a scientific result from the crash.

Gymnasium PassiveEnvChecker warnings about Torch Tensor observation
and reward are separate warnings; they did not cause this exception.
The log's traceback was executed in the `pi05` Conda env, which was
built for the official JAX/OpenPI method, whereas the custom
`custom-physics-dev` script was designed for `agcofl4090`.

## Code change

- `agcofl/v140_custom_pick_push_physics.py`: explicitly cast
  `released_on_table` and `grounded` to Python bool.
- All emitted JSON source/target traces, status and summary now use a
  shared `_json_dumps` function with `default=_json_default`.
  It accepts measured NumPy scalars and arrays and, if Torch is
  installed, detached CPU scalar/tensor data; it refuses unsupported
  objects and nonfinite measurements rather than silently
  converting all data to strings. No change to the physics policy,
  agent controller, source/target goals, environment setup or seeds.
- `tests/test_v140_custom_pick_push.py` now checks NumPy bool_ and
  numeric roundtrip, optional Torch tensor roundtrip, and rejection of
  invalid values. These tests are committed but have not been run
  on the operator's server.

## Exact rerun

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate agcofl4090
    python -m pytest -q tests/test_v140_custom_pick_push.py
    bash scripts/run_v141_next_gate.sh custom-physics-dev

Inspect these outputs after the rerun:

    outputs/v140_custom_pick_push_dev/source_pick/source_summary.json
    outputs/v140_custom_pick_push_dev/status.json

The source must satisfy actual grasp observed, lift >=0.055 m,
wall overflight (via real PhysX cube position), release on table,
and final goal success before target controls run. If invalid, the
script records `SOURCE_INVALID_NO_TARGET_COMPARISON` and stops
without fabricating a target success result.

Output `source_summary.json` can be used to diagnose further
physical-teacher failure; nothing here is a trained learned policy.

## Official baseline still blocked independently

Official RoboReel π0.5 Uniform has its 6.2GB checkpoint but
original SigLIP is missing. The public dataset audit found
zero original precomputed EVAL SigLIP files. Original GCS base
model download stalled at 0/11.6GB. Do not confuse this new
custom task crash with π0.5 evaluation or alter frozen Vid2Robot
3/60. Custom PhysX is an independent scientific feasibility DEV track.
