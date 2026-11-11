# 2026-11-11 — v1.6.3 Original π0.5 SigLIP restored; process isolation for remaining 13 EVAL videos

## Actual new user-server result

The server now has the original OpenPI pi05_base checkpoint staged locally
and the original-author SigLipTokenizer restored/exported
`OPENPI_DATA_HOME/pi05_vision_encoder/siglip_params.pkl` with
**1,659,216,368 bytes**. The `pi05` environment successfully checked
JAX CUDA device `gpu:0`, `numpy==1.26.4` and OpenPI imports.

The shared-process SigLIP extraction:
- EVAL front.npy bundles 0080–0086 present (7/20).
- Log explicitly confirms 0081 through 0086 with T=207,176,189,203,203,185;
  0080 was already present before this invocation.
- Logged a warning: `os.fork() was called ... JAX is multithreaded`
  from subprocess handling during later FFmpeg video decode.
- Python process was **killed** by the OS/shell at approximately next demo
  0087 without a Python exception. OOM is plausible but NOT confirmed by
  kernel memory evidence. Do not claim certainty about cause.
- Official assets gate correctly rejected missing IDs 0087–0099.
- `--require-ready` failed; no official π0.5 policy rollout occurred
  (0/60).

## Root cause scope and fix

Original pinned author file:
`third_party/roboreel_models/scripts/compute_vision_features_for_demo.py`
implements `_PersistentEncoder` and was formerly instantiated once inside
`agcofl/v141_pi05_assets.prepare_features` for the entire 20-video job.
That reuses GPU JAX compilation and arrays across video samples and decodes
subsequent MP4 files with FFmpeg after JAX already initialized. The source
author even documents JAX device memory leaks with repeated model setup.
Our wrapper could lead to host/GPU memory accumulation and fork warnings.
The same author encoder successfully processed 7 original videos.

New code changes ONLY the orchestration (not the pinned author algorithm):
- `--feature-process-mode isolated` runs one missing EVAL ID per fresh
  Python subprocess. Each child reads its MP4 *before* initializing JAX.
- Each child imports and uses the *same original pinned author*
  `_PersistentEncoder`, original `image_tools.resize_with_pad`,
  pooling, 3D positional embedding, and original bundled `front.npy`
  output schema and atomic write.
- After each one-video extraction, the child exits, returning JAX GPU/host
  allocations to the OS. This costs reloading the 1.66GB SigLIP pickle and
  re-JIT for each demo, trading speed for bounded per-demo lifetime.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` for workers avoids JAX reserving
  nearly all VRAM; the original input/output numerical semantics stay
  unchanged. This does not ensure the worker fits memory for every video.
- The parent skips already existing nonempty `front.npy` (0080–0086),
  fails fast if a child exits nonzero or SIGKILL, and never fabricates
  a complete bundle. The author's original atomic save prevents a killed
  worker leaving a complete-looking partially written `front.npy`.
- Both `prepare-features` and `smoke-one-feature` runner modes use
  isolation. `--single-feature-id` is an internal child-only selector
  locked to the 0080–0099 evaluation set. No changes to learning or
  evaluation settings.
- CPU/mock unit tests cover worker command, skip-existing, SIGKILL and
  invalid held-out IDs. They are committed, not yet run on the operator
  server.

## Next operator action (NO checkpoint redownload)

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05
    python -m pytest -q tests/test_v141_pi05_assets.py

    # Only 0087..0099 should be attempted, one independent child each.
    bash scripts/run_v141_next_gate.sh prepare-features

    # Do not evaluate until all 20 original bundles pass existing asset gates.
    python -m agcofl.v141_pi05_assets --variant uniform --require-ready
    bash scripts/run_v141_next_gate.sh evaluate-uniform

If a fresh single-video child still dies with SIGKILL, the prior
*shared-process accumulation* hypothesis is not enough. Collect OS OOM
evidence using `journalctl -k -b --no-pager | grep -Ei 'out.of.memory|killed process|oom'`
(read-only, may require sudo), inspect exact per-demo frames and RAM/VRAM
before any further algorithm change. A chunked original-equivalent SigLIP
forward pass could then be considered but MUST be separately labeled as a
preprocessing implementation deviation, with numerical agreement tests
against a previously generated original bundle on the same video.

This remains milestone **M0** (official faithful RoboReel π0.5 Uniform).
The original checkpoint and SigLIP restore are genuine observed progress,
but 13 feature bundles and all 60 official rollouts are still unexecuted.
M1 (same-goal cross-capability scientific problem) and M2 (learned
capability-effect editor with fair baselines) remain unchanged.
