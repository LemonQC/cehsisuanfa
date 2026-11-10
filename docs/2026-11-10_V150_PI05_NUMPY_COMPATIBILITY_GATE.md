# 2026-11-10 v1.5.0 — π0.5 GPU ready, repair NumPy/OpenPI mismatch before SigLIP

## Observed operator log (NOT a physics result)

In the dedicated Python 3.11 `pi05` environment on RTX 4090:
- The author-pinned JAX 0.5.3 is installed and `jax.devices()` returns `gpu:0`.
- The import-only `PI05_RUNTIME_CHECK` in `features` mode initially reported `ready_for_next_gate=true` and all SigLIP/OpenPI imports OK.
- A late `pip install jax==0.5.3 jaxlib==0.5.3` step silently upgraded NumPy from 1.26.4 to 2.4.6.
- pip explicitly warned `openpi-client 0.1.2 requires numpy<2.0.0,>=1.22.4`.
- pip also warned `mani-skill 3.0.0b22 requires gymnasium==0.29.1` while the *official authors' own ENVIRONMENTS.md deliberately overrides to Gymnasium 1.3.0*. This is a documented resolver conflict, **not evidence yet that the author's simulator fails**. Keep original version and test runtime, do not blindly downgrade it.
- SigLIP restored model weights have NOT yet been tested. NO new pi0.5 physical trials.
- The 6.20 GB released π0.5 Uniform checkpoint is already downloaded, no re-download.

## Immediate safe recovery in existing pi05

Do NOT rerun the entire CUDA dependency installer; in the existing pi05 env run only:

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05
    python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
      "numpy==1.26.4" "opencv-python-headless==4.11.0.86"
    python -c 'import numpy, jax; print(numpy.__version__, jax.__version__, jax.devices())'
    bash scripts/run_v141_next_gate.sh check-pi05-env

This preserves downloaded JAX GPU wheels and checkpoint. Check should
report `numpy_openpi_compatibility: OK` and `gpu_visible=true`.
If pip's package resolver flags OTHER conflicts, inspect those rather
than uninstalling GPU packages. The version pin for opencv-python-headless
ensures a currently newer OpenCV dependency does not force NumPy >=2.

Only after a green gate:

    bash scripts/run_v141_next_gate.sh smoke-one-feature

This exercises one genuine EVAL video through *original* SigLIP, not a
cached import-only check. It can reveal an un-cached independent tokenizer
checkpoint or JAX runtime error. In that case report the error and do not
blindly create synthetic `.npy` bundles.

Then, after the one-video smoke succeeds:

    bash scripts/run_v141_next_gate.sh prepare-features
    python -m agcofl.v141_pi05_assets --variant uniform --require-ready
    bash scripts/run_v141_next_gate.sh evaluate-uniform

Official Uniform evaluation is 3 independent banks x 20 EVAL held-out
rollouts; do NOT rerun frozen Vid2Robot 3/60. If any gate fails,
preserve `run.log`, `asset_gate.json` and false-status rather than
claiming an executed benchmark.

## Repository code changes

- `scripts/setup_pi05_author_env.sh`: pin NumPy 1.26.4 in all JAX
  reinstall/re-pin phases, pin compatible OpenCV 4.11.0.86 and
  openpi-client 0.1.2 so later installer reruns cannot silently break
  the working environment.
- `scripts/check_pi05_runtime.py`: reject NumPy >=2 or NumPy below
  the known compatible version even if JAX GPU and imports load.
- `tests/test_check_pi05_runtime.py`: mocked negative NumPy 2.4.6
  and positive NumPy 1.26.4 with GPU available. No real CUDA trials.
- Existing original JAX/OpenPI source, eval frozen protocol and
  RoboReel checkpoints remain unchanged.

The earlier 16/16 CPU tests and successful JAX GPU check came from
the user's real server logs; the NEW tests above are committed but
must still be run after the user's next git pull.
