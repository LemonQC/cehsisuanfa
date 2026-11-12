# TRO1 — Official RoboReel pi0.5 TRAIN-video DEV runtime repair, 2026-11-12

## Operator's real outcome

- Genuine RoboReel TRAIN human-video inputs for `toys_in_drawer` (0000/0001), and bundled SigLIP `front.npy` features, were acquired/generated and preflight reported `READY_FOR_BOUNDED_DEV`.
- Running the earlier `--run --episodes 2 --max-steps 800 --out outputs/roboreel_video_dependency_dev/v2` failed **before any simulator `.step`** while the original `load_train_config()` was trying to expand `${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_base`.
- The same operator execution also reported `SyntaxError: unterminated string literal` in the submitted unit test at line 170. The original `v2/manifest.json` was written before the runtime environment check; retain it as a failed, zero-step configuration attempt rather than overwriting it.
- No learned-method result or policy-level physics evaluation is implied by this log.

## Repair

1. The new `prepare_official_runtime_config()` resolves the **already validated** local OpenPI cache with the same `cache_home()` gate used by successful official SigLIP feature extraction. It sets `OPENPI_DATA_HOME`, `ROBOREEL_CKPT` and `ROBOREEL_DATA` before calling the authors' unmodified `evaluation.utils.load_train_config()`. If no verified OpenPI cache exists, execution fails closed without launching or downloading a model.
2. All variables are expanded, the official checkpoint task is checked, and the final config is read before the run's immutable `manifest.json` is created. No GPU/physics imports occur before these runtime checks. Previous snapshots and manifests remain untouched.
3. Replaced the corrupted test tail with the original complete three immutable-evidence tests and added checks for official M0 environment variables, missing OpenPI cache, and **no manifest on a pre-execution config error**.
4. Default new output directory is `outputs/roboreel_video_dependency_dev/v3` because `v2` already contains a manifest. Do not remove `v2` or reuse it.

## Reproducible, bounded command

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_roboreel_video_dependency_dev.py' -v
# No repeated downloads; new TRAIN media and SigLIP features already exist.
env -u DISPLAY -u WAYLAND_DISPLAY python -u -m scripts.roboreel_video_dependency_dev \
  --run --episodes 2 --max-steps 800 \
  --out outputs/roboreel_video_dependency_dev/v3
```

The script still uses the pinned official RoboReel runner and four **new TRAIN-only DEV** rollouts, saving video/actions/status. It does not rerun the historical 60 PushCube EVAL trials, train the policy, or change an official checkpoint. If this attempt throws an error after writing its manifest, preserve `v3` and examine its logs before trying a different output directory.

## Verification limits

Local synthetic/regression tests: 21/21 pass after reconstruction and Python compilation succeeds. These tests do **not** prove the real robot scene will run on the operator's RTX 4090. The original OpenPI/JAX runtime remains the user's pinned `pi05` installation. Revalidate on-server before reporting physical outcomes.
