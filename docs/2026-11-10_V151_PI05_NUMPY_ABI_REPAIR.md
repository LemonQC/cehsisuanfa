# 2026-11-10 — v1.5.1 π0.5 JAX GPU works but NumPy-1.x ABI has transitive conflicts

## New operator evidence

In the existing isolated `pi05` Python 3.11 environment, the user
successfully replaced `numpy 2.4.6` with `1.26.4` and replaced
`opencv-python-headless 5.0.0.93` with `4.11.0.86`. pip printed
"Successfully installed"; these operations did NOT fail.

However pip reported remaining requirement conflicts:
- numpydantic 2.0.0 -> requires NumPy >=2.0.0;
- pandas 3.1.1 -> requires NumPy >=2.0.2 on Python 3.11;
- ml-dtypes 0.6.0 -> requires NumPy >=2.0.0;
- opencv-python (GUI) 5.0.0.93 -> requires NumPy >=2 in this environment;
- mani-skill 3.0.0b22 -> declares gymnasium==0.29.1 but the **authors'
  own ENVIRONMENTS.md explicitly overrides to gymnasium==1.3.0**. Leave
  this deliberate metadata conflict for runtime verification rather than
  blindly downgrading Gymnasium.

Previously the user observed JAX 0.5.3 with GPU:0 and all SigLIP imports.
This confirms GPU visibility **at that time**; once ABI changes have been
made, verify again before the first actual feature computation. The real
SigLIP tokenizer weights are NOT yet confirmed restored. π0.5 physical
rollouts remain **NOT_RUN**; official Vid2Robot v1.3 3/60 stays frozen.

## Minimal repair now — no large CUDA or checkpoint downloads

Use the newly committed opt-in script which guards against executing
outside `pi05` and only modifies the dedicated Python ABI dependencies:

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05
    bash scripts/repair_pi05_numpy_abi.sh

Exactly pinned small Python packages:
- numpy==1.26.4
- numpydantic==1.6.11
- pandas==2.2.3
- ml-dtypes==0.5.1
- opencv-python-headless==4.11.0.86

If both OpenCV flavors are installed, uninstall only the GUI
`opencv-python` and force-reinstall the **already chosen headless** build,
because both packages provide the same `cv2` modules. The repair script
takes care of this sequence safely in the isolated pi05 env.

No `jax[cuda12]`, cuDNN, PyTorch, tokenizer weights or model checkpoint
are installed or downloaded by this repair. It uses the operator's
configured HTTPS PyPI mirror (default Tsinghua).

The official setup_pi05_author_env.sh script was corrected with the same
pins to prevent future fresh installs from silently upgrading them.

## Runtime checks and next decision

    bash scripts/run_v141_next_gate.sh check-pi05-env
    python -m pip check

The `pip check` command may still report the single ManiSkill/Gymnasium
metadata mismatch described and intentionally allowed in the pinned
author setup. Any OTHER warning is a live blocker that must be inspected.
New `check_pi05_runtime.py` checks not only imports and GPU but also the
target NumPy/pandas/numpydantic/ml-dtypes versions and duplicate OpenCV
distribution status; it now rejects a falsely green GPU-only preflight.

ONLY after a green runtime check, run:

    bash scripts/run_v141_next_gate.sh smoke-one-feature

Do NOT immediately run 20 SigLIP extracts if one-feature smoke fails.
A successful first feature extraction is required to establish that the
original SigLIP weights load and the `front.npy` schema is usable.
Thereafter:

    bash scripts/run_v141_next_gate.sh prepare-features
    python -m agcofl.v141_pi05_assets --variant uniform --require-ready
    bash scripts/run_v141_next_gate.sh evaluate-uniform

Original policy checkpoint remains downloaded; no new benchmark results
are invented. The fresh repair script and updated tests are committed but
have not yet run on the user's 4090.

## Relevant official reference constraints

- RoboReel original environment `third_party/roboreel_models/ENVIRONMENTS.md`,
  section 1, pins JAX 0.5.3 with Python 3.11, sets Gymnasium 1.3.0,
  OpenPI client, ManiSkill and original model imports.
- NumPy-1-compatible published packages: numpydantic 1.6.11, pandas 2.2.3,
  ml-dtypes 0.5.1, OpenCV headless 4.11.0.86.
