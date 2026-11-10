#!/usr/bin/env bash
# One-time, opt-in repair after NumPy 2-only transitive upgrades.
# Does NOT install JAX/CUDA/cuDNN, touch model checkpoints, or alter agcofl4090.
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "pi05" ]]; then
  echo "ERROR: this repair ONLY runs inside the separate 'pi05' Conda environment" >&2
  exit 2
fi
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit("ERROR: pi05 should use Python 3.11")
PY

export PIP_INDEX_URL="${PI05_PIP_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"
case "$PIP_INDEX_URL" in https://*) ;; *) echo "ERROR: HTTPS mirror required" >&2; exit 2 ;; esac
echo "PI05_ABI_REPAIR Mirror=$PIP_INDEX_URL"
echo "Installing only small Python ABI-compatibility wheels, not JAX/CUDA/cuDNN."
python -m pip install --only-binary=:all: \
  "numpy==1.26.4" \
  "numpydantic==1.6.11" \
  "pandas==2.2.3" \
  "ml-dtypes==0.5.1" \
  "opencv-python-headless==4.11.0.86"
if python -m pip show opencv-python >/dev/null 2>&1; then
  echo "Removing conflicting GUI OpenCV wheel; restoring the headless cv2 installation."
  python -m pip uninstall -y opencv-python
  python -m pip install --only-binary=:all: --force-reinstall --no-deps \
    "opencv-python-headless==4.11.0.86"
fi
python - <<'PY'
import cv2, jax, ml_dtypes, numpy, pandas
print("PI05_ABI_REPAIRED", {
    "numpy": numpy.__version__, "pandas": pandas.__version__,
    "ml_dtypes": ml_dtypes.__version__, "opencv": cv2.__version__,
    "jax": jax.__version__,
    "devices": [str(device) for device in jax.devices()],
})
PY
python scripts/check_pi05_runtime.py --mode features
echo "PI05_ABI_REPAIR_OK: next run smoke-one-feature, not the full installer."
