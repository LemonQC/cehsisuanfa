#!/usr/bin/env bash
# Opt-in implementation of section 1 of pinned third_party/roboreel_models/ENVIRONMENTS.md.
# RUN ONLY inside a newly created dedicated Python 3.11 pi05 Conda env.
# Never install packages into agcofl4090 or an existing unrelated rokae_pi05.
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != "pi05" ]]; then
  echo "ERROR: first run 'conda create -n pi05 python=3.11 -y && conda activate pi05'." >&2
  echo "This script will never modify ${CONDA_DEFAULT_ENV:-base}." >&2
  exit 2
fi
python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 11), "Pinned author env requires Python 3.11"
PY
# Public, HTTPS-only mirrors scoped to THIS script; no global pip config changes.
# Override by setting PI05_PIP_INDEX_URL / PI05_TORCH_INDEX_URL before invocation.
# cuda12 pip wheels include large NVIDIA CUDA/cuDNN runtimes; cached wheels are reused.
export PIP_INDEX_URL="${PI05_PIP_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"
PI05_TORCH_INDEX_URL="${PI05_TORCH_INDEX_URL:-https://mirror.nju.edu.cn/pytorch/whl/cu126}"
case "$PIP_INDEX_URL" in https://*) ;; *) echo "Require HTTPS PI05_PIP_INDEX_URL" >&2; exit 2;; esac
case "$PI05_TORCH_INDEX_URL" in https://*) ;; *) echo "Require HTTPS PI05_TORCH_INDEX_URL" >&2; exit 2;; esac
echo "Using PyPI mirror: $PIP_INDEX_URL"
echo "Using PyTorch CUDA index: $PI05_TORCH_INDEX_URL"
python -m pip install "jax[cuda12]==0.5.3" "numpy==1.26.4"
python -m pip install \
  "flax==0.10.2" "optax==0.2.8" "orbax-checkpoint==0.11.13" \
  "equinox==0.13.6" "jaxtyping==0.2.36" "treescope==0.1.10" \
  "augmax==0.4.1" "chex==0.1.90"
python -m pip install "jax==0.5.3" "jaxlib==0.5.3" "numpy==1.26.4"
python -m pip install --index-url "$PI05_TORCH_INDEX_URL" "torch==2.7.1"
python -m pip install \
  "transformers==4.57.6" "sentencepiece==0.2.1" "huggingface-hub>=0.25"
python -m pip install \
  "setuptools<81" "mani_skill==3.0.0b22" \
  "trimesh==4.11.5" "sapien==3.0.2"
python -m pip install "gymnasium==1.3.0"
python -m pip install \
  "einops==0.8.2" "dm-tree==0.1.10" "ml_collections==1.0.0" \
  "numpydantic>=1.6" "fsspec[gcs]>=2024.6" "h5py==3.13.0" \
  "opencv-python-headless==4.11.0.86" "pillow==12.2.0" \
  "imageio==2.37.3" "mediapy==1.2.6" "wandb==0.21.4" \
  "tyro==1.0.12" "tqdm-loggable>=0.2" "loguru==0.7.3" \
  "rich==14.3.3" "hydra-core>=1.3" "beartype==0.19.0" \
  "filelock>=3.16" "typing-extensions>=4.12" \
  "polars>=1.30" "pandas>=2.2" "etils[epath]>=1.0" \
  "imageio-ffmpeg>=0.5"
python -m pip install "openpi-client==0.1.2" "numpy==1.26.4"
# Final re-pin against transitive chex / OpenPI dependency upgrades.
# The original OpenPI client 0.1.2 declares numpy<2.0. Pip otherwise
# upgraded numpy to 2.4.x in the user\x27s environment. Keep NumPy 1.26.4
# together with JAX 0.5.3 and OpenCV 4.11, which support Python 3.11.
python -m pip install "jax==0.5.3" "jaxlib==0.5.3" "numpy==1.26.4"
python scripts/check_pi05_runtime.py --mode features
