#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-ecflow_b12}"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "ERROR: conda environment already exists: $ENV_NAME"
  echo "Refusing to modify an existing environment. Choose a new ENV_NAME."
  exit 2
fi

conda create -y -n "$ENV_NAME" python=3.10 pip
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install \
  torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu124
conda run -n "$ENV_NAME" python -m pip install \
  timm==1.0.15 numpy==1.26.4 pillow

conda run -n "$ENV_NAME" python - <<'PY'
import torch, torchvision, timm
print("TRO1_ECFLOW_B12_ENV_READY", {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "timm": timm.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda": torch.version.cuda,
})
if not torch.cuda.is_available():
    raise SystemExit("CUDA_NOT_AVAILABLE_IN_ECFLOW_ENV")
PY
