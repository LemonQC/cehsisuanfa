#!/usr/bin/env bash
set -euo pipefail
# Call from the extracted project root. This does not change an existing conda environment.
conda create -y -n agcofl4090 python=3.10
# `conda activate` requires shell init; this script intentionally uses conda run.
conda run -n agcofl4090 python -m pip install --upgrade pip
conda run -n agcofl4090 python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
conda run -n agcofl4090 python -m pip install -r requirements.txt
mkdir -p outputs
conda run -n agcofl4090 python -m pip freeze > outputs/environment_freeze.txt
conda run -n agcofl4090 python -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; print(torch.cuda.get_device_name(0))'
echo 'Environment ready. conda activate agcofl4090'
