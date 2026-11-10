# 2026-11-10 v1.4.9 — Slow JAX CUDA/cuDNN wheel installation (no model download)

## Verified operator log

A new dedicated pi05 Conda installation of the authors' JAX 0.5.3 stack
was interrupted by user during the pip CUDA dependency stage:
- jax_cuda12_pjrt 0.5.3 104.7 MB already appeared as cached;
- nvidia_cudnn_cu12 9.28.1.4: 716.6 MB wheel, slow (~0.4–0.7 MB/s);
- pip reported "Operation cancelled by user".

This is an INSTALL-TIME GPU dependency, **not the already completed
6.20 GB RoboReel π0.5 Uniform checkpoint**. No change to scientific results,
no physical rollout, no reason to re-download policy weights. A cancelled
pip install may leave some packages installed and others absent; rerunning
the opt-in script is idempotent in the usual pip sense, but must be validated.

## Immediate operator command (minimal change)

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    conda activate pi05
    export PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    python -m pip install "jax[cuda12]==0.5.3"

If this works, run the remainder of the repo's dedicated setup script.
v1.4.9 has changed scripts/setup_pi05_author_env.sh to set:
- PIP_INDEX_URL to a temporary, script-local HTTPS Tsinghua PyPI mirror
- its single --index-url for CUDA torch to an HTTPS NJU PyTorch mirror

Without any modifications to global pip.conf, direct system libraries or
other Conda environments. The author version pins (JAX 0.5.3, etc.) stay
unchanged. Customize if mirrors inaccessible:

    PI05_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PI05_TORCH_INDEX_URL=https://mirror.nju.edu.cn/pytorch/whl/cu126 \
      bash scripts/setup_pi05_author_env.sh

Preferred exact flow once repo main is fast-forwarded:

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05
    bash scripts/setup_pi05_author_env.sh
    bash scripts/run_v141_next_gate.sh check-pi05-env

If pip downloads are still slow, another already installed CUDA 12/cuDNN
wheel version may exist in a local environment's pip cache. Investigate
without modifying or deleting any environment:

    python -m pip cache dir
    python -m pip show nvidia-cudnn-cu12 nvidia-cublas-cu12 || true
    conda run -n agcofl4090 python -m pip show nvidia-cudnn-cu12 || true

Do NOT move random shared libraries into pi05 or use --no-deps: JAX needs
a compatible complete CUDA runtime and working GPU check. JAX docs recommend
the pip-managed CUDA installation; local CUDA/cuDNN mode is supported but
more complex and should not be used as a blind workaround.

The PIP mirror must be verified over HTTPS. Merely switching the index
does not guarantee the mirror has every NVIDIA or Torch wheel, or that it
will be faster on a specific network.

Once check-pi05-env passes, proceed: smoke-one-feature, prepare-features,
require-ready, evaluate-uniform. v1.3 Vid2Robot 3/60 remains frozen.

References:
- Tsinghua mirror official help https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
- JAX install instructions https://docs.jax.dev/en/latest/installation.html
- NVIDIA cuDNN wheel documentation https://docs.nvidia.com/cudnn/latest/installation/backend/linux/
