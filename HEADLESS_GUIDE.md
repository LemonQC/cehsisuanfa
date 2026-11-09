# RTX 4090 / SSH no-display guide (v0.2.0)

## What changed

Older versions used `render_backend='gpu'` whenever `--video` was enabled. With
`DISPLAY=localhost:10.0` (SSH X11 forwarding without GLX), SAPIEN/Vulkan may segfault
natively before Python can catch an exception. v0.2.0 **always** uses
`sim_backend='physx_cpu', render_backend='none', render_mode=None`, whether
`--video` is present or absent.

`--video` now **only** creates an offline 2D MP4 showing the recorded cube XY trace,
obstacle, checkpoint regions and goal. It is **not a real RGB robot video**.
All numerical success metrics are computed from the actual ManiSkill states, not
the rendered trajectory animation.

## First commands

```bash
# Your SSH server session (RTX 4090)
git clone https://github.com/LemonQC/cehsisuanfa.git
cd cehsisuanfa
conda activate agcofl4090
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
export PYTHONFAULTHANDLER=1

python -m pytest -q tests
python -m agcofl.cli test-loss
python -u -X faulthandler -m agcofl.cli doctor 2>&1 | tee doctor.log
```

Only after `STOCK_PHYSICS_OK` and `SMOKE_OK`:

```bash
python -u -m agcofl.cli collect --episodes 1 --seed 100 --out outputs/episodes --video \
  2>&1 | tee collect.log
cat outputs/episodes/collection_summary.json
```

If neither side has a successful teacher physical rollout, **stop** and share
`doctor.log`, `collect.log`, `outputs/episodes/*.json`. Do not train on failed
rollouts. You can rerun collection without `--video` (faster). `--video` no
longer depends on GPU rendering.

## Failure isolation

- `STOCK_CREATE` or `STOCK_RESET` crashes: stock ManiSkill/SAPIEN/PhysX or native
  graphics libraries; unrelated to the custom task or supervision loss.
- Stock stage succeeds but `SMOKE_CREATE` crashes: custom task initialization.
- Both stages succeed but teacher fails: `mplib`, grasp pose, collision/path or
  push contact geometry; inspect errors in per-episode JSON.
- `LOSS_CHECK_OK` does **not** validate any ManiSkill physics. Equal loss values
  are possible with uniform test logits even though gradients touch different frames.

Do not blindly upgrade NVIDIA/CUDA/PyTorch. Record `nvidia-smi`, `python -m pip
freeze`, and the exact stage marker where a failure occurs.
