# AGCOFL 0.1.1: SSH GLX missing / segfault fix

The previous `LOSS_CHECK_OK` is a pass. The three loss values are equal because the test uses deliberately uniform logits/masks. The different gradient frame indices are expected.

The `GLX missing on localhost:10.0` warning refers to X11 forwarding. A segfault during `smoke` is a native graphics/driver or simulator issue, not a Python exception from the loss function.

The original `make_env()` set only `sim_backend='physx_cpu'` and omitted `render_backend`, whose ManiSkill default is GPU rendering. **v0.1.1 explicitly uses `render_backend='none'` and `render_mode=None` when video is disabled**. This prevents graphical initialization in the physics-only task.

## Run first on the 4090 server

```bash
conda activate agcofl4090
cd /path/to/agcofl_partialflow_4090
unset DISPLAY WAYLAND_DISPLAY
bash scripts/run_headless_smoke.sh
```

Expected: JSON for side=-1 and side=1, ending in `SMOKE_OK: physics stepping and reference masks (render_backend=none)`.

Then, without `--video`, attempt a single physical demonstration per side:

```bash
python -m agcofl.cli collect --episodes 1 --seed 100 --out outputs/episodes
```

Do not train the policy unless the physical expert provides successful trajectories; do not turn on `--video` yet.

## If headless smoke still segfaults

Get diagnostic logs without changing dependencies:

```bash
bash scripts/check_gpu_rendering.sh 2>&1 | tee gpu_render_diagnostics.log
python -X faulthandler -m agcofl.cli smoke 2>&1 | tee smoke_failed.log
```

Try the **official** task with explicit no-render settings. This distinguishes driver / simulator issues from our task-specific source code:

```bash
unset DISPLAY WAYLAND_DISPLAY
python - <<'OFFICIAL_PHYSICS_EOF'
import gymnasium as gym
import mani_skill.envs
import numpy as np
env = gym.make('PushCube-v1', obs_mode='state_dict', control_mode='pd_joint_pos',
               num_envs=1, sim_backend='physx_cpu', render_backend='none', render_mode=None)
obs, info = env.reset(seed=42)
env.step(np.zeros(env.action_space.shape, dtype=np.float32))
print('OFFICIAL_PUSH_PHYSICS_OK')
env.close()
OFFICIAL_PHYSICS_EOF
```

If this official task also segfaults, verify the Vulkan/NVIDIA driver/ICD; capture `nvidia-smi`, `vulkaninfo --summary` and the error. If official task works but custom task fails, send the Python traceback or faulthandler output so we can patch the exact task integration issue.

If `smoke` works but `--video` crashes, **keep running without video**. Collect real physics state traces, numpy data, and matplotlib trajectory plots; video is secondary.

## Optional GPU Vulkan diagnostics for offscreen video

```bash
unset DISPLAY WAYLAND_DISPLAY
ls /usr/share/vulkan/icd.d/
# Only if this file really exists and matches your GPU driver:
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
vulkaninfo --summary
```

Never point `VK_ICD_FILENAMES` to a non-existing ICD or reinstall GPU drivers just to address a Python dependency.

Official refs:
- https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html
- https://maniskill.readthedocs.io/en/latest/api/mani_skill/envs/tasks/tabletop/pick_cube/index.html
