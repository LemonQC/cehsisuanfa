> **Headless v0.1.1**: Start with [README_HEADLESS_FIX.md](README_HEADLESS_FIX.md) if your SSH session reports GLX errors; execute `bash scripts/run_headless_smoke.sh`, and omit `--video`.

# AG-COFL / Partial Object-Flow Supervision — ManiSkill 3 pilot

**Purpose:** verify the essential Gate A before investing in full/anchor/end high-level training. This is a real `env.step()` robotic experiment with a Panda in a custom two-route tabletop task, **not a precomputed success illustration**.

**Version:** v0.1 (2026-11-09). **Simulation runtime status:** API cross-checked against ManiSkill 3.0.1 documentation and official source; Python syntax and framework-independent unit tests checked in authoring environment; **NOT physically run with ManiSkill/SAPIEN in this container**. Your RTX 4090 Linux machine must run the included smoke test. Do not cite this code as a verified success-rate baseline until you obtain rollouts.

## What is delivered

* `AGCOFL-TwoRoute-v1`: Panda, rigid cube, central static obstacle, top/bottom path around obstacle, inspection checkpoint, final goal.
* `teacher.py`: four-leg Panda physical pushing expert using official `PandaArmMotionPlanningSolver`; stores all joint actions and full object position trace. Failed motion plans are failures, never converted to successful examples.
* `representation.py`: four 32×32 future binary square masks encoding upper vs lower desired object route. They are **synthetic geometric mask references**, not measured human video segmentation and not dense optical flow.
* `learning.py`: actual trainable CNN + proprioceptive behavioral-cloning controller that consumes future masks and outputs `pd_joint_pos` actions; held-out physical evaluation on both routes from the same starting state.
* `partial_supervision.py`: torch `full`, `anchor`, `end` losses including selected-frame mask union and central-window anchor matching. Not yet wired into a trained high-level predictor—**Gate B intentionally gated on Gate A**.
* Reproducible command-line scripts, JSON episode results, optional MP4 capture, tests.

## Hardware / software

Linux x86-64 with NVIDIA RTX 4090, CUDA driver compatible with PyTorch cu121, headless possible. Requires `conda`. `maniskill`/`sapien`/`mplib` are unavailable in the authoring container; physics API compatibility is checked against the 3.0.1 official source but is **not runtime verified**.

### 1. Environment install

```bash
unzip agcofl_partialflow_4090.zip
cd agcofl_partialflow_4090
bash scripts/setup_linux4090.sh
conda activate agcofl4090
```

If your server already has a stable ManiSkill 3 environment, use it first and install only missing dependencies, rather than overriding your current installation. To check CUDA and versions:

```bash
python -c 'import torch,mani_skill,sapien,mplib; print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))'
python -m pytest -q tests
```

If the specified ManiSkill release or its dependencies conflict with your system, record pip error, installed versions and `nvidia-smi` output; avoid blind upgrades.

### 2. One-step smoke test (do this BEFORE collecting data)

```bash
python -m agcofl.cli test-loss
python -m agcofl.cli smoke
```

Expected: two lines with side ±1, matching detected flow side, end with `SMOKE_OK`. `smoke` only verifies physics stepping, registration, masks and action shape; **not expert success**.

### 3. Collect real Panda pushing trajectories

```bash
python -m agcofl.cli collect --episodes 3 --seed 100 --out outputs/episodes --video
cat outputs/episodes/collection_summary.json
```

This can be CPU-physics slow because the official mplib solver is used. Success is required for data inclusion. The reporter tells exactly which seed and motion-plan failed; for failures first inspect `outputs/episodes/route_*.json` and validate joint reach/contact geometry before scaling.

**STOP:** if neither route has a physically successful demonstrated episode, don't attempt training. Do not invent a successful “teacher” trajectory or use kinematic teleportation to fill gaps. The exact workspace dimensions and controller behavior depend on the installed ManiSkill/SAPIEN version and need server validation.

### 4. Train Flow-conditioned low-level policy on RTX 4090

```bash
python -m agcofl.cli collect --episodes 15 --seed 200 --out outputs/episodes
python -m agcofl.cli train-bc --data outputs/episodes --epochs 100 --ckpt outputs/models/flow_bc.pt
```

The training uses GPU automatically if available. Since collection records **real Panda joint actions**, BC has no access to teacher planning targets other than the provided future occupancy masks and robot/object state. All successful expert time steps are used. It is a feasibility baseline, not official Diffusion Policy and not the exact Chen et al. policy.

### 5. Gate A: same seed, different reference mask

```bash
python -m agcofl.cli gate-a --ckpt outputs/models/flow_bc.pt --seed 9000 --episodes 5 --max-steps 500 --video
cat outputs/gate_a/gateA_report.json
```

Success must include visiting the **side-specific checkpoint and reaching goal while on table**, not only reaching either goal. A single success on both sides is an initial pilot, **not** sufficient statistical evidence for T-RO. Keep failure trajectories; report pass counts out of N and reasons. If BC chooses the same side even though reference masks differ, **Gate A fails**, and high-level full/anchor/end comparisons cannot be scientifically interpreted.

For a fast single-command trial after install:

```bash
bash scripts/run_gate_a.sh
```

### 6. Supervision module (not a trained policy)

```python
from agcofl.partial_supervision import torch_segment_loss
robot_loss = torch_segment_loss(robot_logits, robot_masks, 'full')
human_loss = torch_segment_loss(human_logits, human_masks, 'anchor')
loss = robot_loss + 0.5 * human_loss
```

All tensors above are `[T,H,W]`, logit inputs, target binary masks. The anchor is matched in the middle 50% of the prediction horizon. The end is **end of the supervised segment**, and must not be mislabeled as the final task goal for short windows that do not cover that event. The supplied unit tests verify that unselected prediction frames have no direct gradient. The full high-level experiment (two phases × three modes, nine tables) needs human clips and a working Flow-conditioned low-level controller; do NOT claim it was run here.

## Experimental integrity / what can and cannot be claimed

**Can establish if executed successfully:** a learned Panda policy can condition on future mask references to execute distinct physical upper/lower paths, in a controlled geometry-based pilot.

**Does not establish:** human-to-robot transfer, benefits of anchor/end over full, model innovation, end-to-end imitation from RGB, publication readiness, or superiority over Chen et al. (2026).

**Route information leakage:** task `route_side` changes checkpoint location and success evaluation; policy state deliberately excludes it. The only side-specific policy input is the binary reference masks. Same random seed is used for upper/lower tests. Static marker is non-colliding and visible only in optional rendering, not policy state. To rule out action-level leakage rigorously, compare matching start robot/object states and verify route-side inference from masks.

**Critical baseline caveat:** `mask_sequence` is currently a handcrafted future occupancy label, not raw RGB human masks. This is an Oracle-control Gate A; do not label it as “object flow predictor” or final proposed method.

## Important sources / API reference

- [ManiSkill 3.0.1 PushCubeEnv official implementation](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/push_cube.py)
- [Official Panda PushCube motion-planning solution](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/examples/motionplanning/panda/solutions/push_cube.py)
- [Official Panda motion-planning solver](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/examples/motionplanning/panda/motionplanner.py)
- [ManiSkill task building documentation](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/intro.html)
- [ManiSkill observation documentation](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html)
- Source paper: [Cross-Behavior Learning With Object Flow Prediction for Robotic Manipulation](https://eprints.whiterose.ac.uk/id/eprint/242108/1/26-0166_02_MS.pdf)

## Troubleshooting

* `ModuleNotFoundError: mani_skill`: activate the conda env and check pip install.
* `ImportError: mplib`: install official `mplib` compatible with your Python/ManiSkill, record exact version.
* `motion planning failed`: inspect per-seed JSON. Check Panda work envelope, barrier size, target pose, RRT cloud and contact height; don't change all parameters simultaneously.
* No physical expert episodes: Gate A cannot proceed. Report a NO-GO for this **implementation**, not a NO-GO for the scientific hypothesis.
* Renderer problems on headless server: omit `--video`; numerical physics and `npz` logging still run.
* GPU running but collection is slow: `physx_cpu` and mplib are CPU workloads; GPU is used for neural network training. This is intentional for minimal deterministic verification.
