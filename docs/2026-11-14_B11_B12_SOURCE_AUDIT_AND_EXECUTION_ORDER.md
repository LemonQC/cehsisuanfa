# B11/B12 source audit and execution order — 2026-11-14

## Decision

Start with B11 Im2Flow2Act and B12 EC-Flow as the next external flow baselines. The source repositories and exact public revisions were inspected. Neither is a direct RoboReel Panda policy. First reproduce each method in its native environment with author weights; only then estimate the cost and scientific validity of a RoboReel port.

## Source facts verified

| Item | B11 Im2Flow2Act | B12 EC-Flow |
|---|---|---|
| Public source | real-stanford/im2Flow2Act | YixiangChen515/EC-Flow |
| Pinned revision | `f44b368b76195c30bf38074f6419fc8b9f8feb5f` | `7d0a6243fd8b92500b5f864d70b3b60d6b6c853a` |
| Paper | CoRL 2024 | ICCV 2025 |
| Native stack | UR5e / MuJoCo | Sawyer / Meta-World |
| Native policy/action | 7-D UR5e action | task-specific Cartesian controller |
| Released assets | checkpoints.zip + SD1.5 + simulation buffers | flow.pt + goal_img.pt + SAM2 + GroundingDINO + CoTracker |
| GPU note | README says >=24 GB for online tracking + policy inference | README says 8×4090 for full training; use released weights first |
| RoboReel port | substantial action/camera/robot retarget | substantial action/camera/robot retarget |

## Critical implementation findings

1. B11 official full evaluator is `scripts/controll/evaluate_from_flow_generation.py`; default config expects both flow and manipulation-policy checkpoints, calibration files, Stable Diffusion 1.5, and the author's simulation evaluation zarr buffers. Its UR5e/MuJoCo action cannot be sent directly to RoboReel Panda.
2. B12 official evaluation is `experiment/eval_my_flow_policy.py` and launches nine Meta-World tasks × 25 episodes in `eval_policy.sh`. The pinned evaluation script counts `len(policy.images) <= 500` as success, while the policy loop separately sees `info['success']`. Record both; do not conflate the proxy with an explicit environment success flag.
3. Do not install either environment into `pi05`. B11 has a large Python 3.10/CUDA/FAISS environment file; B12's install script force-installs Torch 2.4.0 + CUDA 12.4 and several large perception dependencies. Isolate environments on the project disk.
4. No new training or PhysX was launched by this source audit. No RoboReel port or performance result is claimed.

## Execution order

- Gate 0 (now): read-only local audit with `python scripts/audit_b11_b12_native_assets.py --out outputs/research_runs/b11_b12_asset_audit_20261114.json`.
- Gate 1: if B11 native assets are complete and disk/GPU allow, run exactly one official native evaluation episode from the existing eval buffers; preserve author code/metrics. Otherwise download only missing assets after checking their sizes and storage.
- Gate 2: perform B12 one-task, one-episode native checkpoint inference; independently log the original <=500-step proxy and `info['success']`. Do not run all 9×25 yet.
- Gate 3: inspect trajectories and failure modes; only then design a RoboReel adapter. Native scores are not RoboReel scores and do not establish the cross-behavior hypothesis.

## Audit script behavior

`scripts/audit_b11_b12_native_assets.py` only inspects local files, git revisions, Python module availability, GPU memory via `nvidia-smi`, and free disk. It does not download, install, extract, train, or run a simulator. It refuses to overwrite an existing JSON report.


## User server audit result (2026-11-14 17:35 +08)

The read-only audit ran successfully after adding the CLI wrapper. Server facts:
- Python 3.11.18 in the current `pi05` shell.
- RTX 4090: 24,564 MiB total, 23,762 MiB free at audit time.
- Project disk: 3,034,625,794,048 bytes free (~2.76 TiB).
- Both official source repositories are absent locally (`SOURCE_NOT_PRESENT_LOCALLY`).
- No local B11/B12 asset readiness can be inferred until the source trees are cloned.

The 4090 meets the nominal 24-GB class, but B11's online tracking + policy inference still has a tight memory margin. Do not run B11 end-to-end until weights are staged and a low-memory preflight is prepared. B12 should use released checkpoints for a one-task native inference smoke; do not attempt the authors' 8-GPU training recipe.

## Pinned-source clone gate

Added `scripts/clone_b11_b12_pinned_sources.sh`. It clones only the two source repositories at exact audited commits into `third_party/im2Flow2Act` and `third_party/EC-Flow`. It refuses to overwrite any existing non-matching or dirty tree and is safe to rerun when both clones are already clean and pinned. It does not install packages, download checkpoints/datasets, create environments, train, or launch simulators.

Run:

```bash
bash scripts/clone_b11_b12_pinned_sources.sh
python scripts/audit_b11_b12_native_assets.py \
  --out outputs/research_runs/b11_b12_asset_audit_after_clone_20261114.json
```

Expected source revisions:
- B11: `f44b368b76195c30bf38074f6419fc8b9f8feb5f`
- B12: `7d0a6243fd8b92500b5f864d70b3b60d6b6c853a`

Only after this second audit should any asset download or isolated environment creation be planned.


## Project hardware hard gate (user decision, 2026-11-14)

All baselines must fit on the project's single RTX 4090. Any method whose required inference/training memory exceeds this single-GPU budget is **temporarily excluded**; do not use multi-GPU workarounds, do not spend time on an infeasible port, and do not treat nominal "24 GB" marketing capacity as proof of fit. Methods close to the limit require a measured peak-memory preflight with practical headroom before full evaluation.

Consequences:
- **B11 Im2Flow2Act: HOLD / likely resource-excluded** until its simultaneous online point-tracking + policy inference path is measured and shown to fit one RTX 4090 with headroom. No environment creation, large checkpoint downloads, or inference attempt before that gate passes.
- **B12 EC-Flow: candidate only**, not yet approved. First inspect released checkpoint sizes and run a small, isolated one-task memory preflight. Proceed only if it fits the single GPU; do not reproduce the authors' 8×4090 training.
- Apply the same single-GPU gate to every remaining baseline, not only B11/B12. Prioritize methods that can run with released weights and fit the hardware.



## B12 native policy evaluation: dependency narrowing

Inspection of the pinned source clarified that the official Meta-World policy evaluator and the standalone flow-visualization demo have different runtime requirements.

- `experiment/eval_my_flow_policy.py` constructs `FlowPredModel`.
- `FlowPredModel.__init__` loads the flow DiT checkpoint `ckpt/flow.pt` and CLIP RN50.
- The policy evaluator does not instantiate `GoundedSam2`, `PointTracker`, or the goal-image DiT. Their checkpoint files are therefore not loaded by this policy-evaluation path.
- Caveat: `inference.py` imports `sam_and_track.utils` at module import time, and that module imports SAM2, GroundingDINO and CoTracker Python packages. Those code dependencies may still be needed unless we make a minimal lazy-import-only patch; their large weight files are not needed for this evaluator.
- The separate standalone `inference.py` visualization path does load the goal-image model and invokes SAM2/GroundingDINO/CoTracker.

The Hugging Face model card lists `flow.pt` at 9.46 GB and `goal_img.pt` at 715 MB. The flow file size is a disk/host-RAM concern, not a GPU VRAM estimate. Check available system RAM before loading because the official helper first loads the checkpoint to CPU. The updated asset audit reports this execution profile and host RAM availability.

**Current decision:** B12 remains a single-4090 candidate, not yet approved. First check the updated audit's `B12_native_policy_eval` profile and whether `flow.pt` / CLIP RN50 are already local. Do not download the goal-image or SAM2/GroundingDINO/CoTracker weights for the native policy evaluator. After the actual required assets and import dependencies are verified, run a model-only FP32 GPU preflight and require at least 4 GiB of VRAM headroom before starting Meta-World.
