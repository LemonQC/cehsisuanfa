# v1.1 results → v1.2 baseline-first decision (2026-11-09)

## Empirical evidence: measured on RTX 4090 / GitHub e535046

Actual source files: `research_reports/2026-11-09_v110/`. The v1.1 run used `facebook/dinov2-small` pretrained ViT-S/14 with frozen image features, trained a temporal Transformer on six RoboReel **TRAIN** human push-cube MP4s (4 train/1 val/1 held-out test). Three modes Full/Anchor/End trained for 15 epochs on `NVIDIA GeForce RTX 4090`. This is **2D HSV color-candidate centroid prediction**, not calibrated object 3D motion or robot actions.

| Mode | Heldout test windows | Mean XY error px ↓ | Endpoint error px ↓ | Moving windows (n=3) mean px ↓ |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 10 | 9.89 | 17.57 | 32.27 |
| Constant Velocity | 10 | **7.58** | 17.69 | **23.34** |
| Full ViT | 10 | 9.73 | **16.20** | 26.97 |
| Anchor ViT | 10 | 14.50 | 19.55 | 30.66 |
| End ViT | 10 | 11.85 | 21.49 | 30.36 |

**Limits**: 10 windows from one heldout video are not 10 independent trials; 3 moving windows cannot support significance tests. Full ViT improves endpoint error vs persistence by ~1.36 pixels, but loses to constant velocity on mean error by ~2.15 pixels. NOT evidence of superior trajectory prediction, let alone robot task transfer. The old degenerate CNN pilot is an implementation failure only. No reason to abandon the T-RO study.

## Official Baseline code: what is and isn't available

- Official code and selected pinned commit `79a0fb0099e73c5560114753a8706276f93aefc2` are present in `third_party/roboreel_models` for B03/B04/B05/B06/B07. This confirms CODE availability ONLY. None has yet loaded released weights or executed official robot-action evaluation.
- Original authors explicitly advise per-method environments due JAX / PyTorch CUDA version conflicts; do not force one conda environment at the expense of faithful reproduction.
- B10/B11/B12 require matching problem and source adaptation; a larger algorithm count is NOT a valid excuse to fabricate runs.
- Official RoboReel PickCube-v2 and PushCube-v2 have **different task success objectives**; same-goal cross-behavior experiments will need a clearly labeled additional paired simulation task.

## Exact v1.1 blockers and fix

1. **REAL robot TRAIN actions 404**: prior helper incorrectly searched `robot_trajectories/without_distraction/push_cube`. Official `src/dataset_layout.py` at pinned commit maps internal `without_distraction` to published `no_distraction`. Correct path: `robot_trajectories/no_distraction/push_cube/`. Code fixed in `agcofl/official_robot_data.py`, with tests.
2. Real human 3D annotations: metadata scan found no verified SE(3) files at inspected first-level directories. Do not claim none exist globally, nor use raw HSV centers as metric 3D pose. Real camera calibration/depth may be present but not parsed.
3. Released model assets / dependencies: no real pretrained control policy checkpoint or official learning-policy rollout yet. First target the official **Vid2Robot** released checkpoint (PyTorch runner), **not** train a custom toy substitute. Official authors' model repo README gives reproducibility dependencies.

## v1.2 first genuine learned robot baseline plan

Execute `scripts/run_v120_real_robot_and_official_baseline.sh` on the RTX 4090 server. It will:

1. Run CPU-only deterministic tests of published dataset layouts and metadata safeguards.
2. Read only two actual published TRAIN robot demonstrations with original state H5 and base/hand camera MP4, budget ≤300MB, inspect action dimensions and Panda joint state offsets.
3. Fetch ONLY the authors' official no-distraction Vid2Robot released model checkpoint (budget ≤8GiB) and 20 original held-out PushCube front eval MP4s (budget ≤250MB), verifying names and actual bytes.
4. Initialize ONLY the officially pinned RoboReel environment submodule, not unrelated large submodules.
5. Attempt one official released Vid2Robot → original ManiSkill simulation episode (`0080`), max 200 steps, dev/integration only. This step is NOT a paper score. It uses no source-privileged oracle, no home-grown random policy. Correctly report if dependencies/checkpoint missing.
6. Publish small machine-readable JSON summaries to `research_reports/2026-11-09_v120/`; large video, H5, weights remain server-local.

**Research stopping rule**: A technical blocker to the official Vid2Robot reproduction calls for fixing the interface or using the next official released baseline. It is NOT grounds to abandon object-effect transfer. Research novelty is accepted/rejected only by same-goal capability mismatch, physically executable target-policy success, and matched-information comparison versus raw flow, Goal-only and Goal+Waypoint Oracle, as well as actual learning baselines.

## Next after v1.2

- If official released Vid2Robot DEV rollout executes, freeze protocol and evaluate whole 20-episode task cell without adjusting checkpoint on EVAL IDs; add second official backbone (video-conditioned π₀.₅) and remaining tasks.
- If blocked by missing checkpoint/repository environment, leave a verifiable dependency list and fix the smallest infrastructure issue; do not replace Vid2Robot with unrelated imitation-learning MLP.
- In parallel, use REAL robot TRAIN H5 observations/actions to build common input/action adapter for a standard ViT-Transformer BC baseline and for future Object-Flow-conditioned policy. No invented labels.
- Characterize ability constraints and same-goal transfer tasks after a working learned robot baseline exists, with contact-aware physics assessment, not only geometric path feasibility.
