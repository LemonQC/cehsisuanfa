# 2026-11-09 — T-RO Cross-Behavior v1.1: frozen baseline & ViT-first execution plan

## Decision (NOT abandonment)

Keep the expert-reviewed question: **source object effects under different embodiment/behavior capability → target-robot-executable effects**. Previous v0.9 CNN was a disposable engineering probe; its 0–0.24% train IoU does NOT refute the research idea. v1.0 learned-mask model lost to persistence (79.88% train IoU) because it resized a tiny object to approximately 2.48 foreground pixels in 32x32, among other defects. **Archive that pilot, do not optimize it.**

The novelty is NOT adopting ViT, not extracting blue HSV masks, not selecting keyframes alone, and not an A* way-point plan. The method will combine:
1. source video / calibrated 3D object-effect extraction (goal, task-relevant intermediate effects, contact/support stages, trajectory);
2. target capability-aware effect **projection** with physically grounded constraints (support, grasp feasibility, contact, obstruction, joint workspace);
3. closed-loop learned robot policy conditioned on projected effects, preserving task-relevant outcomes even when the human/robot behaviors are incompatible.

A geometry-only candidate identical to Goal+Waypoint Oracle is a strong **negative control**, not a reason to abandon. Demonstrate incremental benefit at matched information/budget and in physically realized tasks, or narrow the claim.

## Primary public benchmark & exact reproducibility rules

Official benchmark: https://github.com/gupta-anmol99/roboreel
Official model code + released checkpoints: https://github.com/weiweigu1998/roboreel_models
**Pinned model repository**: 79a0fb0099e73c5560114753a8706276f93aefc2 (as checked on 2026-11-09).
Dataset: https://huggingface.co/datasets/LogosRoboticsLab/roboreel

- Official TRAIN human videos are in human_demos/train/<bucket>/<task>/<id>. **All IDs 0080–0099 under eval are held out**.
- Robot demonstration trajectories live under robot_trajectories/<bucket>/<task>/<id>. Their exact files/actions must be inspected BEFORE claiming robot imitation training.
- Dataset official repo is ~104 GB excluding depth; DON'T download entire dataset yet. Six real training PushCube front MP4s already downloaded; 260 candidates discovered, NOT all downloaded.
- Official evaluation: RoboReel v1 unchanged, the same videos/scene seeds per method, ND then WD and ED, with PD only for compatible pose-based methods. Real bench results should use 3 repetitions x 20 episodes per condition, official 1000-step limit and success checks; preliminary smoke uses 2 episodes and must be labelled DEV ONLY.
- Robot observation: official RGB with full camera calibration, same policy actions where applicable; privileged state baseline separate, **never put oracle-state success in RGB table**.
- Train/test and checkpoint selection only via TRAIN + validation, never official test video ID 0080 to tune checkpoints.
- Source behavior versus target behavior experiments must share **TASK OBJECTIVE**; official PickCube vs PushCube are DIFFERENT goals (lift vs push), so comparing them directly is NOT a legitimate cross-behavior transfer claim. If a novel capability-mismatch task is added, label it CUSTOM-ADDITIONAL and keep official RoboReel evaluation unchanged.

## Mandatory baseline matrix (method, family, priority, reproduction status)

2026-11-11 status: M0 B03 officially evaluated 46/60; Vid2Robot B05 remains 3/60. Other methods and M1 comparisons stay NOT_RUN until actual executed. No paid VLM/LLM API. Prioritize B03 Uniform with local models, official B05 remains frozen, and custom same-goal physical feasibility + learned policy. B04/B07 are deferred, NOT_RUN. No made-up keyframes.

Latest evidence: user 2026-11-11 terminal marker `ACTUAL_PI05_PHYSICS_VERIFIED {"variant":"uniform","k":46,"n":60}`; M0 compact per-trial archive awaiting execution of `scripts/archive_pi05_uniform_m0.py`; earlier failed setup records under v141 are historical, not current state. Custom Pick→Push task still fails source validity and M1 baselines have not run. See `docs/2026-11-11_M0_DONE_M1_FROZEN_PREMETHOD_PROTOCOL.md`. The following table preserves frozen scientific controls but updates execution status only.

| ID | Method | Family | Priority | Status / reproduction rule |
|---|---|---|---|---|
| B01 | Robot-only Transformer BC, RGB, no human | No-human reference | P0 | Needs target robot training trajectories; local implementation clearly REIMPLEMENTED |
| B02 | Robot-only Diffusion Policy, RGB, no human | No-human reference | P1 | Official DP architecture preferred; inspect dependency and training protocol |
| B03 | Video-conditioned pi0.5, Uniform | RoboReel official | P0 | **M0 ACTUAL official ND PushCube 46/60 = 76.67%, 20x3**; operator log verified 2026-11-11; local 60-row SHA256 archive PENDING push; not an M1 same-goal result |
| B04 | Video-conditioned pi0.5, Keyframes | RoboReel official | P2 DEFERRED | NOT_RUN; external VLM API needed for unavailable original keyframe annotations; excluded from current no-API execution, never substitute heuristic frames for official |
| B05 | Vid2Robot | RoboReel official | P0 | v1.3 ACTUAL_60_PHYSICAL_TRIALS: PushCube ND 3/60=5.0%, 20x3; only one official cell; no rerun |
| B06 | UniSkill policy | RoboReel official | P1 | OFFICIAL_CODE_AVAILABLE; external ISD checkpoint/24GB batch constraints |
| B07 | VLM + pi0.5 | RoboReel official | P2 DEFERRED | NOT_RUN; language/VLM annotation pipeline out of current no-API scope |
| B08 | SeeDo | RoboReel paper | P2 | OFFICIAL_MODEL_REPO_NOT_YET_AVAILABLE; no fake reproduction |
| B09 | PointPolicy | RoboReel pose suite | P1 PD | OFFICIAL_MODEL_REPO_NOT_YET_AVAILABLE; use REAL calibrated 3D data |
| B10 | Cross-Behavior Object-Flow, Chen et al., T-RO 2026 | Research-specific | P0 | Need authors' original code/license/dataset before faithful port |
| B11 | Im2Flow2Act, CoRL/PMLR 2025 | Object-flow cross-domain | P1 | OFFICIAL_CODE_AVAILABLE in real-stanford/im2Flow2Act; pin f44b368, faithful conversion to RoboReel NOT VERIFIED |
| B12 | EC-Flow, ICCV 2025 | Embodiment-centric flow | P1 | OFFICIAL_CODE_AVAILABLE in YixiangChen515/EC-Flow; pin 7d0a624, originally Meta-World; conversion to RoboReel NOT VERIFIED |
| B13 | ViT-DINOv2 + temporal decoder, human source only | Modern feature diagnostic | P0 | **OUR REIMPLEMENTED DIAGNOSTIC**, no robot policy yet; NEVER label official baseline |
| B14 | Goal-only / Goal+Waypoint Oracle / persistence | Non-learning controls | P0 | Separate physical privileged / geometric / visual tables |

This is the **full reproducibility matrix**, NOT a fabricated 14-way official success comparison. Compare only valid, actually executed methods under matched inputs and budgets. Official RoboReel model README says five methods have training entrypoints; SeeDo and PointPolicy are still to be added there.

## Locked task plan, not just PushCube

**Official performance:** PushCube-v2 (interface/easy baseline), PutBasket-v1 (multi-object/long-horizon), ObjectInDrawer-v1 (state and contact changes), StackCups-v1 (ordered task constraints). Start with PushCube + PutBasket; later add ObjectInDrawer/StackCups. Repo official evaluates 10 tasks with paired digital twin.

**Cross-behavior science:** Must introduce a **same-goal** capability contrast. Candidate: place one object into a common region using pick-and-place source versus push-only target *when physically possible*, with both a necessary route/check and a feasible-but-irrelevant detour. Before claiming novelty, demonstrate (a) a source flow whose support/contact mode cannot be reproduced by target, (b) a physically executable target route, (c) Goal-only and Waypoint Oracle behaviors under equal actionability information/budget and (d) learned policy outcomes, not precomputed paths. If Push-only physically cannot complete the goal, categorize UNREACHABLE rather than converting failure into success. A custom task is EXTRA; never call it an original RoboReel test.

## Frozen evaluation tables

### Table I: Official RoboReel, RGB, no privileged state
Rows B01-B07 (only methods with runnable code, trained checkpoint and identical official evaluator); columns PushCube/PutBasket/ObjectInDrawer/StackCups; ND/WD/ED; success k/N and 95% Wilson; three independent 20-scene runs for formal results.

### Table II: Source human supervision regime
Robot dataset, robot policy/backbone, training updates and architecture IDENTICAL. One column per source human Full/Anchor/End; uniform-K/random-K controls for annotation budget; no-human Robot-only; fixed test start states. **Use predicted** object flow to condition the action policy. Training on target video itself with no heldout test is forbidden.

### Table III: Capability conflict / physics realization
Same-task same-goal source variants and same robot states; methods: Raw source-flow, 2D geometric warp, Goal-only, Goal+Waypoint Oracle, visual object-flow (B10/11/12 as runnable), our capability projection, privileged upper bound (SEPARATE). Metrics: robot goal success, intermediate-effect preservation, support/contact violation, actual collisions, route and action budget. Balanced necessary vs habitual event cases.

## Stage / GPU execution plan — what to actually do

- P0-A **NOW**: clone pinned RoboReel official model repo **without** its enormous weights or submodules; inspect which exact runnable scripts, configs, action space and local data are present. Produce a reproducibility matrix JSON and lock source commit.
- P0-B **NOW**: on six real TRAIN videos, run DINOv2-S pretrained ViT GPU feature extraction and a small temporal-transformer future-centroid sanity test. RGB 224x224, NOT 32x32 masks; video-level heldout split, compare persistence and constant velocity in **pixel displacement**. This ONLY validates representation and pipeline, cannot prove T-RO. If no checkpoint or dependency, record BLOCKED rather than random-initialized ViT. DINOv2-S can fit a 4090 with frozen encoder.
- P1: retrieve a modest authentic robot TRAIN trajectory subset and official checkpoints for B03/B04/B05; official evaluation runner on 2 **DEV** episodes, not 20 test for tuning; match obs/action APIs. If checkpoint downloading or precompute depends on large resources, label blocked; do not silently fall back to the privileged diagnostic.
- P2: after official baselines and robot data work, train the common RGB/ViT+temporal flow-conditioned policy; run paired Full/Anchor/End, robot-only, keyframe, source-flow variants.
- P3: physical capability conflict on a same-goal task; quantify margins versus B10/B11/12, geometric baseline, and waypoint oracle.

**Do not train new policies from RGB alone without target action labels.** Six TRAIN human videos do not themselves train a robot policy. Need robot trajectories and real matched goal/mode labels.

## Explicit stop conditions (for a hypothesis, NOT abandonment from one CNN)

Stop an implementation when there is a concrete technical blocker; keep the research question and switch baseline or data source, document differences. Stop a particular scientific claim only if paired physical experiments at matched information/budget show no meaningful gap over strong baselines. Neither happened from v0.9 CNN masks.

## Exact reproducibility tracking

- Scripts must emit data SHA/split IDs, backbone checkpoint revision, seed, model type and GPU used.
- Model checkpoint must be real pretrained ViT, verified loaded; NEVER silently fallback to random.
- Large training data/weights/video stay local. Machine-readable small CSV/JSON and MD summaries should be pushed to GitHub.
- All official-vs-port-vs-reimplementation labels in this matrix are **provisional until a code + runtime audit**.

## Verified external reproducibility sources (not yet run in this project)

- **Official RoboReel baseline source**: https://github.com/weiweigu1998/roboreel_models at commit 79a0fb0099e73c5560114753a8706276f93aefc2. Five source entrypoints are available; SeeDo and PointPolicy are still documented as "will be added later". Official model documentation reports 104 GB for RGB/robot data excluding depth, released checkpoints approximately 50 GB total; selective downloads only.
- **Im2Flow2Act**: https://github.com/real-stanford/im2Flow2Act at commit f44b368b76195c30bf38074f6419fc8b9f8feb5f, paper available in PMLR 2025. Its original simulated play data and checkpoints use different tasks; a port requires explicit discrepancy documentation.
- **EC-Flow**: https://github.com/YixiangChen515/EC-Flow at commit 7d0a6243fd8b92500b5f864d70b3b60d6b6c853a, ICCV 2025. Authors' script is Meta-World-focused; SAM2/CoTracker/GroundingDINO dependency makes single-card exact reproduction costly. Do not promise an official RoboReel score from it without a faithful action/observation port.
- **Cross-Behavior Object-Flow TRO 2026**: DOI 10.1109/TRO.2026.3701576; https://eprints.whiterose.ac.uk/id/eprint/242108/. Official public implementation has not been verified, so no "OFFICIAL REPRODUCTION" label yet.

## Actual real-data checks in v1.1 code

- One command: scripts/run_v110_official_baselines_and_vit.sh.
- Official 5-method code audit: agcofl/official_baseline_audit.py, outputs/v110_baselines/official_baseline_audit.json.
- Real TRAIN robot H5+RGB selection: agcofl/official_robot_data.py, outputs/v110_baselines/robot_data_manifest.json.
- Official human Pose/Calibration metadata: agcofl/human_pose_source_audit.py, outputs/v110_baselines/human_pose_annotation_audit.json.
- Frozen pretrained DINOv2 ViT-S/14 + 2-layer temporal Transformer, 224 RGB, six TRAIN videos, full-video split: agcofl/vit_flow_pilot.py.
- Real Video PERSISTENCE and CONSTANT_VELOCITY baselines assessed in native-camera **pixel** errors, not degenerate 32x32-mask IoU.
- This ViT pilot is not a policy / not a Baseline success rate. The real OFFICIAL B03–B05 rollout numbers only enter the main results table after actual released checkpoint loading and official evaluation.
