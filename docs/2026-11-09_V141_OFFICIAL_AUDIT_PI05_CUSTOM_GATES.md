# 2026-11-09 — v1.4.1 official baseline alignment and cross-behavior physics gate

## Provenance and status
Repo: LemonQC/cehsisuanfa (main). Source reference for author baseline:
weiweigu1998/roboreel_models @ 79a0fb0099e73c5560114753a8706276f93aefc2.
Author benchmark: https://roboreel.github.io/.
**This commit creates and audits code; it does NOT execute a new pi0.5 GPU
rollout or a CUSTOM-ADDITIONAL physical rollout.** Never record preflight as success.

### Already completed — Vid2Robot v1.3
research_reports/2026-11-09_v130/official_aggregated_summary.json and status.json:
- Actual author Vid2Robot, clean/clean, push_cube, human EVAL 0080–0099,
  seeded environment-bank slices offsets 0/100000/200000, max 800 steps,
  chunk size 1, Panda pd_joint_pos, original success check;
- 1/20 + 2/20 + 0/20 = 3/60 = 5.00%, three-bank SEM 2.89 percentage points.
- Full 3-pass completion flag TRUE, errors empty.
- Official RoboReel ND PushCube reports Vid2Robot 5.0 ± 2.9%.
  The official site labels uncertainty "std"; our JSON computes SEM.
  Values are numerically consistent; do not claim proven identical error
  definitions or identical action trajectories from aggregate scores.
- The original published checkpoint was strictly loaded; a launch adapter
  skipped timm's unnecessary ImageNet initializer only.
- Verdict: no evidence here of a large protocol mismatch; no more v1.3 reruns.
  This does not prove cross-behavior source Pick to target Push.

### Priority A — actual released video-conditioned pi0.5
RoboReel authors report ND PushCube: Uniform 78.3 ± 6.0%;
Keyframes 100.0 ± 0.0%. These are AUTHOR-REPORTED, **NOT OUR RESULTS**.
Their original JAX/OpenPI demonstration-conditioned evaluation runner is
src/evaluation/demo_pi05_runner.py, invoked by scripts/evaluation.py.
It predicts action chunks with action_horizon=20 and outputs 8-D actions
under the same RoboReel physics and success protocol. Do not copy
Vid2Robot chunk_size=1. Separate pi05 conda environment is expected.
The exact original checkpoint locations are:
- released_checkpoints/nd/vid_cond_pi05_uniform/ckpt/params
- released_checkpoints/nd/vid_cond_pi05_keyframes/ckpt/params
Original train_config.json lives in or next to the checkpoint.
SigLIP front-camera cache is a *bundled dictionary* of embeddings:
- official_eval_data/video_demonstration_features/without_distraction/push_cube/0080/front.npy
  through 0099/front.npy
Keyframes ALSO requires 0080.json through 0099.json under
official_eval_data/bpp_keyframe_annotations/without_distraction/push_cube/.
No fake samples / random initialization allowed. This preprocessing is
test-time conditioning, not training or tuning model on held-out EVAL.

Code prepared:
- agcofl/v140_official_pi05.py — fail-closed source, checkpoint,
  config, cache, annotation and summary gates; original 20×3.
- agcofl/v141_pi05_assets.py — only 20 fixed EVAL demos; selective
  official ND model downloader, upstream SigLIP extractor, upstream
  API-dependent keyframe extractor, all opt-in.
- tests/test_v140_official_pi05.py and tests/test_v141_pi05_assets.py.

Execution from repository root, with author-pinned repo and the
correct dedicated pi05 environment initialized:

    bash scripts/run_v141_next_gate.sh gate
    bash scripts/run_v141_next_gate.sh prepare-uniform
    bash scripts/run_v141_next_gate.sh evaluate-uniform

Then Keyframes after checking original VLM keyframe extraction API cost
and setting OPENAI_API_KEY (keep secret on server, never commit it):

    bash scripts/run_v141_next_gate.sh prepare-keyframes
    bash scripts/run_v141_next_gate.sh evaluate-keyframes

The prep stage can require multiple gigabytes and uses authors' SigLIP;
it MUST NOT download the full 104GB dataset or retrain from scratch.
The 20 official human EVAL videos and v1.3 Vid2Robot weights are already
present; do not re-download them. New pi0.5 weights and cached features
are method-specific and must be verified.

### Priority B — CUSTOM-ADDITIONAL same-final-goal Pick→Push-only
The shared final objective is PUSH TO SAME XY GOAL, not the published
PickCube lift goal. Source picks the same cube over an obstacle and
releases it at the same goal; target keeps it tabletop-supported and pushes
around the obstacle (or correctly reports failure).
agcofl/v140_custom_pick_push_physics.py creates five static PhysX wall
actors, uses the same seeded initial state and goal for source/target, then:
- Requires successful actual source grasp, cube lift >=0.055m,
  measured wall-slab elevated crossing, physical release to tabletop and
  final goal success; only then compare source-target.
- Logs measured cube crossing evidence (wall overflight vs planar detour).
- Requires target gripper to remain commanded open, and disqualifies any
  apparent push success where the cube lifts >0.012m above its start height.
- Includes goal_only, waypoint_oracle and raw_source_xy scripted
  **privileged-state** physical negative controls.
- NO human same-goal source demo, learned policy, full robot action-data
  training, or collision/contact force telemetry has been implemented.
  This is a necessary feasibility smoke, not a proposed T-RO algorithm.

Explicit physical run command (never automatically paired with v1.3):

    bash scripts/run_v141_next_gate.sh custom-physics-dev

This runs one seed in real ManiSkill/PhysX and writes raw source/target
physical JSON to outputs/v140_custom_pick_push_dev/. Before valid source
contact/lift/overflight, it intentionally stops with
SOURCE_INVALID_NO_TARGET_COMPARISON. Do not label a failed source a
human demonstration. Real physics result is **NOT_RUN** in this commit.

### Next scientific stop gates (not claimed complete)
1. Read pi0.5 preflight and selectively fill only missing files. Execute real
   official 20×3 for each variant, publish per-trial and verified summary.
   Then consider UniSkill and robot-only learned strong baselines with real H5.
2. Obtain at least one VALID physically executed Pick-over-wall and
   Push-around-wall pair, then necessary-event vs habitual-detour variants,
   actual unreachable cases, collision/force and support metrics.
3. Collect honest same-goal source demonstrations and target robot
   TRAIN action pairs; learn capability-conditioned effect projection and
   closed-loop target policy rather than only script route planning.
4. Compare Raw Flow, Goal-only, equally privileged Waypoint Oracle,
   simple geometry, matched RGB trained policies and eventual ours under
   fixed state information, action horizon, train budget and paired seeds.
   Privileged Oracle must live in a separate table from RGB-only policies.
5. If capability projection offers no physical advantage over equally
   informed/simple baselines, withdraw that algorithm novelty claim.
