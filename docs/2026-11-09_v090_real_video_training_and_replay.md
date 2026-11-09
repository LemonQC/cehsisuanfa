# 2026-11-09 | v0.9.0 real human video training and measured physics replay

## Ground truth already observed (prior RTX4090 run)

- Original human demonstration: third_party/roboreel/human_demos/eval/no_distraction/push_cube/0080/front.mp4
- Original video: 214 frames; HSV blue-mask candidate detected in 24/24 sampled frames. Coverage is NOT instance-mask accuracy.
- RoboReel PushCube-v2 episode 0: a *privileged-state* hand-coded Panda controller succeeded in 31 actual PhysX steps with 0.045422 m final error, inside 0.05 m official success radius.
- This controller did NOT make its actions depend on human video. The 1/1 result is only an engineering physical-control sanity test.
- Saved physical step states: outputs/roboreel_cn/fixed/episode_0000/physical_trace.json, episode_result.json and actions.npy.

## v0.9.0 source changes

1. agcofl/physics_replay.py creates a playback MP4 and final PNG from the actually measured physical trace; it draws cube, TCP, goal, and phase. Its top-down 2D drawing is NOT the rendered RGB robot arm; the original physics used render_backend=none, so that RGB video was not recorded.
2. agcofl/real_human_supervision.py trains a lightweight convolutional model on *authentic human MP4s* (no synthetic paths). Two observed blue-mask frames are model inputs; strictly future eight masks are labels. Full, Anchor, End use same data, network initialization and training budget; only mask supervision mode changes. It saves checkpoints, per-source quality audit, video-level splits, comparison CSV and the visual prediction montage.
3. If fewer than FIVE distinct real source videos pass data checks, do not form overlapping fake holdout groups: mark TRAIN_ONLY_NOT_GENERALIZATION and report training-fit diagnostics only. For five or more, hold out complete video IDs (never adjacent windows of the same source).
4. RoboReel 0080-0099 human videos are its official EVALUATION videos. Any training on these is a purely INTERNAL DEVELOPMENT exercise and invalidates claims of official RoboReel benchmark evaluation on these clips. Real source video masks are HSV candidates, not verified semantic segmentations.
5. tests/test_real_human_supervision.py checks no future label leakage into the input, temporal video group splitting, network shapes and gradients, real MP4 decoding, and replay MP4 generation.

## Exact single-command run on the existing RTX4090 host

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull origin main
    conda activate agcofl4090
    set -o pipefail
    bash scripts/run_v090_real_video_and_replay.sh 2>&1 | tee real_video_v090.log

No new video download, no new Panda physics simulation and no CUDA stack reinstall.

Watchable files:
- Human actual RGB MP4: third_party/roboreel/human_demos/eval/no_distraction/push_cube/0080/front.mp4
- Human segmentation diagnostic overlay: outputs/roboreel_cn/fixed/episode_0000/human_mask_overlay.png
- Simulated physical XY replay MP4 (NOT camera RGB): outputs/roboreel_cn/fixed/episode_0000/physical_xy_replay.mp4
- Predicted Full/Anchor/End frames: outputs/real_human_supervision_v090/three_mode_predictions.png

Training files in outputs/real_human_supervision_v090:
- report.json: training status and explicit scientific limitations
- source_audit.json: full provenance SHA256 and mask coverage
- video_split.json: no held-out videos when only 0080 available
- mode_summary.csv: train-only mask IoU for three supervision regimes
- all_windows.csv and training_history.csv
- full_real_video.pt, anchor_real_video.pt, end_real_video.pt.

## Decisions for the next nontrivial T-RO gate

A. Obtain real human TRAINING videos (not official evaluation IDs 0080-0099), or newly recorded footage, in multiple independent tasks/scenes. Mask quality requires manual inspection and stage-event annotation.

B. Collect matched robot trajectories under stable benchmark initial states with actual actions, camera masks/poses, success and failure. The current diagnostic controller gives a sanity baseline, not learning.

C. Train the shared high-level predictor with all ROBOT masks receiving Full and the same HUMAN video samples receiving Full/Anchor/End. Match training budget and robot sample counts. Avoid privileged target labels entering model input.

D. Validate camera-to-robot correspondence and train a genuine low-level policy receiving PREDICTED masks; no magical swap from human front pixels to the earlier top-down 4-channel FlowBC.

E. Closed-loop matched robot comparison: ordered task/event success, collision, actual route, runtime; report per-scene differences and a robot-only, uniform-K and random-K baseline.

CRITICAL: RoboReel PushCube-v2 has a straight target push. It does not contain the scientific distinction between necessary intermediary action and feasible-but-irrelevant human habit. It is only an integration platform. To test novelty we need a real task containing both categories of intermediate event and robot capability differences. Do not continue tuning the scripted push until this dataset existence is established.

Stop rule: no physics-aware helpful-vs-habitual contrast in genuinely paired footage -> cannot claim T-RO science, even with perfect supervised mask IoU. The next decision must depend on that evidence, not on a new proxy benchmark.
