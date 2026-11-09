# AG-COFL | Research execution status (v0.9.0, 2026-11-09)

**Current priority: evaluate human/robot behavior supervision, not manually tune a scripted pushing expert.**

## Run now: true human MP4 + three-mode training + robot video replay

Runs with existing authentic RoboReel PushCube demo 0080 and previously saved 31-step physical rollout. **No more downloading, and no rerunning simulation.**

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull origin main
    conda activate agcofl4090
    set -o pipefail
    bash scripts/run_v090_real_video_and_replay.sh 2>&1 | tee real_video_v090.log

### View videos

Actual human video (214 source frames):

    third_party/roboreel/human_demos/eval/no_distraction/push_cube/0080/front.mp4

Actual Panda PHYSICS measured top-down replay, newly generated:

    outputs/roboreel_cn/fixed/episode_0000/physical_xy_replay.mp4

It plots cube, Panda TCP and goal, using stored env.step measurements.
**It is NOT a real rendered camera video of the robot** (RGB rendering was
disabled for headless compatibility).

Human RGB with segmentation candidate highlighted:

    outputs/real_human_supervision_v090/video_0080/human_mask_overlay.png

Full/Anchor/End side-by-side model prediction image:

    outputs/real_human_supervision_v090/three_mode_predictions.png

Open remote files using VS Code Remote SSH or download with SFTP.
The program additionally packages original human MP4, measured physics MP4,
these figures and CSV/JSON into ONE transferable file:

    outputs/real_human_supervision_v090/review_bundle.zip

Upload that ONE ZIP and real_video_v090.log in your next message for
one-pass diagnosis and research go/no-go review.

### What the code ACTUALLY trains

The authentic human front MP4 is sampled at 64 times. The high-level CNN
gets **two observed** object-mask candidates and predicts **eight future**
candidates. Three identical networks with the same initialization, training
samples, budget and seeds are trained with Full, Anchor and End future-mask
losses. No simulated/synthetic future traces enter training.

With only one human video, all windows belong to the train set and the report
is marked TRAIN_ONLY_NOT_GENERALIZATION. Results are a SAME-VIDEO training
diagnostic, NOT a held-out test.

**Important:** the RoboReel human 0080 video belongs to its official
EVALUATION set, so this run is for engineering development ONLY. These
videos cannot be used as official RoboReel training demonstrations in a
subsequent reported benchmark test. The HSV blue-object candidates must
also be visually verified against the original human video.

Full code: agcofl/real_human_supervision.py
Replay: agcofl/physics_replay.py
Bundle: agcofl/package_v090.py
Tests: tests/test_real_human_supervision.py
One command: scripts/run_v090_real_video_and_replay.sh

## What is already physically proven

On the user's RTX4090, RoboReel PushCube-v2 episode 0 ran real CPU PhysX
Panda actions from the official fixed scene successfully in 31 steps,
final goal XY error 4.54 cm against the official 5 cm radius.
This was a privileged-state oracle diagnostic, **not** a learned policy
following human video.

Previously, the synthetic proxy Full/Anchor/End training flow and masks
loss tests passed. Synthetic results are not paper evidence.

## What must still happen before T-RO feasibility can be claimed

1. Real HUMAN TRAINING videos, independent of the official RoboReel eval
   clips, with verified segmentation and annotated event-stage boundaries.
2. Real robot action trajectories under multiple task layouts and robot
   capabilities; all ROBOT masks must get Full supervision.
3. A genuinely shared high-level model trained on the same source clips
   under Full / Anchor / End human labels and matched robot budget.
4. A **learned** low-level robot policy conditioned on the PREDICTED masks,
   with verified human-to-robot camera coordinate alignment. The prior
   synthetic FlowBC accepts a DIFFERENT 4-channel top-down mask tensor;
   simply swapping human camera pixels into it would be invalid.
5. Same-scene, paired closed-loop success/route/violations and strong
   baselines (robot-only, random-K, uniform-K, full human alignment).
6. A task with genuinely necessary AND feasible-but-unnecessary human
   intermediate events. Plain direct PushCube-v2 has neither contrast.
   The T-RO hypothesis cannot be established from plain PushCube success
   or offline mask IoU, regardless of how high the score is.

**Hard stop:** if we cannot get paired real footage with the required
actionability contrast, stop escalating synthetic datasets or hand-coded
push hyperparameters; reevaluate the research design.

## Experiment diary

- docs/2026-11-09_v090_real_video_training_and_replay.md
- docs/2026-11-09_roboreel_episode0_physics_success.md
- docs/2026-11-09_v070_proxy_result_and_next_gate.md
- docs/PARTIAL_FLOW_EXPERIMENT_V070.md
