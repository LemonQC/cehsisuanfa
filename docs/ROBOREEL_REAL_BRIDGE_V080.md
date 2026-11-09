# RoboReel real-video / real-physics integration — v0.8.0

## Decision and scientific boundary

v0.7.0 trained Full/Anchor/End high-level mask prediction on synthetic
geometry only; the 4090 pipeline smoke passed, but it provided no robot data.

v0.8.0 starts G1/G2 by integrating an independent real benchmark:
  Upstream benchmark: https://github.com/gupta-anmol99/roboreel
  Public human videos: https://huggingface.co/datasets/LogosRoboticsLab/roboreel

The RoboReel readme specifically documents 20 evaluation human videos
(IDs 0080–0099) per task, camera front, saved robot initial states,
PushCube-v2, pd_ee_delta_pose 7D and exact 1000-step full evaluations.
The upstream code installs ManiSkill 3.0.0b22 and advertises a Docker image
about 45 GB. The user's current environment has ManiSkill 3.0.1, so
compatibility is an UNVERIFIED assumption. DO NOT upgrade the old environment
or reinstall NVIDIA drivers to guess around a problem.

This adapter loads a REAL human front MP4 from RoboReel, computes an HSV
BLUE-COMPONENT CANDIDATE mask and saves an 8-frame 32x32 sequence, loads
the official scene state via benchmark.reset_to_episode, and runs a real
physical Panda controller with env.step on CPU PhysX, WITHOUT GPU rendering.

It uses PRIVILEGED robot/object/goal state for its diagnostic pushing
controller. The fixed baseline does NOT use video for its actions.
The video_motion variant uses one measured video statistic to scale the
action rate. Neither version is a learned video-to-robot model, nor a
Full/Anchor/End comparison. It is G1/G2 interface / physical-compatibility
diagnostics, not T-RO evidence. If either condition fails, preserve logs.

## Exact one-command start on RTX 4090

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull origin main
    conda activate agcofl4090
    python -c 'import cv2, huggingface_hub' || python -m pip install 'opencv-python-headless==4.10.0.84' huggingface_hub
    set -o pipefail
    bash scripts/run_roboreel_bridge.sh 2>&1 | tee roboreel_v080.log

The script performs these steps:
1. tests/test_real_video_bridge.py — local MP4, mask, 7D action tests.
2. git clone only official RoboReel into third_party/roboreel.
3. Official selective Hugging Face downloader:
     --distraction no --tasks push_cube --cameras front
   Thus it does NOT download the complete 11 GB, multi-task multi-view set.
4. Read human demo 0080/front.mp4; save raw-video mask candidate metadata,
   8-mask array, mask contact sheet and RGB overlay for visual QA.
5. Execute the same saved RoboReel episode 0 with fixed and video_motion
   diagnostic push control, saving actual actions, per-step cube poses,
   physical result and success.
6. Compare two paired episodes' recorded action differences and task results.

Inspect this first:
    outputs/roboreel_real/inspect_0080/human_mask_overlay.png
    outputs/roboreel_real/inspect_0080/human_video_diagnostic.json

The overlay MUST visibly cover the moving cube. If it instead covers clothing,
distractors or background, the HSV candidate is invalid. Do not interpret it
as valid object-flow segmentation; send the PNG for targeted correction.
The video_motion run refuses to use low-coverage masks.

If the official 3.0.0b22 RoboReel scene code fails to import in your existing
3.0.1 conda environment, stop and send the exact traceback. Do not rebuild
the official 45 GB Docker image yet. We will isolate a compatible small
runtime without overwriting the current AG-COFL environment.

There is no EGL/GLX/Vulkan robot rendering in this adapter:
  obs_mode=state; sim_backend=physx_cpu
  render_backend=none; render_mode=None
The RGB frames only come from the downloaded human MP4, not from a simulated
robot camera. They must not be called matched RGB observations.

## What to send back

- roboreel_v080.log (all lines from first error or SUCCESS)
- outputs/roboreel_real/inspect_0080/human_video_diagnostic.json
- outputs/roboreel_real/inspect_0080/human_mask_overlay.png
- outputs/roboreel_real/fixed/report.json (if produced)
- outputs/roboreel_real/video_motion/report.json (if produced)
- outputs/roboreel_real/paired_diagnostic.csv (if produced)

If the simulated Panda fails, we need the actual failure phase and
goal error from physical_trace.json, not another random parameter sweep.

## What this DOES NOT yet prove

RoboReel's PushCube-v2 has a single goal vector, without our essential
same-goal-two-routes or beneficial vs habitual path labels. The real-video
bridge cannot support T-RO's core hypothesis by itself.

The next actual method experiment must freeze human videos, robot data,
scene splits and network, train Full/Anchor/End only differing in HUMAN
stage-mask supervision, and run the SAME learned policy on the SAME held-out
robot scenes with predicted object masks as its condition.

The current v0.7 high-level predictor takes a low-dimensional context,
not actual camera images; it is NOT the original object-flow paper.
The existing v0.6 FlowBC expects four synthetic top-down masks,
while RoboReel's videos have different camera coordinates. We must
train or calibrate an actual low-level flow-conditioned policy; we cannot
simply feed the 8x32 real-camera masks into that old controller.

This note is an honest gate: real-video loading and physics compatibility
first, no further synthetic data claims, no T-RO publication claims.
