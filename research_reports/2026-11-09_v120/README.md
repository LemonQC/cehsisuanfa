# v1.2 — First REAL robot train labels + official released policy baseline

- v1.1 DINOv2 ViT pilot ran correctly on RTX4090, but mean held-out
  pixel error (Full 9.73) is still above constant velocity (7.58);
  the test comprises just ONE independent video. Do NOT terminate the T-RO
  study based on this diagnostic and do not call its 2D HSV centers 3D poses.
- Official model repository (fixed):
  https://github.com/weiweigu1998/roboreel_models
  commit 79a0fb0099e73c5560114753a8706276f93aefc2.
- **Corrected upstream raw robot dataset path**:
  robot_trajectories/no_distraction/push_cube/
  (not the authors' internal alias without_distraction).
- Robot-action files are original RoboReel train H5 + base/hand camera MP4.
  Validated actions are NOT learned-policy success.
- Checkpoint is original released Vid2Robot ND, not our approximation.
- Any 0080 physics smoke uses the authors' original simulator and original
  Vid2Robot inference code, limited to exactly one preselected DEV sample.
  This is NOT a paper score. The original official last20 complete 20-episode
  protocol remains to be evaluated later, with no development on those tests.
- The official original PickCube and PushCube benchmark tasks have DIFFERENT
  objectives: do not call them a matched same-goal Pick-to-Push comparison.
- On failed downloads or dependencies, preserve BLOCKED statuses; do not
  substitute random/synthetic policies or fabricated success numbers.
- Original videos, H5 data and model checkpoints remain LOCAL ONLY.
