# 2026-11-09 — Real RoboReel PushCube-v2 episode 0: physics execution success

## User-supplied JSON (v0.8.1 bridge)

```json
{
  "type": "REAL_VIDEO_REAL_SIM_DIAGNOSTIC_NOT_PAPER",
  "task": "PushCube-v2",
  "mode": "fixed",
  "episodes": 1,
  "successes": 1,
  "success_rate": 1.0,
  "episode_id": 0,
  "human_demo_id": 80,
  "success": true,
  "steps": 31,
  "policy": "diagnostic_privileged_state_push",
  "final_cube": [0.1561201810836792, -0.04533807188272476, 0.020000120624899864],
  "final_goal": [0.20152631402015686, -0.044137824326753616, 0.0010000000474974513],
  "final_goal_error_m": 0.04542199522256851,
  "failure": null,
  "video_frame_count": 214,
  "frames_decoded": 24,
  "valid_blue_candidate_frames": 24,
  "coverage": 1.0,
  "normalized_motion_path": 0.0827115821881191,
  "timing_gain": 0.8327115821881191,
  "video_sha256": "bae643b23f5451875a6dbc6f623d94e41fdec1e4f39a31d71341837685852080",
  "control_mode": "pd_ee_delta_pose",
  "obs_mode": "state",
  "render_backend": "none"
}
```

The success matches the upstream RoboReel PushCube-v2 5 cm XY goal radius (error ~4.54 cm). It is **one episode only**, close to the threshold; no statistical success-rate claim can be made from 1/1.

Blue HSV detection coverage 1.0 only indicates candidates are detected in every sampled frame; **it does not verify object identity** without reviewing the `human_mask_overlay.png`.

## Proven

- The user obtained genuine RoboReel evaluation video 0080 front in mainland China.
- The local ManiSkill environment can load official saved initial state and run a headless physical Panda controller successfully for this one episode.
- 7D `pd_ee_delta_pose` controller interface, CPU PhysX and output artifacts run.

## Not proven

- `fixed` mode actions do not depend on the human video. Controller is explicitly `diagnostic_privileged_state_push`, reading oracle object/goal state.
- No learned video-conditioned policy, and no Full/Anchor/End comparison has been performed.
- No route-dependent selection, useful/habitual trajectory intervention, cross-ability adaptation, multi-layout holdout or T-RO-level novelty.
- One near-threshold success cannot support a reliability claim.

## Decision

Freeze the physics interface as a working diagnostic control. Do not return to hand-tuning distance offsets as the research focus.

Optional one-time paired sanity test (still *diagnostic*, not paper evidence):

```bash
python -u -X faulthandler -m agcofl.roboreel_bridge evaluate \
  --roboreel-root third_party/roboreel --episodes 0 \
  --mode video_motion --max-steps 250 \
  --out outputs/roboreel_cn/video_motion \
  2>&1 | tee roboreel_cn_video_motion.log
python -m agcofl.analyze_roboreel --out outputs/roboreel_cn
cat outputs/roboreel_cn/paired_diagnostic.csv
```

The next required research milestone is a learned predictor trained from **real** human object masks on matched RoboReel scenes and three controlled Full/Anchor/End supervision variants, with a trained or verified flow-conditioned low-level policy run under the same initial states. Do not claim these experiments already exist.
