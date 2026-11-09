# v1.3 — Actual original Vid2Robot released policy, official RoboReel protocol

Scope: **PushCube clean human + clean environment only**; no 10-task or
WD/ED claims. Human EVAL videos 0080–0099, original physical simulator,
authors' published Vid2Robot checkpoint. Original pd_joint_pos action,
chunk_size=1, max_rollout_steps=800 and three environment seed offsets
0/100000/200000. No checkpoint tuning on official EVAL videos.

Only official_aggregated_summary.json with full_3pass_complete=true
and status.json.status=OFFICIAL_VID2ROBOT_ND_3PASS_EVALUATED permits a
three-pass performance statement. Missing or interrupted seeds stay partial.

Reproduction delta from unmodified authors' Python evaluator:
a thin launch adapter suppresses timm's redundant ImageNet initializer
download while authors' complete original Vid2Robot state_dict is strictly
loaded. No changes to architecture, learned parameters, action policy,
original RoboReel success criterion or physics.
