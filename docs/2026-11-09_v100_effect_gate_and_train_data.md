# 2026-11-09 v1.0: evidence audit + next science gate

## Actual uploaded v0.9 result (GitHub user commit 00662d3)

The user uploaded real_video_v090.log and review_bundle.zip to the root
of LemonQC/cehsisuanfa. The results below are from the **uploaded log**;
the zipped RGB images have not been independently decoded.

- All 11 tests passed.
- Genuine human RoboReel evaluation MP4 0080: 214 source frames, 64 sampled,
  14 overlapping two-past/eight-future mask windows from ONE video.
- Same-source train only, no validation/test. Three CUDA models trained
  20 epochs: Full, Anchor, End.
- Training full-mask IoU: Full 0.0000, Anchor 0.0023967, End 0.0023569.
- Training endpoint IoU: 0 for ALL three.
- Predicted mask coverage: Full 5.13%, Anchor 50.44%, End 38.68% of
  32x32 image pixels, indicating possible degenerate mask predictions.
- Human source MP4 is from official RoboReel **EVALUATION** IDs 0080-0099.
  It must not be treated as train data in a reported official RoboReel score.
- 31-step saved official PushCube Panda real PhysX episode succeeded with
  final XY error 4.54 cm. The privileged-state controller did NOT use
  the human video for its actions.

## Why the scientific goal changes

The expert-agreed core research concept is a source Pick object effect
E_s=(Goal, Waypoints, Mode/Support, SE3 Object Trajectory) transformed to
a target Push-only robot that cannot lift, must maintain table support and
must respect workspace/obstacles/actual contact physics.

Directly Reachable / Projectable / Unresolved must be based on real robot
physics, not simply geometric 2D path existence. Human source Pick over an
obstacle cannot just be flattened to an XY Push path. A Goal+Waypoint Oracle
using the same route and budget is a strong required baseline.

## New code in v1.0

- agcofl/analyze_v090.py: compare the real MP4 three trained modes to
  persistence / constant-velocity no-learning motion baselines.
- agcofl/effect_projection_gate.py: validate calibrated 3D source/object
  pose, table support and obstacle constraints; produce conservative
  geometry-only path candidates with Goal-only/Waypoint-Oracle comparisons.
- agcofl/train_data_discovery.py: discover real upstream TRAIN files via
  configurable China HF mirror, exclude official eval/test, optionally
  download at most six known-size real training MP4 files below 300 MB.
- agcofl/train_video_audit.py: genuine training MP4 RGB mask overlays,
  occupancy and temporal consistency; never equate HSV coverage to
  correct instance identity.
- scripts/run_v100_next_gate.sh: single 4090 command; no new physics runs,
  no CUDA reinstalls, no synthetic fake positives.
- scripts/publish_v100_report.sh: publish ONLY small derived JSON/CSV/MD to
  research_reports/2026-11-09_v100 in the user's existing GitHub remote.
  Does not upload source human videos, ZIP or model binaries.
- tests/test_effect_projection_gate.py / tests/test_analyze_v090.py.

## Single command

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull origin main
    conda activate agcofl4090
    set -o pipefail
    bash scripts/run_v100_next_gate.sh 2>&1 | tee next_gate_v100.log

Expected outputs:
    outputs/next_gate_v100/learnability_gate.json
    outputs/next_gate_v100/learning_vs_motion_baselines.csv
    outputs/next_gate_v100/effect_gate.json
    outputs/next_gate_v100/train_data/train_discovery.json
    outputs/next_gate_v100/train_video_audit.json

The automatic GitHub push may fail without access or credentials; local
files are retained and a clear status is printed. No repeated manual ZIP
upload is necessary when AUTOPUBLISH_OK appears.

## Precise next data constraint

Source Pick effect JSON must contain ACTUAL calibrated
source_object_xyz[N,3], source_mode (pick/push/unknown), actual source
provenance, table_center_z, goal_xy, bounds_xy, obstacles_xyxy and optional
required_via_xy. Do not fabricate XYZ from raw front-camera 2D HSV masks.

The current single saved PushCube trace is REAL simulated object XYZ but
does NOT record true contact mode and is NOT a human Pick source. It is used
solely to sanity-check the input interface.

The current geometric projection is identical to a Goal+Waypoint Oracle
under the same geometry. The code reports this explicitly. It cannot be
advertised as a novel T-RO contribution until physical actionability,
mode adaptation or learned projection outperforms meaningful baselines.

## Stop / go

GO only if non-test source human/robot data are available, target object
masks/3D poses can be verified, and a genuine source Pick vs target Push
capability conflict with task-essential intermediate effect is observed.
Then implement physical source-effect projection and directly compare
Goal-only, filtering, waypoint oracle and learned projection under paired
ManiSkill states.

STOP increasing synthetic data/epochs or tuning contact gaps if the video
model loses to persistence, or if the new projection just duplicates a
traditional waypoint planner. 14 overlapping windows and one 31-step
success cannot support T-RO statistical claims.
