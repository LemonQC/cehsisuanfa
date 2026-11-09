# AG-COFL — Actionability-Guided Cross-Behavior Object Flow

**Current execution gate: v1.0 (2026-11-09).**

## One command on your RTX4090 (existing environment, no new installs)

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull origin main
conda activate agcofl4090
set -o pipefail
bash scripts/run_v100_next_gate.sh 2>&1 | tee next_gate_v100.log
```

The script checks four questions in the same run:

1. **Did v0.9 real-mask learning actually work?** On one authentic RoboReel evaluation MP4, 11 tests passed but train IoU was Full=0, Anchor≈0.00240, End≈0.00236, endpoint IoU=0 for all. New test compares against simple real-video persistence and constant-velocity baselines; no further meaningless synthetic training.
2. **Can our 3D source effect / target Push-only interface produce geometrically valid task-relevant paths?** Uses previously measured 31-step PushCube robot PhysX object trajectory as an interface check only. It does not relabel it a human Pick source, infer unmeasured contact or claim physically successful new projection.
3. **Can we access real *training* source videos/annotations without VPN?** Metadata-first official RoboReel train split discovery through optional hf-mirror.com, with <=6 actual train front MP4s, <=300 MB. Official EVAL demo IDs 0080–0099 are explicitly excluded from train-data discovery.
4. **Can the results reach GitHub automatically?** Only small JSON/CSV/MD research reports are staged and pushed to [research_reports/2026-11-09_v100](research_reports/2026-11-09_v100/README.md) if Git credentials/network permit. MP4, ZIP, checkpoints and personal files are NOT uploaded.

See [full v1.0 plan and provenance](docs/2026-11-09_v100_effect_gate_and_train_data.md).

## Files to review after the run

```
outputs/next_gate_v100/
  learnability_gate.json
  learning_vs_motion_baselines.csv
  effect_gate.json
  train_data/train_discovery.json
  train_video_audit.json
```

Status lines: `REAL_LEARNABILITY_GATE`, `EFFECT_GEOMETRY_GATE`, `TRAIN_DATA_GATE`, `TRAIN_MASK_GATE`, `AUTOPUBLISH_OK` or `AUTOPUBLISH_PUSH_BLOCKED`, `V100_GATE_COMPLETE`.

If automatic GitHub push succeeds, you do not need to upload a ZIP again. Send only the short final status or the link to the results folder.

## What is proven so far

- Existing official RoboReel PushCube-v2 episode 0 physics **succeeded 1/1 in 31 steps**, goal error 0.0454 m within 0.05 m radius. It used a privileged-state scripted controller, not video-conditioned learning.
- Genuine source human evaluation video `third_party/roboreel/human_demos/eval/no_distraction/push_cube/0080/front.mp4` was downloaded. A saved physical XY animation exists at `outputs/roboreel_cn/fixed/episode_0000/physical_xy_replay.mp4` (not RGB robot footage).
- Full / Anchor / End CNN code trains, but real video training IoUs are near zero and there is **NO** independently held-out human source sample. It is not a positive research result.

## Crucial T-RO research requirements

The actual proposed concept is **Pick-source object effect -> capability-aware Push-only Reachable Effect Projection -> robot-executable object flow**, not simply mask video reconstruction. Source effect needs task goal, essential intermediate waypoints, observed contact/support modes and calibrated 3D object poses. Geometry-only candidate planning is the SAME as Goal+Waypoint Oracle today, so it has **not established a new algorithmic gap**.

Before further policy training, obtain genuine non-evaluation source demonstrations/3D annotations, prove at least one real Pick-vs-Push capability incompatibility and task-essential intermediate constraint, then compare physics-backed projection against Goal-only, filtering, random candidates and Goal+Waypoint Oracle on paired simulated scenes and fixed action budget.

**Do not claim T-RO novelty from code running, one robot success or one video train IoU.**

## Archive

- [Actual uploaded v0.9 evidence / v1.0 plan](docs/2026-11-09_v100_effect_gate_and_train_data.md)
- [v0.9 video and measured replay](docs/2026-11-09_v090_real_video_training_and_replay.md)
- [Expert baseline and 3D effect plan](docs/PARTIAL_FLOW_EXPERIMENT_V070.md)
