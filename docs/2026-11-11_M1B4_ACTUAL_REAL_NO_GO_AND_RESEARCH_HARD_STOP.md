# 2026-11-11 — M1-B4 REAL SOURCE NO-GO: HARD STOP for Pick→Push blocker-insertion formulation

## What was ACTUALLY run on the user's server

The user ran the repaired M1-B4 runner, and the four preflight unit tests completed with `OK`. The runner **then** printed its real-data marker
`M1B4_UNIT_TESTS_PASSED_NOW_READING_REAL_FROZEN_M1B3_DIAGNOSIS`.
Only after that, the existing actual M1-B3 `diagnosis.json` returned:

```json
{
  "old_scene": "AGCOFLM1B3Corridor-v0",
  "old_source_error": "RuntimeError: official_mplib_lower_failed",
  "source_actual_grasp_steps": 0,
  "source_actual_max_task_z_m": 0.019999999552965164,
  "source_actual_task_net_xy_m": 4.30453004975207e-06,
  "source_actual_task_total_xy_m": 4.676188555490802e-06,
  "source_lower_planner_fallback_is_justified": false,
  "old_push_stall_xy_m": [0.12527310848236084,-0.030821576714515686],
  "wall_front_cube_center_x_limit_m": 0.126,
  "free_cube_center_half_band_y_m": 0.005000000000000001,
  "stuck_xy_consistent_with_throat_collision": true,
  "geometry_unchanged": true,
  "prior_trial_rerun": false
}
```

Final marker:
`M1B4_NO_GO_SOURCE_DID_NOT_PHYSICALLY_CARRY_CUBE_IN_FROZEN_RUN`.

**Unlike the older synthetic-unit-test marker, this one is genuinely supported by the archived user-server trajectory.** Source cube never registered a grasp, stayed at approximately its 20mm table height, and moved only 4.7 micrometres of total planar path. Upstream MPLib stage return flags `grasp/close/lift/across=true` documented only planner execution, not the task cube actually being transported. Therefore trying a different `lower` planner cannot fix this source run. The script correctly refused new PhysX. No M1-B4 source or target physical case was launched. Do not claim there are two newly failed rollouts; the hard stop came **before** simulation.

## Frozen, distinct empirical results

| Experiment | Actual recorded outcome | What it supports |
|---|---|---|
| Official RoboReel π0.5 Uniform ND PushCube | 46/60, real 3×20 per trial archived at GitHub `d5de864` | Reproduced original published benchmark competence |
| Official Vid2Robot ND PushCube | 3/60, historic official 3×20 | Existing weaker video imitation benchmark |
| Official ManiSkill PickCube motion planner | 3/3, original H5/JSON and video | Original **different-goal** grasp source works |
| M1-B2 first custom same-goal scene | Pick/Direct/Clear all succeed; Direct Push 31 steps | No evidence that target-only clearance is necessary |
| M1-B3 size-selective corridor | Blocked source FAIL, direct FAIL, clear FAIL, open direct succeeds | Geometry separates conditions, but **no successful paired blocked source+target** |
| M1-B3 saved source trace + M1-B4 real preflight | Grasp steps 0; lift none; total XY ~4.7 μm | Adapted source actually never moved the cube; source lower fallback is unjustified |

## Conclusive decision for THIS formulation and hardware protocol

**NO-GO: stop the current CUSTOM Pick-lift → Push-only mandatory blocker-clearance research formulation.**

This is a **research-readiness STOP**, not a proof that no embodiment-capability adaptation paper can ever exist. We have neither a physically valid matched-goal source/target dataset nor a measured gap against same-information TAMP or goal-only controls. Repeated bespoke scripts and geometry variants would ignore the project's previously committed stop criteria.

- No new M1-B5 corridor, waypoint repair, gripper-gain threshold search, or reexecution of frozen seeds.
- No M2 causal-effect editor training; no T-RO novelty or performance claims.
- No unsupported inference that 3/3 official PickCube source meets same-goal Pick+Place requirements.
- Store original H5/MP4/JSON locally, publish only small authenticated outputs if needed.
- If later reviving the scientific question, start from a **published or independently validated successful matched-goal source/target DATASET**, not another failed from-scratch teacher. Require an information-controlled strong TAMP comparison before M2.

## Decision about unfinished Baselines

Current official benchmark baselines are **NOT all completed**. π0.5 Uniform and Vid2Robot have actual 60-run outcomes, but robot-only BC/DP, source-flow ports, and capability-aware TAMP **remain NOT_RUN**. Running generic official PushCube baselines can establish a future alternative RoboReel paper foundation but **cannot validate the abandoned same-goal effect-insertion hypothesis**. Results from original PushCube and custom M1 must never be pooled.

The *only* scientifically responsible next computational work, if continuing to reuse this project, is a **separate RoboReel baseline foundation** with authentic TRAIN robot trajectories and well-defined goal-only/robot-only policy training and evaluation. No learned Ours, no newly fabricated human Pick video, no fake cross-behavior metrics. A separate decision about a new scientific problem is required before designing another method.

## Operator action

No need to rerun M1B4. This report records the actual result of the already executed read-only gate. Next archival step is repository-only; future baseline assets must use bounded authentic TRAIN data, exclude held-out official EVAL videos and declare any difference in training budget relative to released checkpoint baselines.


## Separate baseline foundation: B01 authentic robot-only BC TRAIN data

After this NO-GO, the official RoboReel baseline matrix remains
incomplete. For future, **separate** research (not as a retroactive M1
success), the next reproducible step is to prepare 64 authentic PushCube
TRAIN robot trajectories (H5 actions plus two real camera videos per
demo), reuse the previous 0000/0001 downloads, and reserve a fixed 51/13
demo-level TRAIN/validation split.

New `scripts/prepare_b01_robot_train.py` is a bounded wrapper
around the already verified upstream dataset downloader
`agcofl.official_robot_data` and real-content checker
`agcofl.real_robot_h5_audit`. It refuses any candidate demo ID
0080–0099, missing video/H5 triplets, files outside the authentic
TRAIN root or more than 300 MB of planned bytes. It stores raw data
in the existing `outputs/v120_baselines/robot_train` (no second data
copy), only small manifest/audit/split JSON in `outputs/b01_robot_only`.
A failed/incomplete download must be treated as BLOCKED, never trained
on synthetic actions or claimed as a trained baseline.

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05
python -m unittest discover -s tests -p 'test_prepare_b01_robot_train.py' -q
# Explicit opt-in to REAL bounded TRAIN-only data download:
python -m scripts.prepare_b01_robot_train --download
```

Expected after all 64 are genuinely present and H5/cameras inspected:
`B01_AUTHENTIC_ROBOT_TRAIN_READY` with
`actual_demos=64`, `train=51`, `validation=13` and observed
8D robot actions. The script itself performs **NO BC training and
NO official rollouts**. If data selection/download is blocked, do not
mislabel the existing two episodes as a 64-demo dataset.

Next, the B01 policy implementation and official RoboReel action
interface must be verified and trained on this exact frozen split
before any B01 success score can be reported. B01 trained on 64
episodes is **not training-budget matched** to a released pretrained
π0.5 and therefore cannot alone establish that human demonstrations
are helpful or useless. Scientific question for any NEW proposed
paper must be selected on existing validated tasks and available
data before inventing another effect editor.
