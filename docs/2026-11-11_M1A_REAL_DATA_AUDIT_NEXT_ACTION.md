# 2026-11-11 — M0 CLOSED; M1-A concrete data+baseline gate (NO Ours training yet)

## Verified M0 completion (FROM REAL COMMITTED FILES)
GitHub main commit d5de864 contains:
- `research_reports/2026-11-11_m0_pi05_uniform/verified_summary.json`
- `research_reports/2026-11-11_m0_pi05_uniform/per_trial_60.csv`
- `research_reports/2026-11-11_m0_pi05_uniform/SUMMARY.md`

Original π0.5 Uniform RoboReel PushCube ND/ND = **46/60**; per seed
0:14/20, 100000:13/20, 200000:19/20. Original author pinned SHA
79a0fb0099e73c5560114753a8706276f93aefc2. Vid2Robot ND/ND 3/60
is separately frozen. No more checkpoint downloads or M0 reruns.

## Evidence and major gap found in existing archived reports

| Resource | Previously measured evidence | Reuse rule |
|---|---|---|
| π0.5 Uniform official | Real evaluated 46/60, 60 raw rows in GitHub | M0 general competence ONLY |
| Vid2Robot official | Real evaluated 3/60, archived v1.3 | M0 competence ONLY |
| RoboReel PushCube robot TRAIN | 2 **real** H5 demos with action shapes (138,8) and (136,8), matching robot camera MP4s and finite joints, see research_reports/2026-11-09_v120/real_robot_h5_audit.json | Can start policy/data loader smoke; **insufficient** by itself for nontrivial M1 blocker insertion |
| RoboReel TRAIN human videos | 260 PushCube front MP4 *listed*, 6 *downloaded*, see research_reports/2026-11-09_v100/train_discovery.json | Some real source clips available but goal is original PushCube |
| Official PickCube human | Published category was listed in human pose source audit, no verified paired same-final-goal video/action evidence | Need actual successful source video & physical final-state match |
| Past custom scripted Pick→Push | seed120 source failed 281-step real PhysX (max lift 0.088mm, no overflight), target comparisons NOT_RUN | Not a positive paired source, do not repeat blind teacher debugging |
| Past geometry effect gate | Pure geometric candidate identical to waypoint oracle | Not a novel method, not a physics result |

**Critical M1 fact**: RoboReel PickCube vs PushCube official tasks have
different goal predicates; simply presenting their videos and robot
actions together does not establish *same-task* cross-behavior transfer.

## M1-A — one-time machine inventory (new runnable code)

A single stdlib-only read-only scanner
`scripts/m1a_inventory_real_assets.py` checks true locally available
TRAIN human front videos (PickCube and PushCube), actual downloaded
TRAIN robot H5 + base/hand camera triplets, archived v120 H5
verification, and M0 provenance. It never counts EVAL 0080–0099
as TRAIN, and never mistakes a listed file for a successfully downloaded
one. It does not copy weights, download video, train or run physics.

Run on user's original project directory (the scanner has no pytest dep):

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05
    python -m unittest discover -s tests -p 'test_m1a_inventory_real_assets.py' -q
    python -m scripts.m1a_inventory_real_assets
    cat outputs/m1a_evidence_gate/local_inventory.json

Only small stdout JSON status needs to be returned to continue; the
complete local report is in outputs/ (not blindly committed). Unknown
local sources remain explicitly unverified. No need to run this many times.

## M1-B — FROZEN FIRST ACTUAL SCIENTIFIC TASK (after inventory)

**Task objective**: final pose of a designated cube lies in a common
physical table goal region G, with actual robot success predicate based
on cube final state (NOT high-level route). Both source Pick+Place
and target Push-only must end with the exact same goal object at G.

**Target-only causal insertion condition**: A moveable blocker occludes
the ONLY push-capable opening. Source is allowed to pick/lift over
the obstruction. Target is restricted to non-grasp pushes, must clear
the blocker before pushing the target cube into G. Record actual physics
contact, blocker clearance, goal region entry, invalid grasps,
timeout, collision and success. Counterfactual: same goal/video/scene,
target capability changes to grasp-enabled; and same goal, blocker
moved out. Need target-dependent effect sequence reversal, not just
different XY path. Start with one successful independent source
and target feasibility pair, then freeze scene seeds before any
learning. Existing source Pick code was NOT validated.

**Mandatory fair baseline order on these SAME M1 scenes**:
1. Capability-aware TAMP with same perception/goal/mode, and separate
   perfect-state privileged oracle (strongest threat to novelty).
2. Goal-only robot BC/DP with same target TRAIN actions and target RGB.
3. Raw source object flow + identical executor (no invented 3D oracle).
4. Geometry-only / waypoint planner with same scene observations.
5. Strong video-conditioned policy + flow methods where a real
   trained checkpoint or faithful documented port exists
   (π0.5 M0 checkpoint does NOT automatically support new M1 task).

Only after their **actual physical** outcomes show an
unmet and source-specific gap may M2 propose/optimize a learned
causal-effect editor. If same-information capability-aware TAMP fully
solves insertion/reject, or source video adds nothing to a fully
observed explicit-goal setup, stop/reframe rather than inventing a
novelty claim.

## Outcomes and next action

M0: **PASS AND ARCHIVED**; no further M0 activity.
M1-A: report/script **COMMITTED NOT YET EXECUTED ON USER SERVER**.
M1-B: matched-goal validated source, target feasibility and baseline
physical outcomes **NOT YET AVAILABLE**.
M2: **NOT STARTED**. Reproducibility and novel method claims must not
be inferred from existing M0 rollout performance.


## 2026-11-11 服务器 M1-A 首次结果与精确纠错（v1.6.4）

用户原服务器运行 `unittest` **3 tests passed**，并执行了 M1-A 扫描：
`m0_verified=true`、历史 H5 验证 `2`、本地机器人 TRAIN `0`、
本地 Pick TRAIN 视频 `0`、本地 Push TRAIN 视频 `6`，
`same_goal_pair_verified=false`、`target_insert_verified=false`。

检查代码后确认 **本地 TRAIN=0 不能成立为数据缺失证据**：
`scripts/run_v120_real_robot_and_official_baseline.sh` 实际将
`official_robot_data --data-root outputs/v120_baselines/robot_train`，
并由 `real_robot_h5_audit` 在该位置真实检验了 0000/0001 H5 和双摄像头。
最初的 M1-A 扫描器却只检查
`outputs/v120_baselines/official_dataset` 等目录，漏掉真实路径。
已修复 `scripts/m1a_inventory_real_assets.py` 加入 `robot_train`，
新增一个模拟实际 v120 保存结构的回归用例，同时不把历史数据
必然等同于当前磁盘文件是否仍在。**修复后的扫描器尚未在用户机器执行**，
若文件仍在应找到 2 条。

本次不重新下载机器人数据，不重新跑 M0，不训练 Ours，
不再盲调从零编写的 Pick 教师控制器。新命令仅一次：
`python -m scripts.m1a_inventory_real_assets`。
下一步 **M1-B** 是真正的物理 paired-task：
先选/生成有真实成功来源的同终态 Pick 演示、再验证 target-only
blocker insertion 的 Push 可执行性，和 capability-aware TAMP 同场比较。
若真实物理配对不可建立，则标阻塞并更换可靠专家演示来源，不把
`m0=46/60` 或孤立的几何 waypoint 当成 M1 结果。
