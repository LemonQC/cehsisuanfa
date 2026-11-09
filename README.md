# v1.3 — 正式官方 Vid2Robot 20场景 × 3 seed 物理评测（代码已提交）

**v1.2.1 已验证成功的环节：** 2.37GB 原版 Vid2Robot 预训练权重严格加载；20/20 官方 EVAL front 视频完整；作者的物理评测器运行成功。单独的 0080 集成测试结果为失败（0/1），不能代表正式成功率，更不能作为放弃 T-RO 研究的依据。

**现在进入正式官方 Baseline**：PushCube clean-human / clean-environment 条件；评测原始 0080–0099 共20场景，每场景3个官方环境 seed offset 0/100000/200000；保持官方控制方式 chunk=1、单轮800步，不做 EVAL 数据上的模型调参。

运行：

\`\`\`bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --rebase --autostash origin main
conda activate agcofl4090
set -o pipefail
bash scripts/run_v130_official_vid2robot_3pass.sh 2>&1 | tee baseline_v130.log
\`\`\`

**只核对三个标志：** OFFICIAL_V130_PASS_VERIFIED（本seed完整20条物理记录）、OFFICIAL_V130_EVALUATION_GATE（三seed结果）、V130_AUTOPUBLISH_OK（小型数据报告已提交）。若某seed中断，下一次运行会验证并复用之前已完成的整个seed。不重下2.37GB权重、20段视频和真实机器人训练H5，也不重训ViT。

实验JSON自动上传至 research_reports/2026-11-09_v130/；此处仅覆盖官方 PushCube ND 单格，不代表所有任务/干扰条件的完整 RoboReel benchmark。

研究决策与算法创新实验计划：[v1.2.1结果与v1.3方案](docs/2026-11-09_V121_SUCCESS_V130_BASELINE_PLAN.md)。

---

# AG-COFL v1.2.1 — Official Vid2Robot 权重续传与真实物理 Baseline

**v1.2 已实测成功**：2段官方 RoboReel TRAIN 机器人 H5 动作 + base/hand RGB 视频全部通过检查（动作形状 138×8、136×8）。**还未执行官方学习型机器人物理闭环**：作者公开模型权重和 EVAL 视频的 Hugging Face 下载不完整，官方 rollout 状态 false。

**下一步只需运行这一条（避免重复ViT和H5数据）：**

\`\`\`bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --rebase --autostash origin main
conda activate agcofl4090
set -o pipefail
bash scripts/run_v121_official_baseline_resume.sh 2>&1 | tee baseline_v121.log
\`\`\`

v1.2.1：重用已有机器人 TRAIN H5、核对/复用原来可能已下载完整的2.37GB官方 Vid2Robot checkpoint；文件缺失时通过显式 Hugging Face 镜像、官方地址与HTTP Range断点续传；下载20段约97MB固定评测视频；在**作者官方模型与官方ManiSkill**执行单段预先固定的集成测试；报告自动提交 \`research_reports/2026-11-09_v121/\`。**不上传权重、视频或H5**，不以一次调试成功冒充论文正式成绩，也不使用测试视频调参。

**只以下述状态判成功：** \`OFFICIAL_ASSET_READY\`（具体文件）；\`VID2ROBOT_OFFICIAL_RELEASE_GATE.status=OFFICIAL_VID2ROBOT_RELEASE_AND_EVAL_DATA_READY\`（资源就绪）；\`VID2ROBOT_ROLLOUT_GATE.rollout_executed=true\`（真正执行物理运行）。最终 \`V121_AUTOPUBLISH_OK\` 仅说明归档完成。

详见 [v1.2实验诊断与v1.2.1研究计划](docs/2026-11-09_V120_FINDINGS_V121_RECOVERY.md)。

---

# AG-COFL v1.2 — 真实机器人动作和官方 Vid2Robot 闭环 Baseline（已提交，待4090执行）

**v1.1 已实际成功：** DINOv2 ViT-S/14 + Transformer 在 RTX 4090 上训练完毕，并归档7份报告。独立测试视频（仅1段）的平均位置误差：恒速 7.58px、Full ViT 9.73px、Anchor 14.50px、End 11.85px。该实验不是机器人策略，也不是跨行为创新验证。

**现在优先跑官方 Baseline，不再重复旧ViT训练：**

\`\`\`bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --rebase --autostash origin main
conda activate agcofl4090
set -o pipefail
bash scripts/run_v120_real_robot_and_official_baseline.sh 2>&1 | tee baseline_v120.log
\`\`\`

**v1.2 工作内容：** 修正官方发布目录 \`robot_trajectories/no_distraction/push_cube/\`（此前使用内部别名 \`without_distraction\` 导致404）；限量获取并审计真实机器人 H5+base/hand 视频；下载作者发布的 Vid2Robot 权重（上限8GiB）和固定20段 PushCube EVAL 视频（≤250MB）；尝试一段 0080 **DEV ONLY** 的真正学得策略→物理模拟器闭环，明确记载阻碍、真实成功及其证据；实验 JSON/MD 自动提交 GitHub \`research_reports/2026-11-09_v120/\`。源视频、权重和H5不上传。

详细数值与决策：[v1.1 结果及 v1.2 任务](docs/2026-11-09_V110_RESULTS_V120_DECISIONS.md)。

**准确理解状态：** \`V120_COMPLETE_IS_NOT_A_SUCCESS_FLAG\` 只表示流程结束；只有 \`OFFICIAL_VID2ROBOT_DEV_SMOKE_EXECUTED\` + \`rollout_executed:true\` 才说明真正执行过学习型机器人闭环，\`official_success\` 才表示该次物理成功与否。一次 DEV 结果不能替代完整官方 Baseline 比较。

---

# AG-COFL | v1.1 — 官方 Baseline 优先 + 预训练 ViT GPU 验证

**研究方向继续推进，不因 v0.9 的简易 CNN 失效而放弃。** 创新问题保持：源人类物体运动效应如何在目标机器人不同操作能力下转化为物理可执行行为。CNN 只是旧的工程烟雾测试；从 v1.1 开始使用预训练 DINOv2 ViT + 时序 Transformer，固定官方 Baseline、数据协议和闭环实验。

## RTX 4090 一键执行 v1.1

\`\`\`bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --rebase --autostash origin main
conda activate agcofl4090
set -o pipefail
bash scripts/run_v110_official_baselines_and_vit.sh 2>&1 | tee baseline_v110.log
\`\`\`

这次脚本会：

1. 检查已有真实训练视频6段，以及 **RoboReel 官方 Baseline 仓库** [weiweigu1998/roboreel_models](https://github.com/weiweigu1998/roboreel_models)，固定源码提交 79a0fb0099e73c5560114753a8706276f93aefc2（无需下载全部大模型）。
2. 从官方数据集抽取最多2段 **真实机器人训练轨迹**，每段必须有 H5 动作及 base/hand MP4，硬上限300MB；缺任何文件都标记 BLOCKED。
3. **在 CUDA/4090 上**加载 \`facebook/dinov2-small\` 的真实预训练权重（不可用则报 BLOCKED，禁止随机初始化替代），以224×224真实RGB提取视觉特征，训练相同结构的 Full/Anchor/End 时序 Transformer 预测头，并对完整视频级 held-out 数据进行评估。仅供数据/表征 sanity check，不是完整机器人策略。
4. 将少量结果 JSON/CSV 自动归档到 \`research_reports/2026-11-09_v110/\`。不推送视频、H5或权重。

**输出路径** \`outputs/v110_baselines/\`：\`official_baseline_audit.json\`, \`robot_data_manifest.json\`, \`vit_pilot/report.json\`, \`vit_pilot/summary.csv\`, \`vit_pilot/split.json\`。跑完若看到 \`VIT_GPU_TRAINING_COMPLETED\` 和 \`AUTOPUBLISH_OK\`，只回复“v1.1 完成”即可，我会直接读 GitHub 结果。

**注意**：只有6段训练视频，1段测试视频的 ViT 结果只能验证学习和技术流程。不能用它作为论文统计证据，也不能以某个回归数值判定整项 T-RO 研究无效。现在还需要原始机器人动作示范、源/目标同目标跨行为任务、真实接触/支撑的物理评测。

完整 Baseline 矩阵（14项分层、官方vs改写标注、任务、预算、停止条件）：
[docs/BASELINE_MATRIX_TRO_V110.md](docs/BASELINE_MATRIX_TRO_V110.md)。

---

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
