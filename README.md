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
