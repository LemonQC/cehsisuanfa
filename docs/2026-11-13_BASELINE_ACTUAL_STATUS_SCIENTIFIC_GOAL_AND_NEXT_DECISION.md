# TRO1：2026-11-13 Baseline 真实执行清单、科学问题与硬门槛（已核对 GitHub）

> 判定依据：当前 main 分支、research_reports 及独立 tro1-experiment-evidence 分支。**有代码 ≠ 已训练；有 2×2 DEV ≠ 官方正式评估；轨迹分叉 ≠ 视频信息有效；单任务成功 ≠ 跨行为算法贡献。** 本文优先于旧 BASELINE_MATRIX_TRO_V110.md 对当前状态的描述。

## 1. 新上传的真实结果（在用户要求“看 GitHub”时再次核查）

- 原始视频条件 π0.5 在 **ObjectInDrawer-v1 (toys_in_drawer)**，真实已执行四条 PhysX，任务成功 2/4，配对的两个条件结局无反转（0/2），全程动作 L2 为 1.340732694 / 0.270444125；前 20 步 L2 仅 0.001442696 / 0.001331879。
- 第三份 **最新时限审计**：**400 步前 0/4；800 步前 2/4**；四条首次成功步数分别 NULL、NULL、634、650。服务器完成该审计并自动发布 `drawer_horizon_v1_e57085942a82872c`，只读，不新增 PhysX。
- **非常重要**：400 是 ObjectInDrawer 原始 Gym 注册默认 TimeLimit；已复现原作者 PushCube 正式评价配置的 max_rollout_steps=800。因此「400 内 0/4」只是同组 DEV 轨迹的前缀观察，**不是一项已经执行的官方 heldout 400-step 基准**，也不能把跨任务的 2/4 当作官方 3×20 的比较值。
- 最新证据：
  - [真实四条轨迹](https://github.com/LemonQC/cehsisuanfa/blob/tro1-experiment-evidence/evidence/runs/drawer_hf_v1_540ff3963863f5ed/evidence.json)
  - [同场景动作差异诊断](https://github.com/LemonQC/cehsisuanfa/blob/tro1-experiment-evidence/evidence/runs/drawer_hf_v1_40a7bc0821b412be/evidence.json)
  - [400/800 时限审计](https://github.com/LemonQC/cehsisuanfa/blob/tro1-experiment-evidence/evidence/runs/drawer_horizon_v1_e57085942a82872c/evidence.json)

### 真实 Baseline 状态（B01–B14；不要把同名脚本的局部执行与原论文正式结果混为一谈）

| ID | Baseline | 截至本次核查结果 | 能否计入同一新方法任务的对比表 |
|---|---|---|---|
| B01 | Robot-only Transformer BC | **NOT_RUN**：当前 main 中 `scripts/prepare_b01_robot_train.py` 和同名测试均为 404；已有少量真实机器人 TRAIN H5，但目标数据准备/训练/正式评测没有完成 | 否 |
| B02 | Robot-only Diffusion Policy | **NOT_RUN**：当前 RoboReel 统一任务上没有训练/评测回执 | 否 |
| B03 | 官方 π0.5 Uniform (RoboReel) | **实际完成**：ND PushCube 46/60，官方 20×3 评价；ObjectInDrawer 2/4 @800 DEV；StackCups 0/4 @800 DEV | 仅 PushCube ND 是正式已复现 cell；后两项 DEV 不能混列 |
| B04 | π0.5 Keyframe | **NOT_RUN / DEFERRED**：原作者关键帧链依赖尚未具备的 VLM 接口/标注 | 否 |
| B05 | 官方 Vid2Robot | **实际完成**：ND PushCube 3/60，20×3，正式原作者评价；其余任务无对应回执 | 仅 PushCube ND |
| B06 | UniSkill policy | **CODE_AVAILABLE, NOT_EVALUATED**：没有在当前协议下完成训练或 60 次真实评测 | 否 |
| B07 | VLM + π0.5 | **NOT_RUN / DEFERRED**：不引入付费/外部 API | 否 |
| B08 | SeeDo | **NOT_RUN**：尚未验证可获得能运行的官方实现/权重 | 否 |
| B09 | PointPolicy | **NOT_RUN**：未确认完整官方可运行实现/权重和校准的 3D 评价 | 否 |
| B10 | Cross-Behavior Object Flow (T-RO 2026) | **NOT_RUN**：未验证合法可复现的官方实现/权重；需忠实改写时标差异 | 否 |
| B11 | Im2Flow2Act | **CODE_AVAILABLE, NOT_PORTED/NOT_RUN**：官方源码与旧任务不等于当前 RoboReel 对照 | 否 |
| B12 | EC-Flow | **CODE_AVAILABLE, NOT_PORTED/NOT_RUN**：Meta-World 原任务与当前 RoboReel 不同 | 否 |
| B13 | DINOv2/时序解码器 | **PARTIAL DIAGNOSTIC**：既有表征/视频 Pilot，并非完成的机器人策略性能对照 | 否 |
| B14 | Goal-only、Goal+Waypoint Oracle、TAMP、Persistence | **PARTIAL PHYSICAL/GEOMETRIC PROBES**（M1 中部分真实 Push/清障路径），但没有在同一个有意义的科学反事实问题上完成一套信息匹配对照与正式结果 | 否 |

**结论：14 个登记项中，仅 B03 和 B05 在同一个官方 PushCube ND cell 有完整的 60 次正式复现结果；并非 14 个算法全部完成。** 两者也不是在 M1 能力差异/信息必要性任务上对比过。B01 旧状态“代码已提交”过期且错误，已通过当前 main 路径检验纠正。

## 2. 为什么做这项研究：科学问题与不能偷换的假设

原始 T-RO 目标不是“再训练一个更快的模仿学习模型”或“让抽屉更容易拉开”。

**核心问题（M1 原定）**：源真人视频描述了期望的物体效果与操作阶段，目标机器人可能没有源演示中的抓持/移动/支撑能力，或目标环境在接触、障碍和可行动作方面不同。我们想学习**目标相关的效果与真实先决条件**，并在目标能力限制下保持终态、删除非必要源动作、替换不可能的动作、插入必须的新动作，同时让机器人真正完成物理任务。

**为什么需要视频？这是额外必须证明的前提**：如果最终目标和目标端机器人 RGB/几何/动作能力已足以唯一决定任务，那么没有源视频的普通 Goal-only policy/TAMP 也能完成；不能宣称视频提高决策有科学上的必要性。另一候选“源视频指明不同目标/隐含约束”的 G0 仍为 **G0_BLOCKED**（没有经过真人语义和授权核验的相反目标视频对），已批准的 PartNet/ShapeNet 模型资产不自动解决视频数据要求。

**不可替代、可证伪的三个预测**：
1. 同一终态/机器人初始场景，仅改变目标执行能力或物理阻碍，正确的目标效果顺序必须实际发生可观察的反转（如必须清障 vs 不必清障），不是只改变仿真 A* 路线。
2. 与相同目标/感知/能力信息的 Goal-only、几何规划、强任务-运动规划、直接视频条件 policy 和 flow-based policy 比较，**学习的效果模型**在真实 PhysX 执行及未见场景中解决了其失败模式（不能只换输入预算或加入 privileged GT）。
3. 人类源视频传递的附加任务相关信息（例如隐含目标、接触/顺序要求）确实会改变正确选择；交换或删除视频带来可预期的错误，而正确视频恢复成功。若任务并不依赖视频，则不应把这种实验做成主创新点。

**M1 旧版本已被实际反证**：
- B2 测试中源 Pick 与目标 Direct Push 都可以完成目标，清障不是必要的；所谓必需的目标-only Insert 不成立。
- B3 虽有目标端动作成功，但源端未产生实际同目标成功演示；不能建立合法的源-目标同目标配对。
- 所以 M1“先插入清障效果”的当前具体机制是 NO-GO，不应因现在下载到了 PartNet 模型就重新启动已失败的 M1。
- StackCups 0/4 与抽屉 DEV 2/4 仅证实可复现基础设施和冻结 π0.5 的阶段能力差异，**不能逆转 M1 科学性否定结论**。

## 3. 接下来到底跑哪些基线

在数据/物理合法性之前，盲跑所有 14 项会造成伪比较和严重时间浪费。采用**两张互不混列的基线表**。

**表 A｜官方可复现 Benchmark 能力（非论文创新证据）**：
- 已有：B03 π0.5 Uniform 46/60、B05 Vid2Robot 3/60（clean PushCube）。保留原始协议与哈希。
- 若要将 ObjectInDrawer 纳入作者的正式 benchmark 表，需要 B03 同条件 heldout 20×3、Vid2Robot/同任务可用策略在同一数据/步数下评测；当前抽屉 2×2 DEV 不能代替这些数。B01 Robot-only BC 需先核对并补齐真实动作标签数据，再训练后评测。**不要仅为凑 baseline 个数立刻投入数十小时。**

**表 B｜真正支持方法的 same-goal/能力差异实验（当前无资格开跑）**，在确认真人示范、两类可执行解及非平凡依赖后，最低 6 个不同信息对照：
1. Goal-only / robot-only BC or DP，真正目标与目标机器人状态给足；
2. Raw flow → 同一执行器，检验直接迁移的物理不兼容；
3. Geometry+Waypoint / replanning，检验是否纯几何可解；
4. **Capability-aware TAMP**，使用与新方法相同目标、视觉/几何及动作能力信息；
5. Direct video-conditioned π0.5 / flow-conditioned BC/DP，检验有无显式效果编辑必要性；
6. 代表性的官方相关 Flow 算法：优先 Im2Flow2Act、EC-Flow、Cross-Behavior Object Flow 中能**忠实复现/可公开取得**的算法；明确官方/重实现/不兼容。

额外加 privileged-state oracle 为单独上界，不与 RGB 组比较；训练量、任务分布、测试种子和低层控制器完全匹配。论文完整版本至少还应有跨对象/布局 split、消融与真实机器人验证，但没有科学缺口前不能先跑一堆 GPU。

## 4. 硬决策 / 下一项具体执行，不再无限扩展诊断

**当前：基线能力工程部分验证通过；T-RO 科学创新主假设 NO-GO；G0 新反事实目标的数据前提仍 BLOCKED。**

接下来不是开发 CEPS、M2、另一种“抽屉阶段修复”，也不是把 400-step 0/4 做成一个新的创新点，而是**冻结固定目标视频对的开发研究，选择/收集能真实展示不同目标或接触约束的合法人类视频 + 必要的目标侧 PhysX 反事实**。优先使用已有 RoboReel TRAIN 数据/已经获得授权的 PartNet 和 ShapeNet 物体资产作数据与环境；它们可帮助真实场景构建但不应被当作现成反事实示范。短期必须先有真正符合判据的 pair；没有则 NO-GO。

**具体停止门槛**：找不到真实合法的视频/同目标源目标结果、或者等信息 TAMP 已完全解决可执行任务 -> 停止，不投入新的模型开发；有科学有效的反事实和可学习强对照差距 -> 才展开正式 B01/B02/B06/B11/B12 等 baseline 和我们的方法。

**禁令**：不得重复已完成的 PushCube 60 次和 Drawer DEV 4 次；不得声称 400-step 0/4 是正式官方评价；不得把源 Video 0000/0001 当作相反目标；不得把 scripted M1 策略当成 learned algorithm baseline；不得说所有 14 个算法都已跑完。

## 2026-11-13 追加更正：B06 UniSkill 作者 IDM 已找到，执行入口已就绪（非结果）

此前正文将 B06 标为需进一步确认 ISD 权重。现已通过作者 Hugging Face 仓库文件清单确认 `UniSkill_final_weight/idm.pth` 可获取（约 80.3 MB）；RoboReel 官方仓库提供 `uniskill_policy` ND checkpoint 下载入口。原状态表的“NOT_RUN”仍成立，因为**服务器端尚未下载/提取/运行**，但阻塞原因已经消除。

已新增 B06 自动化入口：
- `scripts/setup_b06_uniskill_stackcups.sh`：单独创建 `uniskill` Python 3.12 环境；所有 HF cache/checkpoints/IDM 写入 `outputs/v120_baselines/`；先核对磁盘空间；只下载 B06 ND policy 和 IDM；锁定 HF revisions 并保存 release manifest。
- `scripts/roboreel_uniskill_video_dev.py`：官方 UniSkillRunner，使用 StackCups TRAIN IDs 0000/0001，2 paired episodes / 4 DEV PhysX rollout；同 pair 相同环境 state 和 Torch RNG；保存视频/action trace/hash，并发布脱敏结果。
- `tests/test_roboreel_uniskill_video_dev.py`：纯 CPU **7/7 passed**；setup shell `bash -n` passed。
- 执行说明：`docs/2026-11-13_B06_UNISKILL_STACKCUPS_DEV_REPRODUCTION.md`。

运行命令：`bash scripts/setup_b06_uniskill_stackcups.sh`。成功输出 `ROBOREEL_B06_UNISKILL_PUBLISHED` 后，再由助手读取 `tro1-experiment-evidence/evidence/runs/` 中的实际回执。

本轮只属 TRAIN-video bounded DEV，不能作为正式 benchmark 成绩；正式 UniSkill 对比仍需 EVAL IDs 0080–0099 × 3 pass。没有服务器端回执前，B06 不计为已复现。