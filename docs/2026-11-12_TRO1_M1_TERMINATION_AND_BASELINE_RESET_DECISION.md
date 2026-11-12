# TRO1 / 2026-11-12 — M1 Insert 研究支线终止，RoboReel 基础资产与下一课题重审

> **当前立项结论：NO-GO for the existing M1 capability-effect insertion method.**
>
> 该判断不等于否定所有跨行为机器人操作，也不等于放弃已成功运行的 4090 / RoboReel / pi0.5 工程资产。**现阶段不启动 M2 因果效果编辑网络，不再产生 B5 或修补 B3。**

## 1. 实验事实：操作者于 2026-11-12 19:47 回传

操作者在服务器 `/mnt/4Tdata/LemonData/cehsisuanfa`、Conda `pi05` 执行已提交的 `scripts.m1_existing_asset_decision`：

```json
{
  "decision": "NO_GO_FOR_CURRENT_M1_INSERT_METHOD_START",
  "b2_pair": true,
  "b2_direct_push_success": true,
  "b2_source_video_file_on_disk": true,
  "b3_source_success": false,
  "b3_target_success": true,
  "m1_insert_scientifically_validated": false,
  "new_physx_actions": 0,
  "new_training_runs": 0
}
```

真实原始文件仍仅保存在操作者服务器；本报告引用对话中的终端结果，没有对大型 H5/视频进行独立校验。

- **B2** x=.150：源端 Pick 170 步真成功；目标端 Direct Push 31 步真成功；先清障后 Push 51 步真成功。同一初始化/目标三路验证通过。**直接 Push 成功，否定“必须清障”这一具体 Insert 必要性主张。** 源端 MP4/H5 文件存在，但这是机器人仿真轨迹，不是人类示范；也不能证明视频具有额外信息。
- **B3** x=.210：目标端冻结 v6 66 动作真成功；三次源端 Pick 接近/抓持未能形成同目标成功示范。离线规划报告：MPLib 0.1.1，IK 候选 0、Screw 失败，自/环境碰撞状态未知，新增物理动作 0。不能宣称该场景所有抓取绝对不可能。
- 官方训练人类 Pick 视频**本地 0**；已有 robot Push H5+双摄像头三件套 **2**。官方 PickCube 和 PushCube 任务目标不同，不可私自配对。
- **Goal-only vs video 信息增益 NOT_RUN；等信息强 TAMP NOT_RUN；Ours M2 NOT_RUN。**

## 2. 文献覆盖与方法风险（2026-11-12 外部资料核对）

| 研究方向/先验 | 相关工作 | 当前 M1 不能把什么声称为首创 |
| --- | --- | --- |
| 跨行为物体流 | [Cross-Behavior Learning With Object Flow Prediction for Robotic Manipulation, IEEE T-RO 2026](https://doi.org/10.1109/TRO.2026.3701576) | 对象中心 flow、从不同操作行为学习的基本想法 |
| 人类视频中转移因果子任务信息 | [Cross-Embodiment Causal Transfer from Human Videos to Robot Manipulation (2026-10 arXiv)](https://arxiv.org/abs/2610.21798) | “人类行为不直接复制，转移因果上下文/子任务关系”这一广泛论点 |
| 视频引导多步任务-运动规划 | [Multi-step Manipulation Task and Motion Planning Guided by Video Demonstration (2025)](https://arxiv.org/abs/2505.08949) | 视频提取接触和中间状态来辅助 TAMP |
| Flow 引导机器人行动 | [Im2Flow2Act (CoRL 2025)](https://im-flow-act.github.io/) | flow 作为跨域操作接口 |

这些工作与当前设想有实质重叠；**本报告不是完整审稿级先验调查**。尚不能从“我们也用因果图、插入动作、几何约束”推断有 T-RO 可发表的新贡献。

## 3. 当前事实对下一方法的决定

### M1：关闭，不再补工程

- B2 只作为已验证低难度同目标 sanity 资产，B3 只作为原执行器 NO-GO 和目标端可执行性资产。
- 不新增 B5/不同狭窄走廊，不更换抓取角度或 MPLib 版本后再次物理采集，不凭单 seed 创造成功率。
- 不实现 Effect Editor / flow 后处理网络，不训练 M2，不报告未执行 Baseline。

### 基线工程资产：保留，但不伪装为 T-RO 科学证据

- 官方 RoboReel M0：pi0.5 Uniform clean PushCube 46/60；Vid2Robot 3/60；已报告的官方基线结果不属于 M1 Insert 任务。
- 旧 `docs/BASELINE_MATRIX_TRO_V110.md` 声称 B01 的 64-demo 下载准备脚本 `scripts/prepare_b01_robot_train.py` 已提交。但在 2026-11-12 当前主分支 GitHub 路径核查，该脚本及 `tests/test_prepare_b01_robot_train.py` **均返回 404**。该文件存在性声称与当前仓库不一致。不得直接引用旧文档给操作者执行该不存在的入口；必须先重新核对代码、仓库历史与实际目录。
- 现有 `scripts/m1a_inventory_real_assets.py` 能只读盘点官方训练资产，B01 现阶段仅 2 组真实已落盘机器人训练样本。额外 Robot-only BC/DP、Im2Flow2Act、EC-Flow、TAMP 等仍不可报告成绩。
- 如果未来决定以官方 RoboReel 训练策略作为**独立的新研究基础工作**，先锁定一个可证伪的新科学问题和信息预算，再补齐数据准备、真正训练和相同任务官方评测。不能因为 4090 空着就先把 14 个 baseline 全跑一遍。

## 4. 新的 T-RO 可行性问题：先理论审查，再做实证

旧问题“源 Pick、目标 Push，目标必须插入清障”不成立。若仍尝试以源视频推动 T-RO，需要先清晰写出：

- **视频独有的潜变量是什么？** 在给出相同最终目标、目标侧全部观测、动作能力约束后，是否还有人类示范独有、可验证且与目标动作有关的物理条件、偏好或隐藏意图？没有的话，额外视频是冗余观测，不能承诺超越强 goal-only planner。
- **反事实是否改变正确目标策略？** 固定目标环境、最终目标和机器人能力，系统地切换唯一受控信息来源；理想比较还要考虑相同信息传给传统 TAMP 的上界。仅靠更强的动作脚本不是新方法。
- **方法是否有超出现有跨行为 Flow、CBST 和 Video-TAMP 的可验证机制？** 如果不能明确到输入、可学习的约束、输出和真实失败类型，不进入网络设计。

### 紧缩 GO/NO-GO 规则

1. **无现成可验证的数据/对照任务**：不立即新建模拟场景，不运行模型训练，判定该具体设想 `DATA_BLOCKED`。
2. **Goal-only 已能完整解出且视频没有未提供的任务信息**：视频必要性 `NO_GO`。
3. **有视频特有的信息，但等信息 TAMP 已能同等完成**：规划/约束编辑算法创新性 `NO_GO`，只能考虑是否存在真正的新视觉理解/估计贡献。
4. **只有在上述门槛均通过，且有足够新的算法机制**，才开一个与 M1 不混的研究分支，预注册数据集、强 Baseline、停止门槛。

## 5. 推荐的下一实际任务

**目前没有理由再在 4090 上启动任何 M1 实验。** 下一次研究投入用于审查 RoboReel 可复用的公开数据集与已发布源码是否支持一个具体、与 T-RO 2026 跨行为物体流和 2026-10 CBST 明确不同的科学问题。未确认以前不进行任何 GPU 开销或新专用环境搭建。

如目标仅是保留今后可用的 Benchmark 基础设施，可以另列 `B01/BC` 工程任务，但它不是当前 NO-GO 课题的“下一步方法”。

**会话结论**：M1 研究支线科学 NO-GO；原 4090 基础设施和官方 M0 Baseline 实验资产仍有效；新的 T-RO 论文问题尚待独立审查，不把工程熟练度当作研究新颖性。
