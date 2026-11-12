# TRO1 / M1-B3 — 2026-11-12 真实离线规划结果与正式工程停止门槛

> 状态：**ENGINEERING NO-GO for tested M1-B3 same-goal source Pick implementation**。不是证明 Panda 在原场景绝对不可能抓取；也不意味着整个 TRO1 课题 NO-GO。
>
> 依据：操作者在 RTX 4090 服务器、Conda `pi05`、项目 `/mnt/4Tdata/LemonData/cehsisuanfa` 上提供的 `frozen_planning_audit.json` 字段终端输出。原始 JSON/H5/视频在用户服务器，本次仅根据**操作者提供的打印结果**更新科学状态；未独立下载或校验该次报告的 SHA。禁止将其误写成 GitHub 上已有的可复核原始二进制结果。

## 一、冻结条件

- 仓库：`LemonQC/cehsisuanfa`；原场景 `AGCOFLM1B3Corridor-v0`，seed 120，源、目标最终方块目标 x=0.210m。
- 原目标端 v6：特权状态 Push-only 控制器，单 seed、66 个真实 PhysX 动作，cube 末位置 x≈0.2018075m，目标 XY 误差≈8.193mm；保留，不重跑、不当成学习方法成功率。
- 原三条同场景源端抓取尝试：335/106/65 真实动作，均零真实夹持/抬升；最后一次 65 步尝试中，手掌和双指对静态围栏所记录的力均为 0N，预抓取跟踪位置误差 16.75mm，末端距 cube 中心 52.98mm；正式抓取 `grasp_pose` Screw 阶段零执行动作。
- 冻结的离线脚本：`scripts/m1_source_frozen_planning_gate.py`，接口修复提交 `5d6b9c871cabb651899cac75ee83ce61781469aa`；真正本机 MPLib `0.1.1`。
- H5 有效状态：`traj_0/env_states/articulations/panda` [66,31]，解码 `ROOT13_QPOS9_QVEL9`；双指末态 [0.02300095744431019, 0.02300083078444004]m。

## 二、真实离线规划终端结果（operator-reported，非合成测试）

```text
DECISION: INCONCLUSIVE_IK_OR_COLLISION_FAILURE
MPLIB_VERSION: 0.1.1
IK_INPUT: mplib_0.1.1_flat_7d_xyz_wxyz
IK_STATUS: IK Failed! Cannot find valid solution.
IK_SOLUTIONS: 0
COLLISION:
  environment: {observed: null, reason: NO_IK_SOLUTION}
  self:        {observed: null, reason: NO_IK_SOLUTION}
SCREW_STATUS: screw plan failed
NEW_PHYSX_ACTIONS: 0
AUTHORIZES_PHYSICS: False
```

本轮诊断的合成单元测试此前已通过；现在有了**服务器真实的规划器返回状态**，而不再是 `Pose` API 报错。

## 三、严格解释

1. 在当前冻结的末态关节、原 OBB 抓取目标重建、原规划器和原采样障碍点云条件下，MPLib IK **未返回碰撞筛选后的有效候选**（0 个）。因此没有可用的终态配置开展明确的 self/env collision 查询。
2. Screw 规划返回 `screw plan failed`。注意：这是对当前规划器和当前重建目标的失败，不是对全部机器人的可达空间的不可行性证明。
3. MPLib 0.1.1 的 IK 内部可能执行碰撞过滤，`IK_SOLUTIONS=0` **不能区分**纯运动学 IK 无解、碰撞过滤失败、规划初值/求解收敛问题。两个 collision 字段 `observed=null` 是**未知**，不是无碰撞或已证实碰撞。
4. 原失败时 `grasp_pose` 的完整 7D 目标没有序列化；此次目标是从冻结脚本配方重建的，不能保证逐比特复原。诊断只支持保守的规划结论。
5. `NEW_PHYSX_ACTIONS=0` 说明没有执行新的机器人动作；此脚本为了构造规划器仍创建/重置过 ManiSkill/PhysX 场景。不能表述为完全没有调用物理引擎。
6. 因 `AUTHORIZES_PHYSICS=False`，该结果**不允许**第四次抓取、修改朝向、增加求解时间、换 planner 再物理试错。

## 四、正式决策：关闭这条工程支线

- **停止**原 M1-B3 x=.210 场景的当前源端 Panda Pick 执行器修补与追加 PhysX rollout。可将同场景的 target v6、三条失败源端轨迹和本次离线诊断保留为构建基准的工程诊断证据；不可当作成功跨行为源—目标配对。
- **不宣称** Panda 绝对不可抓取、不宣称某个具体连杆/围栏发生碰撞、不宣称场景/算法存在理论不可达证明。
- **没有完成**同目标成功 source Pick 视频 → target Push 的物理配对；没有证明视频相比 goal-only 的信息优势；没有等信息 strong TAMP 实验；没有 Ours/M2 的训练或物理结果。因此当前不具备 T-RO 方法立项证据。
- 本场景 **不再进行**“一次额外纯 IK 查验”或各种模型版本/API 迭代来绕过预注册停止门槛。若将来有独立必要性和强规划证据，必须新立审查协议，不能悄然改写本次 NO-GO。

## 五、下一个研究门槛（不是本次已有结果）

重点从“如何修原执行器”转为“是否存在一个**真实可行且具算法区分度**的跨行为任务”。

**Gate A：资产优先，停止新方法。** 盘点已有成功 PhysX source/target 轨迹的**初始状态、最终任务目标、目标容差、行为能力、物理障碍、成功证据**。M1B2 x=.150 的成功 Pick 只作为资产候选，**不能与 M1B3 x=.210 的成功 target 拼接**。只认可严格同目标、同成功定义的真实成对任务。缺少成对任务时先把数据/任务构造评为 BLOCKED。

**Gate B：科学必要性优先。** 对合格成对任务做目标信息对照：Goal-only（充分状态和目标）、Video-conditioned、Capability-aware TAMP（同信息、同动作约束），明确何种任务事实只有源演示能提供，强 TAMP 能否已完整解出目标操作序列。不要在缺少此证据之前编码因果效果编辑器或训练 M2。

**Gate C：条件性 GO。** 只有存在可复现的物理成对任务、源示范带来可归因的信息增益，且强同信息现有规划/学习算法有具体缺口，才允许开发唯一的最小方法假设；否则研究方向或 benchmark 需要调整。工程 NO-GO 与 T-RO 科学 NO-GO 分别记录。

## 六、重现与追溯

- 只读本地诊断结果（本次用户已读取）：`outputs/m1_source_geom_aligned_topdown_v1/frozen_planning_audit.json`。
- 真实原始数据在 4090 机器，不上传 H5/视频；如需要进一步外部审查，只单独提交**小型摘要及 SHA256**，同时说明来源为本地操作者机器。
- 冻结协议及曾经的兼容性修复记录：`docs/M1B3_FROZEN_PLANNING_GATE_2026-11-12.md`。
- 早期 65 步真实接触审计：`docs/2026-11-12_M1_SOURCE_FORCE_ZERO_PLANNER_FAILURE_NO_GO_DECISION.md`。

**结论：** 当前 M1-B3 源端实验证明“现有抓取执行路线尚不可用”，不足以证明“跨行为适应方法有独特科学价值”。停止该工程支线、保存证据、迁移研究判据至真实同目标成对基准与同信息强基线，才是下一步。
