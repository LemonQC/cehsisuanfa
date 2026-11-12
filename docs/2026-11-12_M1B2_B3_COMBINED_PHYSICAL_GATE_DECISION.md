# TRO1 — 2026-11-12 M1-B2/B3 真实物理证据合并审查与算法立项门槛

**状态：M1 Insert regime NOT ESTABLISHED / 当前 B2+B3 不能支撑方法训练立项。** 区分：B2 是成功的同目标物理配对但不具因果必要性；B3 的目标端单次成功但源端缺少有效同目标视频。项目级 T-RO 创新性仍未确立，不能将两个不同场景拼接为同一个成功案例。

## 1. 来源、可复现性和冻结范围

本报告依据用户在 RTX 4090 服务器 `/mnt/4Tdata/LemonData/cehsisuanfa`、`pi05` 环境运行现有只读程序 `scripts/m1b2_validate_paired_physics.summarize(Path("outputs/m1b2_same_goal_dev"))` 后，在对话中提供的终端输出。该脚本审查原有三份 status 的起点、目标、seed 和真实 rollout 结果；本报告**没有**下载源 H5 或视频，也没有对用户服务器上的原始数据做独立哈希复核。绝不改动这些现有数据。

之前 B3 的冻结离线 IK/Screw 诊断与工程停止决策见 [B3 最终报告](2026-11-12_M1B3_FINAL_OFFLINE_PLANNING_STOP_AND_NEXT_GATE.md)。

## 2. 用户提供的 M1-B2 已执行真实物理汇总

```json
{
  "type": "M1B2_SAME_GOAL_PHYSX_GATE",
  "status": "PILOT_INCONCLUSIVE_OR_FAILED",
  "source_pick_place_valid": true,
  "target_direct_push_success": true,
  "target_clear_then_push_success": true,
  "target_only_blocker_displacement_observed": true,
  "matched_physics_start_and_goal": true,
  "all_three_real_rollout_traces_present": true,
  "pilot_contrast_pass": false,
  "causal_necessity_proven": false,
  "strong_capability_aware_tamp_evaluated": false,
  "learned_method_evaluated": false,
  "paper_ready": false,
  "per_case": {
    "source": {"success": true, "steps": 170, "error": null},
    "target_direct": {"success": true, "steps": 31, "error": null},
    "target_clear": {"success": true, "steps": 51, "error": null}
  }
}
```

原任务的源端目标为 x≈0.150m，不得与 B3 的 x=0.210m 混合。这里的 `matched_physics_start_and_goal: true` 是 **B2 内部三组** 的成功配对检查，不意味着它与 B3 匹配。

### B2 的因果必要性结论

- Source Pick-and-Place **成功**，170 动作；证明当前 B2 低层操作可执行，真实行为能力成立。
- Target Direct Push **成功**，31 动作；是决定性反例：清障并非本场景达到目标的必要前置条件。
- Target Clear-then-Push **成功**，51 动作，确实观察到 blocker 被移开；但成功不能说明其必要，亦不能由单个 trial 推断全局效率。
- 原冻结门槛 `Source_success AND NOT Direct_push_success AND Clear_then_push_success AND Blocker_moved` 为 **False**。保留原 `pilot_contrast_pass=false`，不可把场景叙述为成功的目标端 Insert 因果效果反转。
- 31 和 51 动作仅是特定一次脚本执行的动作数量；未匹配规划器和动作成本，不作为任何算法效率定论。

## 3. B2+B3 联合证据

| 判据 | M1-B2 (x=.150m) | M1-B3 (x=.210m) |
| --- | --- | --- |
| 真实源端 Pick 成功 | 是，170 步 | 否，三次源端执行均无实际抓取/抬升 |
| 真实目标端 Push 成功 | 是，直接31步、清障51步 | 是，冻结 v6 清障后Push共66步 |
| 同一场景同一终点的源—目标成功配对 | 是（B2 内部） | 否 |
| “不清障失败、清障成功”被物理对照验证 | **否：直接推成功** | **未验证为因果必要**：仅有目标端脚本成功，缺乏足以排除其他目标策略的证据 |
| 视频相比目标+可见场景有不可替代信息 | NOT_EVALUATED | NOT_EVALUATED |
| 等信息强 TAMP | NOT_RUN | NOT_RUN |
| 新算法 / M2 | NOT_RUN | NOT_RUN |

**联合判据：现有证据中不存在同时满足“严格成功同目标源—目标物理配对”与“必要的目标端新子步骤”两个条件的任务。**

B2 是有价值的 **non-discriminative/sanity control**：源和目标都能成功，但直接推动足够；B3 保留作为 **engineering negative / target-only feasibility**：目标端单次有物理解，但当前源端候选被正式工程停止。不可把 B2 的源成功与 B3 的目标成功拼接。

## 4. 预注册研究决策

**GO / NO-GO 不同层级应分开：**

1. **当前 M1 Insert 主场景作为算法必要性证据：NO-GO。** 关闭原 B2/B3 因果插入的“已验证必要性”声称；停止基于该证据立刻训练所谓 Effect Editor。
2. **B2 作为正控制/普通迁移 smoke：RETAIN。** 仅保留 170/31/51 真实轨迹与成功状态，不重跑；未来可用于验证代码正常或简单跨行为。
3. **B3 原 Panda Pick 执行器：ENGINEERING NO-GO。** 遵守先前源端计划失败后的停止门槛，不重试抓取姿态、MPLib API 和物理完整 rollout；目标 v6 66 步及原始失败记录冻结。
4. **整体 T-RO 跨行为科学问题：NOT YET DECIDED。** 该负面实验足以拒绝当前**基准支持的新方法立项**，却未证明所有跨行为迁移问题都缺乏研究空间。

## 5. 下一步唯一有成本上限的判别工作（无新物理轨迹）

**先盘点仓库/本地已有可用物理资产，之后才决定是否更换任务场景。不要先想网络结构。**

建立一张不超过一页的候选任务审查表，字段必须包括：

- 源端真实视频/H5/状态/原目标、成功 flag 和完整控制策略；不得用源脚本 waypoint、不同目标数据替代真实成功源视频；
- 目标端同一任务目标与初始几何、可用行为能力、成功 flag；
- `direct`、`adapted` 是否在同一物理状态下进行过可比较的测试；未测就写 NOT_RUN，不能自动当作失败；
- 任务效果约束/隐含前置条件是否由源演示独有地提供，还是目标状态+目标已经充分给出；
- 强等信息视频条件/能力约束 TAMP 的竞争关系；不能把**故意不给基线的任务信息**当成算法创新。

**早停规则：**

- 若只有 B2 这类 `Direct Push=True` 的容易场景，且没有已验证的额外源信息：现有 Insert 论文故事 **NO-GO**。
- 若有候选符合严格同目标和真实行为差异、且具有可识别的源演示信息贡献，再预注册 **一个**新的小规模科学反事实，对照 `goal-only`、`video-conditioned`、`capability-aware TAMP`。此时要合理披露信息差别：goal-only 不应被描述为同信息 TAMP。
- 若仍未找到候选，停止“换场景—调执行器”循环，重新审查研究问题及强方法覆盖边界，而非启动 M2 训练。

## 6. 指标与学术声明禁区

- 当前不能声称“目标端插入清障动作是成功必要条件”；B2 的直接成功已反驳这一点。
- 当前不能声称“B3 是不可通过抓取完成的任务”；只证实当前 Planner/抓取配置失败。
- 当前不能声称“源视频带来性能收益”“强 TAMP 不会做”“Ours 优于 baselines”，均尚未运行有效对照。
- 不把两个场景的单次成功数整理成统计成功率；可只描述它们是有监督的固定场景工程试验。

**科学结论：** 经由真实 B2 物理三路对照，我们获得了一个明确反例：清障不是必要步骤；加上 B3 同目标源执行失败，当前主实验没有建立新算法必要性。应停止用这两个场景为 T-RO Method 背书，优先查证真实的、具可区分度的源—目标任务资产。
