# TRO1 · 主动研究决策与下一轮最小可证伪方案（2026-11-13）

## 决定：冻结失效路线，保留强运行基线，转向真实目标反事实

不再推进 `CEPS / StackCups` 的阶段同步故事。已归档 `StackCups-v1` 四条真实 PhysX 轨迹，任务成功 `0/4`；两对成功反转 `0/2`。首 20 步配对动作 8D L2 差异约 `0.0016`，全程均值 `0.758/1.222`，后者不能被误判为视频目标信息必要性的因果证明。真实 montage：Episode 0 两条已有红杯抬起和局部叠放，Episode 1 两条没有稳定两杯堆叠的可见证据；因此失败机制并不统一。当前科研结论仍是 **NO-GO for video-stage synchronization as framed**。

保存此前官方 `RoboReel PushCube-v2` 的 `demo_pi05` 46/60 成功和已执行的 Vid2Robot 3/60（原始任务结果，不得虚假外推）；**PushCube 只用作机器人执行能力/评估基础设施的正基线**。

### 为什么不直接回到官方 PushCube 做更大规模视频配对

核查已固定的 `gupta-anmol99/roboreel@afaf4a03` 官方源码 `src/envs/push_cube.py`：
- `PUSH_DISTANCE=0.15`；
- 环境初始化 `target_xyz[...,0] += self.PUSH_DISTANCE`；
- `evaluate()` 检查方块距离这个已定义目标的 XY 距离（`goal_radius=0.05`）。
即该任务的正确目标总是**相对于方块初始位置的 +X 推动**，不是两段人类视频决定左推/右推的交互式指令。`StackCups-v1` 的目标杯子顺序也是固定的 dark_blue → red → light_blue。**在这两个原始任务上更换不同演示而得到相同目标，不能回答“视频对正确决策是否具有必要信息”**。源文件：https://github.com/gupta-anmol99/roboreel/blob/afaf4a03df0be1e2347becde96da4f34e0de83fc/src/envs/push_cube.py 。

## 真正新的候选科学问题（待数据可行性验证；不是已经成立）

**跨具身视频约束下，能否区分外观相近而目标/接触结果不同的操作意图，生成满足源视频功能约束且适合目标机器人可行域的动作？**

区分“看完视频完成一个默认任务”与“视频是正确动作的必要信息”，必须形成真正的目标反事实：
`(same o_0, v_A, g_A)` vs `(same o_0, v_B, g_B)`，其中 `g_A != g_B`，两个行为在相同初态均可以合法完成。可以以 **同一方块放在中心，推向左侧/右侧两个目标区**作为最小环境示例；但两个带方向的真实**人类**示范必须真实存在、合法取得、人工验证源视频确实表现该目标。不得把 RoboReel 的 +X 人类视频改名为 -X，不能伪造训练数据或使用 held-out EVAL 样本作为方法开发素材。

这项研究并非凭改变任务成功函数就有论文贡献；“两个方向”只是验证数据中确有不同意图的 **科学实验装置**。方法创新须在跨具身、目标/接触可行性建模方面提出可推广的机制，并超越目标参数直接给策略、视频目标解码后规划、动作重放等同信息强基线。

## 下一步工程协议：四个硬门槛，先证据后方法

**G0（只读、无需 GPU，立即）**：审计现有合法可用的真人视频素材/公开数据中是否有 *两种真实操作目标*、各自的时间顺序、是否能映射到统一场景。核查授权、是否真人演示、镜头和对象对应、真实目标区域以及文件哈希。现有 `PushCube` +X 和 `StackCups` 固定顺序的两段 TRAIN 演示 **不能充当 G0 通过**。缺少真实目标相反的可用真人视频时，报告 `G0_BLOCKED`，不得开始训练或宣称新算法必要性。

**G1（任务定义、无模型训练）**：仅在 G0 存在真实可用数据后，创建**独立命名的新实验任务**（如 `VideoGoalPush-v0`，不得覆盖官方 `PushCube-v2`），固定 ManiSkill/Panda 版本、碰撞场景、两种目标判定函数、同一个初始状态和机器人初始观察；目标及条件提示不允许进入 agent 的观测或 prompt，目标只能由对应真人视频提供。记录完整资产/轨迹源、证据分割，验证两种目标在仿真中都物理可行。

**G2（必要性门槛，最小样本）**：在完全一致的机器人观测、固定随机数和无遮挡低层状态下，分别输入 `v_A` 与 `v_B`，首先进行**不执行物理动作的第一行动作分布/目标方向投影比较**，然后才在小规模固定场景里与 oracle-target controller 和 no-video / swapped-video 对照。必要证据是正确/错误视频导致的目标相关、方向一致的动作变化以及显著的任务结果差别；全程 8D L2 变大不够。若现成冻结 π0.5 对两种目标均缺乏执行能力，应停止此政策基线上的视频改进故事，不能将控制不足包装成研究缺口。

**G3（研究资格门槛）**：只有视频信息在 G2 被证明确有区分力，同时使用**同等目标信息**的解析器+规划器、目标 token conditioned policy、视频 replay、无视频政策等基线仍存在可量化的可迁移性或接触可行性缺口时，才开始方法创新和统一基准扩充。若目标 token + 常规规划或简单规则即可解决，则明确 `NO_NOVEL_GAP`，停止，不靠增加模型参数或跑 10 个无意义 baselines 提高故事性。

### 强基线与诚实状态

- **已执行且冻结**：官方 `PushCube-v2` π0.5 46/60；官方 Vid2Robot PushCube 3/60；StackCups π0.5 0/4。
- **本候选未运行**：oracle target-aware controller、目标 token conditioned π0.5、no-video、video-swap、video-goal extraction + classical control、BC、ACT、Diffusion Policy、Im2Flow2Act、XSkill/UniSkill/CORE；只有通过 G0–G2 后再确定可复现子集，任何结果必须区分 official/faithful reimplementation/NOT_RUN。
- 研究背景可以关注 H2R-Bench（2026，source-grounded goal/action/contact 与 embodiment）和 CoRL 2025 X-Sim 等论文，但它们并未直接证明存在与当前 ManiSkill 完全匹配、合法可用的目标反事实视频对，不能假定即插即用。

## 立刻做什么、不做什么

1. **不再运行** StackCups 当前分支任何 PhysX、模型训练或另选随机数；保持 `outputs/roboreel_video_dependency_dev/stack_v1` 只读。
2. **不再运行** 官方 PushCube 的“换两个同目标视频看 L2”实验；该任务 +X 目标恒定，不具识别目标必要性的辨识条件。
3. **下一项产出是 G0 素材审计表**（明确 `HAS_VALID_OPPOSING_HUMAN_GOAL_PAIR` / `G0_BLOCKED`），无须先改仿真框架。若发现两个合法来源且可对应同一操控对象/双目标区域，才直接实现 G1+G2 的最小代码；失败则终止这个候选。
4. 所有讨论和决策归档 GitHub。禁止删除已经成立的负面证据，也不把没有运行的 baseline 当作结果。

**总的投稿判断：目前不能宣称 T-RO GO；下一轮的投入上限为一个目标可辨识性审计和一个最小任务可行性验证，不承诺论文成功或先做大规模训练。**
