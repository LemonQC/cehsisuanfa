# T-RO 跨行为机器人操作研究：Baseline 全覆盖复现优先与论文问题重审

**专家评议稿 v1.0｜2026-11-13｜仓库：`LemonQC/cehsisuanfa`**  
**阶段：复现计划与科学问题审议；不是新算法论文成果，也不是 14 个 Baseline 全部完成的声明。**

## 0. 请专家先给出的判断

我们暂时**不继续实现 Ours**，将首要工作切换为：①逐项核实已登记 14 个 Baseline 的源码、权重、训练数据、原任务执行与统一 RoboReel 评测可行性；②基于可执行方法和真实失败案例，最后选择有科学价值、可被现有方法反驳的论文命题。需要专家判断的是：**这条“先复现，再定故事”的技术路线是否合理；怎样定义真正有意义的跨行为操作问题；哪组 Baseline 具有最强否定力。**

**重要边界**：一台 RTX 4090（24 GB），项目磁盘 `/mnt/4Tdata/LemonData/cehsisuanfa`；既有环境 `pi05`。不购买 VLM/LLM API，不下载无关大数据集，不允许将未运行代码计为复现，不把单种子脚本成功计为学习基线。优先共用 ManiSkill/PhysX 场景、目标、控制接口和评测种子；JAX/PyTorch 等互斥依赖允许**分离推理环境**，不可为了单 Conda 而篡改作者实现。

## 1. 已经发生的真实实验（不是计划）

| 来源 | 任务与协议 | 实际结果 | 可支持什么 |
|---|---|---|---|
| B03 RoboReel 官方 video-conditioned π0.5 Uniform | PushCube ND，官方 heldout 20×3，`max_rollout_steps=800` | **46/60（76.67%）** | 官方模型复现/有效运行，不证明我们的方法 |
| B05 官方 Vid2Robot | 同一 PushCube ND，heldout 20×3 | **3/60（5.00%）** | 官方模型复现；与 B03 可按该 cell 比较 |
| B03 π0.5 DEV | StackCups，两个 TRAIN 视频×两个相同初态 | **0/4** | 开发级复杂任务失败，非正式基准 |
| B03 π0.5 DEV | ObjectInDrawer（PartNet-Mobility 44962），两个 TRAIN 视频×两个初态 | **800 步 2/4；400 步前 0/4** | 作者场景及已授权模型可运行；成功首次出现于第 634、650 步；**不是**作者 heldout 20×3 正式成功率 |
| M1 定制同目标物理可行性 | B2/B3 挡块、Pick→Push、闭环脚本 | B2 直接 Push 已可成功，故“必须先清障”是假命题；B3 目标 Push 可成功但源 Pick 未形成同目标真实成功示范 | 原 M1 因果步骤插入命题 **NO-GO**，这不是任何学习型 Baseline 的完整结果 |

重要：抽屉同初态交换人类视频，**成功反转 0/2**；首 20 步平均动作 L2 约 `0.001443/0.001332`，全程 `1.3407/0.2704`，后期差别可能来自闭环状态分叉，**不能据此证明视频携带必要目标信息或 Stage-Sync 优越性**。

仓库证据：
- [B03 60 次逐样本统计](https://github.com/LemonQC/cehsisuanfa/blob/main/research_reports/2026-11-11_m0_pi05_uniform/verified_summary.json)
- [B05 官方复现状态](https://github.com/LemonQC/cehsisuanfa/blob/main/research_reports/2026-11-09_v130/official_aggregated_summary.json)
- [抽屉四条真实轨迹](https://github.com/LemonQC/cehsisuanfa/blob/tro1-experiment-evidence/evidence/runs/drawer_hf_v1_540ff3963863f5ed/evidence.json)
- [抽屉 400/800 步只读审计](https://github.com/LemonQC/cehsisuanfa/blob/tro1-experiment-evidence/evidence/runs/drawer_horizon_v1_e57085942a82872c/evidence.json)
- [M1 科学否定证据](https://github.com/LemonQC/cehsisuanfa/blob/main/docs/2026-11-12_TRO1_M1_TERMINATION_AND_BASELINE_RESET_DECISION.md)

## 2. 首先厘清：“14 个 Baseline”并不等于 14 套可直接运行的学习算法

下面所有状态是**截至 2026-11-13 的仓库与公开代码审查结果**。`OFFICIAL` 表示作者发布的实现；`PORT` 是我们对不同任务做的可追溯适配；`REIMPL` 为按论文忠实重写；`NOT_RUN` 为未提供实测证据；`BLOCKED` 为缺少关键公开资源。必须分开记录这些标签。

| ID | 对照方法与家族 | 源码/依赖现状 | 本项目实测状态 | 优先级与复现方式 |
|---|---|---|---|---|
| **B01** | Robot-only Transformer BC（无真人视频） | 当前 `main` 里旧表声称存在的 `scripts/prepare_b01_robot_train.py` **实际 404**；可利用真实 RoboReel Panda TRAIN H5 构造对照 | **NOT_RUN** | **P0**；补真实数据清单、BC 训练、标准物理评测；属 `REIMPL`，不得标“作者基线” |
| **B02** | Robot-only Diffusion Policy（无真人视频） | 公开基础方法可实现；必须锁定 architecture/action horizon 和训练标签 | **NOT_RUN** | **P0**；与 B01 共享数据、split、动作接口，`REIMPL` |
| **B03** | RoboReel π0.5 Uniform | 官方 5-method 代码与发布 checkpoint 已验证可用 | **PushCube ND 46/60 正式完成**；Drawer/StackCups 只有 DEV | **P0 已有结果**；无须重跑 PushCube；其他任务待选择性 heldout 扩展 |
| **B04** | RoboReel π0.5 Keyframes | 官方 checkpoint 和训练入口有；**原作者提关键帧脚本明确要求 `OPENAI_API_KEY`**，评测还需要对应 JSON | **NOT_RUN** | **P1 审计**：先看授权预制标注是否可合法获取；不能则 `BLOCKED_API_ANNOTATIONS`。本地启发式替代只可另列 `REIMPL` |
| **B05** | Vid2Robot | 官方 checkpoint 与原始评测代码可用 | **PushCube ND 3/60 正式完成** | **P0 已有结果**；保持冻结 |
| **B06** | UniSkill（特指 `uniskill_policy`，不是同名 Uni-Skill 技能库） | **已找到作者 IDM 权重**：HF `HanjungKim/UniSkill/UniSkill_final_weight/idm.pth`（约 80.3 MB）；RoboReel 官方 ND policy checkpoint 有下载入口；独立 `uniskill` 环境与 DEV runner 已提交 | **READY_TO_RUN_NOT_EXECUTED** | **P0 当前首个执行项**：固定上游/HF revision → 下载 policy+IDM → 提取两个 TRAIN skill → 4 条 StackCups DEV PhysX；成功后再准备 held-out last20×3 |
| **B07** | RoboReel VLM + π0.5 | **更正此前判断**：官方 `annotate_human_videos.py` 使用可本地推理的 **Qwen3-VL-4B-Instruct**，不是必需付费 API | **NOT_RUN** | **P1 可做**：项目盘缓存 Qwen3-VL，单卡完成视频标注/官方权重推理；检查推理峰值内存和 Qwen 许可证 |
| **B08** | SeeDo | RoboReel 官方 `roboreel_models/README.md` 仍写“将来加入”，未在该基线库中发布其直接评测 runner | **NOT_RUN** | **P2 独立作者源审计**；若完整可运行实现不可获得，`BLOCKED_AUTHOR_RELEASE`；不伪造结果 |
| **B09** | Point Policy / PointPolicy（3D keypoint） | **更正**：独立原作者仓库 `siddhanthaldar/Point-Policy` 已公开代码、数据；但 RoboReel 作者的 PointPolicy 统一 runner 尚未提供 | **NOT_RUN** | **P1**：先执行 Point Policy 原生入口，再用 RoboReel PD 校准关键点接入，明确 `OFFICIAL_NATIVE` 和 `PORT` 差异 |
| **B10** | Cross-Behavior Object Flow（IEEE T-RO 2026） | 正式论文与 accepted manuscript 可查；**独立完整可运行作者代码／权重尚未核实** | **NOT_RUN** | **P0 重要论文**；搜作者代码、直接核对算法/许可/数据；缺失则请求作者，有限度 `REIMPL` 且逐项注明偏差 |
| **B11** | Im2Flow2Act | 官方 `real-stanford/im2Flow2Act` 有代码、原始数据/weights 下载说明；原作者任务不等于 RoboReel | **NOT_RUN on RoboReel** | **P0**：先 `OFFICIAL_NATIVE` 运行原任务，再 `PORT` 观察/flow/控制器到同一 RoboReel 物理场景；官方要求约 24 GB GPU 同时跑在线 tracking+policy，必要时顺序预计算并标注差异 |
| **B12** | EC-Flow（ICCV 2025） | `YixiangChen515/EC-Flow` 官方代码、Meta-World 数据与 checkpoint；公开 README 描述原始训练为 **8× RTX 4090** | **NOT_RUN** | **P1**：**优先复现作者现成权重的原生 inference/eval**；一张 4090 不承诺完全复现 8 卡从零训练；再尝试 `PORT`，凡修改显式标注 |
| **B13** | DINOv2 + temporal decoder、Persistence 等视觉诊断 | 本项目部分已有表征 pilot；**不是原论文学习机器人策略** | **PARTIAL_DIAGNOSTIC** | **P1 诊断对照**；报告光流/状态预测误差，不能挤入策略成功率表充数 |
| **B14** | Goal-only、Goal+Waypoint、TAMP、privileged Oracle | 四类信息预算不同，不应笼统算“一种算法”；M1 的脚本只是局部物理可行性 | **PARTIAL_PHYSICAL_PROBES；非完整基线** | **P0**：拆 B14a–d 并严格分 visual planning / privileged oracle，记录真实闭环成功率 |

**按原版/适配两层统计**：目前**正式运行**只有 B03、B05 同一官方 PushCube ND cell；B01/B02/B04/B06/B07/B08/B09/B10/B11/B12 尚无本项目相同任务完整评测。B06 已进入 `READY_TO_RUN_NOT_EXECUTED`，但服务器未执行前仍不得计为结果；B13 仅诊断，B14 为局部物理验证。不要用“14/14”做结果指标。`B08` 无法取得作者必要资产时，复现的**结果就是可证伪的 BLOCKED 报告**，不能编造一个官方实现。

### 外部公开证据及重要依赖

- RoboReel 官方代码：<https://github.com/weiweigu1998/roboreel_models>；当前 README 列出 **VLM+π0.5、Uniform、Keyframes、Vid2Robot、UniSkill 五类方法**的训练入口，SeeDo/PointPolicy 计划后补；作者明确建议不同方法采用不同 Conda 环境。
- RoboReel benchmark、10 个任务及 ND/WD/ED/PD：<https://roboreel.github.io/>；数据 <https://huggingface.co/datasets/LogosRoboticsLab/roboreel>。
- π0.5 Keyframe 作者 `extract_keyframes.py`：需要 `OPENAI_API_KEY`。VLM+π0.5 的 `annotate_human_videos.py`：**本地 Qwen3-VL-4B-Instruct**（路径与上游 pin 需冻结）。
- Point Policy **独立原作者代码**：<https://github.com/siddhanthaldar/Point-Policy>；与 RoboReel 未释放的统一 runner 是两件事。
- UniSkill 原作者源码：<https://github.com/KimHanjung/UniSkill>；作者 HF 仓库已列出 `UniSkill_final_weight/idm.pth`（约 80.3 MB），本项目执行脚本会固定 HF revision 并保存权重 SHA256。
- Im2Flow2Act：<https://github.com/real-stanford/im2Flow2Act>；作者 README 标示 **CoRL 2024**，会务／期刊汇编年份不要混写。
- EC-Flow：<https://github.com/YixiangChen515/EC-Flow>，官方 Meta-World inference/eval 与多卡训练说明。
- 2026 T-RO Cross-Behavior Object Flow：<https://doi.org/10.1109/TRO.2026.3701576>；作者公开手稿 <https://eprints.whiterose.ac.uk/id/eprint/242108/>。不能把本方法“object flow 从行为解耦”再写成首次提出。

## 3. Baseline 复现必须遵守的验收等级

这里的“可用”，不能只看 import、loss 下降或启动成功。为**每个 Bxx × 原生任务/统一任务**记录以下层级：

| 等级 | 验收证据 | 不通过如何处理 |
|---|---|---|
| **L0：资源审查** | 论文、源码仓库和 commit、许可证、权重来源/sha、数据源/授权、输入/输出、官方运行入口完整 | `BLOCKED_SOURCE` / `BLOCKED_WEIGHT` / `BLOCKED_DATA`，附具体缺口 |
| **L1：原生环境运行** | 官方依赖环境清单、导入、自带测试与单 batch 的真实推理；无静默随机初始化、无 fake checkpoint | 修环境或阻塞，不记成功率 |
| **L2：原任务可执行** | 按作者原任务跑最少真实物理场景或离线原版指标，输出文件、video/trace 和 success rule；标明 `OFFICIAL_NATIVE` | `NATIVE_NOT_EXECUTABLE`；不可声称可接入 RoboReel |
| **L3：RoboReel 接口** | 相同目标/场景、camera/calibration、行动维数/单位、控制频率、物理接触、成功判据、相同 heldout；**先 2 个 DEV smoke** | `PORT_BLOCKED`；记录为何不适配、禁止随意修改任务 |
| **L4：统一 benchmark** | 完整原始 RoboReel `EVAL 0080–0099 × 3 seeds`，逐条真实物理 success/k、seed、steps、truncation/error、视频，任务/时限/动作数据一致 | 未实际运行则 `EVAL_NOT_RUN`；DEV 不算 L4 |
| **L5：与新故事公平比较** | 同一个**跨行为/能力差异**、相同目标/输入信息、训练数据预算、物理任务和受控反事实；原论文复现 ≠ 已完成这一层 | 故事未确定前统一记 `SCIENCE_NOT_RUN` |

**状态采用两轴**：`NATIVE=L0..L2` + `ROBOREEL=L3..L4`；`L5` 单独属于未来科学问题。一个方法在自己原任务成功不等于在 RoboReel 官方场景成功，更不等于我们新场景成功。

必备文件：`method_card.md`（来源与差异）、`source_lock.json`、`data_split.json`、`env_lock.txt`、`smoke.json`、`eval_per_trial.csv`、`eval_summary.json`、`failure_taxonomy.json`，没有的文件明确标 `NOT_AVAILABLE`。与旧仓库 `research_reports` 的历史记录不可互相覆盖。

## 4. 按这个顺序真正把 Baseline 做完（每一组都要有停止门槛）

**A. 先统一资源与代码检查（全部 B01–B14，并行只读，不启动 GPU）**
1. 核查每个仓库**实际** commit、LICENSE、checkpoint 的真实文件树及运行入口；不要按论文名猜 GitHub，尤其 B08/B10。
2. 检查已下载数据：RoboReel 官方 TRAIN robot H5、human TRAIN/eval（不混）、当前 `pi05` checkpoint、Gated PartNet-Mobility 44962、ShapeNetCore 的授权。只有数据确需用到时才按文件单独下载；全部放 4T 项目盘。
3. 输出 **全 14 项 L0 资源清单**；分为可直接原版复现 / 需单 GPU 技术适配 / 权重缺失 / 官方接口未发布 / 不适合统一任务。

**B. 先恢复可比性最高的作者原版能力（Released weights first）**
1. **B03/B05**：对已有成功 PushCube ND 不重跑，核查已有 60 真实 trial 哈希作为对照锚点。
2. **B07**：单卡本地 Qwen3-VL 离线标注→官方 VLM+π0.5 checkpoint 推理；不调用云 API。
3. **B06**：ISD 已找到，首个执行脚本已提交；直接跑官方权重与官方 latent extraction 的 StackCups bounded DEV。
4. **B04**：先审核已有合法 keyframe annotations；无 OpenAI API、无作者标注则官方 L2 `BLOCKED`，可以另外做明确的 `OFFLINE_KEYFRAME_REIMPL`，不可同名冒充。
5. **B09**：独立 Point Policy 原版 native smoke，再判断 PD RoboReel 标定/动作 retarget 的合理性。
6. **B11/B12**：作者原始模型下载清单与训练/测试入口核查，优先用作者 released weights 做 native rollout，随后记录转换所需的观测/动作适配；不要先从零完整训练 8 GPU 方法。

**C. 补必须新训练的 Robot-only 控制**
1. **B01** 与 **B02** 共享 RoboReel 真实机器人 TRAIN 轨迹、同样 RGB 相机和机器人 proprioception、锁定 train/validation、动作重采样协议、相同训练数据规模（如 20/50/100 demos 的预算对照，只使用真实有效文件），不能只靠两条演示就写“强 baseline”。
2. 先完成可运行训练、验证 loss 和**真 PhysX** 的 2 个 smoke；若可执行，再 20×3 正式评估。BC/DP 原始论文不同基准，按 `REIMPL_ON_ROBOREEL` 明确区别。
3. **B14** 同时拆为视觉 Goal-only、Waypoint（非特权）、等信息 TAMP、GT-oracle（特权），最后一项不得与 RGB 方法在同一主表比较。

**D. 外部实现缺口单独关闭**
- **B08**：优先找 SeeDo 独立实现是否授权可用；当前 RoboReel 原作者统一接口未发布，缺代码就归档 `BLOCKED`，不要无限“猜论文”。
- **B10**：Cross-Behavior Object Flow 是最贴近论文主题的首要对照；联系作者/核实可公开实现。若无权重和原始数据，对照可忠实 `REIMPL`，但须有公式对应表、单元测试和原任务数据可验证；否则保持 `BLOCKED`。
- **B13**：只在视觉误差/事件检出诊断表列示，不伪装完成整个闭环操作。

**E. 统一任务的执行矩阵（不要求一口气所有方法四任务全跑）**
- 主入口：**PushCube（已有正式锚点）→ ObjectInDrawer（真实多阶段、合法 PartNet）→ StackCups（阶段约束困难）→ PutBasket（多物体）**。优先各方法能接受的共同任务；不能声称原作者没支持的 task 为官方成绩。
- 主榜首做 **ND clean/clean × 20 EVAL demos × 3 seeds**，保留真实 `max_rollout_steps`/TimeLimit 和 controller 配置；通过后再按任务兼容性扩充 WD/ED；PD 单独一张 3D 输入表。
- 需要保证相同场景/seed、success oracle、终止/超时、报告置信区间。不同输入模态（RGB、语言、3D关键点、真值）按输入权限分表，不公平的模型能力比较不强行统一。

**执行约束**：单张 4090 顺序使用；全部 checkpoints、HF/Xet cache、临时解压和数据在 `/mnt/4Tdata/LemonData/cehsisuanfa/outputs/` 或项目下单独 `data/`（gitignored）；只向 `tro1-experiment-evidence` 提交脱敏的 JSON/CSV/关键结果图片，**不上传**大型 H5、受限 PartNet/ShapeNet、模型权重、私有原始视频、令牌。JAX `pi05` 保留；官方不同 Python/CUDA 依赖只用独立专用 Conda 环境，最终同一 ManiSkill 实例/评测逻辑尽量统一（确因原模型接口不同则记严格 PORT 差异）。

## 5. 暂不拍板论文故事：给专家审议的三个候选题目方向

### 方向 A（首选待验证）：跨行为操作的**可执行效果迁移**

- **核心冲突**：真人视频展示一种物体状态变化，但目标机器人抓持、接触、支撑或可达性约束不同，照搬轨迹会失败；且目标需要满足源视频中某些**因果上必要**的物体关系而非每个中间点。
- **潜在贡献**：从视频估计物体效果/先决条件，并预测目标能力允许的效果转换，闭环执行；可能是“保留最终功能、替换接触模式、剔除无关动作、插入必需动作”的统一模型。
- **最强反对意见**：现有 Cross-Behavior Object Flow、Im2Flow2Act、EC-Flow 或 TAMP 已可解决；源示范和目标机器人**同最终目标**的实物配对目前仍不足；旧 M1“必须清障”场景被真实 Direct Push 反例否定。
- **必须先看到的实验**：相同目标的真实源/目标成功案例；同信息 TAMP/Goal-only/Flow 的确切失败；能力改变导致必要效果反事实逆转。没有这三个不进入算法设计。

### 方向 B（备选）：人类视频中**不可由场景直接确定的操作目标或约束**

- **核心冲突**：同样初始机器人 RGB 状态对应两个合法、物理不同的目标/顺序/接触约束，只有真实人类视频说明意图；普通 fixed-goal 策略不知道要做哪一个。
- **最强反对意见**：一旦视频被简单解析为目标 token，普通 Goal-conditioned policy / parser+TAMP 也可解决。这只能证明视频有信息，**不自动形成算法创新**。
- **数据门槛**：两段合法真实、人为验证的相反目标示范；当前 G0 仍 `BLOCKED`，已有 RoboReel PushCube 恒定 +X 和 StackCups 固定顺序不能充当。

### 方向 C（低优先级）：长时序/关键接触阶段的鲁棒执行

- 抽屉 DEV 在原默认 400 步内 `0/4`，扩展 800 为 `2/4`，StackCups `0/4`。说明现有长时序策略尚有能力限制，但**这只是困难指标，不证明需要新的 Stage-Sync 算法**；如果强化普通闭环策略、分段 BC、replanning 即可解决，则创新不足。
- 如选择该方向，必须独立定义可观测阶段条件、无特权传感与完全相同的低层 policy，并证明强长时序 baseline 的实际缺口；不把动作 8D L2 放大写成时序同步问题。

### 论文是否继续的判定

| 实验发现 | 对应科研决策 |
|---|---|
| 所有代表性已有算法原版可运行，迁移到同一任务后性能强，且没有可证明的能力/信息差异 | **停止 A/B，考虑更有辨识性的任务或换方向**；不为了论文硬造编辑器 |
| 同信息 Goal-only / TAMP 达到与 video/flow 等价结果 | 不主张视频必要性；若视频仅便于给 goal，则贡献应转成**可测的目标解析/鲁棒感知**，否则 NO-GO |
| Direct Flow、BC、TAMP 都有明确物理失败，所需效果序列确实随能力和环境而变化 | **有资格再设计方法**，并加入强 flow/规划对照和 heldout 任务 |
| 无法取得作者代码/权重，但已严谨锁定阻塞 | **完成可复现性审计，不是发表级性能对照**；找可信公开替代基线并记录来源 |

## 6. 请专家重点回答的 7 个问题（评议意见直接写在下方）

1. **任务定义**：我们应把最终任务定位为“同目标、不同操作能力的效果转换”（A），还是“视频消解目标/约束歧义”（B）？能否清楚写出与普通 TAMP/goal-conditioned imitation 的不可替代区别？
2. **算法差异**：与 2026 T-RO Cross-Behavior Object Flow、Im2Flow2Act、EC-Flow 相比，哪一个可学习的新机制真正值得 IEEE T-RO？哪些提法属于已知概念？
3. **必须跑的 Baseline**：在 **B01、B02、B03–B12、B14** 中，请圈出审稿人认为绝不能缺席的 6–8 个；B08/B10 无公开作者结果时，接受何种可追溯 `REIMPL`？
4. **公平性**：原生作者任务与 RoboReel port 两级证据能否分表？单 4090 下不重训大型 pretrained VLA，复用作者 checkpoint 进行公平对照是否合理？
5. **数据可行性**：能否建议公开、可合法下载且含**真实不同操作行为但相同功能终态**的示范集？哪些必须自采以支撑接触/约束反事实？
6. **关键难度**：目标侧“能力改变→先决条件/效果顺序改变”是否足够非平凡；如何设计一个**等信息 TAMP 也容易犯错、但并非人为造障碍**的物理案例？
7. **GO/NO-GO**：您认为进入新算法研发前，最低应取得哪些配对实物成功、哪些 baseline 缺口、哪些泛化指标？若得不到，是否建议停止该方向？

> 专家评价：
>
> - 研究问题是否值得做：□ 推荐继续 A　□ 推荐继续 B　□ 推荐继续 C　□ 建议停止/换题
> - 必须优先复现的算法编号：____________________
> - 不可接受的创新重叠/方法漏洞：____________________
> - 建议补充的真实数据与任务：____________________
> - 对等信息 Baseline 和停止门槛的意见：____________________
> - 其他意见：____________________

## 7. 下一批实际交付（按 Baseline 而不是按“故事”排队）

**第一个交付节点：`BASELINE_SOURCE_AUDIT_COMPLETE`**——完整的 14 项源码/数据/权重/环境明细及作者原任务运行入口，阻塞必须有外部可核查证据。**第二节点：`NATIVE_SMOKE_AVAILABLE`**——能获得官方资源的所有方法都完成原任务真推理/物理 smoke，失败留下错误类别；B01/02 完成真实 TRAIN 数据及训练入口。**第三节点：`ROBOREEL_PORT_SMOKE_AVAILABLE`**——相容方法在至少一个相同机器人环境及任务下 2 次真 PhysX。**第四节点：`FAIR_EVAL_MATRIX_READY`**——在可比输入模态与 locked heldout 中形成可核查的定量结果。节点二／三／四**尚未完成**，不能随意自动标记 GO。

本专家评议文档不要求专家现在批准某个网络结构。**先确认对照实验能比较什么；算法创新要由真实失败模式倒推出，而不是从最新模块名拼出来。**

## 8. 已实现的第一步：全 14 项本地 L0 检查与 GitHub 自动归档

新增 `scripts/audit_baseline_reproduction_l0.py` 与 `tests/test_audit_baseline_reproduction_l0.py`。先用**纯 CPU、本地只读源码/现有证据**确认 14 个登记项是否具备预期入口、已归档的 B03/B05 官方 60 次结果是否存在；将每项的缺口作为可复核的结构化 JSON 上传到独立 `tro1-experiment-evidence` 分支。该检查绝不声称完成 L1–L5，且预期当前只有部分入口在本地，不代表外部开源代码不存在。**模拟项目上的 9/9 离线单元测试已通过；用户服务器的实际 L0 检查尚未执行。**

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_audit_baseline_reproduction_l0.py' -v
python -u -m scripts.audit_baseline_reproduction_l0 --publish
```

结果本地仅保存 `outputs/research_runs/baseline_readiness_l0/inventory_<hash>.json`（项目盘），远程自动保存 `tro1-experiment-evidence/evidence/baseline_l0/baseline_source_inventory_<hash>/inventory.json`；不上传任何受限原始视频、HF token、权重或大数据。后续根据真实 L0 报告逐个编写/完善 L1–L4 运行适配器，而不是先猜所有 Baseline 都能运行。

## 8. 2026-11-13：B06 UniSkill 已从“权重待核实”转为首个执行项

- 作者 HF 仓库当前文件列表确认 `UniSkill_final_weight/idm.pth` 存在（约 80.3 MB）；本轮只下载这个文件，不下载整个 Diffusers model 仓库。
- RoboReel 官方 checkpoint 下载脚本可选择 `uniskill_policy` + `nd`，不需要下载全部约 50 GB checkpoint。
- 新增并提交：
  - `scripts/roboreel_uniskill_video_dev.py`
  - `scripts/setup_b06_uniskill_stackcups.sh`
  - `tests/test_roboreel_uniskill_video_dev.py`
  - `docs/2026-11-13_B06_UNISKILL_STACKCUPS_DEV_REPRODUCTION.md`
- 本地 CPU 单元测试 **7/7 通过**，shell `bash -n` 检查通过。自动化脚本会固定 HF revision、记录 policy/IDM SHA256、检查项目盘空间，并把脱敏结果发布到证据分支。
- **关键状态：服务器端尚未运行。** 当前仅为可执行入口准备，不是 UniSkill 复现成功。命令：`bash scripts/setup_b06_uniskill_stackcups.sh`。
- 本轮仅使用 StackCups TRAIN IDs 0000/0001 做 2 paired episodes / 4 DEV rollouts；不能当成 held-out 正式结果。正式结果仍需 EVAL IDs 0080–0099 × 3 pass。
- 详见 [B06 执行单](2026-11-13_B06_UNISKILL_STACKCUPS_DEV_REPRODUCTION.md)。

### 2026-11-13 B06 独立交付与连接中断保护（不代表 GPU 实验完成）

- B06 现已单独完成全部代码修补并提交：`setup_b06_uniskill_stackcups.sh`、`roboreel_uniskill_video_dev.py`、`b06_failure_record.py` 及对应测试。
- 已修复默认 Conda 环境可能写到项目盘外的问题；改为项目内 prefix，且 pip/conda/HF/torch/cache/临时文件均由项目盘承载。
- 通过仅包含两个 TRAIN 视频软链接的受控数据根目录调用官方 UniSkill extractor，避免误处理全部 TRAIN 数据。
- 锁定 HF repo revisions，保留完整实验结果自动补传功能和失败状态自动脱敏发布。已有部分物理结果时禁止自动重跑。
- 本地 12/12 CPU 单元测试、Shell 语法以及 Python 编译通过。**尚无用户 GPU 上的 B06 真实 rollout 完成回执**。
- 仅在 B06 获得正式执行回执或可核查阻塞报告后，启动下一个算法，不同时大改 B07/B09/B11。
- [B06 具体执行文档](2026-11-13_B06_UNISKILL_STACKCUPS_DEV_REPRODUCTION.md)。
