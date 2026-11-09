# AG-COFL / T-RO 跨行为机器人研究：完整交接与历史索引（2026-11-09）

> **新聊天窗口必读。** 唯一当前项目仓库：https://github.com/LemonQC/cehsisuanfa
>
> **归档范围说明：** 本文综合了当前可访问聊天内容、先前相关研究摘要以及此仓库里 v0.4—v1.3 的原始 Markdown、JSON 和代码，尽可能完整保留历次研究决策、真正运行的结果、失败原因、后续实验协议与复制命令。**这不是逐字聊天记录**：我无法从现有接口直接提取每条完整历史发言，因此不能声称100%原文导出。每项关键实验都有指向仓库原始数据的位置。尚未实际执行的步骤严格标注。
>
> **安全：** GitHub PAT、密码、HF token 等不得放到 GitHub；模型权重、H5和原始 MP4 仅留服务器。

## 0. 项目、服务器、用户不可变要求

- GitHub 主项目：https://github.com/LemonQC/cehsisuanfa 。所有阶段性研究决定、代码变化、失败教训、实验结果和下一步计划提交 Markdown 和小型 JSON/CSV，便于任何新窗口恢复。
- RTX4090 Ubuntu 服务器路径：/mnt/4Tdata/LemonData/cehsisuanfa；Conda：agcofl4090；github main 分支为代码同步与结果归档来源。安装了 GitHub CLI 并通过 gh auth setup-git 解决写权限。
- 目标是 IEEE T-RO 水准：**跨行为人类视频—机器人操作、对象物理效应迁移、真实闭环物理验证**，不是网络叠加、路径规划演示、简单 CNN 更换 ViT 或单任务成功。
- 用户要求先冻结有代表性的近年 Baseline，最好 >=10 项有实质意义的学习基线和强控制，官方代码优先、忠实移植须注明差异；标准 ManiSkill/RoboReel 物理环境、相同数据/场景/测评预算。不要直接因为一个 Baseline 成绩差就否定专家多轮形成的科学方向。
- GPU 很宝贵；不能无目的重复跑早期 CNN/ViT/代理合成数据。旧文件和数据可复用；不下载整个约100GB+数据集；人类 EVAL 不能用于模型训练。
- 用户希望直接读 GitHub 真实结果、改完代码提交仓库、自动发布小结果，不反复问用户已经给出的信息。不得把日志 V100_COMPLETE、V121_AUTOPUBLISH_OK、V130_COMPLETED_SCRIPT 等当作算法成功，必须核验真实 rollout、模型参数加载、per_trial 和 full_3pass_complete。
- 不要建议 git push --force、git reset --hard；先检查本地未完成 cherry-pick/rebase 与远端差异，不泄漏 GitHub token。

## 1. 核心科学路线与论文逻辑

核心问题：**Source Object Effect → Target Robot Capability-aware Effect Projection → Learned Closed-loop Policy**。

源人类演示可能 Pick、Lift、绕过障碍，目标 Panda 可能只允许 Push、维持接触/桌面支撑，受到物理障碍、机械臂工作空间和关节可达限制。应保留共通目标和真正必要的中间事件与对象状态效应，丢弃人类特有且目标机器人不可执行、无必要性的轨迹片段。

对象效应的候选结构：Goal、必要 Events/Waypoints、接触/支撑 Mode、3D object pose/SE(3) trajectory；**这仍是拟实现表示，不可声称 HSV 蓝色区域中心已经提供真3D标定姿态。**

实验中必须区分：
- **必要中间事件**：不通过就不能实现目标，比如绕过实体障碍、完成支撑状态转移、先满足接触条件；
- **习惯性无关绕行**：源视频特有但可省略，不能强迫目标机器人照做；
- **Target unreachable**：目标只有举起才能达到而 Push-only 不可达，正确输出不可达，不能编造成功；
- **Target reachable but source flow infeasible**：同一最终目标允许通过推行绕路完成，原始 Pick source 的抬升不可直接复现。

必须以真实物理 rollouts 对照：Raw Flow、时间重采样/warp、Goal-only、Goal+Waypoint Oracle、简单几何/A*、相同观测与训练预算的学习策略、我们的能力约束投影，以及单独列示的 privileged-state 上界。若只是做一条与 Oracle 一样的路径，就没有创新贡献。

**特别警告：原始 RoboReel PickCube-v2 和 PushCube-v2 的终极目标不同（Lift vs Push-to-goal）**；不能当作同目标跨行为对照。官方 RoboReel 多任务评测用于通用视频条件机器人策略性能；另建明确标记 CUSTOM-ADDITIONAL 的**同目标 Pick-source → Push-target 能力冲突任务**用于科学主张。

用户此前明确推翻了把“主动探测未知动力学”作为主线的方案（条件难控、必要性不足）。后续不应回到繁琐接触 teacher 调参、无物理差异的泛泛技能拼接、仅2D预测或仅造复杂网络的方向。

## 2. 时间线：阶段结果（运行事实与计划严格分离）

### v0.1—v0.3 基础实现

- 建立 ManiSkill3 / Panda PickCube/PushCube 操作和最初 teacher/BC、Full/Anchor/End 监督框架，headless 服务器测试。
- 这些早期脚手架不是可信的 T-RO 性能。过去用户文件中存在 Object_Flow_TRO_Research_Chat_Complete_2026-11-07.md、AGCOFL_GitHub_Source_Transfer_v020.txt、BASELINE_MATRIX.md；这几份不一定在当前仓库，此处只记名称，不假设已经上传。
- 更早的 PickCube 单任务 DP/ACT 实验曾有运行日志与结果，但其不同任务/训练设置不能混入当前 RoboReel 官方 Vid2Robot 成绩表，除非再核实原始实验协议。

### v0.4—v0.6 接触型物理 teacher 调试

源记录：
- docs/2026-11-09_v040_failure_and_v050_fix.md
- docs/2026-11-09_v050_failure_and_v060_fix.md

v0.4 seed=100 两条边路线 attempted=2、success=0，沿直线段重复尝试深接触，实际 TCP 下压不足但具体 IK 内因未证实。v0.5 attempted=2、success=1，上边路线成功、174步，下边因为中间拐点位置误差阈值宽松而最后推动失败。v0.6 采用更严格路点判定、物体运动和TCP telemetry、限定再接触，方案形成时不能声称结果已成功。科研决策：teacher 调参仅解决物理数据采集，失败示范不可混入正样本，也不作为论文创新。

### v0.7 合成代理 Full/Anchor/End

源记录：
- docs/2026-11-09_v070_proxy_result_and_next_gate.md
- docs/PARTIAL_FLOW_EXPERIMENT_V070.md

实际运行在 4090 的15轮训练：12个合成布局、96个合成mask，layout级 train/val/test 8/2/2；Full、Anchor、End 三种监督方式和梯度链路有效。但这些都是 SYNTHETIC_PROXY_DEBUG_NOT_PAPER 的 IoU，**没有真人数据或学习机器人策略的成功率**。决策：冻结工程接口，停止反复增加合成 epoch/seed，寻找可复现真实基准。

### v0.8—v0.8.1 RoboReel 真实视频与物理诊断

源记录：
- docs/ROBOREEL_REAL_BRIDGE_V080.md
- docs/2026-11-09_v081_mainland_download_fallback.md
- docs/2026-11-09_roboreel_episode0_physics_success.md
- docs/2026-11-09_roboreel_push_xy_regression.md

基准：https://roboreel.github.io/；https://github.com/gupta-anmol99/roboreel ；真实数据：https://huggingface.co/datasets/LogosRoboticsLab/roboreel 。

通过 hf-mirror.com 获得官方 EVAL 0080 front.mp4（214帧）；HSV蓝色候选在抽样帧中全部检测到，但“有候选”不是“真实目标ID正确”。

ManiSkill/PhysX 中一次 **手写 privileged-state PushCube-v2** 成功：31步，最终物体到目标距离0.045422m，处于0.05m阈值以内。这个脚本能读 oracle 物体/目标状态，动作**不由人类视频直接决定**，只是实物理链路诊断，1/1不可混入学习模型成功率。曾修正 cube[:2] XY 与 last_cube XYZ 数组广播错误。

### v0.9 弱 CNN 视频候选 mask

源记录：
- docs/2026-11-09_v090_real_video_training_and_replay.md
- docs/2026-11-09_v100_effect_gate_and_train_data.md

用官方 **EVAL** 0080 一个视频、14个重叠窗口作内部开发训练；模型输入32×32蓝候选mask、两个过去时间戳，未来8帧Full/Anchor/End。20个epoch实际GPU训练；Full 训练 full-mask IoU 0.0000、Anchor 0.002397、End 0.002357，三种末帧IoU全部0，产生大面积错误前景。无独立视频泛化、训练视频属于EVAL，不能正式报告。用户明确指出 CNN 太弱，不可凭CNN失败否定整个方向，后续使用现代预训练ViT并优先官方强Baseline。

### v1.0 真实 TRAIN 数据发现、无学习对照、几何效应门槛

源记录：
- docs/2026-11-09_v100_effect_gate_and_train_data.md
- research_reports/2026-11-09_v100/learnability_gate.json
- research_reports/2026-11-09_v100/effect_gate.json

旧mask平均仅2.482像素；Persistence mean IoU 0.798759、Constant Velocity 0.749486，CNN方法接近0；证明旧表示不合理，不能说明跨行为科学方向无效。几何投影与 Goal+Waypoint Oracle 完全同路且长度0.15m；来源还不是 Pick 的真实三维效果，不能宣称新物理方法成功。

官方真实 TRAIN 人类视频候选约260个，选择性获取6段。重要：官方 human_demos/eval/no_distraction/push_cube/0080-0099 为评测，不作正式训练。Git push/pull 曾有未完成 CHERRY_PICK_HEAD、non-fast-forward、认证问题，已经通过 gh CLI 配置解决，不要强推覆盖远端。

### v1.1 真预训练 ViT-S/14 + temporal Transformer

源记录：
- docs/BASELINE_MATRIX_TRO_V110.md
- docs/2026-11-09_V110_RESULTS_V120_DECISIONS.md
- research_reports/2026-11-09_v110/vit_report.json
- research_reports/2026-11-09_v110/vit_summary.csv

RTX4090 确认运行 facebook/dinov2-small 预训练 ViT，224x224 RGB、冻结视觉特征、2层 temporal Transformer，Full/Anchor/End各训练15轮。6段官方 TRAIN PushCube 视频以**整视频**分4 train/1 val/1 test；测试仅1独立视频、10个窗口（非10独立样本）。

| 方法 | 平均未来位置误差（px，越低越好） | 末帧误差 |
| --- | ---: | ---: |
| Constant Velocity | **7.58** | 17.69 |
| Persistence | 9.89 | 17.57 |
| Full ViT | 9.73 | **16.20** |
| Anchor ViT | 14.50 | 19.55 |
| End ViT | 11.85 | 21.49 |

结论：ViT 链路可以训练、Full末帧略好，但全时序不及恒速；只是**二维HSV中心**诊断，非标定三维效应，非真实机器人动作学习。不要继续反复训练这一小规模诊断。

核查官方模型仓库 https://github.com/weiweigu1998/roboreel_models ，固定commit **79a0fb0099e73c5560114753a8706276f93aefc2**：Video-conditioned π0.5 Uniform/Keyframes、Vid2Robot、UniSkill、VLM+π0.5有代码入口，不能因为脚本存在就说“运行过全部Baselines”。

### v1.2 真实机器人 TRAIN 动作数据首次验证

源记录：
- docs/2026-11-09_V120_FINDINGS_V121_RECOVERY.md
- docs/2026-11-09_V120_API_MOCK_FIX.md
- research_reports/2026-11-09_v120/real_robot_h5_audit.json
- research_reports/2026-11-09_v120/official_vid2robot_assets.json

官方发布 raw 路径是 **robot_trajectories/no_distraction/push_cube/**；原先误用作者代码内部别名 without_distraction 导致404，已修正。

真实 robot TRAIN demo 0000、0001：H5 动作形状**138×8、136×8**，Panda articulation**139×31、137×31**，base_camera/hand_camera 512x512 MP4均可解码，是真正可用的视觉-动作标签入口。权重和20评测视频的元数据已找到，但当时 hf_hub_download 报 LocalEntryNotFoundError，策略尚未能运行。单测曾因模拟 HfApi MockApi 丢失 repo_id 参数导致4失败，后已修复；这是测试桩错误不是研究模型错误。

### v1.2.1 真正的官方学习型策略物理闭环成功连通

源报告：
- research_reports/2026-11-09_v121/official_vid2robot_assets.json
- research_reports/2026-11-09_v121/official_vid2robot_smoke.json
- research_reports/2026-11-09_v121/official_runner_dev_summary.json
- 用户4090自动归档提交 **ede31ff2f3355ecd03cb8d8879a0a866eba7df9b**。

原版官方 Vid2Robot ND checkpoint：
- ckpt.pt **2,368,873,705 bytes**，SHA256 **8902745333e46e30e7988574e5bdaf870fa29278dfbc65f71e884a7193377197**
- train_config.json 911bytes
- 官方 EVAL PushCube front MP4 **20/20**（IDs 0080—0099），合计**97,448,129 bytes**。完整验证大小与SHA256。
- 模型**严格加载完整官方 state_dict**，使用作者原本 Vid2Robot + ManiSkill 物理 simulator，Panda视觉输入和8维动作，非 privileged-state 手写策略。
- 由于 timm 会先下载一套随后被完整checkpoint覆盖的ImageNet初始化权重，适配层仅暂时关闭冗余初始化下载，在完整严格参数加载后恢复；已透明记录这项工程性差异。
- 0080固定DEV集成试验**真实执行**、200步最大预算、success=false (0/1)；**不是正式成功率**。该0/1不能否定算法，也不能回头“优化0080”导致评测泄漏。

### v1.3 正在等待4090执行的官方真实Baseline（不要误称已完成）

源决策：docs/2026-11-09_V121_SUCCESS_V130_BASELINE_PLAN.md 。

截至本归档创建时，v1.3代码/测试已在 main（此前核验HEAD 03b77a865aefaea1f95a74a379eb8fcf9a3c9cdd），**没有看到v1.3真正20×3的测评报告**。

已冻结协议：
- 官方原始模型 + checkpoint + RoboReel ManiSkill Panda，**PushCube No Distraction clean human/clean env 单一评测条件**，原始 EVAL 0080—0099 共20个视频。
- seed_offset_base 0、100000、200000，**20次×3个seed pass=60物理rollouts**。chunk_size=1，max_rollout_steps=800，官方 pd_joint_pos，原始 human FRONT 与 robot base_camera 视觉历史。
- 不修改 EVAL demo scope、不对0080失败调参。每个pass只有20个不同demo ID、真实boolean success、表格统计一致时才计入。输出每pass k/20、三pass均值和SEM，不能把60次当作60个不同人类演示。
- 可验证复用已经完整完成的seed；中断的pass不能冒充0/20，完整20×3才是该cell正式复现。只是一个ND任务格，不等于全部RoboReel基准十任务、WD、ED、PD。
- 源：agcofl/vid2robot_official_full_eval.py、agcofl/official_v130_protocol.py、tests/test_v130_official_protocol.py；执行：scripts/run_v130_official_vid2robot_3pass.sh；自动上传：scripts/publish_v130_official_eval.sh。

**当前唯一优先服务器执行命令：**

~~~bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --rebase --autostash origin main
conda activate agcofl4090
set -o pipefail
bash scripts/run_v130_official_vid2robot_3pass.sh 2>&1 | tee baseline_v130.log
~~~

结果预期自动归档 research_reports/2026-11-09_v130/（**仅实际运行并推送后才出现**）。查看 OFFICIAL_V130_PASS_VERIFIED（每20条物理记录）、OFFICIAL_V130_EVALUATION_GATE（三pass状态）、V130_AUTOPUBLISH_OK（GitHub提交，不代表算法高成功率）。

## 3. 研究 Baseline 矩阵及当前真实性状态

完整矩阵详见 docs/BASELINE_MATRIX_TRO_V110.md 。所有方法需要明确 OFFICIAL_CODE_AVAILABLE / OFFICIAL_EXECUTED / FAITHFUL_PORT / REIMPLEMENTED / BLOCKED / NOT_RUN 标签。

| 方法 | 2026-11-09 当前事实 | 下一步 |
| --- | --- | --- |
| Robot-only RGB ViT Transformer BC | 仅2段真实TRAIN H5已验，正式训练/测试尚未建立 | 扩真实训练集，按原始8维动作和视觉对齐 |
| Robot-only Diffusion Policy | 曾在更早单任务中测试，不可视作本RoboReel官方成绩 | 任务/动作空间/物理场景对齐 |
| Vid.Cond.π0.5 Uniform | 作者提供代码；当前未验证原版权重闭环 | 建兼容环境、冻结评测 |
| Vid.Cond.π0.5 Keyframes | 作者提供代码；当前未验证原版权重闭环 | 同基座同数据预算对照 |
| **Vid2Robot 原版** | **已真实严格加载与0080物理运行0/1**；v1.3 20×3尚待运行 | 现在首先执行 |
| UniSkill | 官方源码已确认，尚未完整复现 | 独立模型依赖环境 |
| VLM+π0.5 | 官方源码可查，尚未实际跑 | 考虑输入/模型资源预算 |
| SeeDo | 尚未确认本代码库可运行官方发布版 | 不能冒充已复现 |
| PointPolicy | 需要真实标定3D输入；尚未建立 | 标定/真正3D标签可用后 |
| Cross-Behavior Learning With Object Flow Prediction for Robotic Manipulation（T-RO 2026） | 核心研究竞争者，尚未确认可直接执行的原作者源码 | 核实后忠实复现 |
| Im2Flow2Act | 作者开源代码，尚未在RoboReel物理协议忠实移植 | 明确环境差异 |
| EC-Flow | 作者开源代码，原实验Meta-World，尚未RoboReel port | 不能直接照抄成绩 |
| DINOv2-S/14 + Temporal Transformer | 真4090曾训练，只有小规模2D候选中心测试 | 工程诊断完成，不重复 |
| Persistence / Constant Velocity / Goal-only / Goal+Waypoint Oracle | 存在部分真实先前对照，但无统一完整机器人策略评测 | 在同目标物理机制试验中独立报告 |
| privileged-state Panda teacher | 旧PushCube物理1/1诊断成功 | **oracle上界独立成表**，不得与视觉策略公平混比 |

用户要求足够扎实的 Baseline（包括近年SOTA、真实reproduction）而非“找十几个名字”；正式论文需在相同任务/动作/信息条件下覆盖相互补充的强算法，若不同环境/模型需诚实注释。

## 4. 下一步 T-RO 科学创新具体实验

**两张主结果表不能混为一谈：**

A. RoboReel 官方基准：Vid2Robot、π0.5 Uniform/Keyframes、UniSkill、Robot-only BC/DP等学习机器人策略，在相同物理评测协议下给出真正k/N、跨seed均值与SEM/置信区间。Clean ND、WD、ED不可混写；对官方已发布结果与我们真实重跑结果分别注明。

B. **新增同目标 Capability-conflict 任务**（单独注明CUSTOM-ADDITIONAL）：source可Pick/抬起，target只可Push/桌面支持；同一个对象、终点、初始状态与物理世界。构造“源抬升不可移植但推行可达”，“必要中间事件与可省略冗余绕行”，“目标Push确实不可达”三类，且实际引入可核验的障碍、接触、支撑和关节限制。

物理强对照：Goal-only、Goal+Waypoint Oracle、Raw Object Flow、time resampling、2D warp、仅几何A*、简单可执行性过滤、同骨干flow-conditioned learned policy、提出的capability-aware effect projection、privileged oracle（单独表）。观测数据/动作预算/训练步骤/随机种子须配平。衡量任务成功、保持必需中间状态、删除无关轨迹、接触/支撑违反、碰撞和任务步数。

**止损标准是针对具体创新主张，而非整个项目。** 如果只得到与Waypoint Oracle完全相同的几何路线并且物理执行上无增益，不能包装成T-RO贡献。若真实配对任务仍没有一个明确的非平凡科学gap，应调整/推翻该主张，而不是堆更多模块或仅增大ViT。

## 5. 代码/数据/学术复现硬约束

- 人类真实训练只用 official TRAIN 目录，不能在官方EVAL 0080—0099上继续调参/训练；必须按video/scene划分训练、验证和测试，不能误把重叠窗口算作独立样本。
- 真实机器人动作：robot_trajectories/no_distraction/<task>/<demo>/，H5 traj_0/actions 为8维，对齐真实 RGB base_camera和hand_camera；源代码的 without_distraction 是内部映射。
- 官方模型源码PIN commit 79a0fb0099e73c5560114753a8706276f93aefc2；作者 Vid2Robot PyTorch 环境依赖可能不同于π0.5的JAX、UniSkill等，不必勉强一个conda，但保证**物理成功判据和观测/action协议一致**。注明任何port/compatibility差异。
- 使用GPU但需边界清楚：单元测试通过、权重传输成功、strict checkpoint load、真实 rollout、正式paper-ready多pass统计是**五件不同的事**。
- 下载采用有界文件清单/镜像续传；避免全量下载百万文件；不要重下已核验权重；不要上传任何大型权重、视频、H5、个人密钥。
- headless server，保留工作CUDA/PyTorch/ManiSkill，避免无故升级驱动。
- GitHub权限先前已成功；若push因本地落后失败，先安全同步远端，不force覆盖。

## 6. 新窗口固定执行顺序

1. **首先读此交接文件和最新真实GitHub报告**；尤其查 research_reports/2026-11-09_v130/status.json 与 official_aggregated_summary.json 是否出现。若没出现，v1.3**还没得到真实结果**。
2. 若v1.3完成，立即分析每seed成功/20与总体均值/SEM，审计异常、任务失败与官方原始benchmark差异；继续添加下一强Baseline的真实权重/官方物理评测。
3. 若v1.3被依赖/资源/代码阻塞，直接读取 run_status.json 错误并修复代码推送，不要让用户再次跑旧ViT或下载2.37GB。
4. 并行凝练同目标源Pick/目标Push的能力冲突机制和强Baseline，先做明确可重复的物理对照，然后决定Full/Anchor/End或事件表示的论文增益是否真实。
5. 每一次真实决策、代码版本、结果和下一步都写入Markdown并提交**LemonQC/cehsisuanfa**。

## 7. 可供新ChatGPT窗口复制的启动提示

> 我们继续 T-RO 项目 AG-COFL / Cross-Behavior Object Effect Transfer。GitHub 唯一当前项目仓库是 https://github.com/LemonQC/cehsisuanfa 。请先读取 docs/RESEARCH_CONVERSATION_HANDOFF_FULL_2026-11-09.md 全文、docs/BASELINE_MATRIX_TRO_V110.md、docs/2026-11-09_V121_SUCCESS_V130_BASELINE_PLAN.md 和最新 research_reports/；不要凭历史记忆猜新结果。
>
> 当前已真4090跑过小规模 DINOv2 ViT诊断；已拿到官方真正 TRAIN Panda H5 2段；Vid2Robot 2.37GB官方 checkpoint 和20 EVAL 视频已验证并严格加载；0080固定官方机器人学习策略物理 smoke 真实运行 success=false，但不是paper分数。v1.3的20视频×3seed正式评测代码已提交，先从GitHub确认是否完成（不能直接认为已完成）。
>
> 科学主线是 source Pick object effect 到 target Push-only 的 capability-aware physical effect projection + learned robot policy，在同一任务目标下保留必要事件、抑制无关绕行、识别不可达，必须与真实视频条件策略、raw object flow、Goal-only、Goal+Waypoint Oracle、简单几何和物理上界形成信息预算匹配的对照。原始 RoboReel PickCube和PushCube目标不同，不能伪称同目标任务。
>
> 请直接检查代码和实验结果、作出下一步决定并提交可运行代码和GitHub Markdown记录；禁止虚构Baseline成功、无意义重复CNN/ViT、小数据评测泄漏、主观否定多轮形成的研究课题。
