# TRO1 / 2026-11-12 — 不再自建 Pick→Push：RoboReel 官方长序列视频控制 48h 执行合同

**决策：保留 RTX 4090 / pi05 / 已验证的 π0.5 和 RoboReel；停止当前 M1 Insert 工程支线。重新工作的对象是官方 Human Video → Panda long-horizon manipulation，而不是复活旧的 M1-B5。**

本文件是 Codex 可执行任务说明；不是已经完成的新实验证据，也不是保证 IEEE T-RO 的创新性。不可在完成实测前把假设写成成果。**只允许在已固定的官方环境、已发布源视频/动作、单 RTX 4090 上工作。**

## 1. 科学问题与两个公开任务

**新候选问题（需实验验证）：** 在“人类视频条件化的大模型机器人操作”中，源视频作为固定整体表示可能无法与目标机器人当下已完成的物体效果事件保持同步。能否在不改原策略主干、无测试时 privileged simulator state 的前提下，用物体中心的**事件完成证据**（如 object-in-drawer、drawer-closed）更新源视频条件/执行阶段，改善长序列任务的成功率与中断恢复？这与先前的“必须清障”完全不同，不依赖不成立的 M1B3 配对。

- **主任务：** RoboReel 官方 *Place in Drawer & Close*（仓库实际任务 ID 必须由官方配置核对，**不可假设**固定为某个字符串）。人类视频和机器人数字孪生属于同任务/同结果，包含 put-object → close-drawer 的真实任务依赖；遵循官方成功判据。
- **对照任务：** *Push Cube*。原 π0.5 Uniform ND 46/60、Vid2Robot 3/60 已归档，**不可重跑这 60 个 eval case 或把它们当开发数据**。已归档 14 个失败事件：offset0 为 6，offset100000 为 7，offset200000 为 1；demo 92、93 各在前两个 offset 失败。该分布只说明 seed/demo 关联，不能证明具体感知/接触/控制失败机制。
- RoboReel 论文报告 *Place in Drawer & Close*（网站表格 *Toy in Drawer*）π0.5 Uniform ND ≈16.7%，PushCube ≈78.3%，但这是作者公布的统计结果，**不是本地已经跑过的长序列实验**。严格核对官方环境/配置名称、checkpoint `train_config.json` 与 `tasks` 字段，防止把非目标任务的 checkpoint 用到长序列任务。

## 2. 新方法只作为可证伪候选，不凭概念认定创新

暂名 **CEPS (Cross-Embodiment Event Progress Synchronization)**，候选组件：

1. 源人类视频提取**物体事件时间索引**：原始 video feature 的片段与已发布目标关系定义结合；只能训练在 TRAIN/VAL 上，不能用 eval seed GT 事件制作测试时特征。
2. 目标机器人从 RGB(必要时加官方允许的 proprioception) 估计事件完成概率 `P(e_j done | o_0:t, a_0:t-1)`，同时输出失败/未知置信度；不使用测试时物体真值。
3. 根据当前阶段的**可观测完成证据**，动态调整提供给**冻结 π0.5** 的视频 tokens/上下文；必须先核查官方输入 schema 和 token buffer 可否支持在线切换。不能直接换成手工目标 pose 或 privileged oracle。
4. 目标事件未证实则保留/重试阶段；证据达标后向后推进。不能因为算法自己预测 success 就跳过真实官方任务 success gate。

**必须正面对照：** 原 Uniform；作者 Keyframes（只有作者实现和必要元数据可用才称 official，否则标注改写）；同计算预算的简单均匀重采样、online nearest-frame/DTW 或 ORCA-style sequence alignment；同等观察量的状态式 stage controller / VLA-Corrector-style recovery（若可忠实实现）；无视频 Robot-only（单独信息条件，不冒充同信息）；privileged state oracle（**只**作为另表 upper bound）。已有 ORCA (2025), VLA-Corrector (2026), Im2Flow2Act / Cross-Behavior Object Flow 覆盖广泛的时间对齐/恢复/Flow 机制，因此**不能**将对齐、阶段或闭环本身作为首创。

## 3. Codex 第一阶段：只做一次 4090 可运行性 + 视频因果探针，不直接训练 CEPS

在服务器：
```
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
```

**先做代码工作（立即执行，所有数据存 repo 下）：**

- 阅读 `docs/2026-11-12_TRO1_M1_TERMINATION_AND_BASELINE_RESET_DECISION.md`、`docs/BASELINE_MATRIX_TRO_V110.md`、M0 `research_reports/2026-11-11_m0_pi05_uniform/verified_summary.json`。
- 检查本地 pinned `third_party/roboreel_models`，应为 upstream commit `79a0fb0099e73c5560114753a8706276f93aefc2` 或精确记录差异。官方入口 `scripts/evaluation.py` → `evaluation.utils.run_eval_protocol`，直接查本地 `train_config.json`、runner 的 **demo index / video embedding / episode env seed** 映射及配置项，不能假设有 `--task`、`--demo` 等未核实 CLI。
- 盘点：长序列任务所需的真实训练和用于 DEV 的人类视频、checkpoint、train_config/vision embeddings 是否存在。不自动下载 104GB 或 50GB 全量文件；若确实缺失，生成**精确列表+单任务预估字节+镜像/断点续传方案**，保留已缓存 π0.5/SigLIP 基座；所有新增下载需任务局部、有日志和 SHA。
- 在主仓库新增 `scripts/ceps_video_dependency_probe.py`（以 pinned 官方 evaluator 的**已核实接口**为基础），`tests/test_ceps_video_dependency_probe.py`；不能改原 checkpoint、原 eval 文件、已成功 M0 报告。输出到新路径 `outputs/ceps_v0_dependency_probe/`，每个新实验单独 seed/video 输入/config/hash。
- **只用官方 TRAIN/validation 演示**进行开发实验，不能用 EVAL 0080–0099 为 CEPS 选参数。选择开发 split 不超过 12 个 demos，先使用单一预注册 seed，且使用相同初态/goal/动作预算。对照：A 正确配对视频与原 Uniform；B 同任务但其他 train demo 的真实视频 token（合理的替代，不是空输入）；C 顺序打乱或时间反转（**仅作 OOD 敏感性辅助**，不可拿其性能下降证明视频决策因果必要）；D 仅当在线片段插入接口可证实时，做一个简单 nearest-event/ordered-coverage 对照。所有版本必须保持相同冻结策略与目标观测信息预算。
- 第一轮真实 rollouts 总数上限 **24**，最多 12 对 A/B，每个最多 800 环境动作。不要重复运行此前官方 60 次；只要能有一组配对 A/B 的真实奖励、事件、视频配置哈希就可以验证闭环接入。**必须记录 completed/failed/timeouts 并保存 rollout videos**（仅此新 DEV 目录）；保留 raw state 只用于离线诊断，不可喂给 CEPS。
- 解析 A/B 的**配对差异与置信区间**（n 小只能作 DEV，不宣称统计显著）。原视频敏感性若很弱，先用主任务带序列依赖开展同类 probe，不要仅因为 PushCube 相同目标而宣布“视频没有价值”。

**输出必须包含** `results.json`，`per_trial.csv`，`SUMMARY.md`，每次 checkpoint/source-embedding/scene config 的 SHA 和源码 commit；(每个失败应有) 关键帧视频，失败分类允许 UNKNOWN；审查人工报告中不存在由 EVAL 选择算法配置的泄漏。

## 4. 第二阶段（要有第一阶段实测后，才能实现最小机制）

- 优先写简单、透明的 RGB 对象事件读取器，离线用发布的人工注释验证检测精度，明确 train/val/test partition；禁止把真值事件直接喂给测试算法。
- 比较 **静态视频**, **ordered sequence/ORCA-style 对齐**, **可观测事件完成证据条件化**。先 frozen π0.5、小模型可单卡训练，CEPS 新模块要有移除/替换/延迟鲁棒性消融；不升级主干大模型，不使用付费 VLM/LLM API。
- 学术门槛不是只要“有一点提升”。必须有跨多个 episode/布局/任务的统计提升，且在同等信息的强对齐/恢复基线下存在差异；否则 CEPS 只是已知 ORCA / VLA-Corrector 换壳，**不得对外宣称 T-RO 创新**。

## 5. 失败与执行纪律

本协议不是另一轮“B5/调夹爪/造走廊”。不触碰原 M1B2/B3 的任何物理结果；不把 B2 成功 source 与 B3 target 拼成同一配对；不重新打开禁用实验。可积极推进 RoboReel 原基准的基础设施、训练样本和有监督控制，不再对早已判为 NO-GO 的自定义任务耗时。

Codex 完成后应在新分支提交代码和小型证据，核对 HEAD 后合并；不上传大型 MP4/H5/模型权重。**如果发现任务缺 checkpoint/开发数据，则先提交确定的缺失清单与精确下载脚本，不能谎报做完物理 probe。** 本轮尚未实测 CEPS，理论新颖性仅处于候选阶段。
