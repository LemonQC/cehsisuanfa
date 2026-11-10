## ★ v1.5.9 — 2026-11-10 统一目标：必要 checkpoint 应下载；锁定 M0/M1/M2 科学验收

研究纠偏：不再把“不要无效重复下载/调试”等同于“遇到需要的模型权重就放弃”。**M0** 忠实完成 RoboReel 官方 π0.5 Uniform ND PushCube 20×3 物理评测（Vid2Robot 3/60 已归档，不重跑）；**M1** 构造同目标源 Pick→目标 Push-only 的必要新效果插入任务、明确源视频额外信息与强 TAMP 对照；**M2** 训练能力约束 Effect Editor + 闭环机器人策略并公平比较，若无新意则如实停止。发现可公开访问的 OpenPI 基础 pi05_base 的 HF 第三方镜像 `robotgeneralist/openpi_checkpoint_mirrors2`，锁定 SHA `1fda6b04e83d625eb1a55d68db32050e248cac2b`；新 `scripts/prepare_pi05_base_from_hf.py` 只下载原版基础 Orbax `pi05_base/params`（约 12.4GB），做文件完整性检查并在项目目录创建 OpenPI 缓存链接，不替换 RoboReel Uniform 6.2GB 微调模型，不碰原作者代码。使用 `plan-pi05-base-hf`（纯元数据）后 `prepare-pi05-base-hf`（显式下载），随后 `smoke-one-feature`、`prepare-features`、`--require-ready`、`evaluate-uniform`。Hugging Face 镜像声称是 GCS 原件镜像，尚未独立逐字节比对，保留来源说明。此次仅提交代码，服务器未执行新下载与物理试验。见 [冻结研究计划](docs/2026-11-10_V159_LOCKED_M0_M1_M2_AND_PI05_BASE_MIRROR.md)。

---

## ★ v1.5.6 — 2026-11-10 同目标 Pick→Push 源端 seed120 真实物理测试失败

服务器返回实际 PhysX JSON：源端 281 步，`ever_grasped=true`，但物体最高抬升仅 `0.00008825 m`（0.088 mm），没有越障，目标误差 `0.12924 m`，`env_success=false`、`end_phase=failed`、`valid_pick_source=false`。门槛正确阻止目标 Push-only 对照（`targets=[]`），不可宣称跨行为可行性成立。源控制器只有 grasp 阶段超时才转入 failed；`ever_grasped` 在任意阶段的动作后取样，因此并不意味着 grasp 阶段持续抓住。已新增 **无需运行物理实验** 的 `bash scripts/run_v141_next_gate.sh diagnose-custom-source`，读取已保存的 `source_trace.json`，按阶段输出夹爪接触、TCP/物体位置及抓取命令统计。先诊断已有 281 步，避免盲目重跑调参。记录见 [v1.5.6 物理失败诊断](docs/2026-11-10_V156_SOURCE_GRASP_FAILED_TRACE_DIAG.md)。π0.5 Uniform 官方测试仍因 SigLIP 原版权重缺失阻塞，Vid2Robot 3/60 冻结。

---

## ★ v1.5.2 — 2026-11-10 SigLIP 意外请求额外 11.6GB 基础权重，必须停止自动下载

用户执行真实一视频 SigLIP smoke 时，原作者 `SigLipTokenizer` 因找不到 `$OPENPI_DATA_HOME/pi05_vision_encoder/siglip_params.pkl` 自动访问 `gs://openpi-assets/checkpoints/pi05_base/params`，gcsfs 94 秒显示 0/11.6G 并反复输出 ADC 凭证警告，用户手动中断。**这是 OpenPI π0.5 基础模型，和已经下载的 6.2GB RoboReel Uniform 微调模型不是同一文件。** 原作者只需从基础模型 `PaliGemma.img` 导出 SigLIP pickle；不能在不验证权重等效性的情况下用微调模型替换它并声称官方复现。已提交 `bash scripts/run_v141_next_gate.sh check-siglip-cache`（无网络），并为 `smoke-one-feature` / `prepare-features` 增加默认 fail-closed 缓存门槛，避免再次意外请求 11.6GB。先检查本地是否有原版 `siglip_params.pkl` 或官方预生成 EVAL `front.npy`，否则通过可靠网络按原版权重获取。用户没有实际运行 pi05 物理评测；Vid2Robot 3/60 冻结。详见 [v1.5.2 GCS/缓存审计](docs/2026-11-10_V152_SIGLIP_BASE_11GB_BLOCKER.md)。

---

## ★ v1.5.1 — 2026-11-10 π0.5 环境仅剩 NumPy ABI 依赖冲突

用户新日志：在独立 `pi05` 环境中已成功恢复 `numpy==1.26.4` 和 `opencv-python-headless==4.11.0.86`；pip 仍提示 `numpydantic==2.0.0`、`pandas==3.1.1`、`ml-dtypes==0.6.0`、`opencv-python==5.0.0.93` 依赖 NumPy2。作者明确要求的 Gymnasium 1.3.0 与 ManiSkill 声明的 0.29.1 metadata 警告须通过真实运行验证，不盲目降级。新增 `bash scripts/repair_pi05_numpy_abi.sh` 仅修复 pi05 内上述小型 Python wheel、保持 JAX/GPU/716MB cuDNN 和 6.2GB 官方权重不变；`check_pi05_runtime.py` 新增 ABI 与双 OpenCV wheel 检查。执行 `git pull --ff-only origin main` → `conda activate pi05` → repair → `check-pi05-env` → 单视频 `smoke-one-feature`。本轮未执行 GPU 特征提取或真实物理 π0.5 评测；Vid2Robot 3/60 冻结。详见 [v1.5.1 依赖修复文档](docs/2026-11-10_V151_PI05_NUMPY_ABI_REPAIR.md)。

---

## ★ v1.4.9 — 2026-11-10 π0.5 环境安装的 716MB NVIDIA cuDNN 下载过慢

用户服务器新日志：在独立 `pi05` Conda 环境执行 JAX GPU 依赖安装时，`nvidia-cudnn-cu12==9.28.1.4` 716.6 MB 下载仅约 0.4–0.7MB/s，中途主动取消。**这不是已下载完成的 π0.5 Uniform 6.2GB 权重。** 已将 `scripts/setup_pi05_author_env.sh` 的普通 PyPI 源默认改为清华 TUNA HTTPS 镜像、PyTorch cu126 改为南京大学 HTTPS 镜像（可由 `PI05_PIP_INDEX_URL`、`PI05_TORCH_INDEX_URL` 覆盖）；不更改官方 JAX/其他包版本、全局 pip 配置或已有 Conda 环境。重新激活 `pi05` 后 `git pull --ff-only origin main`，执行 `bash scripts/setup_pi05_author_env.sh`，然后 `check-pi05-env`，通过后才提取 SigLIP 并评测。此轮没有新增物理评测；Vid2Robot 3/60 仍冻结。见 [安装镜像恢复手册](docs/2026-11-10_V149_JAX_CUDNN_MIRROR_RECOVERY.md)。

---

## ★ v1.4.8 — 2026-11-10 π0.5 已通过权重资源预检，只缺 20 个 SigLIP 特征

用户服务器实测：16 个 CPU 测试通过；Uniform 资源预检 checkpoint/train_config/协议检查均通过，唯一缺失 EVAL 0080–0099 的 20 个原版 SigLIP 缓存。Conda 中没有名为 pi05 的环境，但发现旧路径 /home/ubuntu/miniforge3/envs/rokae_pi05（是否有效未知）。已新增 check-pi05-env 环境校验、独立 pi05 原作者依赖安装脚本和相应离线测试：优先验证这个旧路径，不应破坏 agcofl4090 或改装他人环境；不通则另建 Python 3.11 pi05。然后依次 smoke-one-feature → prepare-features → --require-ready → evaluate-uniform（20×3），不得重跑 Vid2Robot。此次没有真实 π0.5 trial。见 [详细步骤](docs/2026-11-10_V148_PI05_ENVIRONMENT_GATE.md)。

---

## ★ v1.4.7 — 2026-11-10 官方 π0.5 Uniform 权重用户反馈已下完，进入特征与物理实验

**用户反馈下载完成；GitHub 尚未核实 RTX4090 本地文件完整性，不等于 checkpoint 已成功加载。** π0.5 官方物理评测仍 NOT_RUN。下一步先运行 `python -m agcofl.v141_pi05_assets --variant uniform` 和 CPU 测试检查原版权重及协议；独立 `pi05` JAX/OpenPI 环境先执行 `bash scripts/run_v141_next_gate.sh smoke-one-feature`（仅 1 段 EVAL 视频原版 SigLIP），再 `prepare-features`（余下 20 段缺失自动跳过已存在特征），确认 `--require-ready` 为 true 后才执行 `evaluate-uniform` 正式 3 seed×20。并行 `agcofl4090` 运行 `custom-physics-dev` 检查同目标 Pick→Push 物理有效性，不当作学习方法结果。**禁止重新下载原权重、重跑 Vid2Robot v1.3 60 次或开启 Keyframes/VLM API。** 详见 [v1.4.7 完整命令](docs/2026-11-10_V147_POSTDOWNLOAD_NEXT_GATES.md)。

---

## ★ v1.4.6 — 2026-11-10 镜像全 308，代理慢：只补齐 13 个缺失官方权重文件

服务器真实状态：HF-Mirror 的 25/25 HEAD 全部 `308 → huggingface.co`，12 个文件本地大小一致（**尚不能当成哈希验证**），13 个文件缺失；用户已有部分权重通过慢代理获得。新增 `scripts/recover_pi05_uniform_missing.py`：默认仅用镜像读取模型元数据并核对本地文件大小/哈希，`bash scripts/run_v141_next_gate.sh plan-checkpoint-repair` 不下载；主动指定 `--download-missing` 时才通过可信代理走官方 HF 原始文件，`.part` 支持 HTTP Range 续传，避免重复下载。可用 `--max-files 1` 先小规模测试，或在系统级 VPN/TUN 具备可用出口时指定 `--routed-vpn`。详见 [v1.4.6 下载恢复协议](docs/2026-11-10_V146_PI05_PARTIAL_PROXY_RECOVERY.md)。当前 **未实际恢复任何新文件或运行 π0.5 物理评测**，v1.3 3/60 冻结；同目标自定义 PhysX 仍优先推进。

---

## ★ v1.4.5 — HF-Mirror 下载 17/25 失败，新增逐文件 HEAD 检查（2026-11-10）

服务器真实反馈：HF-Mirror 可以列出 `LogosRoboticsLab/roboreel_models` 官方 ND Uniform 的 25 文件（6.20 GB），但 `snapshot_download` 在 17/25 报 `FileMetadataError`（可能是镜像文件 308 重定向使 `x-repo-commit` 丢失；与 Hugging Face 已知 issue #4637 相符，**仍需核实确切失败路径**）。已提交 `bash scripts/run_v141_next_gate.sh probe-checkpoint-files`：只发 25 次 HEAD、不下载权重，显示各文件镜像重定向目标和现有文件大小。新版 `prepare-checkpoint-mirror` 会先检查这些路径，若重定向到被阻断的官方站则提前终止，避免再重复下载 6.20 GB。详见 [逐文件排查](docs/2026-11-10_V143_PI05_TLS_RECOVERY.md)。π0.5 真实物理 trial 仍为 0，v1.3 Vid2Robot 3/60 保持冻结。

---

## ★ v1.4.4 — 2026-11-10 DNS/IP 路由异常，改用安全镜像预检查或离线传输

服务器新日志显示系统 DNS `104.244.43.104`、指定 1.1.1.1 查询 `202.160.130.145` 均命中 X/Twitter 网段，强制连接 CDN IP 的 TLS 握手也被重置。**不再重试原域名、不改 hosts、不禁用证书验证**。已提交 `bash scripts/run_v141_next_gate.sh probe-checkpoint-mirror`（只检查公开模型元数据，无下载），通过后可自愿执行 `prepare-checkpoint-mirror` 下载官方 ND Uniform 子集；镜像可能重定向回原站，失败则从联网电脑下载并传输权重。同时继续本地 `custom-physics-dev`。详见 [排障记录](docs/2026-11-10_V143_PI05_TLS_RECOVERY.md)。π0.5 正式物理评测仍 **NOT_RUN**；v1.3 Vid2Robot 3/60 不重跑。

---

## ★ v1.4.3 (2026-11-10) π0.5 Uniform TLS 下载失败与修复

用户服务器实测：访问 huggingface.co 的官方 HfApi API 时发生证书域名不匹配，**官方 checkpoint 未下载**；20 个评测视频存在但对应 SigLIP 特征全部缺失，π0.5 尚未开始真实物理评测，成功率依旧 `null`。已修复评测脚本的缺失资源 fail-closed 门控（`--require-ready`），并拆分 `prepare-checkpoint` 与 `prepare-features`，提供不泄露 token/代理密码的 `python scripts/diagnose_hf_tls.py`。不要关闭 SSL 验证或假造模型缓存；不重跑 Vid2Robot v1.3。诊断与恢复步骤见 [v1.4.3 TLS 文档](docs/2026-11-10_V143_PI05_TLS_RECOVERY.md)。

---

## ★ v1.4.2 研究决策（2026-11-10）：暂停一切依赖付费大模型 API 的 Baseline

按最新研究决策，**不再运行依赖 VLM/LLM 外部 API 的 Keyframes 标注、VLM+π0.5 标注等流程**。这类方法即便有论文报告成绩，也不是目前资源应投向的科学主线。现阶段仅推进不依赖付费 API 的官方 π0.5 **Uniform**（本地 JAX/OpenPI 模型与 SigLIP 缓存，仍可能需要下载公开权重）和 **CUSTOM-ADDITIONAL 同目标 Pick→Push-only 物理验证**；随后构建训练数据和真正学习型闭环策略，比较具有同等信息的 Baselines。任何纯几何绕路、伪人工关键帧或仅运行代码，均不得宣称创新成功。

已经收窄执行入口为 `bash scripts/run_v141_next_gate.sh {gate|prepare-uniform|evaluate-uniform|custom-physics-dev}`，不包含 Keyframes 和任何 VLM API 调用。**本次只更新代码和研究协议，未执行新的物理 trial。** 旧 v1.3 Vid2Robot 3/60 保持冻结。

---

# v1.4.1 — 官方 Vid2Robot 复现对齐；π0.5 与同目标 Pick→Push 物理关卡已提交（尚未执行新trial）

**真实已完成：** Vid2Robot PushCube ND 20×3 = 3/60（5.0%），与 RoboReel 作者公开的 5.0% 数值相符，严禁重复运行。
**新代码已提交、尚未在服务器运行：** π0.5 Uniform/Keyframes 官方 checkpoint/SigLIP/关键帧资产准备及原版物理评测入口，以及 CUSTOM-ADDITIONAL Pick-source→Push-only-target 刚性障碍 PhysX 验证。未有新增实验成绩，不能将 NOT_RUN 写成成功。

阅读 [v1.4.1 研究审计及执行顺序](docs/2026-11-09_V141_OFFICIAL_AUDIT_PI05_CUSTOM_GATES.md)、[未执行状态](research_reports/2026-11-09_v141/status.json)、[新增执行脚本](scripts/run_v141_next_gate.sh)。先用 `bash scripts/run_v141_next_gate.sh gate` 做纯预检查；独立 pi05 环境、按需选定权重和特征，才运行新的物理实验。后文关于“旧 v1.3 进程仍在运行”的内容仅是历史记录，当前不再适用。

---

# 换新聊天窗口先看这里：研究聊天与实验完整交接

## 最新研究状态：v1.3 已完成并成功归档（2026-11-09）

**正式官方 Vid2Robot PushCube ND 单格，20个 EVAL 视频 × 3 seed：1/20、2/20、0/20，累计3/60=5.0%，跨seed SEM=2.89个百分点。**

GitHub 已核实 [正式实验汇总](research_reports/2026-11-09_v130/official_aggregated_summary.json) 和 [正式评测状态](research_reports/2026-11-09_v130/status.json)：full_3pass_complete=true，errors=[]。**不需要再执行 v1.3 这60次评测**；本文下方早期“正在运行第三组、下一步跑v1.3”的文字为历史快照，不是当前状态。

新聊天请首先阅读 [完整研究交接（含Git修复和最新科研任务）](docs/RESEARCH_CONVERSATION_HANDOFF_FULL_2026-11-09.md)。下一步优先核实作者原版评测对齐程度、开展视频条件 π0.5 等强 Baseline，并行开始 CUSTOM-ADDITIONAL 同目标 source Pick→target Push-only 的物理效应可执行性实验。

---


## v1.3 实时进度查看（不要中断正在执行的评测）

此前前两组已真实完成：seed=0 1/20、seed=100000 2/20，第三组已启动；这不是最终3-seed结果。原先每20个 demo 才有完整输出，现在已增加逐demo控制台消息和原子更新的 progress.json（**仅在下一次启动的 Python 评测进程中生效**，无法给运行中的旧进程补日志）。

**当前旧进程还在运行时**，第二终端只能看已有日志和GPU，不要为了获得新脚本而提前 git pull：

~~~bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
tail -n 15 outputs/v130_official_vid2robot/seed_offset_200000/run.log
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
~~~

**本轮评测进程结束后**才同步新代码，之后新启动的评测进程将逐demo写 progress.json，运行 bash scripts/show_v130_progress.sh 或 watch -n 3 bash scripts/show_v130_progress.sh 可查看。原有运行尚未结束时请勿 git pull、不要 Ctrl+C。完整说明与GPU显存占用/GPU-Util区分：[v1.3 实时进度与GPU记录](docs/2026-11-09_V130_LIVE_PROGRESS_GPU_DIAGNOSTIC.md)。

---

**[AG-COFL / T-RO 研究对话交接全文（2026-11-09）](docs/RESEARCH_CONVERSATION_HANDOFF_FULL_2026-11-09.md)**：历次研究决策、从 v0.1 到 v1.3 的已运行结果与计划、Baseline状态、失败原因、现有代码和新窗口可复制提示词均已归档。本文是可核验的研究内容整理，不是逐条原始聊天记录转储。

---

# v1.3 — 正式官方 Vid2Robot 20场景 × 3 seed 物理评测（代码已提交）

**v1.2.1 已验证成功的环节：** 2.37GB 原版 Vid2Robot 预训练权重严格加载；20/20 官方 EVAL front 视频完整；作者的物理评测器运行成功。单独的 0080 集成测试结果为失败（0/1），不能代表正式成功率，更不能作为放弃 T-RO 研究的依据。

**现在进入正式官方 Baseline**：PushCube clean-human / clean-environment 条件；评测原始 0080–0099 共20场景，每场景3个官方环境 seed offset 0/100000/200000；保持官方控制方式 chunk=1、单轮800步，不做 EVAL 数据上的模型调参。

运行：

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --rebase --autostash origin main
conda activate agcofl4090
set -o pipefail
bash scripts/run_v130_official_vid2robot_3pass.sh 2>&1 | tee baseline_v130.log
```

**只核对三个标志：** OFFICIAL_V130_PASS_VERIFIED（本seed完整20条物理记录）、OFFICIAL_V130_EVALUATION_GATE（三seed结果）、V130_AUTOPUBLISH_OK（小型数据报告已提交）。若某seed中断，下一次运行会验证并复用之前已完成的整个seed。不重下2.37GB权重、20段视频和真实机器人训练H5，也不重训ViT。

实验JSON自动上传至 research_reports/2026-11-09_v130/；此处仅覆盖官方 PushCube ND 单格，不代表所有任务/干扰条件的完整 RoboReel benchmark。

研究决策与算法创新实验计划：[v1.2.1结果与v1.3方案](docs/2026-11-09_V121_SUCCESS_V130_BASELINE_PLAN.md)。

---

# AG-COFL v1.2.1 — Official Vid2Robot 权重续传与真实物理 Baseline

**v1.2 已实测成功**：2段官方 RoboReel TRAIN 机器人 H5 动作 + base/hand RGB 视频全部通过检查（动作形状 138×8、136×8）。**还未执行官方学习型机器人物理闭环**：作者公开模型权重和 EVAL 视频的 Hugging Face 下载不完整，官方 rollout 状态 false。

**下一步只需运行这一条（避免重复ViT和H5数据）：**

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --rebase --autostash origin main
conda activate agcofl4090
set -o pipefail
bash scripts/run_v121_official_baseline_resume.sh 2>&1 | tee baseline_v121.log
```

v1.2.1：重用已有机器人 TRAIN H5、核对/复用原来可能已下载完整的2.37GB官方 Vid2Robot checkpoint；文件缺失时通过显式 Hugging Face 镜像、官方地址与HTTP Range断点续传；下载20段约97MB固定评测视频；在**作者官方模型与官方ManiSkill**执行单段预先固定的集成测试；报告自动提交 `research_reports/2026-11-09_v121/`。**不上传权重、视频或H5**，不以一次调试成功冒充论文正式成绩，也不使用测试视频调参。

**只以下述状态判成功：** `OFFICIAL_ASSET_READY`（具体文件）；`VID2ROBOT_OFFICIAL_RELEASE_GATE.status=OFFICIAL_VID2ROBOT_RELEASE_AND_EVAL_DATA_READY`（资源就绪）；`VID2ROBOT_ROLLOUT_GATE.rollout_executed=true`（真正执行物理运行）。最终 `V121_AUTOPUBLISH_OK` 仅说明归档完成。

详见 [v1.2实验诊断与v1.2.1研究计划](docs/2026-11-09_V120_FINDINGS_V121_RECOVERY.md)。

---

# AG-COFL v1.2 — 真实机器人动作和官方 Vid2Robot 闭环 Baseline（已提交，待4090执行）

**v1.1 已实际成功：** DINOv2 ViT-S/14 + Transformer 在 RTX 4090 上训练完毕，并归档7份报告。独立测试视频（仅1段）的平均位置误差：恒速 7.58px、Full ViT 9.73px、Anchor 14.50px、End 11.85px。该实验不是机器人策略，也不是跨行为创新验证。

**现在优先跑官方 Baseline，不再重复旧ViT训练：**

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --rebase --autostash origin main
conda activate agcofl4090
set -o pipefail
bash scripts/run_v120_real_robot_and_official_baseline.sh 2>&1 | tee baseline_v120.log
```

**v1.2 工作内容：** 修正官方发布目录 `robot_trajectories/no_distraction/push_cube/`（此前使用内部别名 `without_distraction` 导致404）；限量获取并审计真实机器人 H5+base/hand 视频；下载作者发布的 Vid2Robot 权重（上限8GiB）和固定20段 PushCube EVAL 视频（≤250MB）；尝试一段 0080 **DEV ONLY** 的真正学得策略→物理模拟器闭环，明确记载阻碍、真实成功及其证据；实验 JSON/MD 自动提交 GitHub `research_reports/2026-11-09_v120/`。源视频、权重和H5不上传。

详细数值与决策：[v1.1 结果及 v1.2 任务](docs/2026-11-09_V110_RESULTS_V120_DECISIONS.md)。

**准确理解状态：** `V120_COMPLETE_IS_NOT_A_SUCCESS_FLAG` 只表示流程结束；只有 `OFFICIAL_VID2ROBOT_DEV_SMOKE_EXECUTED` + `rollout_executed:true` 才说明真正执行过学习型机器人闭环，`official_success` 才表示该次物理成功与否。一次 DEV 结果不能替代完整官方 Baseline 比较。

---

# AG-COFL | v1.1 — 官方 Baseline 优先 + 预训练 ViT GPU 验证

**研究方向继续推进，不因 v0.9 的简易 CNN 失效而放弃。** 创新问题保持：源人类物体运动效应如何在目标机器人不同操作能力下转化为物理可执行行为。CNN 只是旧的工程烟雾测试；从 v1.1 开始使用预训练 DINOv2 ViT + 时序 Transformer，固定官方 Baseline、数据协议和闭环实验。

## RTX 4090 一键执行 v1.1

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --rebase --autostash origin main
conda activate agcofl4090
set -o pipefail
bash scripts/run_v110_official_baselines_and_vit.sh 2>&1 | tee baseline_v110.log
```

这次脚本会：

1. 检查已有真实训练视频6段，以及 **RoboReel 官方 Baseline 仓库** [weiweigu1998/roboreel_models](https://github.com/weiweigu1998/roboreel_models)，固定源码提交 79a0fb0099e73c5560114753a8706276f93aefc2（无需下载全部大模型）。
2. 从官方数据集抽取最多2段 **真实机器人训练轨迹**，每段必须有 H5 动作及 base/hand MP4，硬上限300MB；缺任何文件都标记 BLOCKED。
3. **在 CUDA/4090 上**加载 `facebook/dinov2-small` 的真实预训练权重（不可用则报 BLOCKED，禁止随机初始化替代），以224×224真实RGB提取视觉特征，训练相同结构的 Full/Anchor/End 时序 Transformer 预测头，并对完整视频级 held-out 数据进行评估。仅供数据/表征 sanity check，不是完整机器人策略。
4. 将少量结果 JSON/CSV 自动归档到 `research_reports/2026-11-09_v110/`。不推送视频、H5或权重。

**输出路径** `outputs/v110_baselines/`：`official_baseline_audit.json`, `robot_data_manifest.json`, `vit_pilot/report.json`, `vit_pilot/summary.csv`, `vit_pilot/split.json`。跑完若看到 `VIT_GPU_TRAINING_COMPLETED` 和 `AUTOPUBLISH_OK`，只回复“v1.1 完成”即可，我会直接读 GitHub 结果。

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
