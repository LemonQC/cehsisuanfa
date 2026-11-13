# B06 UniSkill：RoboReel StackCups 首轮复现执行单

**日期：2026-11-13｜状态：代码与自动化执行入口已提交；服务器端复现尚待执行**

## 1. 为什么先跑 B06

最新 L0 资源清单确认：B06 的 RoboReel 官方训练入口和评测入口已在本机 `third_party/roboreel_models` 中；但这只证明代码路径存在，不证明权重、IDM、特征和运行时已就绪。

B06 适合作为下一项真实复现：

- RoboReel 官方模型仓库提供 `uniskill_policy` 的 ND checkpoint 下载入口：`scripts/download_checkpoints.py --method uniskill_policy --condition nd`。
- UniSkill 作者的 Hugging Face 仓库当前可取得 `UniSkill_final_weight/idm.pth`（约 80.3 MB）。只下载这个 IDM 文件，不下载整个约 3.52 GB Diffusers 仓库。
- RoboReel 官方 `ENVIRONMENTS.md` 为 UniSkill 单独规定 Python 3.12 / PyTorch 2.10 环境；不要把它硬装进现有 `pi05` JAX 环境。
- StackCups 使用 RoboReel 自带场景网格，不依赖 PartNet-Mobility；本轮不触碰抽屉资产。

上游依据：

- RoboReel model code: https://github.com/weiweigu1998/roboreel_models
- RoboReel UniSkill checkpoint: https://huggingface.co/LogosRoboticsLab/roboreel_models
- UniSkill author code: https://github.com/KimHanjung/UniSkill
- UniSkill author IDM file: https://huggingface.co/HanjungKim/UniSkill/tree/main/UniSkill_final_weight
- UniSkill paper: https://arxiv.org/abs/2505.08787

## 2. 本轮到底复现什么

这是 **B06 官方权重 + 官方 IDM + 官方特征提取流程 + RoboReel 官方 UniSkill runner** 的 bounded DEV。不是重训，也不是自制 UniSkill 替代品。

| 项目 | 冻结设置 |
|---|---|
| 任务 | `stack_cups` / `StackCups-v1` |
| 来源视频 | RoboReel TRAIN / no-distraction / IDs 0000、0001 / front camera |
| 源技能 | 官方 IDM，`skill_interval=20`，skill dim 64 |
| policy | RoboReel 官方 `nd/uniskill_policy/ckpt.pth` |
| IDM | UniSkill 作者 `idm.pth`，不训练、不微调 |
| 机器人 | RoboReel 官方 Panda 环境，8D action |
| 配对 | 2 个 episode × 2 个 source video = 4 次真实 PhysX rollout |
| 初始状态 | 每个 episode 内两次 rollout 完全复用同一 `state_dict` |
| 随机性 | 每个 episode 内 source pair 使用相同 Torch RNG seed，两个 episode 交换 source 顺序 |
| 步数 | 每条最多 800 |
| 视频 | 默认保存每条 rollout MP4 |
| 输出 | `outputs/research_runs/b06_uniskill_stackcups_dev_v1/` |
| 远程证据 | 自动发布脱敏 JSON 至 `tro1-experiment-evidence` |

**解释边界：** IDs 0000/0001 是 TRAIN 视频，不能作为正式测试集。该轮只用于证明 B06 官方推理链能否在我们的 RoboReel StackCups 环境执行，并与已有 π0.5 StackCups 0/4 DEV 做同条件工程诊断。不能写入论文正式 success-rate 表，不能替代官方 held-out IDs 0080–0099 × 3 pass。

## 3. 自动化脚本做什么

`scripts/setup_b06_uniskill_stackcups.sh` 一次完成：

1. 验证 `third_party/roboreel_models` 是锁定的官方 commit `79a0fb0099e73c5560114753a8706276f93aefc2`。
2. 只初始化 UniSkill 和 UniSkill-Policy 两个必要上游 submodule。
3. 建立独立 `uniskill` Python 3.12 环境，按 RoboReel 官方 `ENVIRONMENTS.md` 安装依赖；不修改 `pi05`。
4. 所有 HF cache、checkpoint、IDM 都放在项目盘 `outputs/v120_baselines/`。
5. 先通过 Hugging Face 元数据核算待下载 checkpoint 大小和磁盘剩余空间；空间不足时在下载前停止。
6. 只下载 B06 ND policy，不下载 RoboReel 全部约 50 GB checkpoints。
7. 只下载 UniSkill 的 `idm.pth`，不下载整个 3.52 GB Diffusers repo。
8. 从现有两条 StackCups TRAIN 视频提取 skill，batch size 固定为 8，适配单张 24 GB RTX 4090。
9. 通过本地 preflight 后，执行 4 条配对 PhysX rollout，并保存 MP4、action trace、success trace、SHA256。
10. 将脱敏结果自动推送至独立 GitHub 证据分支；原始 MP4、权重、HF token 不上传。

## 4. 服务器执行

先确认工作区无需要保留的未提交修改，再更新主分支：

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git status --short
git pull --ff-only origin main
```

然后只需运行：

```bash
bash scripts/setup_b06_uniskill_stackcups.sh
```

正常结束应出现：

```text
ROBOREEL_B06_UNISKILL_PREFLIGHT ... "status": "READY_FOR_DEV"
ROBOREEL_B06_UNISKILL_TRIAL ...
ROBOREEL_B06_UNISKILL_RESULT ...
ROBOREEL_B06_UNISKILL_PUBLISHED ...
```

若任何资源、版本、checkpoint、IDM、显存或特征不满足，脚本应在物理 rollout 前阻断；请把 `B06_BLOCKED` 或 traceback 原样回传，不要自行跳过门槛。

## 5. 结果出来后如何决策

- **4 条 rollout 正常结束：** 先审查视频、动作轨迹和真实成功标志，再准备官方 held-out IDs 0080–0099 的 StackCups 视频及 UniSkill skills，进入官方 `last20 × 3` 评价。
- **出现 0/4 或低成功：** 先诊断 checkpoint/task/camera/action normalization 是否与官方训练配置一致；不擅自改算法或改任务容差。
- **推理错误或环境初始化失败：** 标为复现阻塞，不把它记作算法失败率。
- **任何结果：** 都不能据 4 条 DEV rollout 得出方法优劣或 T-RO 创新结论。

## 6. 本地代码验证

新增文件：

- `scripts/roboreel_uniskill_video_dev.py`
- `scripts/setup_b06_uniskill_stackcups.sh`
- `tests/test_roboreel_uniskill_video_dev.py`

本地纯 CPU 单元测试：**7/7 通过**；shell 脚本 `bash -n` 语法检查通过。它们尚不代表用户服务器已完成权重下载、特征提取或 PhysX rollout。

## 7. B06 v2 安全复现修补（2026-11-13）

**下方修补后的单命令将取代旧脚本的安装逻辑；尚未在用户服务器执行。**

1. **项目盘唯一落盘**：以前按名字 `conda create -n uniskill` 默认可能将大型 CUDA 环境安装在其他磁盘；现在使用 `conda create -p outputs/v120_baselines/conda_envs/uniskill`，并同时限定 `CONDA_PKGS_DIRS`、`PIP_CACHE_DIR`、`TMPDIR`、`XDG_CACHE_HOME`、`TORCH_HOME`、`HF_HOME` 为项目盘目录，不改变已有 `pi05`。
2. **只提取两条真人视频**：官方 extractor 会遍历给定数据根目录下的全部视频；以前直接把 `official_eval_data` 作为根目录，会额外处理数十个训练样本。现在建立 `outputs/v120_baselines/b06_train_subset/` 两条只读软链接（ID 0000/0001）再调用官方 extractor，源文件不复制、不修改、不触及 EVAL。
3. **权重版本固定**：第一次从 Hugging Face 官方元数据冻结 RoboReel policy / UniSkill IDM 的 repo commit SHA 到 `outputs/v120_baselines/b06_upstream_hf_lock_v1.json`，下载明确指定 revisions，二次执行不会默默换权重；同时检查下载缺口和项目盘剩余空间。
4. **中断保护**：任何 setup 失败都会生成脱敏 `outputs/research_runs/b06_blocked/b06_blocked_<hash>.json`，尽力自动发布到 `tro1-experiment-evidence/evidence/runs/b06_blocked_<hash>/evidence.json`；若网络中断，至少保留本地文件，不包含 HF 令牌、原始视频、路径、traceback。
5. **防止重跑**：有完整 `outputs/research_runs/b06_uniskill_stackcups_dev_v1/results.json` 时，重新执行同一脚本只做真实性验证并**补传现有结果**，不会重新下载模型或执行新的 PhysX；若仅有部分物理结果但没有完整 `results.json`，直接阻断，先人工审查，不得自动重跑覆盖。
6. **离线补传命令**：如果完整物理结果已保存但只是 GitHub 上传失败，在项目根目录运行 `python -m scripts.roboreel_uniskill_video_dev --publish-existing`，不需要 GPU 或再装模型；它会核对每条轨迹与视频 SHA256 后才发送脱敏 JSON。
7. **失败补传命令**：如果网络恢复但 `B06_BLOCKED_LOCAL_ONLY_PUBLISH_FAILED`，可针对对应 `stage`/`exit_code` 重跑 `python -m scripts.b06_failure_record --stage <stage> --exit-code <code> --publish`。不会重复物理实验。

新增：`scripts/b06_failure_record.py` 和 `tests/test_b06_safe_restart.py`。**本地合并测试 12/12 passed，`bash -n`、`py_compile` passed**（测试环境不具备用户 GPU/数据，不代表成功复现）。

执行顺序严格按 Baseline 串行：**B06 DEV → B06 held-out 正式评测（成功后）→ B07 或 B04 → B09/B11/B12 → B01/B02**；每个算法必须留下作者版/适配版分类、真实可运行状态、可复核证据与停止原因后才前进。

## 8. 真实服务器最新阻塞及单点修复（2026-11-13）

- 证据：`tro1-experiment-evidence/evidence/runs/b06_blocked_102d4d611a3280de/evidence.json`；`stage=author_source_submodules`，`exit_code=1`，本次尚未进入任何 PhysX。不是算法成功率数据。
- 已验证 RoboReel 官方两个 Git submodule 引用的确切 gitlink 提交：UniSkill `bd9843b3168f889951bc117c2fc70be59590f8f9`、UniSkill-Policy `02ab999609b9e698a197a6786a7edebdb69faac5`；这两个提交在原作者 GitHub 仓库可读取。
- 改进 `scripts/setup_b06_uniskill_stackcups.sh`：对每一子模块首先识别已正确安装的相同 SHA（无需重复下载）；否则尝试 RoboReel 官方 fork；官方 fork 下载失败时再尝试原作者 `KimHanjung/UniSkill` 和 `kang-jaehyun/UniSkill-Policy`。备用链接只写入本地 Git config，不修改受版本控制的 `.gitmodules` 或 root commit。
- 下载后强制校验固定 SHA 与官方入口文件，错误或不一致立即停止，不启动模型/物理评测。该修复为**待服务器验证**，不能声称真实阻塞已解除。
- 继续采用前台执行；**不使用 nohup、tmux、& 等后台运行**：

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
bash scripts/setup_b06_uniskill_stackcups.sh
```

如果重试后 Git 仍失败，本地终端会显示哪一个子模块失败；停止在此一步，不启动 B07。成功后下一步仍是 B06 自己的权重/环境/真实 4-rollout 门槛。

## 9. 单独修复 Git `Needed a single revision`（已提交，服务器待验证）

2026-11-13 用户实际前台日志：`UniSkill` 指定提交 `bd9843b...` 已经 `READY`；对 `UniSkill-Policy` 执行官方地址和作者地址的 `git submodule update` 都返回中文 `需要一个单独的版本`。自动证据 `b06_blocked_102d4d611a3280de` 仍记录 `author_source_submodules`，尚无 PhysX。

**原因定位范围**：不一定是网络或子模块未公开；更符合已经创建子模块 Git 工作树、但 `HEAD` 无法解析到确定提交的状态。原脚本只修改了父仓库 `.git/config` 的备用 URL，不能修复子模块的遗留内部仓库或 unborn HEAD。

已新增 `scripts/b06_recover_submodule_git.sh`：仅当正常 `submodule update` 失败时调用。检查目标目录确实拥有独立 Git 仓库、拒绝修改任何有未提交改动或未跟踪文件的目录、拒绝覆盖一个不同的已检出提交；对空的 unborn Git HEAD，从原作者仓库获取**与 RoboReel gitlink 完全相同**的 SHA，并用 `checkout --detach` 检出它。若 SHA-only fetch 被服务端拒绝，则只获取原作者 `main` ref 再核验 SHA。不会清理/删除目录、修改 `.gitmodules`、重新克隆整个 RoboReel、影响已有实验。

子模块固定提交（已在 GitHub API 验证存在）：

- `UniSkill`: `bd9843b3168f889951bc117c2fc70be59590f8f9`
- `UniSkill-Policy`: `02ab999609b9e698a197a6786a7edebdb69faac5`

新增纯离线 Git 模拟回归测试 `tests/test_b06_submodule_recovery.py`（待服务器执行）；这些测试不下载权重、不运行模拟器。新补丁在作者服务器上的完成与否必须以真实 `B06_SUBMODULE_READY` 输出为准。

**前台执行，不使用 nohup/后台**：

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_b06_submodule_recovery.py' -v
bash scripts/setup_b06_uniskill_stackcups.sh
```

若输出 `B06_RECOVER_BLOCKED: target is not an independent Git working tree`，说明剩余目录并不是完整的独立子模块 Git 仓库；修复脚本会保留原有内容而不是冒险覆盖。须根据 Git 状态再做针对性处理。

## 10. B06 第二次真实阻塞：干净但不同的 UniSkill-Policy HEAD；离线验证后修复

- 用户前台观察：新添的 Git 恢复测试 `Ran 6 tests ... FAILED (failures=2)`；`UniSkill` 已 READY，但 `UniSkill-Policy` 的 `git submodule update` 报 `需要一个单独的版本`，安全恢复报告 `submodule HEAD points to another commit`，实际运行仍是 `author_source_submodules` 阻塞，未开展 B06 PhysX。
- 错误原因分为两部分：① 原测试 fixture `git clone --no-checkout` 后让 HEAD 指向不存在的 ref，却未清空仍含有条目的 Git index，导致模拟出来的目录被正确识别为 dirty（两项测试失败）；② 实际服务器 `UniSkill-Policy` 已有**不同提交的 HEAD**，第一版恢复器无论这个仓库是否干净都强行拒绝。
- 已更正离线 fixture：构造 unborn HEAD 时通过 `git read-tree --empty` 模拟真正的无 checkout 状态；修复恢复器：若不同 HEAD 但 worktree/index 完全干净，则保存 `refs/b06-recovery/preexisting-<原HEAD>` 备份引用，再从原作者库获取要求的准确 gitlink SHA，完成 `checkout --detach`，严格核对脚本入口；如有未提交修改或 untracked 文件仍然立刻停止，不删除和重置任何代码。
- 修复后已在独立本地 Git 模拟仓库（无网络、无 CUDA、无用户数据）执行 **7/7 tests PASS**；`bash -n` 通过。服务器尚未验证本次补丁。

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_b06_submodule_recovery.py' -v
bash scripts/setup_b06_uniskill_stackcups.sh
```

若再次卡住，保留 Git 输出及真实目录内容；不得执行 `git reset --hard`、`git clean -fd` 或清除 `.git/modules`。本轮仍只复现 **B06**。

## 11. B06 实际安装冲突修复（2026-11-13）

用户服务器安装阶段准确回执：`uniskill-policy 0.1.0` 在原作者 Python package metadata 中声明 torch==2.0.1、torchvision==0.15.2、transformers==4.36.2，现有独立 B06 环境是 RoboReel 官方 `ENVIRONMENTS.md` 使用的 torch 2.10.0、torchvision 0.25.0、transformers 5.3.0；还报告 `decord 0.6.0 is not supported on this platform`。`pip check` 以 exit=1 终止在 `install_uniskill_deps`。本次仍未开展 PhysX。

**修复不是降级 torch**：这会与 RoboReel 官方现有 B06 复现环境及 Python 3.12 冲突。保留独立 PyTorch 2.10 环境，调整以下仅 B06 代码：

- `setup_b06_uniskill_stackcups.sh` 不再要求 `decord==0.6.0`；UniSkill 官方 human-video extractor 使用 `imageio.v2` / ffmpeg，UniSkillRunner 与其 wrapper 均未 import decord。
- `roboreel_uniskill_video_dev.py` 不再把 decord 作为 preflight 必选 runtime package。
- 新增 `scripts/b06_dependency_gate.py`：严格校验 torch/torchvision/transformers 的 RoboReel 目标版本；记录但不阻断 **uniskill-policy 0.1.0 自身的旧元数据声明**及上述精确 decord 非必需警告；与 B06 无关的其他依赖冲突仍阻断；保留 `torch,torchvision,transformers,diffusers,mani_skill,sapien,imageio` 的真实 import smoke。
- 新增 `tests/test_b06_dependency_gate.py`，覆盖服务器四条实际警告、旧 package 可选字段、其他包不兼容、空错误、健康 pip check。此测试脚本已提交，**服务器尚未执行**。

**执行方式（全前台，不删环境、不降级版本）**：

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_b06_dependency_gate.py' -v
bash scripts/setup_b06_uniskill_stackcups.sh
```

若仍有新的 pip-check 错误，将以明确的包名/版本打印 `B06_DEPENDENCY_BLOCKED`；请保留原始输出再定点修复，不跳过环境/模型初始化 smoke。只有实际 `ROBOREEL_B06_UNISKILL_TRIAL` 才能算执行了 PhysX。

## 12. 最新真实阻塞：HF 镜像 503 + read timeout；B06 checkpoint 分文件续传修复

用户服务器已通过 B06 依赖阶段，但官方 RoboReel `snapshot_download()` 在并行获取 `nd/uniskill_policy/config.json` 时，`hf-mirror.com` 返回 **HTTP 503 Service Unavailable** 并发生 `httpx.ReadTimeout`。同时日志报告约 **740 MB** 权重传输进度，不能据此断言权重最终已完整落盘。自动阻塞回执：`evidence/runs/b06_blocked_2ef36e3d281eb045/evidence.json`，`stage=download_pinned_official_uniskill_policy`，本次仍未运行物理仿真。

**处理原则：绝不清空约 740 MB 的已有权重、HF 缓存、Conda 环境；不修改官方模型 revision，也不重新训练。**

- 新增 `scripts/b06_download_pinned_policy.py`，只读取此前已写入项目盘的 `b06_upstream_hf_lock_v1.json`。验证 `LogosRoboticsLab/roboreel_models`、40 位固定 commit、限定 `nd/uniskill_policy/` 的文件名和官方文件大小。
- 改为**先下载两个小 JSON，再处理大权重文件**；已完整保存且大小准确、JSON 可解析的文件直接显示 `B06_POLICY_FILE_REUSED` 并跳过。HF 的临时缓存与 `.incomplete` 文件原样保留，交给官方 `hf_hub_download` 尝试续传。遇到已有最终文件但大小或 JSON 无法验证时停止，不覆盖它。
- 对每个缺失文件单独调用官方 Hugging Face Hub 下载函数。先使用现有镜像 (`https://hf-mirror.com`)，失败后尝试 `https://huggingface.co`。每个端点最多两次尝试，`HF_HUB_DOWNLOAD_TIMEOUT` 默认 180 秒、元数据超时默认 30 秒。网络都不可用时保留文件及错误证据，不重复整个 snapshot。
- 上游 RoboReel 的作者权重、模型结构、训练参数和评测代码**都没有改变**；区别仅是下载传输策略从并发整个快照改为按清单逐文件获取。
- `tests/test_b06_download_pinned_policy.py` 已在本地纯 CPU 环境 **9/9 通过**，覆盖已完成的权重、镜像故障转官方、固定 SHA、错误 JSON、错误文件大小、全部端点失败和路径约束；用户服务器真实下载仍待验证。

**继续前台执行**（无需重装、不要删除 `.cache`）：

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_b06_download_pinned_policy.py' -v
bash scripts/setup_b06_uniskill_stackcups.sh
```

若输出 `B06_POLICY_DOWNLOAD_COMPLETE`，说明政策权重已经逐项完成大小和 JSON 校验；之后将继续官方 IDM 下载、提取 TRAIN 技能、再进行 bounded DEV 的真实 PhysX。只有观察到 `ROBOREEL_B06_UNISKILL_TRIAL`，才算进入了物理评测。不得把下载成功标成 baseline 复现成功。

## 13. B06 又一次真实阻塞：错误要求 UniSkill Policy train_config.json（2026-11-13）

本次前台日志：`Ran 9 tests ... OK`；两个固定的 UniSkill 子模块均 READY；HF lock revisions 为 Policy `ef21150c02b2685347fe3e7c8ba5a2e313ff1b7e`、IDM `b8d6cbad5f0d9634c8da889d9b9e4e6fcffd59d9`；`missing_download_bytes=80348072`、项目盘剩余约 3 TB；但下载器校验抛出 `B06_REQUIRED_FILES_NOT_IN_LOCK`，产生 `b06_blocked_73934173abc2ebfc`。本轮没有进入 PhysX。

**根因已定位为我们此前复现脚本的错误假设，非模型下载再次失败。** RoboReel 上游 `src/evaluation/uniskill_runner.py` 顶部明确声明：UniSkill 不是按 LfO 视频模型训练脚本训练的，**不存在随 Policy checkpoint 附带的 `train_config.json`**；官方 eval config `configs/eval/nd/uniskill_policy.json` 直接指向 `nd/uniskill_policy/ckpt.pth`，而策略架构来自锁定源码的 `src/external_models/UniSkill-Policy/configs/uniskill_policy.json`；checkpoint 本身含 `shape_metadata` 和动作归一化信息。

因此统一修复三处：

1. `scripts/b06_download_pinned_policy.py` 只要求官方确切的 `nd/uniskill_policy/ckpt.pth`，不再要求不存在的 `train_config.json`；其他 lock revision、目录越界、大小/JSON 校验不变。
2. `scripts/setup_b06_uniskill_stackcups.sh` 不再在下载后硬性检查 Policy 同级的 `train_config.json`，改为检查锁定 UniSkill-Policy 源码下的 `configs/uniskill_policy.json`。
3. `scripts/roboreel_uniskill_video_dev.py` 移除不正确的 preflight training-config gate，增加真实策略架构配置必选检查。

回归测试：`tests/test_b06_download_pinned_policy.py` 新增没有 `train_config.json` 的真实目录结构测试、缺少真正 checkpoint 的失败测试；`tests/test_b06_no_train_config_preflight.py` 新增无独立 train_config、但真实必需文件齐全时允许进入特征提取的模拟校验。**这些新增测试在用户服务器尚未执行，不能视为 GPU 复现。**

`missing_download_bytes=80348072` 强烈提示已有 Policy 文件均按原始 lock 大小存在，可能只缺 IDM；但在逐文件核查、加载模型成功之前，**不得将 Policy 标记为已验证完成**。

前台执行，不清空权重/缓存、不重装环境：

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_b06_download_pinned_policy.py' -v
python -m unittest discover -s tests -p 'test_b06_no_train_config_preflight.py' -v
bash scripts/setup_b06_uniskill_stackcups.sh
```

期望先看到 `B06_POLICY_FILE_REUSED ...` 和 `B06_POLICY_DOWNLOAD_COMPLETE ...`，随后下载缺少的 IDM，再做模型真实导入/特征提取和 PhysX。若官方文件列表不包含 `ckpt.pth`，脚本会输出精确文件名列表并立即停止；不能凭空改名。
