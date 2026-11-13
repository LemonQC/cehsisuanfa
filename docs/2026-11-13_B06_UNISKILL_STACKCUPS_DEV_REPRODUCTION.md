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
