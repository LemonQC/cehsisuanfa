# TRO1 · 2026-11-13 · No-PartNet RoboReel StackCups 替代方案与执行协议

## 核查后的明确选择

操作者确认 PartNet-Mobility 相关数据目前不对其中国使用条件开放。因此，本项目**不再要求任何 PartNet-Mobility 授权模型、下载器、数据包或转存镜像**。此前 `toys_in_drawer` 的 v0–v3 实验记录保留作为不可覆盖的工程负例，停止该任务后续试验。

选定 **RoboReel 官方 `StackCups-v1`（checkpoint task ID: `stack_cups`）**。这是三个杯子按照 dark_blue（底）→ red（中）→ light_blue（顶）堆叠的长序列同机器人操作任务。已核查作者固定子模块 `gupta-anmol99/roboreel@afaf4a03`：`src/envs/stack_cups.py` 使用仓库随附的 `assets/scenes/stack_cups/{dark_blue_cup,red_cup,light_blue_cup}/{visual.glb,collision_vhacd.obj,orientation_result.json}`，**不调用 PartNet-Mobility**。本地已报告的 `vid_cond_pi05_uniform/ckpt` 训练任务包含 `stack_cups`，不需要新的 π0.5 权重或训练。

这不是同任务实测成功的宣称；新场景还没有执行物理 rollouts。

## 改动原则

1. 在成功运行过的 `scripts.prepare_roboreel_train_drawer` 上增加显式 `--task stack_cups`：只获取同任务的真实 human TRAIN `0000..0079/front.mp4`，选最前两条；最多 2 个视频、合计 ≤ 300 MiB，Hugging Face 官方/现有镜像，原始字节长度和 SHA256 验证。保留 `--task toys_in_drawer` 向后兼容；不同任务输出到不同目录，不能混视频和标签。
2. 复用已验证的官方 SigLIP 权重与 `frames-per-batch=2`，单视频单子进程，不重新下载 pi0.5_base，不重新计算旧视频特征。
3. `scripts.roboreel_video_dependency_dev --task stack_cups` 调用官方 `StackCups-v1`，启动前校验三杯共 9 个原始资源文件，**无 PartNet 访问**。仅两组固定机器人初态 × 两条不同同任务 HUMAN TRAIN 视频，总共 4 条 DEV rollouts，最大 800 步，独立 `stack_v1` 目录。
4. 保留原 M0 PushCube 官方 π0.5 46/60 和 Vid2Robot 3/60 测试证据，不重跑 EVAL 0080..0099，不拿原来 `toys_in_drawer` 视频冒充 `stack_cups` 演示。实验是视频输入敏感性 DEV，**不是已验证 CEPS 算法，也不是 T-RO 新颖性或人类视频因果必要性结论**。

## RTX 4090 操作步骤

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main

python -m unittest discover -s tests -p 'test_roboreel_stack_no_partnet.py' -v
python -m unittest discover -s tests -p 'test_prepare_roboreel_train_drawer.py' -v
python -m unittest discover -s tests -p 'test_roboreel_video_dependency_dev.py' -v

# 只读取公开的 TRAIN 视频文件大小，不下载：
python -m scripts.prepare_roboreel_train_drawer --task stack_cups --discover
```

确认 `ROBOREEL_TRAIN_STACK_CUPS_PLAN` 两个原始 TRAIN 视频、总量 ≤300MiB 后：
```bash
env -u DISPLAY -u WAYLAND_DISPLAY python -u -m scripts.prepare_roboreel_train_drawer \
  --task stack_cups --download --features --frames-per-batch 2

python -m scripts.roboreel_video_dependency_dev \
  --task stack_cups --out outputs/roboreel_video_dependency_dev/stack_v1
```

仅当前述预检查返回 `READY_FOR_BOUNDED_DEV`，才允许执行：
```bash
env -u DISPLAY -u WAYLAND_DISPLAY python -u -m scripts.roboreel_video_dependency_dev \
  --task stack_cups --run --episodes 2 --max-steps 800 \
  --out outputs/roboreel_video_dependency_dev/stack_v1
```

若元数据不存在、模型现场资产缺失或 GPU 资源不足，只记录准确错误；禁止下载 PartNet 或使用不明来源的替代碰撞几何。若真实仿真出错，保留已生成的 `manifest.json` 与部分结果，不直接重跑同一目录。

## 学术检验界限

StackCups 已有固定顺序和明确阶段依赖，更适合作为“观测证据与视频时序对齐”的研究对象。但两段同任务视频均可能给出同一有效策略，A/B 产生的动作差异**不等于视频有独有信息**。之后必须跟等信息视频序列对齐、规则阶段控制、视觉事件检测与恢复基线公平对比。不能基于两组 DEV 即宣称 T-RO GO。

## 2026-11-13 StackCups TRAIN metadata 与测试修复回执

真实服务器 `pi05` 已执行 StackCups 的只读 Hugging Face 镜像发现：

- `stack_cups/0000/front.mp4`: **11,602,881 bytes**
- `stack_cups/0001/front.mp4`: **14,569,453 bytes**
- 两条总计 **26,172,334 bytes**，≤300 MiB。
- 原元数据审计显示 `no_eval_data_or_robot_actions=true`、`download_executed=false`，因此视频尚未下载，StackCups SigLIP 特征尚未生成，物理实验 `NOT_RUN`。

原 `test_openpi_config_failure_does_not_create_manifest` 的旧样例仍传入 `toys_in_drawer`，因此在新的柜体资产检查上提前失败，没有实际测试预期的 OpenPI 缺失场景；这是测试夹具与新分支不一致，而非 StackCups 模型错误。修复：该样例转为 `stack_cups`，创建九个明确标注为虚构的纯文件测试资源，检验缺失 OpenPI 缓存时**不得创建任何实测 manifest**；同时实际 `run_dev` 的 StackCups 分支不再导入 PartNet 资产准备模块。提交 `7614c1e`；服务器重新测试仍待验证。

下一执行门槛：先通过 `test_roboreel_video_dependency_dev.py`（21 项）；再执行 `--task stack_cups --download --features --frames-per-batch 2`；随后在独立输出目录对 `--task stack_cups` 进行只读 preflight。未见 `READY_FOR_BOUNDED_DEV` 前不得启动新的机器人物理操作。
