# TRO1 — B01/B02 真实机器人训练数据共享适配入口（2026-11-14）

## 状态

本次增加 scripts/prepare_b01_robot_train.py，供 B01 Robot-only Transformer BC 和 B02 Robot-only Diffusion Policy 共用真实 RoboReel TRAIN 机器人轨迹与双相机数据。已有 agcofl/official_robot_data.py 提供官方数据的有限下载入口；新增脚本只读项目内已有文件，不联网、不训练模型、不运行 PhysX、不使用 held-out EVAL。

**状态为 DATA_GATE_IMPLEMENTED_OFFLINE_TESTED。B01/B02 模型训练、动作归一化、RGB 时间对齐、权重、原始论文忠实复现、2-episode smoke 和 20×3 正式评测均 NOT_RUN。** B02 的训练入口仍不存在。

## 前台执行

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    conda activate pi05
    python -m unittest discover -s tests -p 'test_prepare_b01_robot_train.py' -v
    python -u scripts/prepare_b01_robot_train.py \
      --data-root outputs/v120_baselines/robot_train/robot_trajectories/no_distraction/push_cube \
      --required-demos 64 --seed 42

需要 h5py、numpy。只接受 0000–0079；不会打开 0080–0099 官方保留集。每例必须有恰好一个非 state-mode 的 H5、base_camera.mp4 与 hand_camera.mp4，H5 内至少一个 actions/action 数据集、合法 MP4 header 和非软链接的文件。记录每个文件 SHA256；固定随机种子 42，64 个有效演示分为 **51 TRAIN / 13 VAL**。

报告仅保存在项目 outputs/research_runs/b01_b02_shared_robot_train/train_manifest_<hash>.json，结果只增不改，不上传任何 H5、视频或权重。真实数据不满 64 demo 时，命令返回状态码 2 和 INSUFFICIENT_AUTHENTIC_TRAIN_TRIPLETS。不要把 2 demo smoke 写为正式 Baseline。

## 当前数据资产说明

服务器只读 find 已确认真实文件位于 `outputs/v120_baselines/robot_train/robot_trajectories/no_distraction/push_cube/{0000,0001}`；两例均包含 H5、base_camera.mp4、hand_camera.mp4。此前推断的 `official_dataset` 目录并不存在。代码默认根目录改为实际 `robot_train` 路径。目前只有 2 条真实演示，只允许 `--allow-small-dev` 工程冒烟，不满足 64 条正式训练门槛。

## 后续真实缺口

1. 核对每个 demo 原始官方来源、文件哈希、H5 action convention、RGB/action 时间对齐。
2. 相同训练/验证划分、相机、机器人本体状态及 action space 下分别实现 B01 Transformer BC 和 B02 Diffusion Policy，标 REIMPL_ON_ROBOREEL。
3. 前台运行真实 RoboReel PhysX 双场景 smoke，再完成严格 20 EVAL × 3 seeds 的正式评测。
4. 继续冻结 B03 PushCube ND 官方 46/60、B05 PushCube ND 官方 3/60；B06 UniSkill StackCups 0/4 为 TRAIN-video DEV。
5. 尊重单独的 G0_BLOCKED 科学门槛：标准 PushCube 能力对比并不证明跨行为视频信息必要性。

本次只在离线合成测试夹具上跑通 **4/4 单元测试**，用户服务器上的真实数据审计和模型训练未运行。禁止 nohup、后台任务及 tmux。
