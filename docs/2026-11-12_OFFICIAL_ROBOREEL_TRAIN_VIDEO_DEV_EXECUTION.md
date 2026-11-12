# TRO1 / 2026-11-12：真正可执行的官方 RoboReel 双视频配对 DEV 实验

这次不使用 Codex，不创建 B5，不重开失败的 M1 Pick→Push。由 ChatGPT 在当前主仓库直接提交代码；操作者仅用 RTX 4090 服务器运行一条真实命令。保留老 `M0` 官方结果：RoboReel π0.5 Uniform ND PushCube 46/60、Vid2Robot 3/60；绝不重复这些正式评测或拿 heldout EVAL 0080–0099 调参。

## 目的与已核实的官方代码

实际官方主任务 ID 是 **`toys_in_drawer`**，ManiSkill 环境 `ObjectInDrawer-v1`，官方任务文本：`Pick up the panda toy and place it in the drawer, then close the drawer.`，为放入物体 + 关抽屉的长序列。旧文字“Place in Drawer & Close”是自然语言描述，不能当成代码 task ID。

上游代码 pin: [`weiweigu1998/roboreel_models` 79a0fb0](https://github.com/weiweigu1998/roboreel_models/tree/79a0fb0099e73c5560114753a8706276f93aefc2)。

- 官方 runner：`src/evaluation/demo_pi05_runner.py`，`DemoPi05Runner.load_demo_static(False, task, demo_id)` 从磁盘 `video_demonstration_features/without_distraction/<task>/<demoid>/` 加载由人类视频产生的 SigLIP tokens，再 `runner.rollout()` 做真实 PhysX 操作。不要另造模型或改变动作接口。
- 评测环境：`evaluation.utils.make_eval_env(task, False, max_episode_steps=...)`、`panda_wristcam_hd`、`pd_joint_pos`。本 DEV 使用自己冻结的 `reset(seed=310000+episode)` 状态，**不加载官方 EVAL `.pkl` bank**；只使用视频训练 ID 0000–0079，绝不使用官方 0080–0099。
- 两个不同的**同任务 TRAIN 人类示范视频**（以及已缓存的 SigLIP 特征）作为 A/B。目标机器人固定同一物理初始状态，且策略采样 RNG 对每对完全一致。记录成功差异与视频。不声称这两个 TRAIN 视频有“一个正确一个错误”的强标签：它们都可能有效，因此这是输入敏感性 DEV，而不是源视频信息因果必要性的证明。
- 容许的最小实验：2 个 seed × 2 个源视频，共 **4 条新的物理 rollout**，最多 800 步/条。仅当确有两个真实 TRAIN 人类 mp4 和相应 video feature 缓存、以及同任务权重时运行。

## 4090 一键检查（完全无物理执行，默认）

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_roboreel_video_dependency_dev.py' -v
python -m scripts.roboreel_video_dependency_dev
```

输出 `ROBOREEL_VIDEO_DEV_PREFLIGHT`，报告只写 `outputs/roboreel_video_dependency_dev/v0/preflight.json`。只有返回 `status=READY_FOR_BOUNDED_DEV` 才允许跑真实 DEV：

```bash
env -u DISPLAY -u WAYLAND_DISPLAY \
  python -u -m scripts.roboreel_video_dependency_dev --run --episodes 2 --max-steps 800
```

若当前官方 checkpoint 没包含 `toys_in_drawer`，或没有真实 TRAIN 视频+预计算特征，脚本会打印具体缺失来源、数据根路径及 `reason`，不会擅自下载 100GB 数据、更不会造随机视频然后声称人类视频可用。需要时可通过 `--config` 和 `--data-dir` 指定现有文件位置，但所有 0000–0079 的训练资产身份门槛不变。

真正执行后输出目录：`outputs/roboreel_video_dependency_dev/v0/`，每例 `episode_XX/source_YYYY/status.json` 和 `rollout.mp4`，末尾 `per_trial.csv`、`results.json`、`SUMMARY.md`。运行中断后保留部分输出且拒绝覆盖；不可把部分结果当成完整 4 条真实轨迹。报告有配置 SHA、模型路径、原始官方 runner SHA 和训练数据路径；不上传模型权重、H5 和 MP4。

**不要再次在官方 EVAL 0080–0099 上选择视频条件或调整本方法。** 同样，官方 M0 的 46/60 不应直接与新 DEV 的 4 条成功率比较。

## 数据不足时下一步

若 `NEED_TWO_GENUINE_TRAIN_VIDEOS_AND_FEATURES`：优先确定确切缓存目录，不重训 π0.5；再只准备此任务**两个实际 TRAIN 视频**和对应预计算特征。禁止用 B2 的仿真机器人示范、不同目标的 PickCube、EVAL 0080–0099 代替。资产不够不代表课题 NO-GO，只是此 DEV 暂不能执行。

若 `CHECKPOINT_NOT_TRAINED_FOR_TASK`：不得把 PushCube 专用模型硬套抽屉；先检查训练配置中的官方长序列 checkpoint 是否已缓存。只需一次确定性的检查即可，不做 B5。

本工具不是 CEPS 模型，也不保证 T-RO 可发表。它的价值是：直接在现成官方 Benchmark 上给出有实测结果的“视频输入条件改变，机器人执行是否发生变化”起点，后续方法必须由实际失败模式驱动。

## 2026-11-12: original M0 environment variables (bug fix)

The working M0 launcher used `ROBOREEL_CKPT=outputs/v120_baselines/released_checkpoints` and `ROBOREEL_DATA=outputs/v120_baselines/official_eval_data` when invoking the original RoboReel evaluator. Its saved config therefore legitimately contains `${ROBOREEL_CKPT}`. The first DEV preflight did not restore these exported variables and stopped with `UNEXPANDED_PATH_VARIABLE` **before running a single PhysX step**. The script now resolves them from the known project layout if unset, honors explicit shell overrides, and exports the resolved paths to the official runner. Unknown variables still block; missing TRAIN demos still block. The original `v0/preflight.json` is immutable; the repaired default output is `outputs/roboreel_video_dependency_dev/v1`.

Run `python -m unittest discover -s tests -p 'test_roboreel_video_dependency_dev.py' -v` followed by `python -m scripts.roboreel_video_dependency_dev`. Do **not** run `--run` until the new preflight says READY_FOR_BOUNDED_DEV.
