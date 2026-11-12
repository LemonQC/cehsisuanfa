# TRO1 — RoboReel `toys_in_drawer` 两条真实 TRAIN 视频及官方 SigLIP 特征准备

**状态：代码已通过本地离线单元测试；尚未在操作者 RTX 4090 上下载/编码，不能宣称数据已齐。**

2026-11-12 4090 预检查报告：`toys_in_drawer` checkpoint 训练任务列表包含 `toys_in_drawer`，但在 `outputs/v120_baselines/official_eval_data` 下 `TRAIN` 人类视频数 0、SigLIP 特征数 0。原官方 PushCube π0.5 46/60 的已冻结 EVAL 资产**不改动、不重新测试、不用于调参**。这不是权重加载错误。

## 代码与强约束

- `scripts/prepare_roboreel_train_drawer.py`（新）：只筛选 Hugging Face 官方 `LogosRoboticsLab/roboreel` 数据集的 `human_demos/train/no_distraction/toys_in_drawer/<0000-0079>/front.mp4`，由官方元数据选取 **前两条真实可用 TRAIN 视频**，累计传输文件大小 **最高 300 MiB**；超上限直接 BLOCKED。不扫描或获取任何 EVAL 0080–0099。
- 复用既有 `agcofl/resumable_hf_assets.get_file` 的 Hugging Face 镜像 + 原站断点续传及大小、SHA 校验；原下载器在**严格限定目标任务 TRAIN front 视频路径**上增加 allowlist，不开放任意数据/模型文件。
- 复用 pin `79a0fb0099e73c5560114753a8706276f93aefc2` 的 `third_party/roboreel_models/scripts/compute_vision_features_for_demo.py` 官方 `_PersistentEncoder`，每条视频单独子进程，产生原 schema 的 `video_demonstration_features/without_distraction/toys_in_drawer/<id>/front.npy`。无随机视觉特征，无人工打标签。
- 在编码前，必须先找到本地 `SigLIP` 参数 pickle 或已由现有 `scripts.check_pi05_siglip_cache.trusted_mirror_install` 证明完整的 OpenPI 本地镜像。不接受未证实的自动 GCS 下载约 12GB 新权重。
- 数据根沿用已跑通原官方评测的数据目录；新报告在 `outputs/roboreel_train_drawer_assets/v1`，只输出两个原始 TRAIN mp4 及 SigLIP 编码后的 npy，**不运行 PhysX、不训练模型、不修改旧报告/模型**。
- `--discover` 只做远程目录元数据读取，**不下载**；无参数仅盘点本地目录；`--download` 仅下载视频；`--features` 仅本地编码；二者可一起使用。

## 4090 命令（所有输出保存在项目目录）

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_prepare_roboreel_train_drawer.py' -v
python -m scripts.prepare_roboreel_train_drawer --discover
# 远端列表可见且两条文件总量 <= 300 MiB，再运行：
env -u DISPLAY -u WAYLAND_DISPLAY python -u -m scripts.prepare_roboreel_train_drawer --download --features
```

最后如果打印 `ROBOREEL_TRAIN_DRAWER_ASSETS_READY`，再用一个**新目录**（此前 v0/v1 的 BLOCKED 记录都保留）核查先前的视频敏感性实验入口：

```bash
python -m scripts.roboreel_video_dependency_dev --out outputs/roboreel_video_dependency_dev/v2
```

只有 `status=READY_FOR_BOUNDED_DEV` 才允许按先前协议新增 `--run --episodes 2 --max-steps 800 --out outputs/roboreel_video_dependency_dev/v2`，这 4 条才是新的真实 PhysX 测试。若视频下载或 SigLIP 编码失败，只保留日志并报告失败，绝不把空 `.npy` 或未完成视频标成 READY。

**注意**：`READY_FOR_BOUNDED_DEV` 只是数据资源已就绪，不是科研可发表结论；视频不同导致动作不同也不是视频不可替代性的证明。后续仍须对照强序列对齐和阶段感知恢复基线。
