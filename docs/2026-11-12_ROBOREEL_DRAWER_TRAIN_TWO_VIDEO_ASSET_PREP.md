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

## 2026-11-12 22:16 真实服务器下载成功；原版全视频 JAX 编码 OOM；分帧修复

**用户服务器已实测并报告**：两条官方 `toys_in_drawer` TRAIN `front.mp4` 下载完成，0 号 `9,177,715` 字节 SHA256 `e798feca10cb6b2d4fefe1f6d1bd82312b7658c0a89a2d678aae9a6e958091eb`，1 号 `8,078,802` 字节 SHA256 `75442bf1c7514532f662b4dfb79e25d7fa7aa7ddc3889e20b7d53d11536eecb3`。Hugging Face hub manager 返回 `LocalEntryNotFoundError` 后，已有工具使用 `https://hf-mirror.com` 的 HTTPS Range 续传，完成文件大小和 SHA 校验。**不需要重新下载视频。**

原官方 `scripts/compute_vision_features_for_demo.py` 中 `_PersistentEncoder.encode_demo_camera` 把所有 `T` 帧组成完整的 GPU/JAX Tensor 再执行 `resize_with_pad(...,224,224)`。在本机环境中，该步骤尝试分配 **6.03 GiB** 时报告 `RESOURCE_EXHAUSTED`，未写出 `front.npy`。后续 DEV 预检查因此准确报告 `video=2, features=0, usable=0`，未执行新 PhysX。

**修复**：新增本仓库 `agcofl/official_siglip_batched.py`，保留原版每帧 `uint8 -> float32/255*2-1`、原 `resize_with_pad`、同一个已经载入官方模型参数的 `_siglip_call`、原 `pool_tokens_to_size` 的 64/16/4 空间 token 池化、原 `_pe` 三档位置编码和 `state_emb/timesteps`，唯一区别为沿视频的 T 维切成连续 2 帧 JAX microbatches，分批 `device_get` 后按原时间顺序拼接。原作者 `_atomic_save` 仍负责原 schema 单个 `front.npy` 的原子落盘；绝不使用假视频、重新初始化未训练权重或修改模型计算。仅修改 **特征提取执行方式**，不是自创新算法。

已新增 `tests/test_official_siglip_batched.py`，通过假 encoder 的 CPU 确定性对照确认：微批 `1/2/4` 与同样公式的整批结果在七个字典键下逐元素一致，原有 TRAIN 下载/校验等 13 个测试未变。**尚未在服务器真实 JAX 上完成 SigLIP 特征计算或最终配对仿真。** 真实 JAX 数值允许因为 batch-size 编译核函数略有浮点差异，不能承诺所有参数都逐 bit 相同；仍应满足相同模型、视频帧、pool 和输出 schema。

恢复时仅运行（纯本地视频编码，不访问远端，不下载模型）：

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_prepare_roboreel_train_drawer.py' -v
python -m unittest discover -s tests -p 'test_official_siglip_batched.py' -v
env -u DISPLAY -u WAYLAND_DISPLAY \
  python -u -m scripts.prepare_roboreel_train_drawer --features --frames-per-batch 2
```

程序会跳过已下载的真实视频，复用现有 `outputs/roboreel_train_drawer_assets/v1/selection.json`，依次编码 0000 和 0001 并保存完成文件。若**仍**出现 JAX OOM，仅在保留原日志的条件下重试 `--features --frames-per-batch 1`（严格不覆盖已完成特征），不要反复运行原全 T GPU 管线，也不要重新下载 17MB 视频。两条真实 `front.npy` 均就绪后，再执行 `python -m scripts.roboreel_video_dependency_dev --out outputs/roboreel_video_dependency_dev/v2`；只有输出 `READY_FOR_BOUNDED_DEV` 才考虑四条新 PhysX DEV。

**科学边界**：这次已取得可复核的真实 TRAIN 视频资产，是实验准备进展；它不是视频事件理解、提升长期操作成功率或 T-RO 创新性的证据。
