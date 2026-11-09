# 2026-11-09 | v1.2 实验实测复盘 → v1.2.1 官方模型传输与闭环修复

## 已获得的真实实验事实（GitHub v1.2, commit f93d380）

- 真实官方机器人训练数据：PushCube (no_distraction) 2/2 H5 + base_camera/hand_camera MP4 获取成功。两个 H5 的动作张量分别为 `[138,8]` 与 `[136,8]`；Panda articulation `[139,31]` 与 `[137,31]`，关节状态截取及图像解码检查通过。这个 **TRAIN 数据真实性** 是有价值的成果，但不是学习策略成功率。
- 作者官方 Vid2Robot ND 预训练 checkpoint 元数据存在，`nd/vid2robot/ckpt.pt` 和 `train_config.json` 预算共 `2,368,874,616` bytes。作者官方 EVAL PushCube clean front 相机 0080–0099 共20段 `97,448,129` bytes。
- 两类 HF 下载均返回 `LocalEntryNotFoundError`。关键细节：实际被列为 `PREFLIGHT_BLOCKED` 的文件是 `train_config.json` 与 `0080/front.mp4`；日志**尚不能证明权重文件是否完整/合法**，不能宣称已加载。
- 官方 Vid2Robot `rollout_executed: false`，`official_success: null`。目前没有任何学习型机器人官方物理成功率，不能填写论文结果表，也不能把已有的privileged-scripted PushCube 1/1混在一张结果表中。

## 当前核心瓶颈和修复

1. 数据路径已解决：`robot_trajectories/no_distraction/push_cube/`，验证结果已经在 GitHub `research_reports/2026-11-09_v120/`。
2. 阶段二缺官方发布文件：修复 `agcofl/official_vid2robot_assets.py` 的获取方式，使用 `agcofl/resumable_hf_assets.py`，明确 `hf_hub_download(endpoint=https://hf-mirror.com)`，失败后仅允许官方 `https://huggingface.co` 和 `https://hf-mirror.com` 的 `/resolve/main/` HTTP Range 尝试；验证官方元数据字节数，逐文件计算SHA256，保留 `.part` 续传进度，避免每次重下2.37GB。
3. 如果前期原始 0080 front MP4 在 `third_party/roboreel/`，只在官方远端字节数一致时 hardlink/复制到这次官方 EVAL 目录，**不是把EVAL数据用于训练**。
4. 已固定的旧测试/方法都保留；只跑 `tests/test_v120_official_assets.py` 和 `tests/test_v121_resumable_hf.py` 确认路径、预算、复用和断点传输；不再重训ViT/下载2段已验证机器人 H5。
5. 资产齐全后，仅做官方已发布 Vid2Robot checkpoint + 官方 evaluator + ManiSkill 的 **单个固定 EVAL 0080 的集成 smoke**。严格标注DEV ONLY；只用于确认链路，不用于参数调整或论文正式分数。最终仍要对全20段和多seed按官方协议评测。

## 下一次服务器执行

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --rebase --autostash origin main
conda activate agcofl4090
set -o pipefail
bash scripts/run_v121_official_baseline_resume.sh 2>&1 | tee baseline_v121.log
```

目标输出：`outputs/v120_baselines/official_vid2robot_assets.json` 和 `outputs/v120_baselines/official_vid2robot_smoke/smoke_result.json`，小体积报告自动存入 `research_reports/2026-11-09_v121/`。这套代码已提交但尚未在用户4090设备上执行；不可预称下载或物理评测成功。

## 官方复现的必要工程差异（透明记录）

作者 Vid2Robot 代码在 eval runner 内使用 `Vid2RobotConfig(vit_pretrained=True)`，创建 ViT-B/16 时默认会再下载 timm ImageNet 初始化权重，随后 `model.load_state_dict(ckpt["model"])` 严格加载作者完整官方模型权重并覆盖初始化结果。我们的 `agcofl/vid2robot_official_smoke.py` **只在构建网络期间临时将 timm 的 pretrained 初始化设为 False**，加载结束立即恢复原函数。作者网络结构、完整 checkpoint、严格权重匹配、推理和 ManiSkill 评测器没有改变；运行报告明确记录该初始化网络调用省略，避免把它伪装成完全原封不动的一键执行。若 checkpoint 缺少权重，严格加载仍会失败，而不会隐藏差异。

另外：官方代码依赖 `timm`、`decord`、`einops`。v1.2.1 只补缺失的小依赖，不替换已运行的 torch/CUDA/ManiSkill。正式可复现论文记录应把这种适配与作者官方环境/提交分开注明。

## 科学进展路线

官方Vid2Robot运行通过后，先完成训练数据与评测协议严格分离的正式Baseline（同源数据、相同任务、相同观测、相同控制模式或明确不同原始实现），再接入强对照：官方Vid.Cond.π₀.₅（keyframes/uniform）、UniSkill、ViT-Transformer robot-only和同预算Object Flow变体。我们的方法重点不应是CNN/ViT换代，而是**同一任务目标下的源行为→目标机器人能力约束的可执行物体效应转换**，必须以物理成功率、有效中间事件、接触/支撑约束和匹配信息预算的Goal+Waypoint Oracle等测量创新优势。

当前不能用原始官方PickCube（举起物体）与PushCube（推到目标）两种不同成功判据，直接证明同目标跨行为迁移。需要另行构造并清楚标明“附加同目标能力冲突实验”，官方RoboReel成绩保持原协议不变。
