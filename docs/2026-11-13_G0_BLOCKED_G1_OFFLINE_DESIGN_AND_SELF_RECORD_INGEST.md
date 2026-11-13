# TRO1 · G0 科学门槛未通过后的工程闭环：双目标几何设计 + 合法真人视频验收

日期：2026-11-13。唯一远端仓库：`LemonQC/cehsisuanfa`。不另建 Conda 环境，不使用额外云 API，不下载受地域许可限制的 PartNet-Mobility。保留 RoboReel 官方 PushCube π0.5 46/60、Vid2Robot 3/60、StackCups 0/4 和四条原始视频的独立只读证据。

## 服务器报告的真实 G0 结果

用户在 `pi05` 上运行 `test_audit_goal_contrast_g0.py`：**15/15 tests OK**。运行原始 `scripts.audit_goal_contrast_g0` 返回 `G0_NOT_PASSED / G0_BLOCKED`，原因是没有已证实的同场景相反目标真人视频对；`real_physx_actions=0`。**单元测试通过 ≠ 科学门槛通过**，不能把 SSv2 的左右推类别、RoboReel 两个同任务视频或几何设计模拟成真的真人反事实数据。

## 本轮已实现可复用代码和离线检查

1. `agcofl/video_goal_push_spec.py`: 纯 Python 双目标数学定义。相同初始方块位置对应世界 X 轴 `-0.15 m / +0.15 m` 两个评测目标；评测半径 `0.05 m`，两目标不会重叠。方块最初 X/Y 均限制 `[-0.10,0.10] m`，生成的两个目标区域落在保守的 `[-0.40,0.40] m` 几何设计边界内。这个简单几何边界**不是 Panda IK、碰撞、接触规划或物理可达性**。
2. `scripts/check_g1_goal_design.py`: 不导入 ManiSkill、JAX 或 checkpoint，仅生成两个固定 seed `310000/310001` 的对应评测目标和防泄漏的观测字段设计。真实策略观测不得直接包含 `goal_pos`, `target_xyz`, `reward`, `desired_direction` 等 evaluator-only 字段。输出的 `G1_GEOMETRY_ONLY_G0_BLOCKED` 明确表示不能进入物理阶段。
3. `scripts/prepare_g0_self_recorded_pair.py`: 将两份**自录真实**视频按绝对路径或项目内相对路径导入文件审计，复用现有 G0 的真实 ffprobe/ffmpeg 解码检验，核对文件大小和 SHA256、禁止同一文件伪装相反方向，严格限制项目路径和 EVAL 禁区。输出内容寻址的 `pair_draft_*.json` 和（支持 cv2/Pillow 时）两行各三张关键帧的 `pair_draft_*_review.jpg`。
4. draft 的所有来源、授权、相同初始场景、相机到机器人 X 轴映射的**人工验收字段预设为 false/未确认**；机器永远不会擅自认定“真人视频对已验证”。必须实际观看完整视频、确认两种真实目标、桌面双目标区同时存在且机器人可映射，再人工填写原有的 `docs/examples/g0_self_recorded_pair.template.json` 需要的真实信息，用 `scripts.audit_goal_contrast_g0 --manifest ...` 再次审计。其输出也只是 `G0_CANDIDATE_REQUIRES_MANUAL_VISUAL_REVIEW`，须专家批准后才进入物理。
5. 本地标准库 `unittest` 12 项全部通过、四个 Python 文件 `py_compile` 通过；纯离线几何 CLI 已执行并真实生成报告。新服务器验收结果尚未运行，所有物理动作=0，训练=0；**没有创建或声称已验证 ManiSkill 新任务的物理成功率**。

## 本轮使用的最短命令

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_g0_selfrecord_g1_design.py' -v
# 离线任务数学设计，不运行物理仿真：
python -m scripts.check_g1_goal_design
# 若尚未自录两段视频，预期打印 TRO1_G0_RECORDING_REQUIRED（退出码2）：
python -m scripts.prepare_g0_self_recorded_pair
```

真实录像的必要外部采集环节（只能由现场人员完成）：将同机位、同桌面、同立方体起点、相同可见双目标区且两种反方向真实人手推送的源视频分别放到：

```
outputs/g0_human_goal_pairs/raw/left.mp4
outputs/g0_human_goal_pairs/raw/right.mp4
```

左/右标签是**世界机器人坐标** `-X/+X`，不是随意把相机画面左右等同世界坐标。应在桌面记录方向标定，保持相机、方块起点与两个可见目标区不变。自录素材需本人合法权利和参与者同意。不能使用视频镜像、剪接反转、换标签方式制造“另一目标”，也不能从 held-out EVAL 视频泄漏。

录制完成后**再次运行同一准备命令**，检查生成的 6 格人工审查图，填写实际合法权利、真人演示、同场景与轴标定信息，再执行：

```bash
python -m scripts.audit_goal_contrast_g0 \
  --manifest outputs/g0_human_goal_pairs/manifest.json
```

该 `manifest.json` 必须依据审查事实填写，程序不会自动给 `true`。未获得真正可用视频对时 **G0 继续 BLOCKED**，不运行 `VideoGoalPush-v0` 的 PhysX、不训练新方法、不将几何自洽冒充物理可行性，不随意转换研究方向。

## 目前门槛与下一动作

| 门槛 | 状态 | 证据 |
|---|---|---|
| G0 真人反目标视频对 | **BLOCKED** | 15 项测试通过，但原始审计明确 `G0_NOT_PASSED` |
| G1 双目标几何设计 | **OFFLINE_SPEC_PREPARED** | 两个目标区域互斥、seed 固定、无目标字段传入政策设计；仅本地测试通过 |
| G1 物理可达性 | **NOT_RUN** | 无 ManiSkill/Panda 双向物理轨迹，禁止推断物理可行性 |
| G2 视频必要性 | **NOT_RUN** | 无合法两种真人视频输入对 |
| G3 T-RO 创新性 | **NO_GO_PENDING_FALSIFIABLE_EVIDENCE** | 不借复现运行成功包装创新 |

**改造意义**：今后只需补真实真人素材并核查，代码中不再含伪造的成功标注或未经验证的反目标视频。未来如果 G0 审核通过，可将此稳定的几何定义移入单独新环境和具名评测，不能修改官方 `PushCube-v2` 的成功函数和历史 baseline 数据。
