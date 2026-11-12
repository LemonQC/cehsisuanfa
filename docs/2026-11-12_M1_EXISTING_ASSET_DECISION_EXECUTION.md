# TRO1 / RTX 4090 — M1 现有物理资产审查与后续研究决策

**任务类型：纯只读分析（新建小型 JSON/Markdown 报告除外）**。不运行 ManiSkill、不创建 PhysX 场景、不用 GPU、不调用网络、不重新训练，不修改旧实验，不重新抓取。仓库：`LemonQC/cehsisuanfa`，本地路径：`/mnt/4Tdata/LemonData/cehsisuanfa`。

## 1. 已有证据（用户服务器终端回报）

2026-11-12 原 B2 的 `scripts/m1b2_validate_paired_physics.summarize()` 输出：

- `source_pick_place_valid=true`、170 步。
- `target_direct_push_success=true`、31 步。
- `target_clear_then_push_success=true`、51 步。
- `target_only_blocker_displacement_observed=true`；三个案例起点和目标严格一致，且记录有实际 rollout。
- `pilot_contrast_pass=false`。**B2 的直接 Push 已成功，清障不是必要前置步骤**。不能把 target-clear 的成功说成 Insert 因果优势。

B3 在冻结的 x=.210 原始狭窄场景：目标端 v6 在 66 步中完成清障＋推送，末态误差 8.193mm；三条原源端 Pick 执行均未抓持。原始 65 步 H5 中 Panda 31D 末状态已恢复，MPLib 0.1.1 离线规划 `IK_SOLUTIONS=0`、`screw plan failed`，不能唯一归因于纯 IK 或碰撞；严格 `authorizes_physics=false`。不允许追加源端 PhysX 抓取。既有停止记录见 [`2026-11-12_M1B3_FINAL_OFFLINE_PLANNING_STOP_AND_NEXT_GATE.md`](2026-11-12_M1B3_FINAL_OFFLINE_PLANNING_STOP_AND_NEXT_GATE.md)。

**当前决策**：B2 是有效的同目标“简单成功”对照，但不是必要清障的例子；B3 有目标端工程成功但没有同目标成功源端视频。**没有获得同时满足同目标真实源—目标成功及 Insert 必要性两个条件的已验证场景。** 更不能声称有视频信息增益、TAMP 缺口、新模型性能优势。

## 2. 新增只读扫描器

- `scripts/m1_existing_asset_decision.py`
- `tests/test_m1_existing_asset_decision.py`

程序复用此前已提交的 `scripts/m1b2_validate_paired_physics.summarize()` 和 `scripts/m1a_inventory_real_assets.audit()`，不引入 ManiSkill、torch、sapien、mplib 等运行时。它只检查：

1. B2 三份原 `status.json` 的真实成功标签、相同起点/终点和 seed；单独检查相应 `physical_trace.json` 的行数、H5/MP4 文件**存在性**（不打开大型二进制）。
2. B3 原 65 步失败源端、66 步成功 target v6 的版本／seed／goal／动作数及 trace 完整性；如果本地有冻结 `frozen_planning_audit.json`，核对其真正的 IK-zero 停止结果。
3. 通过旧 M1A 资产盘点，汇总已下载的官方 TRAIN Pick 人类视频、TRAIN Push 机器人 H5+相机三件套及 M0 结果存档；**官方 PickCube 和 PushCube 的任务定义不同，不允许自动配成相同目标。**
4. 生成一个机器可读 JSON，以及一个 Markdown 决策表，清楚标记 `video_information_gain_tested=false`、`equal_information_strong_tamp_executed=false`、`learned_method_executed=false`。

**注意：MP4 文件存在不是视频内容已经人工验证，H5 文件存在不是物理抓取成功；B2 的 source 是仿真机器人 MotionPlanning 视频，不是人类示范。** 原始大文件没有被重算 SHA256 或拷贝；只对相关小 JSON 保存 SHA256。扫描范围是当前被冻结的 B2/B3 + 旧 M1A 指定的官方本地路径，不是对全部互联网机器人数据集的穷举调研。

## 3. 原服务器执行（Conda `pi05`）

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_m1_existing_asset_decision.py' -v
python -m scripts.m1_existing_asset_decision
cat outputs/m1_existing_asset_decision/decision.md
```

JSON：`outputs/m1_existing_asset_decision/decision.json`；Markdown：`outputs/m1_existing_asset_decision/decision.md`。新报告**只创建一次**，重复输出一致可通过；如果原始输入改变造成不同报告，工具**拒绝覆盖已保存版本**。不主动把 H5、MP4、日志上传 GitHub。

## 4. 研究停/走门槛

- **B2 direct Push 成功**：保留 B2 作为可执行 sanity/低难度对照；不再声称 blocker-clear 的 Insert 必要性。
- **B3 source 失败**：继续冻结 v6 target 和原三次源失败；不重启 B3 抓取执行器，不搞 M1-B5。
- 只有找到**独立物理验证**的匹配源—目标成功任务，且源视频携带目标状态/显式 goal 无法提供的特定信息，才可讨论跨视频信息迁移。
- 即使找到上述实例，也须让同等任务信息/能力约束的强 TAMP 成为真正对照；不能仅以目标端多步成功作为新算法贡献。
- 当前不启动 M2/Effect Editor 训练、不宣布 T-RO GO，不把单个 seed 伪装成统计性能。

## 5. 此轮之后唯一值得进行的工作

请优先核实**已有**其他录像/机器人轨迹资产是否存在“源、目标同一最终目标，但可用能力不同且目标正确行为随约束改变”的**真实**样本。先做只读数据可识别性与算法覆盖审查，不新建多版专用狭窄走廊。若只是与 B2 一样 goal-only 直接推动就能完成，或像 B3 一样源执行无效，则当前 Insert 故事应停止，而不是投入更多 RTX4090 计算。
