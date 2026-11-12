# TRO1 / M1B3：冻结源端抓取目标的离线规划判别（不追加物理执行）

## 研究边界（继承 d7898e2）

- 执行服务器：RTX 4090，`/mnt/4Tdata/LemonData/cehsisuanfa`，Conda `pi05`；该任务只需要 CPU 规划，不需 GPU 训练。
- 任务、Panda、seed120、`AGCOFLM1B3Corridor-v0`、`x=.210m`目标、静态围栏和动态 blocker、原碰撞采样配方和完整 65 步失败记录均**冻结**。
- 已证实手/指对静态围栏的接触力在 65 个动作内为 0 N；实际关节跟踪误差约 16.75mm；`grasp_pose` Screw 规划失败且执行 0 动作。不能推断整个机械臂无碰撞，也不能推断精确抓取位姿运动学无解。
- 目标 v6 66 步真实任务成功继续冻结，不重跑。源端尚无同目标真实抓取演示。M2、强 TAMP、学习策略未运行。
- 此脚本只做原始 H5 取证和单次目标位姿规划调用，不调用 `env.step`、`follow_path`、`RecordEpisode`，不触发新任务 rollout。**`--plan` 会创建并 reset 一个新 ManiSkill/PhysX 场景，框架内部初始化可能涉及引擎操作；不得称其完全不接触 PhysX，只能称没有新的动作执行/任务仿真轨迹。** 如要求连物理场景初始化都不允许，只运行 `--inventory`。

## 2026-11-12 实机只读回传与修复：31维打包关节状态（新证据）

第一次 `--inventory` 在 `pi05` 服务器报 `NO_QPOS_DATASET`，但其数据集清单明确提供 `traj_0/env_states/articulations/panda` `[66,31] float32`。这是旧版工具**未识别打包格式**，不能认定 H5 缺失 qpos。官方 ManiSkill `Articulation.get_state/set_state` 确定布局：

- `[0:3]` 根位置、`[3:7]` 根四元数 wxyz；
- `[7:10]` 根线速度、`[10:13]` 根角速度；
- **`[13:22]` 9关节 `qpos`（7臂＋2夹爪）**；
- `[22:31]` 9关节 `qvel`。

对应 `13+9+9=31`，且源 H5 物理动作形状 `[65,8]`，关节状态长度 `66`。新版只接受显式 66x9 qpos 或**精确命名** `traj_0/env_states/articulations/panda` 的 66x31 打包状态；不再将其错误归类为“缺失 qpos”。来源见 [官方结构定义](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/advanced.html) 与 [官方 Articulation 实现](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/utils/structs/articulation.py)。

**安全核查**：检查全部根四元数、9关节物理合理值、最后双指值与真实 JSON 一致、同文件 cube/blocker 末帧与物理 trace 一致；在 `--plan` 中要求真实本地 `get_state()` 在 `set_state()` 后与原保存 31维数值一致，并核对末端 FK。保留所有旧原始文件及 SHA256；从 H5 直接读取，不通过 TCP 逆解瞎猜关节。新版 9 项单元测试均通过（本地合成 fixture，不等于 4090 的真实 H5 已通过）。

只有用户在 4090 上重跑后才能认定原始关节恢复成功。`--plan` 需要先通过只读 `--inventory`，且不触发新的物理轨迹。即使 `IK Success` 也不授权源端抓取。

## 原始数据可识别性问题

65 步 JSON 保存了逐步 TCP / cube / blocker 的位置、TCP quaternion、双指 qpos，**却没有保存失败时 `grasp_pose` 的完整 7D 目标，也没有保存 9D 手臂关节轨迹**。上游 `RecordEpisode` H5 可能保存机器人 qpos，也可能只有扁平 `obs`；不能在不知道其字段语义的情况下把扁平状态任意 9 个数当作 qpos。

`--inventory` 寻找唯一 `.h5`，接受明确的 `*/qpos` `[66,9]`/`[66,1,9]` 或已确认路径 `traj_0/env_states/articulations/panda` `[66,31]`。新版首先从该打包状态的 `[13:22]` 提取真实 qpos、`[22:31]` 提取 qvel；根状态为前 13 维。若多个独立 qpos 记录不一致则拒绝，不能通过 `--dataset` 强行选其中之一。验证根四元数、最后双指值与真实记录、cube/blocker 末帧后输出原数据 SHA256。原存盘仍**没有失败目标 pose7**；只可使用冻结代码重新生成候选目标，禁止从 TCP 反解假想抓取目标。

由于原失败脚本未保存目标 pose7，`--plan` 只能在原场景/同 seed/同 OBB grasp 实现下**重建**目标，不能宣称比特级精确恢复之前未存储的目标。若目标是否逐位一致无法确认，任何规划通过都仍是候选，不是已经验证的真实原路径成功。

## 单次规划流程（全部在原场景，无参数扫荡）

1. 仅从已存 H5 的 Panda articulation 打包状态读取第 65 个动作之后的**完整 31维状态**（65 个动作对应 66 个状态），取 `[13:22]` 真实 qpos；检查根四元数、qpos/qvel、双指日志、cube/blocker 最后一帧均有效。恢复原 reset(seed=120) 场景后，调用本地 articulation `set_state` 恢复**原 31维根＋关节状态**，再用 `get_state` 检查逐值一致；比较 cube/blocker 与日志差别小于 1mm、TCP FK 与日志差别小于 5mm。任何检查不一致立即停止。
2. 调用**原先写在** `agcofl/m1_source_geom_aligned_topdown_v1.py` 的 top-down OBB 配方：接近轴 `[0,0,-1]`、闭爪轴 `[1,0,0]`、depth 0.025、`build_grasp_pose`。不优化目标，不修改障碍物，不修改抓手，不重新运行 65 步。
3. 原 `get_planner` 创建 PandaArmMotionPlanningSolver；加载原 `original_obstacle_points()`，启用其 `use_point_cloud=True`（点云仅是原规划碰撞近似，非精确 PhysX 几何证明）。
4. 单次 `planner.IK`（20 个内部初始化点）查询；**上游 IK 常带碰撞过滤，因此其失败本身不能区分纯运动学不可达与碰撞过滤**。若有解，只检查第一个返回解的显式 self/environment collision API，不新增随机重试。不支持 API 或出错均记 UNKNOWN。
5. 原 `plan_screw(..., use_point_cloud=True)` 进行一次**规划计算**；不执行结果、不跟随路径。`IK Success` + `Screw Failed` 仅说明已找到一个候选终态但原 Screw 方法拒绝；对搜索空间的可达性不构成证明。只报告真实 planner status。

## 运行命令（只读盘点优先）

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05
python -m unittest discover -s tests -p 'test_m1_source_frozen_planning_gate.py' -v
# 第一阶段：不创建 PhysX 场景；只读取以前的 H5/JSON
python -m scripts.m1_source_frozen_planning_gate
# 若输出 INVENTORY_DONE_PLANNING_NOT_RUN 且 qpos 数据身份正确，再执行：
python -m scripts.m1_source_frozen_planning_gate --plan
```

若无唯一 H5、打包布局损坏或数据集歧义，程序会输出 `M1_SOURCE_FROZEN_PLANNING_INCONCLUSIVE` 和现有 H5 数据集 shape；只核对原始文件位置或结构，**不可自造缺失的原始关节状态**。如验证确实存在另一个原生 qpos 数据集，可显式指定 `--h5` 和 `--dataset`，要求记录完整路径与 SHA。若没有，直接终止此门槛。

输出：
- `outputs/m1_source_geom_aligned_topdown_v1/frozen_planning_inventory.json`
- `outputs/m1_source_geom_aligned_topdown_v1/frozen_planning_audit.json`

两个文件只新增、不覆盖不同内容；旧 H5、status、trace 的 SHA 在读前读后校验。产物可以只提交小型 JSON 摘要和路径/SHA 到 GitHub，不提交大 H5 或视频。

## 预注册停止门槛 / 不产生新的正式成功率

| 结果 | 允许的科学表述 | 下一步 |
| --- | --- | --- |
| `INCONCLUSIVE`（H5 无 qpos，FK/姿态不一致，原版本或 API 不兼容） | 不能区分 IK、碰撞、Screw 的根因 | 停止本次 M1B3 源端试探，不增加其他 seed/抓取姿态 |
| `INCONCLUSIVE_IK_OR_COLLISION_FAILURE` | 规划器 IK（可能含碰撞过滤）未给出解，不是纯运动学不可达证明 | 停止当前目标的直接重试 |
| `GOAL_COLLISION_REPORTED_BY_PLANNER` | 规划器模型检测到当前候选配置碰撞，未证明任何其他配置都碰撞 | 不以原失败目标执行物理抓取 |
| `IK_CANDIDATE_BUT_SCREW_REJECTED` | 至少有候选 IK，原 Screw 路径仍失败；碰撞检查的覆盖范围必须记录 | 单独定位 Screw vs 规划器模型，不直接试物理抓取 |
| `PLANNING_CANDIDATE_RECONSTRUCTED_POSE_ONLY` | 重建目标存在规划候选，未确认原目标逐位一致，未证明实际接触 | **仅允许人工审查**原模型/目标身份；不得自动执行物理抓取 |
| 任何 API/碰撞审查缺失 | 不具备肯定/否定证明 | INCONCLUSIVE、停止 |

该工具永远输出 `authorizes_physics=false`：是否再允许**至多一次**原场景接触探针，必须有比本工具更强的、由人审阅的 exact-pose、原规划碰撞模型和真实状态一致性证据。单次规划成功绝不自动放行。

### 当前科学结论（在服务器新证据出现前不变）

目标端 v6 是一个**特权状态脚本控制器的单 seed 物理成功**；M1B3 源端三条真实执行路线都没有形成有效抓持；没有有效同目标 cross-behavior 示范对，没有等信息强 TAMP 优势，没有新学习方法。这个离线门槛只处理工程可行性，不构成论文创新或 T-RO GO。若数据不齐或规划判别不能提供干净证据，**本场景以基准构造工程 NO-GO 结案**，并保留 M0 和目标 v6 成果，不继续修改源执行器。
