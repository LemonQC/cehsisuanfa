## 2026-11-12 — 原 M1B3 当前 v3 真实下降阶段：TCP 在 41.3mm 高度停滞、X 向前漂移；未到接触位姿，不可放宽阈值

用户从已存 `outputs/m1_current_target_safe_approach_v3/target_safe_approach/physical_trace.json` 抽取 `descend_behind_cube` 的25个 PhysX step：t=144 TCP≈[-.0631,0,.1602]；t=159≈[-.06745,-.00055,.06690]；t=164≈[-.06198,.00006,.04134]；t=168≈[-.05478,.00136,.04128]，底部5步 Z 几乎不动，X 向前7.2mm，Cube仍在[0,0,.020]、闭合指关节qpos均0。MPLib Screw 返回成功但实际TCP不符合下降指令[-.065,0,.020]，故 `M1_TARGET_NO_VALID_REAR_CONTACT_POSE_BEFORE_PUSH`，从未开始+X推送；此前6+82步清障、18步上升、38步高空平移均有效。**现阶段不能确定是手指/桌面碰撞、关节限制、末端跟踪误差还是固定工具姿态定义失配**；当前 `q` 直接沿用从+Y清障到+X推进，可能不合适。禁止简单放宽18mm阈值或重跑v3。下一次仅比较冻结成功 `outputs/m1b3_corridor_dev/open_direct/physical_trace.json` 的TCP接触高度、真实方块运动起点和夹爪状态与本次v3的无扰动接近轨迹，确定接触框架后才允许定向修复。无新PhysX、无新M2。详见 [v3真实下降诊断](2026-11-12_M1_TARGET_V3_DESCENT_TRACKING_DIAG.md)。

---

## ★ 2026-11-11 — 目标端真实后362步轨迹发现反向推动：同一 M1 场景 v3 安全后方接近与向 +X 推送，待物理验证

用户 pi05 服务器运行新离线脚本5测试 `OK`，读取**实际已保存的450步PhysX** 得到：闭合夹爪 MPLib 清障已82步成功，清障后362步方块从 x≈0 移到 `x=-0.050044m`、y≈0.00167m，目标 x=+0.210m，净目标进度 **−0.050049m**，末15步完全不动，TCP 最近距方块中心37mm。原失败代码第二阶段从当前机器人清障结束位置通过 RRTConnect 前往 `precontact_x=-0.050m`，与方块最终负位移数值吻合，**高度提示返回接近过程可能把方块误推向−X，但未记录具体碰撞时刻，不能声称已证明**。已提交 `agcofl/m1_current_target_safe_approach_v3.py`：保持原M1B3场景、seed120、goal x=.210、20mm阈值、`pd_joint_pos`、闭合夹爪、450步预算和**原v2已成功的MPLib清障过程完全不变**；只改清障后的接近——先在当前x/y垂直退到Z=.160m、悬空到方块−X后方65mm、竖直下降并实测 TCP在后方，之后分段沿+X前进(.110/.180/.260m)，每个 PhysX step 遇负向位移、横向漂移、抬升/抓取或狭口失配立即停止并存证。旧任意记录不改、不再跑原450步、新v3最多**一次**真实目标端实验。脚本 `bash scripts/run_m1_target_safe_approach_once.sh` 先5项纯CPU单测、原几何门槛、只读原450步记录和真实SHA256（验证既有−50mm失败）才启动PhysX；输出 `M1_TARGET_SAFE_APPROACH_GATE`，新结果于 `outputs/m1_current_target_safe_approach_v3/`。**代码已提交，尚未由用户服务器测试或执行**；source当前仍是106步真实双指抓取0，强TAMP/M2未做，不能冒充整套方案可行。详见 [当前M1安全接近修复的完整协议](2026-11-11_M1_TARGET_SAFE_REAR_APPROACH_V3.md)。

---

## ★ 2026-11-11 — 当前 M1 闭合夹爪 MPLib **真实清障成功**，后续 Push 触发 450 步预算；仅审计现成后362步轨迹

用户从服务器取得已封存的 `target_closed_mplib` 真 PhysX 输出：`success=false`、`steps=450`、错误 `M1B2_STEP_BUDGET_EXCEEDED`。第一段闭合夹爪后 82 步**确实将挡块推到侧槽** `[0.1048923,0.1290073]`，停车误差6.324mm；450步全程夹爪闭合（Panda `-1`，末端双指qpos均0m），真实 `is_grasping(task)=false` 且 `is_grasping(blocker)=false`，说明“闭合工具作为推杆”可满足**不抓取**而非“张开夹爪”的 Push-only 定义。原 `DiagnosticPushTeacher` 在同场景更早32步清障亦已成功，清障阶段不再需要重复开发。原新闭合MPLib 共6步闭夹爪+82步清障，之后**有362步真实目标方块阶段尝试**未完成，`phases` 未列出因第451动作触发保护异常；最终目标误差260.05mm，不能仅凭误差推断物体是否被推反/卡障碍。已提交纯标准库离线脚本 `scripts/m1_current_closed_pusher_trace_audit.py` 及5项回归测试，严格读取旧450步H5派生 `physical_trace.json` 和原 M1B3 35步成功 `open_direct`，对比真实方块位移、x/y变化、TCP距离、末15步停滞、夹爪状态、所有原文件SHA256；**不启动 PhysX / 不重跑旧种子、不训练模型**。命令 `python -m unittest discover -s tests -p 'test_m1_current_closed_pusher_trace_audit.py' -q; python -m scripts.m1_current_closed_pusher_trace_audit`，输出 `M1_CLOSED_PUSHER_450_STEP_TRACE_DIAG`。源端新的侧向抓取106步无物理抓取仍待离线比对，目标端清障成功**不是**同目标跨行为完整成功或TRO创新。见 [已完成物理结果与后362步轨迹审计](2026-11-11_M1_CLOSED_PUSHER_CLEARANCE_PASS_POST_CLEAR_TRACE_AUDIT.md)。

---

## ★ 2026-11-11 — 当前 M1 目标端 MPLib 真实失败88步，发现 Panda 夹爪符号反转的旧验收错误

用户在**相同M1B3原始障碍场景、同seed120、同目标x=.210**独立执行 `target_official_push`：实际88步失败，挡块从约[0.100,0]只移动到[0.10738,0.03198]m，距离侧槽停车目标93.3mm，未进入推目标方块阶段，完整结果在服务器 `outputs/m1_current_executor_swap_smoke/target_official_push`。此前 M1B3 **已有成功的挡块清障**，32步用 `DiagnosticPushTeacher` 移至约y=.1198m，因此不应盲目换掉已成功执行器。新核查官方 ManiSkill `TwoFingerGripperMotionPlanningSolver` 的 `OPEN=+1`、`CLOSED=-1`，并查明官方 Panda `panda/solutions/push_cube.py` **先 `planner.close_gripper()`**。但先前失败的 MPLib 改写代码用了 `open_gripper(+1)`；原先32步成功清障的旧程序用了 `action[..., -1]=-1`，即**闭合**。更严重的是旧 `M1B3 gripper_never_closed=all(action<=0)` 和 `M1B2 saw_gripper_close=any(action>0)` 都错误地把真正闭合行为叫作“未闭合”。旧 H5/视频不动，纠正科学文字解释；**Push-only 要求两个物体都未被夹住、未被举起，并不强制手指张开**。已提交针对当前 M1 相同任务的**唯一一项单因素闭合推杆检验** `agcofl/m1_current_closed_pusher_ablation.py` 与 `bash scripts/run_m1_target_closed_pusher_once.sh`，先实测文件验证旧“32步-1成功”和新“88步+1失败”，再仅将原MPLib第一步夹爪 OPEN→CLOSED，保持原规划目标/几何/seed/动作预算，逐步记录左右手指实际位置及对挡块和目标的物理抓取标志。新输出 `outputs/m1_current_closed_pusher_v2/`，**代码已提交，服务器尚未执行**。不重新跑旧 Baseline、不调整源端106步失败、不设计新任务。科学问题/强TAMP/M2仍待物理配对和正式对照。完整说明 [真实夹爪符号错误与单因素试验](2026-11-11_CURRENT_M1_TARGET_OPEN_VS_CLOSED_PUSHER_CONTROL.md)。

---

## ★ 2026-11-11 — 当前 M1 成熟执行器替换源端真实抓取失败，目标端改为独立单次验证（同实验，不重新选题）

用户当前原 M1B3 障碍场景实际跑完官方 Panda/MPLib 侧向抓取替换的一次 106步 PhysX：`M1_CURRENT_EXECUTOR_SWAP {"mode":"source_side_grasp","success":false,"steps":106,"grasp_steps":0,"error":"CONTACT_NO_GO...did not physically grasp cube"}`，`observed_lift_m=0`、最终误差0.21000036m。证明**这次侧向抓取规划成功但物理夹持失败**，不能再声称低层执行器已修复；旧runner因源NO_GO未启动新目标端 Push。保持原 M1研究问题、真实场景、障碍、终点、seed和全部已封存H5/视频。新增离线 `scripts/m1_current_compare_grasp_contact.py` 读取这次失败的106步轨迹与旧 M1B2 **实际成功**的170步抓取轨迹，对照真实夹爪 close 动作、TCP距离/位姿、抓取证据，记录原文件SHA256，**不执行任何仿真**。新增 `scripts/run_m1_current_target_only_once.sh`，允许源端NO_GO时独立测试已提交的 ManiSkill官方Panda PushCube式 MPLib 底层执行器（同 M1B3 原场景，仅一次边界物理），不修改原 source 或再跑旧四组；此处 Target 成功只能证明低层推动可行，**不能冒充已经完成同目标跨行为配对/TAMP**。执行 `python -m unittest discover -s tests -p 'test_m1_current_compare_grasp_contact.py' -q; python -m scripts.m1_current_compare_grasp_contact; bash scripts/run_m1_current_target_only_once.sh`。这两项新检查还未在用户服务器运行。详见 [当前 M1 真实抓取失败与独立目标执行验收](2026-11-11_CURRENT_M1_SOURCE_CONTACT_FAIL_TARGET_INDEPENDENT.md)。

---

## ★ 2026-11-11 — 当前 M1 场景执行器直接替换已编码，不换任务、不重跑旧实验

用户明确要求解决**当前**同目标 Pick 源端→Push-only 目标端清障操作里的失效执行模块，不能擅自换成新论文。旧 M1B3 真实轨迹源抓取步数为0、物体无升降、XY总位移约4.7微米；目标端清障32步真实完成，但旧 `DiagnosticPushTeacher` 将目标方块推至狭口前 x≈.125m、y≈−.031m 卡住。已提交直接替换 `agcofl/m1_current_executor_swap.py`：保持原始 `AGCOFLM1B3Corridor-v0`、seed120、goal x=.210、50mm狭口/动态挡块、同一 Panda，不动任何已保存 H5/视频。源端用 ManiSkill 官方 Panda OBB + MPLib 规划接口**从机器人所在的−X侧向接近抓取**，关夹爪后马上验证真实 `is_grasping`，否则停止抬升/搬运；目标端替换手写比例 Push 为 ManiSkill 官方 `panda/solutions/push_cube.py` 同款 MPLib 关节空间规划接口，分别控制清障和推方块，夹爪继续保持 open。均为**有明确适配差异的现成接口组合**，不是原版未修改算法、训练策略或 TAMP。新入口 `bash scripts/run_m1_current_executor_swap_once.sh`：先4项纯 Python unittest、原几何检查、只做一次源执行冒烟；实际抓取/抬升/落点在相同目标中通过后，才启动最多一次目标执行冒烟；否则立刻停，不重跑已有4组基线或调参循环。新代码已提交但用户GPU机器**尚未验证**。科学门槛（同目标源视频、等信息强 TAMP、非平凡方法必要性）仍未通过。详见 [当前实验的执行模块替换操作](2026-11-11_CURRENT_M1_OFFICIAL_EXECUTOR_SWAP_SMOKE.md)。

---

## ★ 2026-11-11 — 用户纠偏：当前 M1 实验难题**替换失效执行模块**，不换研究问题、不重开实验

用户明确指出“我说的是当前实验遇到的难题，不是再重新做实验啊”。因此取消我擅自提出的长序列任务 M1C 和独立 B01 64 条 BC 数据下载作为下一步：它们的未执行脚本和测试已从仓库删除。继续当前**同目标 Pick 源端 → Push-only 目标端，需要清障插入的跨行为任务**。真实失败来自 M1B3 自写的官方 Panda 抓取移植在狭窄障碍布局下，方块真实抓取次数0、未抬升、总XY位移仅约4.7微米；这只证明**这个抓取移植失效**，不能自动否定整个方法科学问题。已有可信替换源是原版 ManiSkill PickCube 专家3/3的 H5/视频、原M1B2成功 Pick 的 H5/视频，但必须核对目标物体终态效果，尤其原M1B2目标x=.15m、现M1B3目标x=.21m，**不能直接冒充同终态配对**。目标低层执行器替换优先选择 ManiSkill 官方 `mani_skill/examples/motionplanning/panda/solutions/push_cube.py` 中真实 Panda MPLib 运动规划和推动接口，保留同一障碍/目标/Push-only 判据、原始四组轨迹不改，不再使用不可靠的 `DiagnosticPushTeacher` 重复盲调。原版官方 PushCube 仅是低层推动专家，并非已经解决复杂清障序列；强 TAMP 和视频增益仍未评估，不擅自宣称创新。详见 [针对现有技术问题替换模块的执行边界](2026-11-11_SCOPE_CORRECTION_REPLACE_CURRENT_EXECUTOR_NOT_RESTART_RESEARCH.md)。

---

## ★ 2026-11-11 — M1-B4 **真实**源演示 NO-GO（硬停止）：旧物体始终未抓取、未抬升、未搬运；转入独立 B01 官方训练数据基线

用户执行修复后的4项 unittest **OK**；出现专门的真实数据标记 `M1B4_UNIT_TESTS_PASSED_NOW_READING_REAL_FROZEN_M1B3_DIAGNOSIS` 后，实际只读旧 M1-B3 轨迹得到 `source_actual_grasp_steps=0`、`source_actual_max_task_z_m=0.01999999955m`、`source_actual_task_total_xy_m=4.67618855549e-06m`，因此真实输出 `M1B4_NO_GO_SOURCE_DID_NOT_PHYSICALLY_CARRY_CUBE_IN_FROZEN_RUN`。这次**不是临时 unittest 假负样本**；旧源端从未物理搬运 cube，调整 `lower` 的 RRTConnect 没有作用，按冻结规则直接终止，不执行任何 M1-B4 新 PhysX。**停止当前同目标 Pick→Push 必须清障场景，不继续 M1-B5、不改控制器、不训练 M2；之前的官方 π0.5 46/60、Vid2Robot 3/60、官方 Pick 3/3 均冻结归档。** 已更新 [正式科研停止决定](2026-11-11_M1B4_ACTUAL_REAL_NO_GO_AND_RESEARCH_HARD_STOP.md)。已有 Baseline 并未全部完成：B01/B02 等仍 NOT_RUN，且即使完成也不能挽救已失败的同目标科学问题。为后续**独立** RoboReel Benchmark 研究准备真正 B01 robot-only 数据，新增 `scripts/prepare_b01_robot_train.py`，限制真实 PushCube TRAIN 64 条 H5+双摄像头（≤300MB，复用原来的2条）、不可使用 EVAL0080–0099、固定51/13 train/val split、H5 action 维度8内容验证。命令 `python -m unittest discover -s tests -p 'test_prepare_b01_robot_train.py' -q`；明确同意实际下载时 `python -m scripts.prepare_b01_robot_train --download`。此 B01 下载和训练**均尚未由用户服务器执行**，不得虚构任何 BC 成绩。后续必须重新锁定新的科学问题，才能开展 Method。

---

## 2026-11-11 — M1B4 preflight 中假的 NO_GO 与真正单元测试失败（已修复）

用户运行 `bash scripts/run_m1b4_one_time_feasibility.sh`，4项 unittest 中1项失败：`pre_throat_lateral_alignment_needed([.1252731,-.0308216,.02])` 预期类别字符串，但旧代码只有 x>.129m 才标记狭口前撞墙，而真实几何中障碍前接触位置约 x=.126m。已改为 x>=.123m 提前识别并补充边界测试，不改物理场景或控制器。**同一日志显示的 `M1B4_EVIDENCE_PREFLIGHT {...source_actual_grasp_steps:0,...task_total_xy_m:1e-5}` 以及 `M1B4_NO_GO_...` 都来自 unittest 中故意构造的临时模拟负样本**，并不是用户原 M1-B3 的真实 source 轨迹！由于 bash `set -e` 在测试失败后退出，真正的 `outputs/m1b3_corridor_dev/diagnosis.json` 门槛没有被执行，新 PhysX 案例0次。已将测试 fixture 的标准输出捕获、不再展示假的 NO_GO；runner 会在测试成功后显示 `M1B4_UNIT_TESTS_PASSED_NOW_READING_REAL_FROZEN_M1B3_DIAGNOSIS`，其后的真实 SOURCE evidence gate 才能决定是立刻止损还是最多两次新 PhysX。执行 `git pull --ff-only origin main; conda activate pi05; bash scripts/run_m1b4_one_time_feasibility.sh`；如果**真实门槛**随后 NO_GO，则严格停止当前场景、不继续控制器调整或 M2。详见 [M1-B3 实际诊断与 M1-B4 最终补救](2026-11-11_M1B3_ACTUAL_DIAG_AND_M1B4_FINAL_RESCUE.md)。

---

## ★ 2026-11-11 — M1-B3 真实轨迹已离线诊断，M1-B4 证据先行的**最后一次**固定场景可行性补救

用户服务器实际报告 `M1B3_OFFLINE_DIAG ran_simulator=false`，3项标准库测试通过。原场景源端 MPLib 的 `approach/grasp_pose/close_gripper/lift/across` **规划器返回成功**但 `lower` 失败；不能由这些“规划器成功”断定真实物体确实被夹持/搬运。旧 `blocked_clear` 清障 **32步真实成功**（块移约120mm），接下来推目标 **418步失败**，方块卡在 **[x=0.125273, y=−0.030822]m**、末15步没有位移。根据实际墙面 `x=0.146` m、方块半宽0.020m，障碍前接触中心约 `x=0.126`m；狭口中方块中心允许 y≈±0.005m；因此是**高度符合狭口前横向失配**的几何机制，但没有接触力原始测量。已提交一次性新版本 `agcofl/m1b4_one_time_feasibility.py` + `scripts/run_m1b4_one_time_feasibility.sh`，沿用**完全不变的 M1-B3 场景、seed、20mm 成功阈值和旧推操作器**。先只读原诊断并核验**源 Cube 真实抓住、抬起、运输**的物理证据；否则输出 `M1B4_NO_GO_SOURCE_DID_NOT_PHYSICALLY_CARRY_CUBE_IN_FROZEN_RUN`，**不再启动物理仿真**。如果物理证据成立，则最多两个固定新案例：源端仅在 `lower` Screw 失败时尝试一次官方 MPLib RRTConnect、目标端清障后在狭口前对中再推行。先过 stdlib unittest，命令 `bash scripts/run_m1b4_one_time_feasibility.sh`。原四个 M1-B3、全部 M0/M1-B1/B2 原始数据不改不重跑。这是传统特权状态控制的物理可行性试验，**不是 TAMP、不是学习跨行为、更不是 T-RO 创新**。无论旧证据门槛或新场景物理任一失败，都停止在此场景反复控制器修补；强等信息 TAMP 和人类视频独特信息价值仍未验证。详见 [M1-B3 真实诊断及 M1-B4 最后补救协议](2026-11-11_M1B3_ACTUAL_DIAG_AND_M1B4_FINAL_RESCUE.md)。

---

## ★ 2026-11-11 — M1-B3 四组真实 PhysX 执行完毕，**同目标配对门槛未通过**，仅做离线轨迹诊断

用户4090实测 M1-B3 新狭窄通道：`blocked_source` **失败335步**（`official_mplib_lower_failed`，误差210.0mm）；`blocked_direct` **失败91步**（误差142.1mm，挡块仅移动17mm）；`blocked_clear` **失败450步**（挡块确实移到侧槽，但目标方块仍离终点90.2mm）；`open_direct` **成功35步**（误差14.3mm）。四组是实际物理结果，`geometry_preflight_pass=true`、`matched_task_start_and_goal=true`，但 `pilot_contrast_pass=false`、`causal_necessity_proven=false`、`strong_tamp_evaluated=false`。该场景没有得到成功的同目标源/目标配对，**不得据此声称需要新增因果效果或进入 M2**。保持完整原始 `outputs/m1b3_corridor_dev/` H5/视频/物理轨迹不改不重跑。新提交零GPU标准库 `python -m scripts.m1b3_diagnose_existing` 离线读取四组保存的 `status.json` 和 `physical_trace.json`，提取源 MPLib `planner_phases` 及 `blocked_clear` 清障后推方块分段实测、TCP和物体位移/停滞、SHA256；先用 `python -m unittest discover -s tests -p 'test_m1b3_diagnose_existing.py' -q` 验证。输出 `M1B3_OFFLINE_DIAG` 和小文件 `outputs/m1b3_corridor_dev/DIAGNOSIS.md`，**离线脚本尚未在用户服务器执行**。根据真实轨迹才决定是否值得一次明确版本化修正，否则及时停止此科学命题。详见 [真实M1-B3结果与下一关](2026-11-11_M1B3_ACTUAL_PHYSICS_NO_GO_AND_OFFLINE_DIAG.md)。

---

## ★ 2026-11-11 — M1-B2 实测成功但科学对比失败，M1-B3 新物理通道反事实实验（待执行）

**真实用户服务器记录**：原版 RoboReel π0.5 M0 46/60 已完整归档；原版 ManiSkill PickCube 专家 3/3；M1-B2 CUSTOM 同目标 PhysX 三组 `source=成功(170步)`、`target_direct=成功(31步)`、`target_clear=成功(51步)`，其 `pilot_contrast_pass=false`，因为**直接推动即可完成，清障没有显示必要性**。因此冻结旧场景/旧数据，不在其上训练 Method，也不重新运行原版 Baseline。新建 `AGCOFLM1B3Corridor-v0`：真实 static walls + dynamic blocker、狭窄“尺寸选择性”通道及侧向停车槽（目标 Cube 40mm 可通过 50mm 狭口、挡块 58mm 不能），用无 PhysX 的 C-space BFS 验证阻塞/开放状态连通性；然后在同一最终目标、同一 seed=120 上运行 4 个物理反事实场景：`blocked_source`（复用官方 MPLib Pick+Place）、`blocked_direct`、`blocked_clear`、`open_direct`（三种目标 Push 使用旧 privileged Push 执行器）。运行入口 `bash scripts/run_m1b3_corridor_once.sh`，自动先过 stdlib unittest 与 geometry gate，再限时真实物理并保存 H5/JSON/MP4 和逐帧物体状态；**这批代码已提交但物理未执行**，不声称必然成功。即使 blocked-clear 物理反事实成功，也需要强等信息 TAMP、动态替代路径检验和源视频独特信息验证，才能进入 M2。详见 [M1-B3 反事实物理实验与科研停止条件](2026-11-11_M1B3_CORRIDOR_COUNTERFACTUAL_PRE_REGISTERED.md)。

---

## 2026-11-11 — M1-B2 测试前置阻塞：坐标列表与 Pose 接口修复，物理实验尚未启动

用户执行 `bash scripts/run_m1b2_same_goal_once.sh`，首先进行标准库 unittest（4 tests，2 errors）：`actual_gate` 的 Python `final[:2]-goal[:2]`（列表不能直接减）及 `TracedEnv.step` 调用 `_pose(agent.tcp_pose)` 报 `AttributeError`（TCP 已经是 Pose 类型，只有 `.p`；原 `_pose` 只支持 actor.pose.p）。已在 `agcofl/m1b2_same_goal_effect_physics.py` 对测量 XYZ 统一 NumPy 数组、三维有限数检查，并兼容 Actor.pose.p 和直接 Pose.p；在 `tests/test_m1b2_same_goal_effect_physics.py` 新增列表/数组一致性和 TCP/Actor 访问回归测试。原 runner 因 `set -e` 在测试失败后**没有运行 source/target_direct/target_clear 任何一个 PhysX 案例**。用户需要 `git pull --ff-only`，先执行 `python -m unittest discover -s tests -p 'test_m1b2_same_goal_effect_physics.py' -q`，通过后再执行 `bash scripts/run_m1b2_same_goal_once.sh`。M0 46/60、M1-B1 官方 Pick 3/3 不受影响；M1-B2 和强 TAMP 仍未得到真实结果。详见 [M1-B2 原协议及修复日志](2026-11-11_M1B2_REAL_PHYSX_PAIRED_PILOT.md)。

---

## ★ 2026-11-11 — M1-B1 官方 Pick 3/3 实测成功，M1-B2 同目标物理对照开始

**用户实际服务器证据**：`M1B_OFFICIAL_PICK_EXISTING_VERIFIED {"requested_attempts":3,"recorded_episodes":3,"successes":3,"any_video":true,"usable_reference":true,"same_goal_pair_validated":false}`；此前 RoboReel 官方 π0.5 M0 46/60（已归档）、M1-A 本地 2 条 PushCube TRAIN H5 + 6 条 Push 人类视频不变。抓取专家已经可用，**原版 PickCube 的 3/3 不能冒充同目标跨行为结果**。下一步已提交 `agcofl/m1b2_same_goal_effect_physics.py` + `scripts/run_m1b2_same_goal_once.sh` + `scripts/m1b2_validate_paired_physics.py`：同一固定 PhysX 场景/最终 Cube 目标、同一 seed=120，实际运行三个独立物理案例（复用原作者 OBB+MPLib 的 Pick+Place、直接 Push-only、先推走动态挡块再 Push-only），保留每段 H5/JSON/MP4 和逐时刻 object/contact 数据，严格判定源 Pick 是否抓起/越障/释放和目标是否真实移开挡块/不抓取地到达目标。控制器中的清障顺序是人工给定的**特权状态可行性上界**，不是学习算法、不是完整能力约束 TAMP；有限短墙场景也**尚不能证明必须先移挡块**。单次运行入口 `bash scripts/run_m1b2_same_goal_once.sh`（自带 stdlib unittest 和 3 次真实物理）已提交，但用户服务器尚未执行。下一阶段必须用强等信息 TAMP 和真实阻断拓扑测试必要性，再决定 M2。说明见 [M1-B2 首次真实 PhysX 对照](2026-11-11_M1B2_REAL_PHYSX_PAIRED_PILOT.md)。

---

## 2026-11-11 — M1-B1 官方 Pick 求解器已真实运行，旧校验器结构错误；仅离线复核禁止重跑

用户服务器真实执行 `m1b_run_official_pick_reference.py`，单元测试报 `KeyError: usable_successful_pick_reference`（仅测试 fixture 漏字段）；正式 3 次 ManiSkill motionplanning 命令返回后，旧 `validate_record` 报 `ValueError: Not the original ManiSkill PickCube-v1 demo metadata`。对照 ManiSkill 官方 `mani_skill/utils/wrappers/record.py` 确认此前包装器误读了 JSON：真实是 `env_info.env_id`，每个 episode 的完成标签是 `episodes[i].success`，不是 `env_id` 与 `episodes[i].info.success`。官方原始 H5/JSON/MP4 应已生成，**尚未核实成功次数**，不可报告源专家成功。已修复格式、测试 fixture 与实际 H5/JSON 一致性核验，新增 `--verify-only`，只验证已有文件并生成小型报告，不调用/重跑仿真。操作：pull 后 `python -m unittest discover -s tests -p 'test_m1b_official_pick_reference.py' -q` 和 `python -m scripts.m1b_run_official_pick_reference --verify-only`，观察 `M1B_OFFICIAL_PICK_EXISTING_VERIFIED` 的 `successes`；若无成功轨迹，查原 `official_solver.log`。M0 46/60 已封存；M1 同目标源/target 插入任务仍未做，不能把原版 PickCube 结果当跨行为胜利。详见 [M1-B 冻结协议及修复](2026-11-11_M1B_EXECUTION_EXPERT_PICK_THEN_BLOCKER_TAMP.md)。

---

## ★ 2026-11-11 — M1-A 真实资产二次核验通过，M1-B 启动官方 Pick 物理专家

用户服务器再次运行修复版 `m1a_inventory_real_assets`，确认 `m0_verified=true`、`local_robot_action_demos=2`（真正位于 `outputs/v120_baselines/robot_train`）、`local_source_push_train_videos=6`、`local_source_pick_train_videos=0`；同目标 Pick→Push 物理配对与 blocker insertion 仍未验证。M0 π0.5 Uniform 46/60 三个 seed 的原始 60 条 CSV/JSON/MD 已归档 GitHub `d5de864`，不再重跑。下一步 M1-B1 不再手写 Pick 控制器：提交 `scripts/m1b_run_official_pick_reference.py`，调用 ManiSkill 原版 `mani_skill.examples.motionplanning.panda.run -e PickCube-v1`，最多 3 次真实 PhysX 专家尝试，保存原作者 H5/JSON/MP4 到 `outputs/m1b_official_pick_reference`，逐 episode 验证 H5 action 及 JSON 成功标记；`python -m scripts.m1b_run_official_pick_reference` 为执行入口，附标准库 unittest，无 pytest、无模型下载。**原版 PickCube 目标≠我们同目标 Pick→Push 的目标，因此只能验证可靠抓取专家；绝不作为 M1 同目标成功结果。** M1-B2 必须在自定义同终态场景中复用可靠专家完成真实 Pick+Place、完成 Push-only 先挪 blocker 再推 cube 的物理行为，再将强 TAMP/Goal-only/Raw Flow/Geometry 等同信息 Baseline 跑完，之后才进入新算法。详见 [M1-B 真实实验计划](2026-11-11_M1B_EXECUTION_EXPERT_PICK_THEN_BLOCKER_TAMP.md)。B1 新代码尚未在用户服务器执行。

---

## 2026-11-11 — M1-A 本地资产门槛首次执行：扫描脚本路径漏报已修复

服务器输出：`3 tests in 0.002s OK`；`M1A_LOCAL_EVIDENCE_GATE` 为 `m0_verified=true`、`historical_robot_h5_valid=2`、`local_robot_action_demos=0`、`local_source_pick_train_videos=0`、`local_source_push_train_videos=6`、`same_goal_pair_verified=false`、`target_insert_verified=false`。经阅读 **真实已执行下载脚本** `scripts/run_v120_real_robot_and_official_baseline.sh`，可证明其 H5 和两路机器人视频原本存放于 `outputs/v120_baselines/robot_train`；而最初的 `scripts/m1a_inventory_real_assets.py` 错查 `official_dataset`，导致本地 H5 假零。已修正扫描目录并新增回归测试；用户机器仍需要一次实际运行才能确认文件目前是否在原处，不能把历史验证记录等同于当前仍有文件。无需重下。M1 原始同目标源 Pick 视频和 target-only blocker insertion 尚未验证，任何 M1 baseline 实绩仍为 NOT_RUN。M0 GitHub d5de864 真实 46/60 关闭；下一项真正研究任务是复用可信 Pick 专家操作源，重新执行并记录相同最终目标的源视频与 target Push 清障仿真，比较等信息强 TAMP/Goal-only/flow，不以旧手写教师或单纯几何图替代。记录见 [M1-A 后续](2026-11-11_M1A_REAL_DATA_AUDIT_NEXT_ACTION.md)。

---

## ★ 2026-11-11 — M0 真实结果全部归档（d5de864），下一步执行 M1-A 数据/配对可行性门槛

GitHub 已核验 `research_reports/2026-11-11_m0_pi05_uniform/verified_summary.json` 和 `per_trial_60.csv`：RoboReel 官方 π0.5 Uniform ND PushCube **46/60 = 76.67%**，逐 seed 分别为 `14/20`、`13/20`、`19/20`，原作者 commit `79a0fb0099e73c5560114753a8706276f93aefc2`。M0 关闭、不再下载/跑官方 π0.5；旧 Vid2Robot 3/60 保留。当前重心转向 M1：此前真实 H5 审核有 2 条 RoboReel PushCube TRAIN 动作轨迹、260 个 TRAIN 人类 PushCube 前视视频远程候选（实际只下载 6 个），**尚无验证通过的源 Pick→目标 Push 同一最终目标配对**。新标准库脚本 `python -m scripts.m1a_inventory_real_assets` 仅核对用户机器上真实文件，排除官方 EVAL，报告 M1 缺口；测试命令用 Python 自带 unittest，不需要 pytest。科学首个任务锁定为：源端 Pick 越障达成同一物体目标，目标 Push-only 需先清理可移动 blocker 再推目标物体；先跑与 Goal-only、Raw Flow、Geometry 以及强 TAMP 的等信息物理对照，**这些 M1 Baseline 仍 NOT_RUN**。详见 [M1-A 数据门槛与首个实验](2026-11-11_M1A_REAL_DATA_AUDIT_NEXT_ACTION.md)。研究方法 M2 尚未启动。

---

## 2026-11-11 — M0 归档命令兼容性修复（不影响正式评测）

真实服务器执行 M0 离线归档时，`pi05` 环境中没有 `pytest`（`No module named pytest`），直接 `python scripts/archive_pi05_uniform_m0.py` 时 Python 模块搜索路径不包含仓库根目录（`No module named agcofl`）。这两项只影响额外测试与结果导出，**原版 RoboReel π0.5 Uniform 46/60 的正式评测仍成功**。已将归档脚本更正为同时支持项目根目录下 `python -m scripts.archive_pi05_uniform_m0` 和直接 `python scripts/archive_pi05_uniform_m0.py`，无需安装 pytest、修改 Conda 或重跑物理实验。后续服务器只执行前者，输出 `M0_PI05_UNIFORM_ARCHIVED` 后提交 `research_reports/2026-11-11_m0_pi05_uniform` 的小型 CSV/JSON/MD 至 GitHub。已修复的归档仍**尚未在服务器成功执行**，不要视原始记录已提交。

---

## ★ 2026-11-11 — M0 正式完成：官方 π0.5 Uniform ND PushCube 46/60，下一步 M1 强 Baseline 与科学必要性

**用户 RTX4090 服务器真实实测**：原版 RoboReel π0.5 Base SigLIP 已成功生成 `siglip_params.pkl`；0080–0099 共 20/20 段官方 EVAL `front.npy` 已全部提取，`PI05_SELECTIVE_ASSET_GATE ready=true`，`PI05_RUNTIME_CHECK ready_for_next_gate=true`，原版 3×20 物理仿真正式输出 `ACTUAL_PI05_PHYSICS_VERIFIED {"variant":"uniform","k":46,"n":60}`，成功率 **76.67%**。旧 Vid2Robot 官方 ND PushCube **3/60=5.00%** 保持不重跑。π0.5 的强基线跑通不等于我们的跨行为创新成立。已新增 `scripts/archive_pi05_uniform_m0.py`：读取服务器实际 `outputs/v140_pi05_uniform/seed_offset_*/summary.json`、官方聚合结果和 config，复核 60 条成功标签并生成带 SHA256 的小型 Markdown/CSV/JSON，后续仅需一次运行并 Git push 归档原始结果。**此原始报告还未由服务器归档到 GitHub**，不得虚构文件已存在。后续 M1 按 [固定协议](2026-11-11_M0_DONE_M1_FROZEN_PREMETHOD_PROTOCOL.md) 核对真实同目标源 Pick→目标 Push 数据与 robot TRAIN action，先执行 Goal-only、Raw Flow、Waypoint/Geometry、能力约束 TAMP（外加相关 flow 方法有验证的 port），证明非平凡 target-only blocker clearance 必要性，再开始 M2 训练；绝不把 M0 官方 PushCube 结果混为同目标 M1 结果。

---

## ★ v1.6.3 — 2026-11-11 原版 SigLIP 权重恢复成功、7/20 视频特征完成、余下 13 段独立进程提取

用户服务器实测：`pi05` 环境 JAX CUDA `gpu:0`、NumPy 1.26.4 均正常；原版 OpenPI Base SigLIP `siglip_params.pkl` 已生成，大小 **1,659,216,368 bytes**；RoboReel 官方 EVAL 视频特征 **0080–0086 共 7/20** 已写入。旧共享 JAX 进程连续处理时出现 `os.fork()`/JAX 多线程警告，随后被系统以 Killed 终止，尚不能仅据此确定 OOM；`0087–0099` 仍缺失，因此 `--require-ready` 正确失败，正式 π0.5 物理评测 **0/60**。新 `--feature-process-mode isolated` 在一个视频一个子进程中调用完全相同的原作者 `_PersistentEncoder`，视频解码发生在 JAX 初始化前，并将 `XLA_PYTHON_CLIENT_PREALLOCATE=false`；每个完成的子进程退出回收内存。已完成的 0080–0086 保持不变，原作者 `front.npy` 原子写入，故可中断续跑而不重新下载任何 checkpoint。主入口 `bash scripts/run_v141_next_gate.sh prepare-features` 已切至独立进程；随后需 `--require-ready` 和 `evaluate-uniform` 验证真实 60 次物理结果。回归测试已提交，服务器尚未重跑。详见 [v1.6.3 提取续跑与内存隔离记录](2026-11-11_V163_PI05_SIGLIP_7OF20_ISOLATED_RESUME.md)。

---

## ★ v1.6.2 — 2026-11-11 已下载 OpenPI Base，修复软链接导致的 `relative_to` 崩溃

用户真实运行原版 SigLIP 一视频 smoke，日志确认 `PI05_PINNED_HF_BASE_STAGED`，说明下载器校验整套基础权重已完成，但原版 OpenPI `maybe_download` 先对缓存软链接调用 `.resolve()`，解析到外部 `openpi_base_hf_mirror`，之后 `local_path.relative_to(cache_dir)` 报错。**不是重新下载 12.4GB 的理由。** 已增加 `bash scripts/run_v141_next_gate.sh repair-pi05-cache`：完全离线从已经核验的原始 Orbax 文件创建 OpenPI 缓存真实目录，利用同文件系统硬链接共享 inode，不重复占用 12.4GB 数据；修复前先验证原目录及旧软链接，采用临时 staging 恢复机制。更新 SigLIP 前置检查，拒绝再把外部软链接误报为“可加载”；补充缓存迁移和无网络测试。服务器尚未实际验证修复；先运行对应 pytest、`repair-pi05-cache`，成功后 `smoke-one-feature`，再进入正式 20 段特征与 60 次评测。原版模型代码不修改；完整记录见 [v1.6.2 缓存迁移修复](2026-11-11_V162_OPENPI_BASE_SYMLINK_RESOLVE_HARDLINK_FIX.md)。

---

## ★ v1.6.1 — 2026-11-10 OpenPI Base 下载已进入真实文件传输，修复 Orbax 零字节标记

用户最新日志确认 `pi05_base/params/array_metadatas/process_0` 已经获得 `git-blob-verified`，并进入固定 20 文件清单的第 5 个文件 `commit_success.txt`，该文件官方 manifest 大小为 **0 字节**。下载器把尚不存在的 `.part` 的大小视作零，误判成已下载完毕，随即校验不存在的文件而报 `missing`。已修复为对空文件执行本地原子创建、验证 Git blob SHA/LFS 摘要，完全不发 HTTP；测试覆盖成功与错误哈希。只需 pull + `pytest tests/test_prepare_pi05_base_from_hf.py` + **原命令** `prepare-pi05-base-hf`，已核验文件自动跳过，禁止删除镜像目录。M0/M1/M2 验收目标不变；checkpoint 尚未全部恢复或正式物理评测。详见 [v1.6.1 零字节 Orbax 修复](2026-11-10_V161_ORBAX_ZERO_BYTE_MARKER_FIX.md)。

---

## ★ v1.6.0 — 2026-11-10 π0.5 Base 下载阻塞修复（M0 目标不变）

用户执行 `prepare-pi05-base-hf` 时，已获取镜像清单但在**第一个极小的** `pi05_base/params/_METADATA` 上遭遇 Hugging Face `FileMetadataError` / `LocalEntryNotFoundError`，尚未下载大权重。原 `hf_hub_download` 的 HEAD/镜像重定向校验不兼容；已改为固定仓库 revision + 公开镜像清单 + 官方 `https://huggingface.co/...` 的 **requests HTTPS GET**（使用用户已配置的 HTTPS_PROXY/VPN），逐文件 `.part`、HTTP Range 断点续传、manifest size/已公布 SHA 校验，失败不覆盖既有完成文件。新增 `probe-pi05-base-hf` 仅验证小文件网络，成功后再运行 `prepare-pi05-base-hf` 下载约12.4GB 原版权重；之后是 `check-siglip-cache` → `smoke-one-feature` → `prepare-features` → `--require-ready` → `evaluate-uniform` 60 次官方评测。此修复尚未在用户 RTX4090 上验证。完整说明见 [v1.6.0 下载恢复步骤](2026-11-10_V160_PI05_BASE_HF_GET_PROXY_RESUME_FIX.md)；科学目标仍按 [M0/M1/M2 冻结协议](2026-11-10_V159_LOCKED_M0_M1_M2_AND_PI05_BASE_MIRROR.md)。

---

## ★ v1.5.9 — 2026-11-10 统一目标：必要 checkpoint 应下载；锁定 M0/M1/M2 科学验收

研究纠偏：不再把“不要无效重复下载/调试”等同于“遇到需要的模型权重就放弃”。**M0** 忠实完成 RoboReel 官方 π0.5 Uniform ND PushCube 20×3 物理评测（Vid2Robot 3/60 已归档，不重跑）；**M1** 构造同目标源 Pick→目标 Push-only 的必要新效果插入任务、明确源视频额外信息与强 TAMP 对照；**M2** 训练能力约束 Effect Editor + 闭环机器人策略并公平比较，若无新意则如实停止。发现可公开访问的 OpenPI 基础 pi05_base 的 HF 第三方镜像 `robotgeneralist/openpi_checkpoint_mirrors2`，锁定 SHA `1fda6b04e83d625eb1a55d68db32050e248cac2b`；新 `scripts/prepare_pi05_base_from_hf.py` 只下载原版基础 Orbax `pi05_base/params`（约 12.4GB），做文件完整性检查并在项目目录创建 OpenPI 缓存链接，不替换 RoboReel Uniform 6.2GB 微调模型，不碰原作者代码。使用 `plan-pi05-base-hf`（纯元数据）后 `prepare-pi05-base-hf`（显式下载），随后 `smoke-one-feature`、`prepare-features`、`--require-ready`、`evaluate-uniform`。Hugging Face 镜像声称是 GCS 原件镜像，尚未独立逐字节比对，保留来源说明。此次仅提交代码，服务器未执行新下载与物理试验。见 [冻结研究计划](2026-11-10_V159_LOCKED_M0_M1_M2_AND_PI05_BASE_MIRROR.md)。

---

## ★ v1.5.6 — 2026-11-10 同目标 Pick→Push 源端 seed120 真实物理测试失败

服务器返回实际 PhysX JSON：源端 281 步，`ever_grasped=true`，但物体最高抬升仅 `0.00008825 m`（0.088 mm），没有越障，目标误差 `0.12924 m`，`env_success=false`、`end_phase=failed`、`valid_pick_source=false`。门槛正确阻止目标 Push-only 对照（`targets=[]`），不可宣称跨行为可行性成立。源控制器只有 grasp 阶段超时才转入 failed；`ever_grasped` 在任意阶段的动作后取样，因此并不意味着 grasp 阶段持续抓住。已新增 **无需运行物理实验** 的 `bash scripts/run_v141_next_gate.sh diagnose-custom-source`，读取已保存的 `source_trace.json`，按阶段输出夹爪接触、TCP/物体位置及抓取命令统计。先诊断已有 281 步，避免盲目重跑调参。记录见 [v1.5.6 物理失败诊断](2026-11-10_V156_SOURCE_GRASP_FAILED_TRACE_DIAG.md)。π0.5 Uniform 官方测试仍因 SigLIP 原版权重缺失阻塞，Vid2Robot 3/60 冻结。

---

## ★ v1.5.2 — 2026-11-10 SigLIP 意外请求额外 11.6GB 基础权重，必须停止自动下载

用户执行真实一视频 SigLIP smoke 时，原作者 `SigLipTokenizer` 因找不到 `$OPENPI_DATA_HOME/pi05_vision_encoder/siglip_params.pkl` 自动访问 `gs://openpi-assets/checkpoints/pi05_base/params`，gcsfs 94 秒显示 0/11.6G 并反复输出 ADC 凭证警告，用户手动中断。**这是 OpenPI π0.5 基础模型，和已经下载的 6.2GB RoboReel Uniform 微调模型不是同一文件。** 原作者只需从基础模型 `PaliGemma.img` 导出 SigLIP pickle；不能在不验证权重等效性的情况下用微调模型替换它并声称官方复现。已提交 `bash scripts/run_v141_next_gate.sh check-siglip-cache`（无网络），并为 `smoke-one-feature` / `prepare-features` 增加默认 fail-closed 缓存门槛，避免再次意外请求 11.6GB。先检查本地是否有原版 `siglip_params.pkl` 或官方预生成 EVAL `front.npy`，否则通过可靠网络按原版权重获取。用户没有实际运行 pi05 物理评测；Vid2Robot 3/60 冻结。详见 [v1.5.2 GCS/缓存审计](2026-11-10_V152_SIGLIP_BASE_11GB_BLOCKER.md)。

---

## ★ v1.5.1 — 2026-11-10 π0.5 环境仅剩 NumPy ABI 依赖冲突

用户新日志：在独立 `pi05` 环境中已成功恢复 `numpy==1.26.4` 和 `opencv-python-headless==4.11.0.86`；pip 仍提示 `numpydantic==2.0.0`、`pandas==3.1.1`、`ml-dtypes==0.6.0`、`opencv-python==5.0.0.93` 依赖 NumPy2。作者明确要求的 Gymnasium 1.3.0 与 ManiSkill 声明的 0.29.1 metadata 警告须通过真实运行验证，不盲目降级。新增 `bash scripts/repair_pi05_numpy_abi.sh` 仅修复 pi05 内上述小型 Python wheel、保持 JAX/GPU/716MB cuDNN 和 6.2GB 官方权重不变；`check_pi05_runtime.py` 新增 ABI 与双 OpenCV wheel 检查。执行 `git pull --ff-only origin main` → `conda activate pi05` → repair → `check-pi05-env` → 单视频 `smoke-one-feature`。本轮未执行 GPU 特征提取或真实物理 π0.5 评测；Vid2Robot 3/60 冻结。详见 [v1.5.1 依赖修复文档](2026-11-10_V151_PI05_NUMPY_ABI_REPAIR.md)。

---

## ★ v1.4.9 — 2026-11-10 π0.5 环境安装的 716MB NVIDIA cuDNN 下载过慢

用户服务器新日志：在独立 `pi05` Conda 环境执行 JAX GPU 依赖安装时，`nvidia-cudnn-cu12==9.28.1.4` 716.6 MB 下载仅约 0.4–0.7MB/s，中途主动取消。**这不是已下载完成的 π0.5 Uniform 6.2GB 权重。** 已将 `scripts/setup_pi05_author_env.sh` 的普通 PyPI 源默认改为清华 TUNA HTTPS 镜像、PyTorch cu126 改为南京大学 HTTPS 镜像（可由 `PI05_PIP_INDEX_URL`、`PI05_TORCH_INDEX_URL` 覆盖）；不更改官方 JAX/其他包版本、全局 pip 配置或已有 Conda 环境。重新激活 `pi05` 后 `git pull --ff-only origin main`，执行 `bash scripts/setup_pi05_author_env.sh`，然后 `check-pi05-env`，通过后才提取 SigLIP 并评测。此轮没有新增物理评测；Vid2Robot 3/60 仍冻结。见 [安装镜像恢复手册](2026-11-10_V149_JAX_CUDNN_MIRROR_RECOVERY.md)。

---

## ★ v1.4.8 — 2026-11-10 π0.5 已通过权重资源预检，只缺 20 个 SigLIP 特征

用户服务器实测：16 个 CPU 测试通过；Uniform 资源预检 checkpoint/train_config/协议检查均通过，唯一缺失 EVAL 0080–0099 的 20 个原版 SigLIP 缓存。Conda 中没有名为 pi05 的环境，但发现旧路径 /home/ubuntu/miniforge3/envs/rokae_pi05（是否有效未知）。已新增 check-pi05-env 环境校验、独立 pi05 原作者依赖安装脚本和相应离线测试：优先验证这个旧路径，不应破坏 agcofl4090 或改装他人环境；不通则另建 Python 3.11 pi05。然后依次 smoke-one-feature → prepare-features → --require-ready → evaluate-uniform（20×3），不得重跑 Vid2Robot。此次没有真实 π0.5 trial。见 [详细步骤](2026-11-10_V148_PI05_ENVIRONMENT_GATE.md)。

---

## ★ v1.4.7 — 2026-11-10 官方 π0.5 Uniform 权重用户反馈已下完，进入特征与物理实验

**用户反馈下载完成；GitHub 尚未核实 RTX4090 本地文件完整性，不等于 checkpoint 已成功加载。** π0.5 官方物理评测仍 NOT_RUN。下一步先运行 `python -m agcofl.v141_pi05_assets --variant uniform` 和 CPU 测试检查原版权重及协议；独立 `pi05` JAX/OpenPI 环境先执行 `bash scripts/run_v141_next_gate.sh smoke-one-feature`（仅 1 段 EVAL 视频原版 SigLIP），再 `prepare-features`（余下 20 段缺失自动跳过已存在特征），确认 `--require-ready` 为 true 后才执行 `evaluate-uniform` 正式 3 seed×20。并行 `agcofl4090` 运行 `custom-physics-dev` 检查同目标 Pick→Push 物理有效性，不当作学习方法结果。**禁止重新下载原权重、重跑 Vid2Robot v1.3 60 次或开启 Keyframes/VLM API。** 详见 [v1.4.7 完整命令](2026-11-10_V147_POSTDOWNLOAD_NEXT_GATES.md)。

---

## ★ v1.4.6 — 2026-11-10 镜像全 308，代理慢：只补齐 13 个缺失官方权重文件

服务器真实状态：HF-Mirror 的 25/25 HEAD 全部 `308 → huggingface.co`，12 个文件本地大小一致（**尚不能当成哈希验证**），13 个文件缺失；用户已有部分权重通过慢代理获得。新增 `scripts/recover_pi05_uniform_missing.py`：默认仅用镜像读取模型元数据并核对本地文件大小/哈希，`bash scripts/run_v141_next_gate.sh plan-checkpoint-repair` 不下载；主动指定 `--download-missing` 时才通过可信代理走官方 HF 原始文件，`.part` 支持 HTTP Range 续传，避免重复下载。可用 `--max-files 1` 先小规模测试，或在系统级 VPN/TUN 具备可用出口时指定 `--routed-vpn`。详见 [v1.4.6 下载恢复协议](2026-11-10_V146_PI05_PARTIAL_PROXY_RECOVERY.md)。当前 **未实际恢复任何新文件或运行 π0.5 物理评测**，v1.3 3/60 冻结；同目标自定义 PhysX 仍优先推进。

---

## ★ v1.4.5 — HF-Mirror 下载 17/25 失败，新增逐文件 HEAD 检查（2026-11-10）

服务器真实反馈：HF-Mirror 可以列出 `LogosRoboticsLab/roboreel_models` 官方 ND Uniform 的 25 文件（6.20 GB），但 `snapshot_download` 在 17/25 报 `FileMetadataError`（可能是镜像文件 308 重定向使 `x-repo-commit` 丢失；与 Hugging Face 已知 issue #4637 相符，**仍需核实确切失败路径**）。已提交 `bash scripts/run_v141_next_gate.sh probe-checkpoint-files`：只发 25 次 HEAD、不下载权重，显示各文件镜像重定向目标和现有文件大小。新版 `prepare-checkpoint-mirror` 会先检查这些路径，若重定向到被阻断的官方站则提前终止，避免再重复下载 6.20 GB。详见 [逐文件排查](2026-11-10_V143_PI05_TLS_RECOVERY.md)。π0.5 真实物理 trial 仍为 0，v1.3 Vid2Robot 3/60 保持冻结。

---

## ★ v1.4.4 — 2026-11-10 DNS/IP 路由异常，改用安全镜像预检查或离线传输

服务器新日志显示系统 DNS `104.244.43.104`、指定 1.1.1.1 查询 `202.160.130.145` 均命中 X/Twitter 网段，强制连接 CDN IP 的 TLS 握手也被重置。**不再重试原域名、不改 hosts、不禁用证书验证**。已提交 `bash scripts/run_v141_next_gate.sh probe-checkpoint-mirror`（只检查公开模型元数据，无下载），通过后可自愿执行 `prepare-checkpoint-mirror` 下载官方 ND Uniform 子集；镜像可能重定向回原站，失败则从联网电脑下载并传输权重。同时继续本地 `custom-physics-dev`。详见 [排障记录](2026-11-10_V143_PI05_TLS_RECOVERY.md)。π0.5 正式物理评测仍 **NOT_RUN**；v1.3 Vid2Robot 3/60 不重跑。

---

## ★ 2026-11-10 — HF 下载根因：服务器 DNS 指向 Dropbox / Facebook

4090 服务端输出：`huggingface.co` 被解析到 `108.160.170.45`（Dropbox 网络）与 `2a03:2880:f10e:83:face:b00c:0:25de`（Meta/Facebook），HTTPS 两个地址均超时，非 Pi0.5 推理故障。外部公共 DNS 返回的是不同的 AWS CloudFront 地址，需排查系统 DNS、/etc/hosts、上游 DNS 或网络出口。GitHub 中 `scripts/diagnose_hf_tls.py` 已新增异常解析警告；[详细只读排查命令](2026-11-10_V143_PI05_TLS_RECOVERY.md)。本轮没有运行真实物理评测，也没有新增成功率；v1.3 3/60 冻结；Keyframes/API 方案仍暂停。

## ★ v1.4.3 (2026-11-10) π0.5 Uniform TLS 下载失败与修复

用户服务器实测：访问 huggingface.co 的官方 HfApi API 时发生证书域名不匹配，**官方 checkpoint 未下载**；20 个评测视频存在但对应 SigLIP 特征全部缺失，π0.5 尚未开始真实物理评测，成功率依旧 `null`。已修复评测脚本的缺失资源 fail-closed 门控（`--require-ready`），并拆分 `prepare-checkpoint` 与 `prepare-features`，提供不泄露 token/代理密码的 `python scripts/diagnose_hf_tls.py`。不要关闭 SSL 验证或假造模型缓存；不重跑 Vid2Robot v1.3。诊断与恢复步骤见 [v1.4.3 TLS 文档](2026-11-10_V143_PI05_TLS_RECOVERY.md)。

---

# AG-COFL / T-RO 跨行为机器人研究：完整交接与历史索引（2026-11-09）

> **新聊天窗口必读。** 唯一当前项目仓库：https://github.com/LemonQC/cehsisuanfa
>
> **归档范围说明：** 本文综合了当前可访问聊天内容、先前相关研究摘要以及此仓库里 v0.4—v1.3 的原始 Markdown、JSON 和代码，尽可能完整保留历次研究决策、真正运行的结果、失败原因、后续实验协议与复制命令。**这不是逐字聊天记录**：我无法从现有接口直接提取每条完整历史发言，因此不能声称100%原文导出。每项关键实验都有指向仓库原始数据的位置。尚未实际执行的步骤严格标注。
>
> **安全：** GitHub PAT、密码、HF token 等不得放到 GitHub；模型权重、H5和原始 MP4 仅留服务器。

## ★ v1.4.2 研究决策（2026-11-10）：暂停一切依赖付费大模型 API 的 Baseline

按最新研究决策，**不再运行依赖 VLM/LLM 外部 API 的 Keyframes 标注、VLM+π0.5 标注等流程**。这类方法即便有论文报告成绩，也不是目前资源应投向的科学主线。现阶段仅推进不依赖付费 API 的官方 π0.5 **Uniform**（本地 JAX/OpenPI 模型与 SigLIP 缓存，仍可能需要下载公开权重）和 **CUSTOM-ADDITIONAL 同目标 Pick→Push-only 物理验证**；随后构建训练数据和真正学习型闭环策略，比较具有同等信息的 Baselines。任何纯几何绕路、伪人工关键帧或仅运行代码，均不得宣称创新成功。

已经收窄执行入口为 `bash scripts/run_v141_next_gate.sh {gate|prepare-uniform|evaluate-uniform|custom-physics-dev}`，不包含 Keyframes 和任何 VLM API 调用。**本次只更新代码和研究协议，未执行新的物理 trial。** 旧 v1.3 Vid2Robot 3/60 保持冻结。


---

## ★ 2026-11-09 v1.4.1 新窗口增量（后于 v1.3 正式结果）

**已向 main 提交 v1.4.1 代码与研究记录，但未运行新的物理 trial。**
阅读 [v1.4.1 审计/下一步](2026-11-09_V141_OFFICIAL_AUDIT_PI05_CUSTOM_GATES.md) 和
[新状态 JSON](../research_reports/2026-11-09_v141/status.json)。
官方 RoboReel 站点披露的 Vid2Robot ND PushCube 成绩为 5.0±2.9%，
本仓库真实 3/60=5.0%，数值一致；作者页面标为 std，本项目计算 SEM，
不能把聚合数值一致等同于逐动作完全一致。

已新增官方 video-conditioned π0.5 Uniform/Keyframes 的**有限20段**
checkpoint/SigLIP/Keyframe准备入口（后者使用作者VLM API，需显式授权并可能计费），
保留上游 action_horizon=20，禁止误用 Vid2Robot chunk=1；
已有评测脚本均带严格协议验证。新脚本：
`scripts/run_v141_next_gate.sh`，命令 `gate` 默认只做预检查，
`prepare-uniform`、`evaluate-uniform`、`prepare-keyframes`、
`evaluate-keyframes` 均需要服务器对应模型环境及资产；
**Uniform和Keyframes目前均未执行物理评测**，不能引用作者报告为本地结果。

同目标 CUSTOM-ADDITIONAL Pick越障→Push-only 绕障场景进一步增加
物体真实墙体越过观测证据、仅推行桌面支撑约束与对应 CPU测试。
`custom-physics-dev` 是一次新的真实 ManiSkill/PhysX scripted
privileged-state 开发试验入口，**尚未运行，非学习型创新结果**。
下一科学门槛是验证真正成功的同目标源/目标物理配对，再提供
同信息预算的 Goal-only、Raw Flow、Waypoint Oracle 与学习型控制
以及必要事件/无关绕路/不可达的独立对照。绝不能只凭几何绕路宣称T-RO创新。

---

## ★ 最新补充：v1.3 正式结果已经完成并上传 GitHub

**本节替换下面历史部分“v1.3 尚未运行”的旧状态。此前那段话记录的是旧时点，不能用于判断当前研究进展。**

### 已核验的真实实验

GitHub main 已包含研究报告：
- research_reports/2026-11-09_v130/official_aggregated_summary.json
- research_reports/2026-11-09_v130/status.json
- research_reports/2026-11-09_v130/README.md

本次报告归档提交为 99377a72358eefea8161d465da5f12807e5cf3e8，已由 GitHub 文件读取确认。报告显示 OFFICIAL_VID2ROBOT_ND_3PASS_EVALUATED，full_3pass_complete=true，errors=[]，官方作者 Vid2Robot 的 PushCube clean-human/clean-env ND 单条件，EVAL 0080–0099 共20段视频，每段在三组环境初始化上评测。官方原 checkpoint 严格加载成功，max_rollout_steps=800，chunk_size=1。

| seed offset | 完整物理 trial | 成功 trial | 成功率 | 服务器日志耗时 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 20 | 1 | 5% | 332秒 |
| 100000 | 20 | 2 | 10% | 319秒 |
| 200000 | 20 | 0 | 0% | 344秒 |
| 合计 | **60** | **3** | **5%** | 995秒 |

原始报告中的三个 seed 成功率均值为 0.05，跨 seed SEM 为 0.02886751345948129，即 **5.00% ± 2.89 个百分点**。成功演示编号：seed 0 为 0095；seed 100000 为 0082 和 0097；seed 200000 无成功。20个人类评测视频在不同环境bank中重复使用，不应将60个物理 rollout 解释成60个独立人类演示。

这个结果仅代表**官方 Vid2Robot 在当前 PushCube 无干扰单格**的已复现实验结果；不是整个 RoboReel 全任务、WD/ED benchmark，也不是我们的创新方法结果，更不等于跨行为研究不值得继续。

### 本轮 Git 问题完整经过及现状

原始正式评测执行成功后，服务器 detached HEAD 下本地 commit b82d0b9 保存了 3 个 v1.3 报告文件，第一次向 origin/main push 因 non-fast-forward 失败。同时主分支在服务器评测期间已收到新提交，因此需要正常整合。

用户曾复制字面命令 git pull ...，其中 ... 被当作仓库路径；普通 git pull 又因 detached HEAD 无所属分支而失败。旧工作树还残留了之前进行到提交 9a116e6 的交互式 rebase。不能在未清理该状态下重开 rebase，也不能在 cherry-pick 时强行切换分支。之前建议先给 b82d0b9 和旧 main 建备份分支，再用 git rebase --quit 安全退出旧 rebase；基于 origin/main 建 publish-v130-results 分支，再 cherry-pick b82d0b9。

新 cherry-pick 在 Git 身份缺失时停下：本次 3 个报告文件已经暂存、没有冲突，但 Git 尚未生成提交，直接 git push 只会报告 Everything up-to-date。正确处理是在当前仓库配置 user.name=LemonQC、user.email=30914380+LemonQC@users.noreply.github.com，继续已有的 git cherry-pick --continue，然后提交并推送；不能重新 cherry-pick，也不能使用 git push --force、git reset --hard 或无理由中止原提交。

**最新远端事实：** 已直接读取到远端 main 的 99377a7 正式实验报告和完整 3/60 数据，因此该报告现在已经保存在 GitHub，无需再次运行60次实验。此聊天没直接看到用户最后一组本地 git status 输出，不能断言服务器本地是否已彻底恢复正常 main；若继续服务器操作，先检查 git status、git branch -vv 和 git log -1 --oneline 与 origin/main 的关系。

### 执行进度、GPU监控和下一步

此前原版作者日志 logger.info 未启用，控制台只在每20个 trial 后报告 PASS_VERIFIED，用户看不到组内进度。后续已向 GitHub 提交 agcofl/official_live_progress.py、tests/test_v130_live_progress.py 与 scripts/show_v130_progress.sh，仅对将来的运行增加每个 demo 的 START、DONE、时长和只读 progress.json。不能对已经运行的旧进程补进度，也不必因此重做这60次评测。

GPU 显存占用约63%不是故障，也不等于计算利用率必须100%。GPU-Util可能受CPU物理仿真、视频解码与串行逐步控制的交替影响。不能为追求100%改变官方控制模式、动作chunk和成功判据。

**下一步行动：**
1. 不再重复 v1.3 Vid2Robot PushCube ND 60次实验，不重训旧CNN/ViT，不下载已核验的官方权重；不要在EVAL 0080–0099上调参。
2. 对照官方论文、官方模型加载、控制/仿真配置和原始物理成功判据分析当前3/60低成功率的复现一致性；明确原版论文任务/条件是否真的相同，不把复现差异误判为方法差。
3. 优先接入官方 video-conditioned π0.5 Uniform/Keyframes、UniSkill 等强学习型Baseline，报告原版与移植差异、真实可执行和实际trial结果，不要仅列算法名。
4. 并行尽快落地独立标注为 CUSTOM-ADDITIONAL 的**同目标 Pick-source → Push-only-target** 物理任务：必要中间事件、可省略绕行、不可达、真实接触与支撑限制。与 Goal-only、Goal+Waypoint Oracle、Raw Flow、几何方案和同信息/训练预算学习策略形成实际闭环对照。
5. 确保所有后续关键决定、原始结果和代码继续提交到 LemonQC/cehsisuanfa。不要单独因为某个基线低成功率就放弃研究方向。

### 新窗口最新版启动提示

> 继续 AG-COFL 的 T-RO 跨行为物体效应迁移项目。仓库 https://github.com/LemonQC/cehsisuanfa 。请先完整读取 docs/RESEARCH_CONVERSATION_HANDOFF_FULL_2026-11-09.md、README、docs/BASELINE_MATRIX_TRO_V110.md 和最新 research_reports。特别是 research_reports/2026-11-09_v130/official_aggregated_summary.json 与 status.json，确认已经跑完的正式结果。**v1.3 原版 Vid2Robot PushCube ND 共20评测视频×3 seed=60次真实物理评测，成功数1/20、2/20、0/20，总计3/60=5.0%，SEM=2.89个百分点；报告已上传 GitHub。绝对不要再跑一遍。** 历次 Git detached HEAD、rebase、cherry-pick 和身份问题详见交接文档。下一步先核实官方复现一致性，再推进真实可运行的 π0.5 Uniform/Keyframes 等强 Baseline，并行落地同最终任务目标下 source Pick → target Push-only 能力约束的 Object Effect Projection + learned policy 物理对照。不得将原版 PickCube 与 PushCube 的不同目标假装为同目标迁移；坚持 Goal-only、Raw Flow、Goal+Waypoint Oracle 等强对照，信息预算相同，实际物理成功率可核验。直接查看 GitHub、分析结果、写代码并提交新的决策/报告，不重复无效实验也不虚构尚未执行的方法。

---

## 0. 项目、服务器、用户不可变要求

- GitHub 主项目：https://github.com/LemonQC/cehsisuanfa 。所有阶段性研究决定、代码变化、失败教训、实验结果和下一步计划提交 Markdown 和小型 JSON/CSV，便于任何新窗口恢复。
- RTX4090 Ubuntu 服务器路径：/mnt/4Tdata/LemonData/cehsisuanfa；Conda：agcofl4090；github main 分支为代码同步与结果归档来源。安装了 GitHub CLI 并通过 gh auth setup-git 解决写权限。
- 目标是 IEEE T-RO 水准：**跨行为人类视频—机器人操作、对象物理效应迁移、真实闭环物理验证**，不是网络叠加、路径规划演示、简单 CNN 更换 ViT 或单任务成功。
- 用户要求先冻结有代表性的近年 Baseline，最好 >=10 项有实质意义的学习基线和强控制，官方代码优先、忠实移植须注明差异；标准 ManiSkill/RoboReel 物理环境、相同数据/场景/测评预算。不要直接因为一个 Baseline 成绩差就否定专家多轮形成的科学方向。
- GPU 很宝贵；不能无目的重复跑早期 CNN/ViT/代理合成数据。旧文件和数据可复用；不下载整个约100GB+数据集；人类 EVAL 不能用于模型训练。
- 用户希望直接读 GitHub 真实结果、改完代码提交仓库、自动发布小结果，不反复问用户已经给出的信息。不得把日志 V100_COMPLETE、V121_AUTOPUBLISH_OK、V130_COMPLETED_SCRIPT 等当作算法成功，必须核验真实 rollout、模型参数加载、per_trial 和 full_3pass_complete。
- 不要建议 git push --force、git reset --hard；先检查本地未完成 cherry-pick/rebase 与远端差异，不泄漏 GitHub token。

## 1. 核心科学路线与论文逻辑

核心问题：**Source Object Effect → Target Robot Capability-aware Effect Projection → Learned Closed-loop Policy**。

源人类演示可能 Pick、Lift、绕过障碍，目标 Panda 可能只允许 Push、维持接触/桌面支撑，受到物理障碍、机械臂工作空间和关节可达限制。应保留共通目标和真正必要的中间事件与对象状态效应，丢弃人类特有且目标机器人不可执行、无必要性的轨迹片段。

对象效应的候选结构：Goal、必要 Events/Waypoints、接触/支撑 Mode、3D object pose/SE(3) trajectory；**这仍是拟实现表示，不可声称 HSV 蓝色区域中心已经提供真3D标定姿态。**

实验中必须区分：
- **必要中间事件**：不通过就不能实现目标，比如绕过实体障碍、完成支撑状态转移、先满足接触条件；
- **习惯性无关绕行**：源视频特有但可省略，不能强迫目标机器人照做；
- **Target unreachable**：目标只有举起才能达到而 Push-only 不可达，正确输出不可达，不能编造成功；
- **Target reachable but source flow infeasible**：同一最终目标允许通过推行绕路完成，原始 Pick source 的抬升不可直接复现。

必须以真实物理 rollouts 对照：Raw Flow、时间重采样/warp、Goal-only、Goal+Waypoint Oracle、简单几何/A*、相同观测与训练预算的学习策略、我们的能力约束投影，以及单独列示的 privileged-state 上界。若只是做一条与 Oracle 一样的路径，就没有创新贡献。

**特别警告：原始 RoboReel PickCube-v2 和 PushCube-v2 的终极目标不同（Lift vs Push-to-goal）**；不能当作同目标跨行为对照。官方 RoboReel 多任务评测用于通用视频条件机器人策略性能；另建明确标记 CUSTOM-ADDITIONAL 的**同目标 Pick-source → Push-target 能力冲突任务**用于科学主张。

用户此前明确推翻了把“主动探测未知动力学”作为主线的方案（条件难控、必要性不足）。后续不应回到繁琐接触 teacher 调参、无物理差异的泛泛技能拼接、仅2D预测或仅造复杂网络的方向。

## 2. 时间线：阶段结果（运行事实与计划严格分离）

### v0.1—v0.3 基础实现

- 建立 ManiSkill3 / Panda PickCube/PushCube 操作和最初 teacher/BC、Full/Anchor/End 监督框架，headless 服务器测试。
- 这些早期脚手架不是可信的 T-RO 性能。过去用户文件中存在 Object_Flow_TRO_Research_Chat_Complete_2026-11-07.md、AGCOFL_GitHub_Source_Transfer_v020.txt、BASELINE_MATRIX.md；这几份不一定在当前仓库，此处只记名称，不假设已经上传。
- 更早的 PickCube 单任务 DP/ACT 实验曾有运行日志与结果，但其不同任务/训练设置不能混入当前 RoboReel 官方 Vid2Robot 成绩表，除非再核实原始实验协议。

### v0.4—v0.6 接触型物理 teacher 调试

源记录：
- docs/2026-11-09_v040_failure_and_v050_fix.md
- docs/2026-11-09_v050_failure_and_v060_fix.md

v0.4 seed=100 两条边路线 attempted=2、success=0，沿直线段重复尝试深接触，实际 TCP 下压不足但具体 IK 内因未证实。v0.5 attempted=2、success=1，上边路线成功、174步，下边因为中间拐点位置误差阈值宽松而最后推动失败。v0.6 采用更严格路点判定、物体运动和TCP telemetry、限定再接触，方案形成时不能声称结果已成功。科研决策：teacher 调参仅解决物理数据采集，失败示范不可混入正样本，也不作为论文创新。

### v0.7 合成代理 Full/Anchor/End

源记录：
- docs/2026-11-09_v070_proxy_result_and_next_gate.md
- docs/PARTIAL_FLOW_EXPERIMENT_V070.md

实际运行在 4090 的15轮训练：12个合成布局、96个合成mask，layout级 train/val/test 8/2/2；Full、Anchor、End 三种监督方式和梯度链路有效。但这些都是 SYNTHETIC_PROXY_DEBUG_NOT_PAPER 的 IoU，**没有真人数据或学习机器人策略的成功率**。决策：冻结工程接口，停止反复增加合成 epoch/seed，寻找可复现真实基准。

### v0.8—v0.8.1 RoboReel 真实视频与物理诊断

源记录：
- docs/ROBOREEL_REAL_BRIDGE_V080.md
- docs/2026-11-09_v081_mainland_download_fallback.md
- docs/2026-11-09_roboreel_episode0_physics_success.md
- docs/2026-11-09_roboreel_push_xy_regression.md

基准：https://roboreel.github.io/；https://github.com/gupta-anmol99/roboreel ；真实数据：https://huggingface.co/datasets/LogosRoboticsLab/roboreel 。

通过 hf-mirror.com 获得官方 EVAL 0080 front.mp4（214帧）；HSV蓝色候选在抽样帧中全部检测到，但“有候选”不是“真实目标ID正确”。

ManiSkill/PhysX 中一次 **手写 privileged-state PushCube-v2** 成功：31步，最终物体到目标距离0.045422m，处于0.05m阈值以内。这个脚本能读 oracle 物体/目标状态，动作**不由人类视频直接决定**，只是实物理链路诊断，1/1不可混入学习模型成功率。曾修正 cube[:2] XY 与 last_cube XYZ 数组广播错误。

### v0.9 弱 CNN 视频候选 mask

源记录：
- docs/2026-11-09_v090_real_video_training_and_replay.md
- docs/2026-11-09_v100_effect_gate_and_train_data.md

用官方 **EVAL** 0080 一个视频、14个重叠窗口作内部开发训练；模型输入32×32蓝候选mask、两个过去时间戳，未来8帧Full/Anchor/End。20个epoch实际GPU训练；Full 训练 full-mask IoU 0.0000、Anchor 0.002397、End 0.002357，三种末帧IoU全部0，产生大面积错误前景。无独立视频泛化、训练视频属于EVAL，不能正式报告。用户明确指出 CNN 太弱，不可凭CNN失败否定整个方向，后续使用现代预训练ViT并优先官方强Baseline。

### v1.0 真实 TRAIN 数据发现、无学习对照、几何效应门槛

源记录：
- docs/2026-11-09_v100_effect_gate_and_train_data.md
- research_reports/2026-11-09_v100/learnability_gate.json
- research_reports/2026-11-09_v100/effect_gate.json

旧mask平均仅2.482像素；Persistence mean IoU 0.798759、Constant Velocity 0.749486，CNN方法接近0；证明旧表示不合理，不能说明跨行为科学方向无效。几何投影与 Goal+Waypoint Oracle 完全同路且长度0.15m；来源还不是 Pick 的真实三维效果，不能宣称新物理方法成功。

官方真实 TRAIN 人类视频候选约260个，选择性获取6段。重要：官方 human_demos/eval/no_distraction/push_cube/0080-0099 为评测，不作正式训练。Git push/pull 曾有未完成 CHERRY_PICK_HEAD、non-fast-forward、认证问题，已经通过 gh CLI 配置解决，不要强推覆盖远端。

### v1.1 真预训练 ViT-S/14 + temporal Transformer

源记录：
- docs/BASELINE_MATRIX_TRO_V110.md
- docs/2026-11-09_V110_RESULTS_V120_DECISIONS.md
- research_reports/2026-11-09_v110/vit_report.json
- research_reports/2026-11-09_v110/vit_summary.csv

RTX4090 确认运行 facebook/dinov2-small 预训练 ViT，224x224 RGB、冻结视觉特征、2层 temporal Transformer，Full/Anchor/End各训练15轮。6段官方 TRAIN PushCube 视频以**整视频**分4 train/1 val/1 test；测试仅1独立视频、10个窗口（非10独立样本）。

| 方法 | 平均未来位置误差（px，越低越好） | 末帧误差 |
| --- | ---: | ---: |
| Constant Velocity | **7.58** | 17.69 |
| Persistence | 9.89 | 17.57 |
| Full ViT | 9.73 | **16.20** |
| Anchor ViT | 14.50 | 19.55 |
| End ViT | 11.85 | 21.49 |

结论：ViT 链路可以训练、Full末帧略好，但全时序不及恒速；只是**二维HSV中心**诊断，非标定三维效应，非真实机器人动作学习。不要继续反复训练这一小规模诊断。

核查官方模型仓库 https://github.com/weiweigu1998/roboreel_models ，固定commit **79a0fb0099e73c5560114753a8706276f93aefc2**：Video-conditioned π0.5 Uniform/Keyframes、Vid2Robot、UniSkill、VLM+π0.5有代码入口，不能因为脚本存在就说“运行过全部Baselines”。

### v1.2 真实机器人 TRAIN 动作数据首次验证

源记录：
- docs/2026-11-09_V120_FINDINGS_V121_RECOVERY.md
- docs/2026-11-09_V120_API_MOCK_FIX.md
- research_reports/2026-11-09_v120/real_robot_h5_audit.json
- research_reports/2026-11-09_v120/official_vid2robot_assets.json

官方发布 raw 路径是 **robot_trajectories/no_distraction/push_cube/**；原先误用作者代码内部别名 without_distraction 导致404，已修正。

真实 robot TRAIN demo 0000、0001：H5 动作形状**138×8、136×8**，Panda articulation**139×31、137×31**，base_camera/hand_camera 512x512 MP4均可解码，是真正可用的视觉-动作标签入口。权重和20评测视频的元数据已找到，但当时 hf_hub_download 报 LocalEntryNotFoundError，策略尚未能运行。单测曾因模拟 HfApi MockApi 丢失 repo_id 参数导致4失败，后已修复；这是测试桩错误不是研究模型错误。

### v1.2.1 真正的官方学习型策略物理闭环成功连通

源报告：
- research_reports/2026-11-09_v121/official_vid2robot_assets.json
- research_reports/2026-11-09_v121/official_vid2robot_smoke.json
- research_reports/2026-11-09_v121/official_runner_dev_summary.json
- 用户4090自动归档提交 **ede31ff2f3355ecd03cb8d8879a0a866eba7df9b**。

原版官方 Vid2Robot ND checkpoint：
- ckpt.pt **2,368,873,705 bytes**，SHA256 **8902745333e46e30e7988574e5bdaf870fa29278dfbc65f71e884a7193377197**
- train_config.json 911bytes
- 官方 EVAL PushCube front MP4 **20/20**（IDs 0080—0099），合计**97,448,129 bytes**。完整验证大小与SHA256。
- 模型**严格加载完整官方 state_dict**，使用作者原本 Vid2Robot + ManiSkill 物理 simulator，Panda视觉输入和8维动作，非 privileged-state 手写策略。
- 由于 timm 会先下载一套随后被完整checkpoint覆盖的ImageNet初始化权重，适配层仅暂时关闭冗余初始化下载，在完整严格参数加载后恢复；已透明记录这项工程性差异。
- 0080固定DEV集成试验**真实执行**、200步最大预算、success=false (0/1)；**不是正式成功率**。该0/1不能否定算法，也不能回头“优化0080”导致评测泄漏。

### v1.3 正在等待4090执行的官方真实Baseline（不要误称已完成）

源决策：docs/2026-11-09_V121_SUCCESS_V130_BASELINE_PLAN.md 。

截至本归档创建时，v1.3代码/测试已在 main（此前核验HEAD 03b77a865aefaea1f95a74a379eb8fcf9a3c9cdd），**没有看到v1.3真正20×3的测评报告**。

已冻结协议：
- 官方原始模型 + checkpoint + RoboReel ManiSkill Panda，**PushCube No Distraction clean human/clean env 单一评测条件**，原始 EVAL 0080—0099 共20个视频。
- seed_offset_base 0、100000、200000，**20次×3个seed pass=60物理rollouts**。chunk_size=1，max_rollout_steps=800，官方 pd_joint_pos，原始 human FRONT 与 robot base_camera 视觉历史。
- 不修改 EVAL demo scope、不对0080失败调参。每个pass只有20个不同demo ID、真实boolean success、表格统计一致时才计入。输出每pass k/20、三pass均值和SEM，不能把60次当作60个不同人类演示。
- 可验证复用已经完整完成的seed；中断的pass不能冒充0/20，完整20×3才是该cell正式复现。只是一个ND任务格，不等于全部RoboReel基准十任务、WD、ED、PD。
- 源：agcofl/vid2robot_official_full_eval.py、agcofl/official_v130_protocol.py、tests/test_v130_official_protocol.py；执行：scripts/run_v130_official_vid2robot_3pass.sh；自动上传：scripts/publish_v130_official_eval.sh。

**当前唯一优先服务器执行命令：**

~~~bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --rebase --autostash origin main
conda activate agcofl4090
set -o pipefail
bash scripts/run_v130_official_vid2robot_3pass.sh 2>&1 | tee baseline_v130.log
~~~

结果预期自动归档 research_reports/2026-11-09_v130/（**仅实际运行并推送后才出现**）。查看 OFFICIAL_V130_PASS_VERIFIED（每20条物理记录）、OFFICIAL_V130_EVALUATION_GATE（三pass状态）、V130_AUTOPUBLISH_OK（GitHub提交，不代表算法高成功率）。

## 3. 研究 Baseline 矩阵及当前真实性状态

完整矩阵详见 docs/BASELINE_MATRIX_TRO_V110.md 。所有方法需要明确 OFFICIAL_CODE_AVAILABLE / OFFICIAL_EXECUTED / FAITHFUL_PORT / REIMPLEMENTED / BLOCKED / NOT_RUN 标签。

| 方法 | 2026-11-09 当前事实 | 下一步 |
| --- | --- | --- |
| Robot-only RGB ViT Transformer BC | 仅2段真实TRAIN H5已验，正式训练/测试尚未建立 | 扩真实训练集，按原始8维动作和视觉对齐 |
| Robot-only Diffusion Policy | 曾在更早单任务中测试，不可视作本RoboReel官方成绩 | 任务/动作空间/物理场景对齐 |
| Vid.Cond.π0.5 Uniform | 作者提供代码；当前未验证原版权重闭环 | 建兼容环境、冻结评测 |
| Vid.Cond.π0.5 Keyframes | 作者提供代码；当前未验证原版权重闭环 | 同基座同数据预算对照 |
| **Vid2Robot 原版** | **已真实严格加载与0080物理运行0/1**；v1.3 20×3尚待运行 | 现在首先执行 |
| UniSkill | 官方源码已确认，尚未完整复现 | 独立模型依赖环境 |
| VLM+π0.5 | 官方源码可查，尚未实际跑 | 考虑输入/模型资源预算 |
| SeeDo | 尚未确认本代码库可运行官方发布版 | 不能冒充已复现 |
| PointPolicy | 需要真实标定3D输入；尚未建立 | 标定/真正3D标签可用后 |
| Cross-Behavior Learning With Object Flow Prediction for Robotic Manipulation（T-RO 2026） | 核心研究竞争者，尚未确认可直接执行的原作者源码 | 核实后忠实复现 |
| Im2Flow2Act | 作者开源代码，尚未在RoboReel物理协议忠实移植 | 明确环境差异 |
| EC-Flow | 作者开源代码，原实验Meta-World，尚未RoboReel port | 不能直接照抄成绩 |
| DINOv2-S/14 + Temporal Transformer | 真4090曾训练，只有小规模2D候选中心测试 | 工程诊断完成，不重复 |
| Persistence / Constant Velocity / Goal-only / Goal+Waypoint Oracle | 存在部分真实先前对照，但无统一完整机器人策略评测 | 在同目标物理机制试验中独立报告 |
| privileged-state Panda teacher | 旧PushCube物理1/1诊断成功 | **oracle上界独立成表**，不得与视觉策略公平混比 |

用户要求足够扎实的 Baseline（包括近年SOTA、真实reproduction）而非“找十几个名字”；正式论文需在相同任务/动作/信息条件下覆盖相互补充的强算法，若不同环境/模型需诚实注释。

## 4. 下一步 T-RO 科学创新具体实验

**两张主结果表不能混为一谈：**

A. RoboReel 官方基准：Vid2Robot、π0.5 Uniform/Keyframes、UniSkill、Robot-only BC/DP等学习机器人策略，在相同物理评测协议下给出真正k/N、跨seed均值与SEM/置信区间。Clean ND、WD、ED不可混写；对官方已发布结果与我们真实重跑结果分别注明。

B. **新增同目标 Capability-conflict 任务**（单独注明CUSTOM-ADDITIONAL）：source可Pick/抬起，target只可Push/桌面支持；同一个对象、终点、初始状态与物理世界。构造“源抬升不可移植但推行可达”，“必要中间事件与可省略冗余绕行”，“目标Push确实不可达”三类，且实际引入可核验的障碍、接触、支撑和关节限制。

物理强对照：Goal-only、Goal+Waypoint Oracle、Raw Object Flow、time resampling、2D warp、仅几何A*、简单可执行性过滤、同骨干flow-conditioned learned policy、提出的capability-aware effect projection、privileged oracle（单独表）。观测数据/动作预算/训练步骤/随机种子须配平。衡量任务成功、保持必需中间状态、删除无关轨迹、接触/支撑违反、碰撞和任务步数。

**止损标准是针对具体创新主张，而非整个项目。** 如果只得到与Waypoint Oracle完全相同的几何路线并且物理执行上无增益，不能包装成T-RO贡献。若真实配对任务仍没有一个明确的非平凡科学gap，应调整/推翻该主张，而不是堆更多模块或仅增大ViT。

## 5. 代码/数据/学术复现硬约束

- 人类真实训练只用 official TRAIN 目录，不能在官方EVAL 0080—0099上继续调参/训练；必须按video/scene划分训练、验证和测试，不能误把重叠窗口算作独立样本。
- 真实机器人动作：robot_trajectories/no_distraction/<task>/<demo>/，H5 traj_0/actions 为8维，对齐真实 RGB base_camera和hand_camera；源代码的 without_distraction 是内部映射。
- 官方模型源码PIN commit 79a0fb0099e73c5560114753a8706276f93aefc2；作者 Vid2Robot PyTorch 环境依赖可能不同于π0.5的JAX、UniSkill等，不必勉强一个conda，但保证**物理成功判据和观测/action协议一致**。注明任何port/compatibility差异。
- 使用GPU但需边界清楚：单元测试通过、权重传输成功、strict checkpoint load、真实 rollout、正式paper-ready多pass统计是**五件不同的事**。
- 下载采用有界文件清单/镜像续传；避免全量下载百万文件；不要重下已核验权重；不要上传任何大型权重、视频、H5、个人密钥。
- headless server，保留工作CUDA/PyTorch/ManiSkill，避免无故升级驱动。
- GitHub权限先前已成功；若push因本地落后失败，先安全同步远端，不force覆盖。

## 6. 新窗口固定执行顺序

1. **首先读此交接文件和最新真实GitHub报告**；尤其查 research_reports/2026-11-09_v130/status.json 与 official_aggregated_summary.json 是否出现。若没出现，v1.3**还没得到真实结果**。
2. 若v1.3完成，立即分析每seed成功/20与总体均值/SEM，审计异常、任务失败与官方原始benchmark差异；继续添加下一强Baseline的真实权重/官方物理评测。
3. 若v1.3被依赖/资源/代码阻塞，直接读取 run_status.json 错误并修复代码推送，不要让用户再次跑旧ViT或下载2.37GB。
4. 并行凝练同目标源Pick/目标Push的能力冲突机制和强Baseline，先做明确可重复的物理对照，然后决定Full/Anchor/End或事件表示的论文增益是否真实。
5. 每一次真实决策、代码版本、结果和下一步都写入Markdown并提交**LemonQC/cehsisuanfa**。

## 7. 可供新ChatGPT窗口复制的启动提示

> 我们继续 T-RO 项目 AG-COFL / Cross-Behavior Object Effect Transfer。GitHub 唯一当前项目仓库是 https://github.com/LemonQC/cehsisuanfa 。请先读取 docs/RESEARCH_CONVERSATION_HANDOFF_FULL_2026-11-09.md 全文、docs/BASELINE_MATRIX_TRO_V110.md、docs/2026-11-09_V121_SUCCESS_V130_BASELINE_PLAN.md 和最新 research_reports/；不要凭历史记忆猜新结果。
>
> 当前已真4090跑过小规模 DINOv2 ViT诊断；已拿到官方真正 TRAIN Panda H5 2段；Vid2Robot 2.37GB官方 checkpoint 和20 EVAL 视频已验证并严格加载；0080固定官方机器人学习策略物理 smoke 真实运行 success=false，但不是paper分数。v1.3的20视频×3seed正式评测代码已提交，先从GitHub确认是否完成（不能直接认为已完成）。
>
> 科学主线是 source Pick object effect 到 target Push-only 的 capability-aware physical effect projection + learned robot policy，在同一任务目标下保留必要事件、抑制无关绕行、识别不可达，必须与真实视频条件策略、raw object flow、Goal-only、Goal+Waypoint Oracle、简单几何和物理上界形成信息预算匹配的对照。原始 RoboReel PickCube和PushCube目标不同，不能伪称同目标任务。
>
> 请直接检查代码和实验结果、作出下一步决定并提交可运行代码和GitHub Markdown记录；禁止虚构Baseline成功、无意义重复CNN/ViT、小数据评测泄漏、主观否定多轮形成的研究课题。
