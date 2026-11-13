# TRO1 · G0 双目标真人视频可辨识性：已执行审计（2026-11-13）

**结论：`G0_BLOCKED`，本轮不准开始 G1/PhysX 或训练。**

这不是一次“建议先做审计”的计划。本轮已核查公开原始场景定义和候选公开数据源，已经生成、实际运行一个可复核的只读审计器，并在本地离线跑通 15 项单元测试与一个真实 CLI 入口（测试夹具输出 `G0_NOT_PASSED`，进程退出码 2）。**这里没有访问或扫描用户服务器的新增本地文件，因此不能声称完成了服务器本地存储中的全部素材普查。** 需要服务器直接执行同一入口才能得到它的本地索引。程序既不会下载数据、运行 JAX/SigLIP、启动 ManiSkill，也不会更改已有 M0 或 StackCups 结果。

## 已核查事实与候选优先级

| 数据来源 | 来源已证实 | 是否已经得到“同场景、反目标、合法可用、真实视频”的配对 | 审计决定 |
|---|---|---|---|
| RoboReel PushCube-v2 | 固定 `PUSH_DISTANCE=0.15`，`target_xyz[...,0] += self.PUSH_DISTANCE`，视频编号不改变目标 | 否 | NO：仅保留官方 π0.5 46/60 的执行能力基线 |
| RoboReel StackCups-v1 | 固定三杯颜色堆叠顺序；π0.5 真实 DEV 0/4，录像有局部执行 | 否 | NO：不再做阶段同步开发 |
| **Qualcomm Something-Something V2** | 官方类别确有 *Pushing something from left to right* / *right to left*；真实人类视频、研究许可协议，整个视频包约 19.4 GB | **否**。目前只验证了类别，未取得任意两条授权视频、未检查同对象、相机映射、目标几何与起始条件 | **最接近候选，但现在禁止作为 G0 通过证据**，也不批量下载 |
| H2RBench (CoRL 2026) | 发布了人类视频与配对仿真/真实机器人场景，四类固定目标任务 | 未找到已核实的同场景、相反目标配对 | NO_PAIR_VERIFIED |
| EgoDex (2025) | 真人第一视角，许多操作任务；大体量 | 没有已核实双目标配对及与 ManiSkill 的统一场景映射 | NO_PAIR_VERIFIED |

核查页面（2026-11-13）：
- RoboReel 官方任务 [PushCube 源码](https://github.com/gupta-anmol99/roboreel/blob/afaf4a03df0be1e2347becde96da4f34e0de83fc/src/envs/push_cube.py) 与 [StackCups 源码](https://github.com/gupta-anmol99/roboreel/blob/afaf4a03df0be1e2347becde96da4f34e0de83fc/src/envs/stack_cups.py)。
- [Qualcomm 官方 Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) 与其 [研究使用许可协议](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/jester_something_something_exercise_research_license_final_qti_28jul2022.pdf)；[官方下载页](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads)。没有接受许可、验证区域可访问性之前，不调用下载器。
- [H2RBench 项目](https://h2rbench.github.io/) 与 [公开仓库](https://github.com/xiaochy/H2RBench)；[EgoDex](https://machinelearning.apple.com/research/egodex-learning-dexterous-manipulation)。

## 直接解决方案：最低成本自录 2 条真人视频（不依赖国外受限数据）

先用手机/固定相机在**同一桌面、同一立方体、相同起点**录制两个短片：A 将方块从中央沿实际机器人工作空间标定 `-X` 方向推向左侧目标，B 将方块从同一起点沿标定 `+X` 方向推到右侧目标。两目标区同时存在于两条视频和之后的机器人初始画面，不能在机器人输入中隐含视频目标信息；拍摄者双手与物体操作过程可见。录制 A、B 期间不得移动相机、桌面、对象起始标记或两个目标区，不得左右镜像生成“反向视频”。**镜头中的屏幕左/右不必然等于机器人世界的 ±X**；用桌面箭头标明机器人 +X 方向并保存标定/场景对应证据。使用自拍/自录且取得所有参与者同意，避免第三方许可与隐私问题。

录制完成后，把视频放在**项目内**：

- `outputs/g0_human_goal_pairs/raw/left.mp4`
- `outputs/g0_human_goal_pairs/raw/right.mp4`

复制 `docs/examples/g0_self_recorded_pair.template.json` 到 `outputs/g0_human_goal_pairs/manifest.json`，填写两个实际视频 SHA256、授权/场景等人工核查字段。**不得将猜测改写成 true**。不要上传原始视频到公共 GitHub。原始内容不够就继续保持 G0_BLOCKED，而不是编造对抗数据。

### 本轮可直接运行的 G0 审计器

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_audit_goal_contrast_g0.py' -v

# 即刻执行：只查看已有 RoboReel 项目记录和公开源审计目录；
# 预期现阶段返回 G0_NOT_PASSED (退出码 2)，不启动仿真。
python -m scripts.audit_goal_contrast_g0

# 当且仅当录制并审查两段真实反目标视频后：
sha256sum outputs/g0_human_goal_pairs/raw/{left,right}.mp4
python -m scripts.audit_goal_contrast_g0 \
  --manifest outputs/g0_human_goal_pairs/manifest.json
```

程序自动检查：项目路径封闭、来源声明、研究使用权、人手示范声明、真实机器人 ±X 目标、相机-世界方向映射声明、两个原视频 SHA256 不同且与标注一致、媒体元数据和第一帧真实解码、拒绝 EVAL 保留集文件。原始视频/轨迹从不修改。审计报告写到 `outputs/g0_goal_pair_audit/v1/g0_audit_<contenthash>.json`，**只增不改**。

`G0_CANDIDATE_REQUIRES_MANUAL_VISUAL_REVIEW` 也**不等于科学验证通过**：操作者/研究者必须实际看视频的起止帧、相机轴和对象终态，并留存人工审核记录；现有代码无法自动判断操作语义，绝不虚构“已验证真实反目标”。人工 G0 过关后才独立新建 `VideoGoalPush-v0`，保留官方 PushCube-v2 完整不变，首先比较同机器人初始观测、相同随机数下两种视频对应的目标动作，与 oracle goal/planner 同信息基线；不能预先宣称大模型有创新。

## STOP 判据与诚实状态

- 当前 G0：`BLOCKED_NO_VERIFIED_OPPOSING_HUMAN_PAIR`。公开类别是线索，不是两条合规视频；旧 RoboReel 同任务视频没有目标反转。
- G1 新物理 rollout：**0**。本次训练：**0**。本次网络视频下载：**0**。
- 不再重跑 StackCups、不重新编码两条旧视频、不下载无授权数据、不修改冻结 46/60、3/60、0/4。
- 如果不能获取合法的人类反目标视频且证明物理可行性，**直接终止 VideoGoalPush 假设**，不要拿任务固定目标的视频硬造。
