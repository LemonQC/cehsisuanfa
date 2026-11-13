# TRO1 | 2026-11-13 | PartNet 44962 后的 ObjectInDrawer DEV 真正结果与下一步

## 证据来自已上传的 GitHub，而不是终端复述

只读现场证据（自动上传分支 \`tro1-experiment-evidence\`）：
- [evidence.json](https://github.com/LemonQC/cehsisuanfa/blob/tro1-experiment-evidence/evidence/runs/drawer_hf_v1_540ff3963863f5ed/evidence.json)
- [README 与机器人预览](https://github.com/LemonQC/cehsisuanfa/tree/tro1-experiment-evidence/evidence/runs/drawer_hf_v1_540ff3963863f5ed)
- 源文件：\`outputs/roboreel_video_dependency_dev/drawer_hf_v1/results.json\`，源 JSON SHA256 为 \`07b0708a0952a70a89d13031ac4bf994df07eab28a1723b63c98b591959e9e40\`。
- 该分支仅发布脱敏指标和机器人预览图，不发布 PartNet/ShapeNet、真实人类演示视频、原始动作数组、Token 或模型权重。

**真实实验执行完毕**：\`DEV_COMPLETE_NOT_FORMAL\`，\`toys_in_drawer\`，4/4 真正 PhysX 轨迹执行；无 rollout 或视频编码错误；冻结 \`demo_pi05\`，两个真实 TRAIN 人类视频 (0000 / 0001)。2/4 成功（50%，只是两种配对场景的开发样本，非正式总体成功率）。

| 配对场景 | TRAIN 0000 | TRAIN 0001 | L2：前两轨迹共有步数的平均 8D 动作差 |
|---|---|---|---|
| Episode 0，reset seed 310000，policy RNG 560000 | False，800 步、超时 | False，800 步、超时 | 1.340732694 |
| Episode 1，reset seed 310001，policy RNG 560001 | True，634 步 | True，650 步 | 0.270444125 |

- 两个场景中均没有选择示范视频造成的二元成功/失败反转：\`0/2\` 对。
- 成功/失败目前与场景相对应，而不是与 0000/0001 的视频编号相对应。
- 全程 L2 差异不能分离源视频直接影响与闭环轨迹逐渐分叉，不代表视频具有独有的任务信息。
- 重要：固定上游 \`ObjectInDrawer-v1\` 的注册默认最大 episode 步数是 **400**，本轮用 \`make_eval_env(..., max_episode_steps=800)\` 显式扩展为 800。**两条成功发生在 634/650 步**；这些不是 400 步默认时限下的成功。上游 eval utils 允许这种显式覆盖，但必须与正式协议指标严格区分。
- 预览图确实在证据分支：\`robot_rollout_preview.jpg\`。本次连接的 GitHub 文本读取能力不能解码该二进制图像；因此本记录没有根据看不到的画面杜撰“失败是关抽屉还是抓取”。

## 判断：任务能力恢复，但源视频必要性仍 NO-GO

PartNet 授权已解决原始抽屉环境的 URDF、碰撞体与启动问题；冻结 π0.5 至少能在一种场景中完成 800 步以内的全部任务。这是 **工程基线恢复**，不是新的 T-RO 算法效果。

两段同任务人类视频目标一致，无法提供“视频决定相反合法决策”的反事实数据。已有 2×2 配对不能证实有意义的阶段对齐、跨具身的因果信息需求或 T-RO 级创新必要性。不得因为观察到个别成功就启动新方法训练或声称强基线已经全部跑完。

## 我已实施的下一行动（不新增 PhysX）

1. 提供 \`scripts/diagnose_drawer_video_dev.py\`：直接读取冻结的 \`results.json\` 和四条 \`actions.npy\`，逐条核查 SHA256、场景/策略种子及真实步数；按 **1、20、50、100、200、400、800（实际共同步数封顶）** 统计配对动作差异，分别报告七关节和夹爪分量；生成机器人预览 montage 并将 \`diagnosis.json\` 写到 **独立** 路径。
2. 提供 \`tests/test_diagnose_drawer_video_dev.py\`：只使用纯 CPU 虚构动作测试夹具，覆盖混合 800/634/650 步、成功组合、摘要复现、哈希防篡改和禁止覆盖旧证据。**本文件属于待服务器测试版本，不能谎称已在服务器执行。**
3. 提供 \`scripts/diagnose_drawer_and_publish.sh\`：若诊断已经完成只发布；若没有，则仅执行离线统计，然后调用既有 \`scripts.publish_research_evidence\` 将小型指标自动发布到 \`tro1-experiment-evidence/evidence/runs/\`。
4. 本轮原来的 \`drawer_hf_v1\`、失败 \`v2/v3\`、StackCups \`0/4\`、PushCube 既有正式结果完全保留。不跑新模型、不修改场景、不追加随机种子。

操作指令：

\`\`\`bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_diagnose_drawer_video_dev.py' -v &&
bash scripts/diagnose_drawer_and_publish.sh
\`\`\`

发布成功后，可直接在专用 evidence 分支读取新目录 \`drawer_hf_v1_<hash>\` 下的 \`evidence.json\` 和可视化。**无需操作者复制任何日志，也无需重跑四条 PhysX。**

## 后续有条件 GO/NO-GO

- **起始 1/20 步差异几乎不可分**，全程差异扩大：优先判断闭环分叉，不能写成视频时序有决定性贡献；当前阶段同步算法路线 \`NO-GO\`。
- **起始动作有视频条件依赖**，但两种视频不代表不同正确目标，且成功率没有偏好反转：最多证明敏感性，仍 \`NO-GO\` 于视频必要性。
- **图像显示某个阶段在失败场景稳定复现**：还必须与使用相同目标信息的规则/规划和已有恢复基线公平对照；本 2×2 数据不自动推导新算法必要性。
- 若不具备合法的、真正目标不同的真人示范对，那么 G0 视频目标反事实研究依旧 \`BLOCKED\`，不能用 drawer 的两条固定目标视频冒充通过。

**总决策：保留原始 ObjectInDrawer 作为多阶段可运行基线；继续只读归因，暂时不投入训练，待可证伪的目标依赖差异出现再开新研究方法。**
