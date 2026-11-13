# TRO1 StackCups official TRAIN-video DEV v1: four real rollouts and post-hoc diagnosis

## Verified results (RTX 4090, user terminal 2026-11-13)

Official RoboReel `StackCups-v1`, two scene states and two same-task genuine human TRAIN videos `0000`, `0001`; frozen `demo_pi05` checkpoint, identical reset states and policy RNG within each paired scene, 800 PhysX steps per rollout.

| Paired episode | Source video IDs (execution order) | Success | Original mean 8D action L2 over common steps |
|---|---|---|---:|
| 0 | 0000, 0001 | False, False | 0.7579299807548523 |
| 1 | 0001, 0000 | False, False | 1.2224546670913696 |

All **4/4 physical rollouts finished at 800 steps without policy runtime errors; 0/4 task completions**. All four `rollout.mp4` paths exist and `video_error=null` in the user's report. This is a successful *benchmark execution*, not successful manipulation. No within-scene binary outcome reversal between video demonstrations, and no evidence of improvement from the video selection. The nonzero full-trajectory mean 8D action difference means trajectories are different, but is not causal proof that the video is useful: closed-loop state divergence can amplify an early perturbation. Arm joints and gripper are combined in that distance, unnormalized. **No CEPS or new recovery method was implemented/tested.** Do not advance T-RO novelty on this evidence.

The JAX / `os.fork()` multithreading warning during MP4 encoding did not prevent completing the rollout; it should be recorded but cannot be identified as the manipulation failure cause.

## Frozen evidence location

The server retains `outputs/roboreel_video_dependency_dev/stack_v1/` containing `results.json`, `SUMMARY.md`, `per_trial.csv`, `episode_*/source_*/actions.npy`, `status.json`, `success_trace.json`, `rollout.mp4`. No uploaded raw data were available to the assistant; all numerical observations above come from the user's terminal output. **Do not rerun or overwrite `stack_v1`.** No PartNet-Mobility required, no EVAL 0080–0099 used.

## Read-only diagnostic tool

`python -m scripts.diagnose_stack_video_dev` operates ONLY on existing completed evidence and explicitly validates all action-file SHA256 against the original trial manifest, expected task, four trials, same initial RNG, and the originally reported full-trajectory L2. It never imports JAX/OpenPI/ManiSkill, calls `env.step`, downloads data, or changes the evidence root. Outputs go into a new directory `outputs/roboreel_video_dependency_diagnosis/stack_v1`:

- `diagnosis.json`: matched pairwise action differences at 1, 20, 50, 100, 200, 400, 800 steps, separated into 7-DOF arm and gripper; first action delta and reference magnitude.
- `SUMMARY.md`: compact comparison for discussion.
- `rollout_montage.jpg`: four rows of 10 timestamps decoded from the original robot rollout MP4s. The montage is only a visual reference; it does not automatically assign manipulation-stage successes or failures.

Test fixture: `tests/test_diagnose_stack_video_dev.py`, 8 offline `unittest` tests. Tested locally using NumPy fixtures; no real user's server videos/actions were inspected by assistant. If the montage can't be generated due to imageio/ffmpeg/Pillow, `diagnosis.json` still records the exact error and metrics; original MP4 files remain intact.

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_diagnose_stack_video_dev.py' -v
python -m scripts.diagnose_stack_video_dev
cat outputs/roboreel_video_dependency_diagnosis/stack_v1/SUMMARY.md
```

The user should send the printed `ROBOREEL_STACK_READONLY_DIAGNOSIS` line and attach `rollout_montage.jpg` to enable a grounded visual classification of reaching/grasping/first stack/second stack stages. Do not invent a failure stage from `success=false` alone. If the montage reveals no meaningful long-horizon stage transition or first-action differences are negligible, stop and report `NO-GO`; do not spend further GPU trials without a new falsifiable hypothesis and a same-information baseline.

## 2026-11-13 用户服务器离线诊断回执与阶段研究判定

服务器实际执行 `python -m unittest discover -s tests -p test_diagnose_stack_video_dev.py -v`：**8/8 passed**。
随后使用已冻结四条机器人轨迹和视频（无新 PhysX）执行 read-only 诊断，原终端结果：

```
ROBOREEL_STACK_READONLY_DIAGNOSIS {
 "status":"READ_ONLY_DIAGNOSIS_NOT_NEW_ROLLOUT",
 "successes":0,
 "first_8d_l2":[0.0013496483038846687,0.0016286211015779217],
 "first20_mean_8d_l2":[0.001622933194751093,0.0016648817314526964],
 "full_mean_8d_l2":[0.7579299807548523,1.2224546670913696]
}
```

| Episode | 起始动作 8D L2 | 前 20 步均值 L2 | 全程 800 步均值 L2 | 任务成功 |
|---|---:|---:|---:|---:|
| 0 | 0.00134965 | 0.00162293 | 0.75792998 | 0/2 |
| 1 | 0.00162862 | 0.00166488 | 1.22245467 | 0/2 |

**限制：** 8D L2 未按关节、夹爪或动作尺度归一化，故“不大于某阈值”或“显著性”不能直接判断。起始 1/20 步输入响应差异较全程数值小约数百倍。随后轨迹差异可能来自闭环环境状态分叉、视频条件累积效应或其他因素；没有对照不能区分，不得宣称“视频不存在贡献”，也不得宣称“视频独立提供有用决策”。任务无一成功，无法从成功率支持新阶段同步思路。

**研究停止门槛（当前状态：PROVISIONAL NO-GO）：** 不追加随机种子、不训练新算法、不改任务。仅待研究者上传已生成的 `outputs/roboreel_video_dependency_diagnosis/stack_v1/rollout_montage.jpg`，目视确认有无第一阶段抓取、对象搬运、第一次堆叠等实际进度；这张图片未上传，所以当前尚未作任何阶段分类。若四条均在基础抓取前失败，明确终止 StackCups 阶段同步方法线；若有真实后阶段失效，也必须先提出可证伪机制与同信息强基线，而非直接以 T-RO GO 决策继续训练。

归档证明仅来自用户回传的运行日志及本仓库现有代码；模型未访问其服务器上的原始 `.npy` 或 `.mp4`，不能臆造图片内容。原 `stack_v1` 与 `diagnosis/stack_v1` 均只读保存。
