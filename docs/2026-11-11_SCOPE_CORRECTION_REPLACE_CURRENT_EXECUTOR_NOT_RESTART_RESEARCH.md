# 2026-11-11 — User-corrected scope: replace failed execution modules IN CURRENT M1, not a new task/paper

## Exact user request and correction

The user clarified: **“我说的是当前实验遇到的难题，不是再重新做实验啊”**.

Therefore, our unsolicited new direction **Execution-Grounded Demonstration Alignment / long-horizon RoboReel** and independent **B01 robot-only BC dataset download** are **CANCELLED as next actions**. Their new (unexecuted) script and unit-test files have been removed from GitHub. No new dataset, no new train/val split, no repeat of M0, M1B1, M1B2, or M1B3 is requested. The existing current project is still **same-goal cross-behavior Pick-source -> Push-only target with blocker intervention**.

The real M1-B4 gate is respected: in the frozen M1-B3 *custom blocked source environment*, source task cube never grasped (0 steps), lifted (~0m), or moved in XY (~4.7 µm). This proves the **particular Panda source-grasp adapter in this obstacle layout is invalid**; it does NOT establish that the overall video-to-robot question or official reference is infeasible. Another unchanged call to lower RRTConnect cannot repair the missing grasp, so do not relaunch that script as-is.

## Reliable replacements already available, with actual provenance

- **Source:** Published **ManiSkill original** Panda PickCube motion planner `mani_skill/examples/motionplanning/panda/solutions/pick_cube.py`. Already actually ran in the user's pi05 environment **3/3 successes** with H5, JSON and RGB video at `outputs/m1b_official_pick_reference`; this is a *different-goal source*, not automatically matched. The project's M1-B2 custom source ALSO genuinely achieved a Pick+Place in its original (less confined) scene, 170 steps, with recorded actual grasp/lift/release; its goal was **x=0.15m**, whereas M1-B3's target goal is **x=0.21m**. The two must not be relabelled as already matched.
- **Target:** ManiSkill has a **separate original Panda PushCube solution** `mani_skill/examples/motionplanning/panda/solutions/push_cube.py`, which calls upstream `PandaArmMotionPlanningSolver` to reach and execute physical pushing. This is a reliable **existing executor interface**, not proof that its default two fixed poses solve the modified blocker/side-pocket geometry. Reuse its planning/actions infrastructure when replacing `DiagnosticPushTeacher` rather than writing another arbitrary Cartesian proportional push loop.
- **Independent physical facts:** M1-B3 target moving the blocker sideways **succeeded in 32 actions**, and the post-clear task cube stopped at `[x=0.125273, y=-0.030822]` near the x=.126 throat-front boundary; the original opening **open_direct succeeded in 35 actions**. Thus the object/robot physics itself runs. Fixing the transfer/executor layer is different from designing yet another obstacle layout.

## Narrow replacement-first path; SAME experiment

1. **Reuse, do not regenerate, source video.** Inspect the already-saved successful H5/MP4 from original Pick 3/3 and M1B2 successful Pick; choose a source only if its observed task-cube transition and goal can be legitimately matched to the current target task under a declared rigid coordinate-frame registration. A rigid transform cannot alter source object displacement magnitude or create an unobserved intervention. Source and target scenes may differ only if the scientific protocol explicitly permits cross-scene demonstration; this must be clearly marked, not quietly counted as the old "identical scene" matched success.
2. **Replace rather than tune the target's low-level push execution.** Retain existing M1-B3 blocker positions, 50mm narrow passage, goal [.210,0], target push-only constraint, physics and held-out baseline protocol. Swap only the old `DiagnosticPushTeacher` contact execution for the already-published Panda `push_cube.py`'s MPLib-based execution, adapting its entry/exit target to actual blocker and cube (open gripper). Do not claim that published `solvePushCube` by itself solves multi-stage clearing or that the manually chosen clear-first ordering is a trained method/TAMP.
3. **Gate first using old evidence, not new whole experiments.** Existing source clip must show real object displacement and the target's goal must match it without fabricating a trajectory; existing target clear trace must expose the physical side-pocket state; unit/interface checks must validate action dimension/control-mode and goal transform. Only a focused existing-scene, new-executor smoke can be considered after those checks; it is NOT a fresh benchmark campaign or a new scientific research direction. No new method training until that succeeds and fair TAMP baselines are honestly run.

## Truthful status

- M0 official π0.5 Uniform **46/60**, Vid2Robot **3/60**, frozen.
- M1-B1 official Panda Pick reference **3/3**, real H5+RGB; **NOT a matched-goal source by itself**.
- M1-B2 same-goal source/direct/clear physical successes but direct needs no extra intervention.
- M1-B3 original custom source failed grasp; target clear physical action worked but stopped at throat; open control succeeded.
- M1-B4 real no-grasp condition stopped **before new physics**.
- **No** source video meeting current target's x=.21 final effect has yet been verified from saved files.
- **No** original push-cube solver integration into this modified blocker scene is yet evaluated.
- Strong equal-information TAMP, learned source-conditioned method and its scientific necessity remain **NOT_RUN/NOT_PROVEN**.

## DO NOT

Do not pivot to long-horizon tasks, download 64 B01 robot-only trajectories, create a new maze, start M1-B5 controller gain tuning, or present a source video from a different goal as a valid same-goal pair. Source replacement and target execution replacement are the only active engineering objectives pending real data matching.
