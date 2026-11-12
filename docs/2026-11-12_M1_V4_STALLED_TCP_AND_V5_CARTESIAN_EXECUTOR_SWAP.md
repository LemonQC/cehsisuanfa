# 2026-11-12 — M1 current-scene v4: 210-step TCP+Cube hard stall at throat; REPLACE failed actuator, not trajectory budget

## Actual operator physical evidence

The user inspected the saved original `outputs/m1_current_target_reference_height_v4/target_reference_height/physical_trace.json`; exactly **210** actions have `execution_phase="push_3_x_0.260"`:

| Physical observation | Recorded value |
|---|---|
| TCP X min / max during stage | **0.130775898695 / 0.131375283003m** |
| TCP X span across 210 actions | **0.000599384308m = 0.5994mm** |
| Cube X min / max during stage | **0.169611617923 / 0.169611722231m** |
| Cube X span | **0.000000104308m = 0.00010mm** |
| TCP step240 | x=.13078907, y=−.00056354, z=.04204354 |
| TCP step449 | x=.13135739, y=−.00217296, z=.04218250 |
| Cube step240 / step449 | x≈.16961169, y≈.00259693, z≈.020 |

**Conclusion:** this is a genuinely stalled executed joint-space Screw trajectory, NOT merely long planned transit or a temporarily lost contact with still-moving TCP. 210 additional steps would be scientifically unjustified. At the end TCP lies about **38.25mm behind** cube in X; both are effectively stationary. Previous M1B3 goal is cube x=.210±.020; cube x=.16961 fails. Original throat solid geometry occupies x=.146..170, y barriers around ±.025 clearance. A 40mm cube centered at x=.1696 still overlaps the throat longitudinally: its rear face is at x=.1496.

The trace does **NOT** capture actual Panda link-wall forces, robot full arm qpos errors, actuator saturation or tool orientation. Therefore do **NOT** write "link collides wall" as a demonstrated causal fact; candidate diagnoses are robot end-effector interference or joint-control inability to realize the x=.260 goal. This observed block is a limit of the **present executor**, not proven topological impossibility of the target Push-only task.

## One justified replacement, using a previously PROVEN real actuator

- Under *the same published user-local source*: M1B3 **open_direct** original `pd_ee_delta_pose` `ActorPush` reached the same goal in **35 real PhysX steps**. This is a genuine actual motion reference, not the custom M1 learned method.
- The original M1B3 `blocked_clear` also physically parked the separate dynamic blocker in **32 actions** using the same `pd_ee_delta_pose` `ActorPush`, without changing any original wall/goal. It then failed to push the task cube due to old handoff and later off-center throat collision.
- v3/v4 validated that **after blocker clearance**, first retreat to high z, move at height over cube to x<0 and then lower at the behind location, avoiding unintended −X precontact motion.
- Thus the most conservative engineering replacement is **reuse the already successful original Cartesian delta-pose +X pushing actuator** with the validated *safe overhead return*. Do NOT call long stuck MPLib Screw again. Do NOT change scene geometry, task, final goal, seed or frozen baseline output.

## Code submitted (NOT YET EXECUTED on operator server)

- `agcofl/m1_original_scene_cartesian_pusher_v5.py`
  - Original `AGCOFLM1B3Corridor-v0`, Panda, `pd_ee_delta_pose`, num_envs=1, physx_cpu, seed120, goal x=.210, 20mm threshold, MAX_STEPS=450.
  - Original M1B3 **ActorPush** `clear_blocker` (fixed target [.100,.125], 22mm tolerance), hard cap 110 new actions (old original clearance succeeded in 32).
  - Explicit clearance-grounding/no-grasp checks. Note: **closed fingers are allowed as a pusher**, but neither actor may be grasped; finger/grasp flags recorded.
  - Same v3/v4 collision-avoiding **safe overhead return** using already proven `delta_pose_action`, first retreat at current XY to Z=.160, move overhead behind cube X=−.060, descend to physically referenced TCP z=.038.
  - Finally **original** `delta_pose_action` with **+X goal TCP x=.280**, closed gripper −1, physically evaluate each action and end on real simulator goal; strict stop for X reverse, transverse drift, object lift or grasp. The physical subbudget is 120 actions, **but stop after only 15 consecutive actions** when both TCP and cube make no meaningful forward progress. No "retry other gains", no added 450-step budget.
  - All raw v5 video/trajectory outputs named `target_cartesian_pusher` and stored ONLY under `outputs/m1_original_scene_cartesian_pusher_v5`. Original old H5/MP4/status/trace never modified.
- `tests/test_m1_original_scene_cartesian_pusher_v5.py`: CPU-only exact initial task/goal/controller/gripper/7D semantics, bounded budgets and immutable-result tests.
- `scripts/run_m1_original_scene_cartesian_pusher_once.sh`: tests + source-level geometry preflight; verifies **real** old 35-step reference and 210-step v4 TCP/cube plateau using SHA256 of both immutable saved traces, and confirms same goal/initial cube. If any mismatch, **zero new PhysX**. Otherwise at most **one** new physical case with 500s hard wall timeout.

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05
bash scripts/run_m1_original_scene_cartesian_pusher_once.sh
```

Expected result `M1_V5_CARTESIAN_PHYSICS_GATE` includes true clearance, actual push progress, last cube XYZ, source still not matched and no learned/TAMP completion.

**If failure:** Freeze new 1-case result. Do not open v6/v7 by varying gains or more seeds. The actual 50mm throat + 40mm object is extremely tight, and original object-only C-space reachability does not establish hand/tool reachability. Next decision should be **pusher geometry / contact-mode verification**, not blindly tuning the task again. The project still has an **independent source grasp problem**: existing x=.210 source showed 0 physically valid grasp steps. A target-only feasibility fix cannot establish cross-behavior scientific novelty.

## Honest evaluation

No fake matched source video, no false claim an official published PushCube solver solved a multi-object clearing task, no "necessary intervention" causal proof, no learned method, no equal-info strong TAMP results. This v5 experiment is a one-time engineering **executor substitution** in the existing environment, not a new research direction or a fair benchmark baseline.
