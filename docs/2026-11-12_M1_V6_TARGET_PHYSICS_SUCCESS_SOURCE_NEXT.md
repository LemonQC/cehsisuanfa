# 2026-11-12 — Current M1B3 target Push-only physics solved in v6; freeze executor, source Pick next

## Frozen operator-provided real PhysX result

The operator ran `bash scripts/run_m1_original_scene_cartesian_pusher_v6_once.sh` on their local Panda/ManiSkill PhysX simulator. Output from `M1_CARTESIAN_PUSHER_RESULT` and `M1_V6_PHYSICAL_GATE` AGREES:

```json
{
 "success":true,
 "steps":66,
 "error":null,
 "goal_error_m":0.008193275891968195,
 "cube_final":[0.20180749893188477,0.00011315263691358268,0.020000118762254715],
 "clearance_valid":true,
 "push_only":{
  "valid":true,
  "ever_grasped_task":false,
  "ever_grasped_blocker":false,
  "closed_gripper_command_steps":66,
  "open_gripper_command_steps":0,
  "finger_joint_qpos_final_m":[0.0,0.0]
 }
}
```

Scene remains original **AGCOFLM1B3Corridor-v0**, seed **120**, cube initially [0,0,.020]m, goal cube x=.210m y=0 with 20mm tolerance, moving blocker initially x=.100 y=0 and parking at [.100,.125], original 50mm narrow throat, original Panda and `pd_ee_delta_pose` control, CLOSED fingers used as physical pusher without grasping either actor. No task redesign, external expert downloads, new baseline, or learned method.

| Physical stage | Actions | Actual measured outcome |
| --- | ---: | --- |
| Privileged original `ActorPush` blocker clearance | 32 | blocker [0.10254146158695221,0.11980852484703064,0.028997577726840973]; parking error **0.005780176577m** |
| Safe vertical retreat | 7 | TCP [0.10030186176300049,0.07285471260547638,0.1505453884601593] |
| Safe high rear traversal | 9 | TCP [−0.04511170834302902,0.00011365443060640246,0.16115841269493103] |
| Final rear descent (v6 accepts error ≤8mm, not v5 22mm) | 8 | TCP [−0.06095581501722336,0.0000418049530708231,0.045633383095264435]; cube unmoved |
| Original Cartesian `pd_ee_delta_pose` forward push | 10 | cube [0.20180749893188477,0.00011315263691358268,0.020000118762254715]; TCP [0.1360277384519577,0.00029119261307641864,0.037984855473041534] |
| **TOTAL** | **66** | **`success=true`**, final XY goal error **8.1933mm** below 20mm |

The last phase reports `stop_reason=SIMULATOR_EPISODE_END`: this can mean termination or truncation as observed by controller; the actual final status separately verifies `_actual_satisfied`, XY error <.020m, physical Push-only and grounded gates, so `success=true` is grounded in task physics, not merely the stop reason.

The original frozen trace + H5/video are LOCAL and have **not been uploaded or independently hashed in this GitHub report**:
`outputs/m1_original_scene_cartesian_pusher_v6/target_cartesian_contact_fixed/`
(`status.json`, `physical_trace.json`, and the original recorded trajectory/video as present).

### What changed and what this DOES NOT imply

v5 used `motion_to(descend_rear, tolerance=.022m)`, prematurely treated actual TCP z=.05689m as contact, and stopped at 54 steps before any forward pushing. v6 alone tightens tolerance to .008m and uses the exact old 7D Cartesian `pd_ee_delta_pose` actuator that had previously completed the 35-step M1B3 open_direct. This demonstrates a *working engineering target-side oracle* on one frozen scene. The older 450-step v4 joint-space MPLib TCP stagnation and other old data remain immutable.

**The original scientific goal is NOT complete:** physically successful blocked-scene **source Pick-and-place** must reach the SAME x=.210m goal while keeping the blocker substantially unmoved. Current M1B3 source actual top-down grasp had 0 grasp and 0 object lift; later v1 side grasp did 106 real actions, 0 valid physically grasped steps and no lift. Original M1B2 source at x=.150 did succeed but is NOT a valid x=.210 source movie. Do not relabel it. Until a source success and goal-pair exists there can be no scientifically valid cross-behavior source-video→target test.

**Baseline/novelty implication:** 66-step success is from a privileged-state scripted Cartesian control oracle, *not a learned video-conditioned baseline and not equal-information strong TAMP*. It does establish that the correct target goal is physically solvable with a clear-first primitive, so failure of a learning method cannot be attributed to task impossibility. However it also means obstacle removal is demonstrably solvable with simple scripted access to full state; any claimed necessity of a new method must be tested against fair strong state/geometry-aware planning or effect-level TAMP. The one-seed successful target does NOT demonstrate a novel transfer advantage or paper-ready benchmark.

### Next actions (no new PhysX yet)

**Freeze v6 target. Do not start v7 or rerun old v6/M0/M1B2/M1B3 cases.**

The existing zero-simulator `scripts/m1_current_compare_grasp_contact.py` compares the 106-step FAILED source-side-grasp with the earlier actually successful M1B2 source, both using true raw per-step TCP, closed-finger commands and grasp flags:

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05
python -m unittest discover -s tests -p 'test_m1_current_compare_grasp_contact.py' -q
python -m scripts.m1_current_compare_grasp_contact
```

Report `M1_SOURCE_CONTACT_ROOT_DIAG` and inspect its original output JSON with hash. If the reference's status/trace identity gate fails, do **not** skip it or use invented values; inspect actual local source filenames and version first. **No source PhysX retrial is authorized by this command.**

Then choose one mature physical source-grasp executor based on confirmed measured failure mode. Keep exact M1B3 source scene/goal and physically validate actual `is_grasping` immediately after closure, cube lift, genuine overflight above 75mm walls, goal placement/release, blocker not cleared, and H5/MP4. Do not re-run a false solved source or call planner success physical grasp.

Only AFTER the matched physical pair exists: finalize original baseline matrix and compare strong information-matched goal-only / explicit geometry / skill sequencing TAMP with source-video-conditioned approaches; no Method/M2 training if video-dependent need or a novel capability gap is absent.

Status at this milestone:
- M0 author original pi05 Uniform **46/60** and Vid2Robot **3/60** — frozen separate task evidence
- M1B1 author original PickCube **3/3** — genuine original task, different objective
- M1B3 blocked target clear-first **1/1 successful smoke seed120**, *not a success rate estimate*
- M1B3 source physically valid — **NO, no matched goal source**
- Equal-info strong TAMP — **NOT_RUN**
- Learned cross-behavior transfer or Ours — **NOT_RUN**
- IEEE T-RO novelty — **NOT ESTABLISHED**
