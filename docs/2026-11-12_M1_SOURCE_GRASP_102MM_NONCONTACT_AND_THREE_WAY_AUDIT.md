# 2026-11-12 — M1 target frozen success; source physical grasp failure: measured 102mm mismatch, compare ALL existing source evidence

## Already measured: no new simulation

The user ran the genuine `scripts/m1_current_compare_grasp_contact.py` on the operator's `pi05` machine. Four CPU tests passed. Output:

```json
{
  "type":"M1_CURRENT_SOURCE_CONTACT_FAILURE_COMPARISON",
  "current_grasp_steps":0,
  "current_close_command_steps":6,
  "current_closest_tcp_cube_distance_m":0.10241956894206888,
  "current_first_close":{
    "step":100,
    "tcp_xyz_m":[-0.008148275315761566,0.02015252225100994,0.12097346037626266],
    "cube_xyz_m":[-3.703731010773481e-7,-7.008002285147086e-7,0.019999923184514046],
    "tcp_to_cube_distance_m":0.10328695932847012,
    "gripper_command":-1.0,
    "is_grasping":false
  },
  "current_six_steps_after_close":{
    "step":105,
    "tcp_xyz_m":[-0.008437223732471466,0.019608577713370323,0.120207779109478],
    "cube_xyz_m":[-3.703731010773481e-7,-7.008002285147086e-7,0.019999923184514046],
    "tcp_to_cube_distance_m":0.10245642335895656,
    "gripper_command":-1.0,
    "is_grasping":false
  },
  "old_reference_grasp_steps":84,
  "old_reference_first_close":{
    "step":73,
    "tcp_xyz_m":[-0.0029121506959199905,-7.424323121085763e-5,0.020009063184261322],
    "cube_xyz_m":[-9.204979960486526e-7,-1.5272332802851452e-6,0.019999919459223747],
    "tcp_to_cube_distance_m":0.0029121525525574506,
    "gripper_command":-1.0,
    "is_grasping":false
  }
}
```

**Well-supported diagnosis:** the failed `source_side_grasp` issued an actual CLOSED gripper command for six physical steps, but its TCP never came closer than **102.42mm** from cube center and remained around **z=120mm** (cube center z=20mm). A planner return status is **not** an actual pose tracking or valid contact guarantee. This is a **pregrasp approach/contact failure**, not evidence that the hand squeezed too weakly, lift speed was inadequate or the object was released after being grasped.

The prior physically successful M1B2 source issued its first close command with TCP **2.91mm** from cube center, and 84 real time steps registered a valid two-finger `is_grasping(obj)`; however M1B2 used **goal x=.150m**, while current M1B3 requires **goal x=.210m in the blocked corridor**. Treat this as a low-level CONTACT reference only, not a matched source video or end-to-end same-goal success. The old successful case first closure snapshot has `is_grasping=false`, which is compatible with subsequent 84 grasping steps; never demand grasp immediately on the first closure step alone.

### The necessary already-collected contrast

There is also a separate **original** M1B3 `blocked_source` top-down Pick trial with saved real PhysX trace:
`outputs/m1b3_corridor_dev/blocked_source/status.json`
and `physical_trace.json`.
It reportedly failed an upstream lower/approach stage with no valid final source success and 0 physically grasped steps; however its precise TCP at first closure, true minimum reach and phase status **have not yet been provided in the current output**. This is the decisive low-cost missing evidence.

Three-way read-only script committed:
`scripts/m1_source_three_way_existing_audit.py`,
`tests/test_m1_source_three_way_existing_audit.py`.
It compares:
1. original SAME-SCENE M1B3 `blocked_source` top-down physical source attempt
2. subsequent SAME-SCENE M1B3 `source_side_grasp` 106-step attempt
3. earlier DIFFERENT-GOAL M1B2 `source` physically valid grasp

Strict gates: original two attempts must have matching x=.210 goals, same seed/initial cube and original scene, old reference must have x=.150 and valid grasp; actual per-step TCP/object, gripper command, grasp flags and step index must be present; old original JSON SHA256s archived. Script prints first CLOSE, closest TCP and actual real grasp indices/peak lift and upstream planner errors.

Run **NO NEW PHYSX**:

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05
python -m unittest discover -s tests -p 'test_m1_source_three_way_existing_audit.py' -q
python -m scripts.m1_source_three_way_existing_audit
```

Expected marker: `M1_SOURCE_THREE_WAY_REAL_TRACE_GATE`.

Output stored **separately**, not over old data:
`outputs/m1_current_executor_swap_smoke/source_three_way_existing_audit.json`.

### Mandatory stop/go after offline comparison

- If the original M1B3 top-down source did reach the cube center (e.g. TCP <10mm distance) **and actually grasped**, do not rerun pregrasp; isolate the downstream physical lift/carry problem from its previous trace and reuse that existing contact recipe.
- If original top-down reached close to cube but registered **no physical grasp**, the bottleneck is actual gripper-object contact despite correct TCP center: inspect approach orientation, individual finger joint qpos, bilateral contact force and wall/tool clearance; one contact-only diagnostic in same scene **may** be justified, not new seeds or full task.
- If original top-down **never reached cube** and side grasp also stays >100mm away, it is a physical reachability / pose execution issue under the original M1B3 obstacle geometry, not a demand for a novel cross-behavior algorithm. Avoid new full source/target rollouts until arm/gripper collision and target pose are verified.
- In all cases: source task success must eventually pass actual same-goal x=.210, cube physically held/lifted *above 75mm walls*, blocker not moved >25mm, placed/released, original goal success. Do not mark earlier source x=.150 video as current source.

### Already frozen target milestone

The M1B3 target side **physically succeeded** with v6 `pd_ee_delta_pose` 66 actions, final cube [0.2018075,0.00011315,0.02000012]m; XY goal error 8.193mm; blocker parked in 32 actions, neither object ever physically grasped. **Stop target v7, no target reruns.**

**Scientific status:** same-goal cross-behavior source/target pair NOT_YET_VALID; strong equal-information TAMP NOT_RUN, video-dependent contribution NOT_SHOWN, learned Ours/M2 NOT_RUN, T-RO novelty NOT_ESTABLISHED.

Current diagnosis is drawn from user-pasted genuine physical traces. New three-way script code is committed but has **not yet run on the operator machine**.
