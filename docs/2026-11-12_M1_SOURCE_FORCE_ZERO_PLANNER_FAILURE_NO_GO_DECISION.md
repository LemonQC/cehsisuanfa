# 2026-11-12 — M1B3 source stop-gate decision after 65-step real force audit

## Newly verified zero-physics operator result

The user executed `python -m scripts.m1_source_65_step_force_vs_planner_audit`. Existing genuine 65-action original-scene physical trace was inspected with no fresh simulation:

```json
{
 "actual_physx_new_runs":0,
 "pregrasp_actual_tcp_xyz_m":[-0.001790609210729599,-0.014053452759981155,0.0710543543100357],
 "pregrasp_tcp_target_position_error_m":0.016753563588883254,
 "tcp_cube_distance_m":0.05298314897233072,
 "tcp_world_x_alignment_abs":0.9999386535384088,
 "tool_static_wall_contact_force_sample_count":65,
 "tool_static_wall_contact_force_missing_count":0,
 "tool_static_wall_force_max_n":0.0,
 "tool_static_wall_force_gt_0p5N_count":0,
 "first_significant_tool_wall_contact":null,
 "failure_classification":"HAND_FINGER_WALL_CONTACT_NOT_OBSERVED_IN_RECORDED_FORCE"
}
```

**Grounded conclusions:** original Panda partial opening q≈.023m physically executed; measured TCP gripper closing axis is aligned with corridor X (abs dot = .99993865). The tool-walls force logger reported **0.0 N in every one of 65 executed actions** for three links (hand and two fingers) versus original STATIC walls. The actual pregrasp TCP tracking residual is 16.75mm; last TCP remains 52.98mm from cube center. `grasp_pose` failed in MPLib `plan_screw` before a SINGLE final-descent action. No physical source bilateral grasp, cube lift or carry occurred.

**Cannot infer:** final grasp is collision-free just because earlier forces were zero. Unexecuted trajectories do not exert forces. Table, blocker, arm/elbow links, robot self-collisions and planner-internal pointcloud collision restrictions were not measured by those force values. Nor can a `screw plan failed` warning differentiate IK, robot limits, start state consistency, continuous joint interpolation or predicted collisions. The 16.75mm pregrasp tracking error can itself matter for planner start state. Prior "finger collision with rails caused failure" was an unproven candidate, **not established** by these real 0N measurements.

## Original source experience and scientific cost gate

Previous ACTUAL same-scene source attempts:

| Technique | PhysX actions | Min TCP–cube distance | Real bilateral grasps | Lift |
| --- | ---: | ---: | ---: | ---: |
| Original M1B3 top-down official MPLib | 335 | 63.97mm | 0 | 0 |
| Side grasp adapted MPLib | 106 | 102.42mm | 0 | 0 |
| Geometry-aligned 46mm-open top-down with original obstacle cloud | 65 | 52.98mm | 0 | 0 |

Old M1B2 successful Pick source at x=.150m had real 84 grasp steps and 127.8mm lift but DIFFERENT source task geometry/goal; it is not a same-goal x=.210 demonstration.

The frozen original same-scene **target** v6 used physically successful closed-finger Push-only **66 steps** to clear blocker and move cube to x=.2018075m; XY terminal error **8.193mm**. It proves target execution is possible for seed120 but **does not** establish source Pick feasibility or source-video necessity.

**Decision:** Mark current **M1B3 source implementation approach NO-GO**. Preserve original target v6 and all three failed source traces. Do not launch more full source episodes (v2/v3 etc.), alternative orientations or solver retries; do not alter environment to make the negative result disappear. This is an ENGINEERING NO-GO for the *tested grasp executors*, not proof that all robotic Pick-and-Place in the original corridor is physically impossible. Do not claim a paired cross-behavior source-target dataset, a trained Ours or a fair strong TAMP comparison.

## Only optional diagnostic that could justify reopening original M1B3

If a future source rescue is considered essential, **first require a non-physics planning feasibility audit of the exact previously failed final pose, in the actual local robot model**, with:

1. The frozen saved robot joint state (not only TCP XYZ), tool quaternion, real partial finger qpos and original pregrasp goal. If available, recover robot qpos from **existing** saved H5/JSON; otherwise note evidence unavailable without an additional simulation.
2. Run IK and collision-only checks for the *single frozen exact grasp target*, distinguishing **IK unavailable**, **target IK exists but in collision**, **IK feasible and collision-free yet straight Screw path fails**, and **pregrasp controller tracking deviation**. Respect exact Panda SRDF, scene wall/blocker geometries and gripper opening, and clearly separate the planner's sampled pointcloud approximation from PhysX contacts.
3. Only a positive, reproducible planning-only feasibility certificate for this *same scene and exact goal* permits reconsidering **one** new monitored contact-only PhysX probe. If the check is negative/inconclusive, stop this M1B3 same-goal paired-pilot and reassess the task as a research benchmark; do not claim a theorem of universal physical impossibility.

The upstream ManiSkill BaseMotionPlanningSolver `move_to_pose_with_screw` calls `planner.plan_screw` twice with `use_point_cloud=True`, then returns `-1` if both statuses are not `Success`; no `follow_path` actions are executed. Thus the observed `grasp_pose:0 steps` is precisely a **planning rejection**, not a physical collision result.

## Publication readiness

- Original target v6: PhysX PASS **1 seed / 66 actions** (single case, not a statistical success rate).
- Same-scene source x=.210: **FAILED** with all three real source grasp approaches, 0 real grasp/lift.
- Matched source-target physical pair: **NOT ESTABLISHED**.
- Strong equal-information planner/TAMP vs video: **NOT_RUN**.
- Novel method/Ours/M2: **NOT_RUN**.
- IEEE T-RO research novelty/necessity: **NOT DEMONSTRATED**.

Do not restart learning/method design or publish success-rate claims based on the 66-step oracle target alone.
