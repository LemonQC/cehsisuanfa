# 2026-11-12 — Current M1 target v4: effective +X push, final Screw trajectory consumes remaining 210 steps

## Physical evidence supplied by operator

The user executed the already committed **v4 reference-height** target executor on the **unchanged** `AGCOFLM1B3Corridor-v0` PhysX scene, seed 120, x-goal=0.210m, 20mm tolerance. The actual server console reports:

| Stage | Real action count | Physics observation |
|---|---:|---|
| Close Panda fingers | 6 | Closed throughout, no reported object grasp in prior executor |
| Blocker precontact RRTConnect | 58 | Planner success |
| Blocker lateral Screw | 24 | Parked blocker at [0.1048923,0.1290073], residual 6.324mm |
| Vertical retreat | 18 | TCP rose to z≈.1578 |
| Overhead travel behind task cube | 38 | TCP reached x≈−.0631,z≈.1602; task undisturbed |
| Descend behind task cube | 22 | TCP [−.068724,−.000895,.04145], physically accepted |
| Push first stage TCP target x=.110 | 37 | Cube [x=.133579,y=−.001787] |
| Push second stage TCP target x=.180 | 37 | Cube [x=.169612,y=.002597]; physical TCP x=.132930 |
| Push third stage TCP target x=.260 | **210 further PhysX steps** (not in completed phases) | Budget 450 reached, no completed phase or terminal goal |
| Final | 450 | Cube [x=.169611692,y=.002596933,z=.019999910]; XY goal error=.040471705m |

These stages sum to 6+58+24+18+38+22+37+37=**240** before the unfinished final stage; **210** further real actions were attempted. Original `TracedEnv.step` aborts before performing step 451, so upstream ManiSkill `follow_path` was interrupted, not necessarily because Screw planning itself returned failure.

The original code uses `PUSH_SEGMENT_TCP_X=(.110,.180,.260)`, with **TCP waypoints**, not object goal positions. The final completed stage's actual TCP x=.13293 is ~36.68mm behind the cube center x=.16961, which is plausible for a pushing tool; the robot needs more forward TCP progress or sustained contact to move the cube into the 20mm goal radius.

**Important geometric correction:** throat walls occupy x≈0.146..0.170. The cube is 40mm wide and its current center is x=.1696, so its body spans roughly x=.1496..0.1896 and is **still partially inside the narrow throat**. Do NOT call it fully through the passage. For the cube's trailing face to exit past x=.170, its center must reach roughly x=.190. The terminal goal x=.210 with 20mm threshold independently requires ~x=.190 at y≈0 (slightly more depending on lateral offset). Current lateral y=.0026 lies within nominal center gap ±.005, but that does not prove robot finger clearance or unobstructed contacts.

## Interpretation: no automatic rerun

The measured v3 failure was safe rear TCP never accepted, with no cube motion. The v4 reference-based TCP z=.038 successfully enabled real +X push of ~170mm while preserving blocker clearance and no logged guard violation. This validates progress **for the target executor**, not source Pick or learned transfer.

The v4 final scalar alone cannot identify whether:
1. Planned Screw path for TCP x=.260 contains a very long 210+ action trajectory that was still executing;
2. Actual TCP stopped advancing inside the 50mm throat because of link/wall/body interference; or
3. TCP advanced but lost effective cube contact or slipped off-center.

Upstream ManiSkill `BaseMotionPlanningSolver.follow_path` iterates every action of the planned path and does not use observed cube progress as a termination check. The custom `TracedEnv` enforces the 450-action global stop; this is **not proof the joint-space solver reported a planning failure**.

**Do not increase 450 steps or run a blind v5.** First read the actual final 210 raw states and compare TCP/cube progress to localize physical cause. This is a pure JSON analysis; the prior 450-step trace must stay immutable.

## Minimum read-only server command

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
python - <<'PY'
import json,math
from pathlib import Path
p=Path("outputs/m1_current_target_reference_height_v4/target_reference_height/physical_trace.json")
rows=json.loads(p.read_text())
assert len(rows)==450, f"unexpected physical length={len(rows)}"
for phase in ["push_1_x_0.110","push_2_x_0.180","push_3_x_0.260"]:
    seg=[r for r in rows if r.get("execution_phase")==phase]
    print("STAGE",phase,"steps",len(seg))
    if seg:
        a,b=seg[0],seg[-1]
        print("  first",a["t"],"tcp",a["tcp"],"cube",a["task_cube"])
        print("  last ",b["t"],"tcp",b["tcp"],"cube",b["task_cube"])
        print("  max_tcp_x",max(r["tcp"][0] for r in seg),
              "max_cube_x",max(r["task_cube"][0] for r in seg),
              "cube_total_xy",sum(math.dist(x["task_cube"][:2],y["task_cube"][:2])
                for x,y in zip(seg,seg[1:])))
        for idx in [0,15,50,100,150,len(seg)-1]:
            if 0<=idx<len(seg):
                r=seg[idx]
                print("  sample",r["t"],"tcp_x",round(r["tcp"][0],5),
                      "cube_x",round(r["task_cube"][0],5),
                      "gap_x",round(r["task_cube"][0]-r["tcp"][0],5))
PY
```

This distinguishes stationary TCP + stationary cube from TCP motion with no cube response. However the logs still do not record finger-wall contact forces, quaternion orientation or commanded-vs-realized joint trajectories. A stalled TCP is not automatically proof of wall collision.

## Next decision

- **TCP X flat and cube X flat through push3:** investigate joint/obstacle interference or infeasible TCP target, not longer trajectory budget.
- **TCP moves while cube flat:** investigate contact loss/tool pose, not blanket gains.
- **TCP and cube both advance toward x=.190 before 450:** investigate planned path length and stopping criteria, but still keep original acceptance; any new bounded execution would need a versioned, evidence-based cap/shortening rather than hidden additional retries.

The source Pick same-goal physical grasp still failed independently (106 actions, zero valid two-finger grasps); current target progress is **not** a successful paired cross-behavior result and does not license TAMP/M2/novelty claims.

This document transcribes real operator-provided console values and source-verified execution behavior. The full v4 trace remains only on the user's machine, and stage-3 metrics above are **not** claimed to have been executed from that trace here.
