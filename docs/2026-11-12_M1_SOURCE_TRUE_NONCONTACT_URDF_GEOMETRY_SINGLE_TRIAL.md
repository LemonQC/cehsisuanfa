# M1 original same-goal SOURCE Pick: real noncontact vs Panda collision envelope

## Operator-verified original physical data

| Source case | Goal x | Actions | Nearest actual TCP to cube | Actual grasp steps | Lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original M1B3 top-down | 0.210m | 335 | 63.966mm | 0 | ~0 |
| Current M1B3 side approach | 0.210m | 106 | 102.420mm | 0 | ~0 |
| Earlier M1B2 original top-down | 0.150m (DIFFERENT) | 170 | 2.608mm | 84 | 127.83mm |

First gripper close in M1B3 top-down at step82: real TCP [0.0173464,-0.00426984,0.0871373] m versus cube [0,0,.020] m. Original planner nominally succeeded in several phases even though the cube NEVER actually grasped or moved. Side-grasp closed at TCP z=.121m, approximately 103mm from cube. Both need physical precontact reach, NOT gripper force tweaks.

## Actual Panda URDF geometry versus ORIGINAL M1B3 walls

Original M1B3 has fixed rails centered y=+/-0.068m, halfwidth .016m and height .075m: interior halfwidth .052m. The official ManiSkill panda_v2.urdf left-finger screw-mount collision box is centered at local y=.0185m, width .015m. Fully open finger q=.040m extends to local .066m; if its closing axis points along world Y, this intrudes **14mm** into each sidewall. A partially open q=.023m extends to .049m (nominal 3mm rail clearance if Y-aligned) with a 46mm gap, still 6mm wider than the 40mm cube. This is a physical geometric hypothesis, not proof of full Panda reachability or attribution of the original failure: the previous top-down target quaternion and actual collision forces are not in its saved trace.

Upstream physical asset: https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/assets/robots/panda/panda_v2.urdf

## Exactly one original scene source contact candidate

New code: agcofl/m1_source_geom_aligned_topdown_v1.py
Runner: scripts/run_m1_source_geom_aligned_topdown_once.sh
CPU regressions: tests/test_m1_source_geom_aligned_topdown_v1.py

Frozen ORIGINAL task: AGCOFLM1B3Corridor-v0, Panda, seed120, cube goal x=.210m with 20mm success tolerance, blocker x=.100 with goal unchanged (must not be cleared), static 50mm throat, max450 PhysX actions. The successful target v6 is FROZEN and NEVER re-executed.

This one documented SOURCE executor adaptation reuses upstream OBB top-down grasp and PandaArmMotionPlanningSolver, but aligns gripper closing with corridor longitudinal world X, commands a 46mm opening (normalized action +0.32), registers original wall and blocker surfaces as deterministic MPLib collision points, and explicitly enables planning with that point cloud. This is NOT original unmodified official solver and NOT a learned method or TAMP.

Actual TCP must reach within **13mm** of cube BEFORE allowing any close command. A hard **155-action precontact cap** prevents another 335-step no-contact source. After closing, actual Panda bilateral is_grasping must hold or lift/carry/release is NOT executed. If grasp is real, it continues the already successful B2 lift/carry/lower/release pattern, but goal remains x=.210 and blocker must stay unmoved. Trace adds actual finger qpos, TCP quaternion and hand/finger vs wall contact forces where PhysX supports them; unavailable forces remain NULL.

## Single-shot command

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05
    bash scripts/run_m1_source_geom_aligned_topdown_once.sh

The runner FIRST checks five CPU tests, original scene geometry, all three previously saved PHYSICAL source traces, the actual locally installed Panda URDF finger collision dimensions and TCP offset, and the immutable actual target-v6 66-step success. A failed preflight starts zero new physics. Otherwise it permits ONE bounded independent original-scene source trial in outputs/m1_source_geom_aligned_topdown_v1/source_aligned_topdown, saving separate real H5/MP4/JSON.

Expected marker: M1_SOURCE_GEOMETRY_ALIGNED_FINAL_GATE. Original H5, video, statuses and reports never overwritten.

## Decision and stop

If TCP still fails to reach <13mm, freeze the source pilot as an approach/geometry NO-GO; inspect measured wall contact force or joint tracking before ever proposing other grasps. If TCP reaches but physical bilateral grasp fails, inspect fingers and contact. If a grasp succeeds but lift/carry/lower fails, report only that actual achieved stage. ONLY an actual grasp, physical overflight above the walls, grounded release within 20mm of x=.210 and blocker displacement <25mm establishes a valid same-goal source. The v6 target 66-step success does not imply source physics success. Strong equal-information TAMP, video-specific necessity and new learned M2 are still NOT_RUN. No additional target episodes or random grasp parameter sweeps.

Observed data in this note came from user's actual saved local traces. New source candidate code is committed, but no user's server has executed its PhysX rollout yet.