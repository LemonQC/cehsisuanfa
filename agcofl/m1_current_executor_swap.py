"""One bounded executor substitution in FROZEN M1B3 geometry; NOT a new research task.

Source: official ManiSkill PandaArmMotionPlanningSolver with lateral
(approach +X from robot on -X) OBB grasp and *actual* grasp verification
before any lift, carrying or lowering.
Target: official ManiSkill Panda PushCube-style MPLib joint-space
move_to_pose_with_screw, repurposed to physically push the blocker to the
same old side pocket and then cube through the same old throat.
Not identical to published official original solvers: approach and
receding poses are adapted to current task, gripper stays OPEN for
push-only. Privileged geometry, NOT a trained policy or TAMP.

Does not edit or rerun any previous M1B2/B3/B4 case. New distinct files
only. Source smoke FIRST, target smoke only if source physical grasp PASS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .m1b2_same_goal_effect_physics import TracedEnv, _pose, _bounded_summary
from .m1b3_corridor_physics import (
    make_env, SEED, MAX_STEPS, GOAL_RADIUS, TASK_HALF,
    BLOCKER_HALF, PARKING_Y, BLOCKER_X, THROAT_X, WALL_HEIGHT,
    _actual_satisfied, geometry_preflight,
)
from .roboreel_bridge import np1

VERSION = "m1_current_task_official_executor_swap_v1"
OUT = Path("outputs/m1_current_executor_swap_smoke")
MODES = ("source_side_grasp","target_official_push")


def geometry_contract():
    # Original corridor has inner free sides y=+/-0.052m. A 40mm
    # cube/parallel gripper approach along X with closing axis along
    # Y is preferable to an unsupported top-down approach obstructed
    # by 75mm high rails. This is a single, declared candidate.
    g=geometry_preflight()
    if not g["pass"]:
        raise RuntimeError("Frozen M1B3 corridor changed")
    return {
        "scene":"AGCOFLM1B3Corridor-v0",
        "source_approaching_xyz":[1.,0.,0.],
        "source_target_closing_axis":[0.,1.,0.],
        "grasp_plan":"ManiSkill official Panda OBB+MPLib",
        "target_plan":"ManiSkill official Panda PushCube-style MPLib joint-space waypoints",
        "target_gripper_action":1.0,
        "target_gripper_never_grasps":True,
        "source_goal_x_m":.210,
        "old_scene_geometry_changed":False,
        "offline_geometry_check":True,
        "old_seed_replayed":False,
    }


def get_planner(traced):
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver)
    b=traced.unwrapped
    if b.control_mode!="pd_joint_pos":
        raise ValueError("Official Panda MPLib requires pd_joint_pos; refuses action-mode mismatch")
    return PandaArmMotionPlanningSolver(
        traced,debug=False,vis=False,base_pose=b.agent.robot.pose,
        visualize_target_grasp_pose=False,print_env_info=False)


def source_once(env):
    import sapien
    from mani_skill.examples.motionplanning.base_motionplanner.utils import (
        compute_grasp_info_by_obb,get_actor_obb)

    traced=TracedEnv(env)
    b=traced.unwrapped
    obj=b.obj
    start=_pose(obj)
    goal=np1(b.goal_pos)[:3].copy()
    blocker_start=_pose(b.blocker)
    planner=get_planner(traced)
    stages=[]
    error=None
    try:
        approaching=np.asarray([1.,0.,0.])
        closing=np.asarray([0.,1.,0.])
        geometry=compute_grasp_info_by_obb(
            get_actor_obb(obj),
            approaching=approaching,target_closing=closing,depth=.025)
        grasp=b.agent.build_grasp_pose(
            approaching,geometry["closing"],obj.pose.sp.p)
        # The upstream PickCube reaches 50mm behind and then closes.
        # Only one bounded adapted physical attempt is permitted here.
        stages.append(dict(stage="original_obb_grasp_pose",
                           measured_initial_cube=start.tolist(),
                           approaching=approaching.tolist()))
        def move(name,pose):
            result=planner.move_to_pose_with_RRTConnect(pose)
            success=result!=-1
            stages.append(dict(stage=name,planner_return_success=success,
                               actual_steps=len(traced.trace)))
            if not success:
                raise RuntimeError(f"upstream_mplib_{name}_failed")
        move("pregrasp_rrt",grasp*sapien.Pose([0,0,-.05]))
        move("grasp_rrt",grasp)
        planner.close_gripper()
        physically_grasped=bool(np1(b.agent.is_grasping(obj))[0])
        stages.append(dict(stage="physical_grasp_check",
                           actual_is_grasping=physically_grasped,
                           tcp_xyz_m=_pose(b.agent.tcp_pose).tolist(),
                           object_xyz_m=_pose(obj).tolist(),
                           actual_steps=len(traced.trace)))
        if not physically_grasped:
            raise RuntimeError("CONTACT_NO_GO: official RRT grasp pose executed, but Panda did not physically grasp cube")
        lift_height=.13
        move("lift",sapien.Pose(grasp.p+[0,0,lift_height],grasp.q))
        if _pose(obj)[2] < WALL_HEIGHT+TASK_HALF+.005:
            raise RuntimeError("CONTACT_NO_GO: physically held cube never cleared original wall")
        offset=np.asarray([goal[0]-start[0],goal[1]-start[1],0.])
        move("carry",sapien.Pose(grasp.p+[0,0,lift_height]+offset,grasp.q))
        if np.linalg.norm(_pose(obj)[:2]-goal[:2])>.055:
            raise RuntimeError("CONTACT_NO_GO: cube did not actually travel with gripper")
        move("place",sapien.Pose(grasp.p+offset,grasp.q))
        planner.open_gripper(t=12)
    except Exception as exc:
        error=f"{type(exc).__name__}: {str(exc)[:500]}"
    finally:
        planner.close()

    final=_pose(obj)
    lifted=max([r["task_cube"][2] for r in traced.trace]+[start[2]])-start[2]
    grasp_steps=sum(bool(r["is_grasping_task"]) for r in traced.trace)
    blocker_moved=np.linalg.norm(_pose(b.blocker)[:2]-blocker_start[:2])
    over=any(abs(r["task_cube"][0]-THROAT_X)<.022
             and r["task_cube"][2]>WALL_HEIGHT+TASK_HALF+.005
             for r in traced.trace)
    goal_error=float(np.linalg.norm(final[:2]-goal[:2]))
    physically_valid=bool(
        error is None and grasp_steps>0 and lifted>.08
        and over and blocker_moved<.025 and goal_error<GOAL_RADIUS
        and abs(final[2]-TASK_HALF)<.009
        and not bool(np1(b.agent.is_grasping(obj))[0])
        and _actual_satisfied(b))
    return {
        "kind":"source_side_grasp",
        "scene_version":"m1b3_corridor_original_geometry",
        "initial_cube":start.tolist(),"goal":goal.tolist(),
        "final_cube":final.tolist(),"initial_blocker":blocker_start.tolist(),
        "success":physically_valid,"error":error,
        "actual_grasp_steps":grasp_steps,"observed_lift_m":float(lifted),
        "over_throat_physically":bool(over),
        "blocker_displacement_m":float(blocker_moved),
        "goal_error_m":goal_error,"planner_phases":stages,
        "steps":len(traced.trace),"trace":traced.trace,
        "is_learned_method":False,"is_official_solver_unmodified":False,
    }


def target_once(env):
    """One full blocker+cube attempt using UPSTREAM joint-space MPLib.

    The original PushCube plan uses a pre-contact pose and screw push;
    here each target actor gets its own observed precontact and push goal.
    This is a privileged physical feasibility check only, NOT TAMP.
    """
    import sapien
    traced=TracedEnv(env)
    b=traced.unwrapped
    start=_pose(b.obj)
    goal=np1(b.goal_pos)[:3].copy()
    blocker_start=_pose(b.blocker)
    planner=get_planner(traced)
    phases=[]
    error=None
    try:
        # Original Panda PushCube closes gripper; our frozen Push-only
        # protocol expressly requires OPEN gripper, so this is a
        # documented executor adaptation, not an exact original run.
        planner.open_gripper(t=6)
        for name,obj,target in (
            ("clear_blocker",b.blocker,np.asarray([BLOCKER_X,PARKING_Y])),
            ("push_task",b.obj,goal[:2]),
        ):
            initial=_pose(obj)
            disp=np.asarray(target)-initial[:2]
            norm=float(np.linalg.norm(disp))
            if norm<.005:
                raise RuntimeError(f"{name}: no meaningful physical action target")
            direction=disp/norm
            # Use upstream PushCube's 50mm pre-contact offset.
            pre=np.r_[initial[:2]-direction*.05,initial[2]]
            post=np.r_[target+direction*.05,initial[2]]
            q=np1(b.agent.tcp.pose.q)[:4]
            before=len(traced.trace)
            rv=planner.move_to_pose_with_RRTConnect(sapien.Pose(p=pre,q=q))
            if rv==-1:
                raise RuntimeError(f"upstream_mplib_{name}_precontact_failed")
            rv=planner.move_to_pose_with_screw(sapien.Pose(p=post,q=q))
            if rv==-1:
                raise RuntimeError(f"upstream_mplib_{name}_push_failed")
            after=_pose(obj)
            residual=float(np.linalg.norm(after[:2]-target))
            phases.append({
                "name":name,"actual_steps":len(traced.trace)-before,
                "initial_xy_m":initial[:2].tolist(),
                "goal_xy_m":target.tolist(),
                "observed_xy_m":after[:2].tolist(),
                "actual_error_m":residual,
                "actual_grasp_steps":sum(
                    bool(s["is_grasping_task"]) for s in traced.trace[before:]),
            })
            if residual>.035:
                raise RuntimeError(f"PHYSICS_{name}_GOAL_NOT_REACHED: error {residual:.4f}m")
    except Exception as exc:
        error=f"{type(exc).__name__}: {str(exc)[:500]}"
    finally:
        planner.close()
    final=_pose(b.obj)
    blocker=_pose(b.blocker)
    grasp_count=sum(bool(s["is_grasping_task"]) for s in traced.trace)
    gripper_closed=any(s["gripper_action"]<0 for s in traced.trace)
    goal_err=float(np.linalg.norm(final[:2]-goal[:2]))
    actual=bool(
        error is None and len(phases)==2
        and goal_err<GOAL_RADIUS and _actual_satisfied(b)
        and abs(final[2]-TASK_HALF)<.009
        and blocker[1]>PARKING_Y-.035
        and np.linalg.norm(blocker[:2]-blocker_start[:2])>.075
        and grasp_count==0 and not gripper_closed
        and traced.trace)
    return {
        "kind":"target_official_push","scene_version":"m1b3_corridor_original_geometry",
        "initial_cube":start.tolist(),"goal":goal.tolist(),
        "final_cube":final.tolist(),"initial_blocker":blocker_start.tolist(),
        "final_blocker":blocker.tolist(),"success":actual,"error":error,
        "actual_grasp_steps":grasp_count,"gripper_closed_during_push":gripper_closed,
        "goal_error_m":goal_err,"phases":phases,
        "steps":len(traced.trace),"trace":traced.trace,
        "is_equal_information_tamp":False,"is_learned_method":False,
        "is_official_solver_unmodified":False,
    }


def run_one(mode,out,official_root):
    if mode not in MODES:
        raise ValueError(mode)
    geometry_contract()
    out=Path(out).resolve()
    case=out/mode
    if (case/"status.json").exists() or any(case.glob("*.h5")) or any(case.glob("*.json")):
        raise FileExistsError(f"Immutable swap smoke evidence exists: {case}")
    case.mkdir(parents=True,exist_ok=True)
    env=None
    artifact_error=None
    try:
        # SAME frozen scene and original robot; BOTH are joint-space so
        # upstream original solver can execute its actual qpos actions.
        env=make_env(official_root,SEED,case,"blocked_source")
        result=source_once(env) if mode==MODES[0] else target_once(env)
    except Exception as exc:
        result={
            "kind":mode,"success":False,"steps":0,"trace":[],
            "error":f"{type(exc).__name__}: {str(exc)[:550]}",
            "status":"EXECUTION_ERROR_NOT_VALID_EVIDENCE",
        }
    finally:
        if env is not None:
            try:
                env.flush_trajectory()
                env.flush_video()
            except Exception as exc:
                artifact_error=f"{type(exc).__name__}: {str(exc)[:300]}"
            finally:
                env.close()
    trace=result.pop("trace",[])
    (case/"physical_trace.json").write_text(
        json.dumps(trace,indent=2,default=_bounded_summary)+"\n")
    result.update(version=VERSION,mode=mode,seed=SEED,
                  max_actions=MAX_STEPS,original_geometry=True,
                  artifact_error=artifact_error,paper_ready=False)
    (case/"status.json").write_text(
        json.dumps(result,indent=2,default=_bounded_summary)+"\n")
    print("M1_CURRENT_EXECUTOR_SWAP",json.dumps({
        "mode":mode,"success":result["success"],"steps":result["steps"],
        "grasp_steps":result.get("actual_grasp_steps"),
        "error":result.get("error"),
        "gate":"SOURCE_REAL_PHYSICAL_GRASP_REQUIRED" if mode==MODES[0]
                else "PUSH_ONLY_TWO_REAL_EFFECTS_REQUIRED",
        "raw":str(case/"status.json"),
    }),flush=True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=MODES)
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--out",default=str(OUT))
    p.add_argument("--official-root",default="third_party/roboreel_models")
    a=p.parse_args(argv)
    if a.preflight:
        print("M1_CURRENT_EXECUTOR_SWAP_PREFLIGHT",json.dumps(
            geometry_contract()),flush=True)
        return
    if not a.mode:
        p.error("--mode required unless --preflight")
    run_one(a.mode,a.out,a.official_root)


if __name__=="__main__":
    main()
