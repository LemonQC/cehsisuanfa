"""M1 current-scene, one-factor target MPLib physical ablation: CLOSED pusher.

Upstream ManiSkill Panda original push_cube.py explicitly calls close_gripper().
The prior adapted target controller used OPEN (+1) and moved blocker 32mm;
the older successful 32-step DiagnosticPushTeacher used CLOSED (-1)
though its legacy validation erroneously labeled that command "open".
This exactly reuses same M1B3 geometry, exact same MPLib approach/push
poses, only changes gripper closure state, recording both finger joints
and BOTH task + blocker actual grasp flags. One shot, no new task/method.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .m1b2_same_goal_effect_physics import TracedEnv, _pose, _bounded_summary
from .m1b3_corridor_physics import (
    BLOCKER_HALF, BLOCKER_X, GOAL_RADIUS, MAX_STEPS, PARKING_Y,
    SEED, TASK_HALF, _actual_satisfied, register_env, geometry_preflight,
)
from .m1_current_executor_swap import get_planner
from .roboreel_bridge import np1

GRIPPER_OPEN_ACTION = 1.0
GRIPPER_CLOSED_ACTION = -1.0
VERSION = "m1_current_target_mplib_closed_pusher_v2"
CASE = "target_closed_mplib"
OUTPUT = "outputs/m1_current_closed_pusher_v2"


def gripper_action_semantics():
    """Grounded in ManiSkill official PandaArmMotionPlanningSolver constants."""
    return {
        "panda_open_action":GRIPPER_OPEN_ACTION,
        "panda_closed_action":GRIPPER_CLOSED_ACTION,
        "previous_successful_clearance_original_command":-1.0,
        "previous_failed_mplib_clearance_original_command":1.0,
        "legacy_m1b3_gripper_never_closed_label_valid":False,
        "legacy_m1b2_saw_gripper_close_label_valid":False,
        "push_only_requires_no_object_grasp_not_open_fingers":True,
        "only_intentional_change_from_prior_mplib_target":
            "gripper state OPEN (+1) to CLOSED (-1); same Panda/MPLib poses and geometry",
    }


class ContactTracer(TracedEnv):
    def step(self, action):
        obs,reward,terminated,truncated,info=super().step(action)
        b=self.unwrapped
        self.trace[-1]["is_grasping_blocker"]=bool(
            np1(b.agent.is_grasping(b.blocker))[0])
        q=np1(b.agent.robot.get_qpos())
        self.trace[-1]["finger_joint_qpos_m"]=[
            float(q[-2]),float(q[-1])]
        return obs,reward,terminated,truncated,info


def make_joint_push_env(official_root,out):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.record import RecordEpisode
    eid=register_env(official_root)
    root=gym.make(
        eid,obs_mode="state",robot_uids="panda",num_envs=1,
        control_mode="pd_joint_pos",render_mode="rgb_array",
        sim_backend="physx_cpu",distract=False,
        max_episode_steps=MAX_STEPS,
    )
    rec=RecordEpisode(
        root,output_dir=str(out),
        trajectory_name=CASE,save_video=True,save_on_reset=False,
        source_type="privileged_diagnostic",
        source_desc="M1 original corridor, closed-tool physical ablation; NOT TAMP or method",
        record_reward=False,
    )
    rec.reset(seed=SEED,options={"m1b3_open":False})
    return rec


def observed_push_only(trace):
    """Correct Panda action signs and check BOTH physical grasp flags."""
    if not trace:
        return {"valid":False,"reason":"no_actions"}
    task_ever_grasped=any(bool(row.get("is_grasping_task")) for row in trace)
    blocker_ever_grasped=any(bool(row.get("is_grasping_blocker")) for row in trace)
    actions=[float(row["gripper_action"]) for row in trace]
    return {
        "valid":not task_ever_grasped and not blocker_ever_grasped,
        "ever_grasped_task":task_ever_grasped,
        "ever_grasped_blocker":blocker_ever_grasped,
        "closed_gripper_command_steps":sum(x<0 for x in actions),
        "open_gripper_command_steps":sum(x>0 for x in actions),
        "finger_joint_qpos_final_m":trace[-1].get("finger_joint_qpos_m"),
        "note":"Closed Panda fingers can PUSH without grasping either object.",
    }


def execute(env):
    import sapien
    traced=ContactTracer(env)
    b=traced.unwrapped
    cube_start=_pose(b.obj)
    goal=np1(b.goal_pos)[:3].copy()
    blocker_start=_pose(b.blocker)
    planner=get_planner(traced)
    phases=[]
    error=None
    try:
        # The *only* intentionally changed action in prior failed MPLib
        # target: open_gripper(t=6) -> close_gripper(t=6).
        # Also matches published ManiSkill panda/solutions/push_cube.py.
        planner.close_gripper(t=6)
        for name,obj,target in (
            ("clear_blocker",b.blocker,np.asarray([BLOCKER_X,PARKING_Y])),
            ("push_task",b.obj,goal[:2]),
        ):
            initial=_pose(obj)
            delta=np.asarray(target)-initial[:2]
            norm=float(np.linalg.norm(delta))
            if norm<.005:
                raise RuntimeError(f"{name}: no meaningful target motion")
            direction=delta/norm
            # Identical predecessor's pre/post XY poses, TCP Z and Q.
            pre=np.r_[initial[:2]-direction*.05,initial[2]]
            post=np.r_[target+direction*.05,initial[2]]
            q=np1(b.agent.tcp.pose.q)[:4]
            t0=len(traced.trace)
            res=planner.move_to_pose_with_RRTConnect(
                sapien.Pose(p=pre,q=q))
            if res==-1:
                raise RuntimeError(f"upstream_mplib_{name}_precontact_failed")
            res=planner.move_to_pose_with_screw(
                sapien.Pose(p=post,q=q))
            if res==-1:
                raise RuntimeError(f"upstream_mplib_{name}_push_failed")
            measured=_pose(obj)
            residual=float(np.linalg.norm(measured[:2]-target))
            phases.append({
                "name":name,"actual_steps":len(traced.trace)-t0,
                "initial_xy_m":initial[:2].tolist(),
                "goal_xy_m":target.tolist(),
                "observed_xy_m":measured[:2].tolist(),
                "actual_error_m":residual,
                "task_ever_grasped":any(r["is_grasping_task"]
                    for r in traced.trace[t0:]),
                "blocker_ever_grasped":any(r["is_grasping_blocker"]
                    for r in traced.trace[t0:]),
            })
            if residual>.035:
                raise RuntimeError(
                    f"PHYSICS_{name}_GOAL_NOT_REACHED: error {residual:.4f}m")
    except Exception as exc:
        error=f"{type(exc).__name__}: {str(exc)[:500]}"
    finally:
        planner.close()

    final_cube=_pose(b.obj)
    final_blocker=_pose(b.blocker)
    report=observed_push_only(traced.trace)
    goal_error=float(np.linalg.norm(final_cube[:2]-goal[:2]))
    task_peak_z=max([cube_start[2]]+
                    [q["task_cube"][2] for q in traced.trace])
    blocker_peak_z=max([blocker_start[2]]+
                       [q["blocker"][2] for q in traced.trace])
    success=bool(
        error is None and len(phases)==2
        and report["valid"] and _actual_satisfied(b)
        and goal_error<GOAL_RADIUS
        and abs(final_cube[2]-TASK_HALF)<.009
        and abs(final_blocker[2]-BLOCKER_HALF)<.012
        and task_peak_z-cube_start[2]<.015
        and blocker_peak_z-blocker_start[2]<.015
        and final_blocker[1]>PARKING_Y-.035
        and np.linalg.norm(final_blocker[:2]-blocker_start[:2])>.075
    )
    return {
        "kind":CASE,"success":success,"error":error,
        "steps":len(traced.trace),"phases":phases,
        "state_before":cube_start.tolist(),"goal":goal.tolist(),
        "cube_final":final_cube.tolist(),
        "blocker_before":blocker_start.tolist(),
        "blocker_final":final_blocker.tolist(),
        "task_goal_error_m":goal_error,
        "push_only":report,"task_peak_z_m":task_peak_z,
        "blocker_peak_z_m":blocker_peak_z,
        "action_semantics":gripper_action_semantics(),
        "scene":"AGCOFLM1B3Corridor-v0",
        "geometry_unchanged":True,
        "only_action_difference_closed_pusher":True,
        "is_original_unmodified_official_solver":False,
        "strong_tamp_evaluated":False,
        "is_learned_method":False,"paper_ready":False,
        "trace":traced.trace,
    }


def run_once(official_root,out):
    if not geometry_preflight()["pass"]:
        raise RuntimeError("Original M1B3 geometry constraint preflight failed")
    root=Path(out).resolve()
    case=root/CASE
    if (case/"status.json").exists() or any(case.glob("*.h5")) or any(case.glob("*.json")):
        raise FileExistsError(f"Preserve immutable physical result: {case}")
    case.mkdir(parents=True,exist_ok=True)
    env=None
    artifact_error=None
    try:
        env=make_joint_push_env(official_root,case)
        result=execute(env)
    except Exception as exc:
        result=dict(kind=CASE,success=False,steps=0,trace=[],
                    error=f"{type(exc).__name__}: {str(exc)[:650]}",
                    status="EXECUTION_ERROR_NOT_VALID_PHYSICS")
    finally:
        if env is not None:
            try:
                env.flush_trajectory()
                env.flush_video()
            except Exception as exc:
                artifact_error=f"{type(exc).__name__}: {str(exc)[:450]}"
            finally:
                env.close()
    trace=result.pop("trace",[])
    (case/"physical_trace.json").write_text(
        json.dumps(trace,indent=2,default=_bounded_summary)+"\n")
    result.update(version=VERSION,seed=SEED,max_steps=MAX_STEPS,
                  artifact_error=artifact_error)
    (case/"status.json").write_text(
        json.dumps(result,indent=2,default=_bounded_summary)+"\n")
    print("M1_TARGET_CLOSED_PUSHER_PHYSICS",json.dumps({
        "success":result["success"],"steps":result["steps"],
        "error":result.get("error"),
        "phases":result.get("phases",[]),
        "push_only":result.get("push_only"),
        "goal_error_m":result.get("task_goal_error_m"),
        "out":str(case/"status.json"),
    }),flush=True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--official-root",default="third_party/roboreel_models")
    p.add_argument("--out",default=OUTPUT)
    a=p.parse_args(argv)
    if a.preflight:
        print("M1_TARGET_GRIPPER_SEMANTICS_PREFLIGHT",
              json.dumps(gripper_action_semantics()),flush=True)
        return
    run_once(a.official_root,a.out)


if __name__=="__main__":
    main()
