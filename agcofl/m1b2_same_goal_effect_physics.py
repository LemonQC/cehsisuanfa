"""M1-B2 one fixed PhysX scene: same-goal source Pick, target Push-only controls.

DEV feasibility only, not a learned method and not a completed fair TAMP.
Source: adapt the published ManiSkill OBB grasp + mplib Panda planner,
not the previously failed custom ScriptedPickSource. Target: original
DiagnosticPushTeacher on actual task/blocker actors, open gripper only.

Important: dynamic blocker ahead of the task cube is observed and tested
in direct-vs-clear control. The current finite geometry does NOT prove
clearing is the *unique* possible strategy; do not report "causally necessary"
until a geometry/topology counterfactual rules out push-around.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass

import numpy as np

from .roboreel_bridge import DiagnosticPushTeacher, np1

ENV_ID = "AGCOFLM1B2SameGoal-v0"
GOAL_DISPLACEMENT_M = 0.15
BLOCKER_X_OFFSET_M = 0.077
BLOCKER_HALF_SIZE_M = 0.025
BARRIER_HALF_SIZE_M = 0.027
BARRIER_Y_OFFSETS_M = (-0.083, 0.083)
BLOCKER_CLEARANCE_Y_M = 0.115
MAX_STEPS = 450


def _pose(actor):
    return np1(actor.pose.p)[:3].astype(float)


def _bounded_summary(v):
    if isinstance(v,np.ndarray):
        return v.tolist()
    if isinstance(v,np.generic):
        return v.item()
    raise TypeError(type(v).__name__)


def build_scene(official_root):
    """Register custom environment, reusing pinned RoboReel PushCube semantics."""
    import gymnasium as gym
    import sapien
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.building import actors
    from mani_skill.utils.structs import Pose
    src=(Path(official_root).resolve()/"roboreel_env/src")
    if not (src/"envs/push_cube.py").is_file():
        raise FileNotFoundError("Pinned RoboReel PushCube source is unavailable")
    if str(src) not in sys.path:
        sys.path.insert(0,str(src))
    from envs.push_cube import PushCubeEnvV2

    if ENV_ID in gym.registry:
        return ENV_ID

    class SameGoalWithBlocker(PushCubeEnvV2):
        """Cube + physically separate movable blocker + two static short walls."""
        def _load_scene(self, options):
            super()._load_scene(options)
            self.blocker = actors.build_cube(
                self.scene,
                half_size=BLOCKER_HALF_SIZE_M,
                name="m1b2_dynamic_blocker",
                body_type="dynamic",
                color=[0.9,0.55,0.12,1],
                initial_pose=sapien.Pose(p=[0.4,0.0,BLOCKER_HALF_SIZE_M]))
            self.corridor_walls=[
                actors.build_cube(self.scene,
                    half_size=BARRIER_HALF_SIZE_M,
                    name=f"m1b2_static_wall_{i}",
                    body_type="static",color=[0.7,0.15,0.18,1],
                    initial_pose=sapien.Pose(p=[0.4,0.2,BARRIER_HALF_SIZE_M]))
                for i in range(len(BARRIER_Y_OFFSETS_M))
            ]

        def _initialize_episode(self, env_idx, options):
            super()._initialize_episode(env_idx, options)
            # Keep the original robot reset and table; freeze single DEV
            # task start/goal at safe within-workspace coordinates.
            start=np.array([0.0,0.0,self.cube_half_size],dtype=np.float32)
            self.obj.set_pose(Pose.create_from_pq(
                p=start[None,:],q=[1,0,0,0]))
            self.goal_pos = self.obj.pose.p.clone()
            self.goal_pos[...,0] += GOAL_DISPLACEMENT_M
            self.goal_pos[...,2] = 1e-3
            blocker=np.array([
                start[0]+BLOCKER_X_OFFSET_M,start[1],BLOCKER_HALF_SIZE_M
            ],dtype=np.float32)
            self.blocker.set_pose(Pose.create_from_pq(
                p=blocker[None,:],q=[1,0,0,0]))
            for actor,offset in zip(self.corridor_walls,BARRIER_Y_OFFSETS_M):
                xyz=np.array([[blocker[0],start[1]+offset,
                              BARRIER_HALF_SIZE_M]],dtype=np.float32)
                actor.set_pose(Pose.create_from_pq(p=xyz,q=[1,0,0,0]))

        def _get_obs_extra(self, info):
            obs=super()._get_obs_extra(info)
            if "state" in self.obs_mode:
                obs["m1b2_blocker_pose"]=self.blocker.pose.raw_pose
            return obs

    gym.register(id=ENV_ID,entry_point=SameGoalWithBlocker,
                 max_episode_steps=MAX_STEPS)
    return ENV_ID


def make_env(official_root, seed, out, name):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.record import RecordEpisode

    eid=build_scene(official_root)
    # Source and target share the identical SAPIEN scene/seed, with
    # only the allowed low-level action behavior changed.
    base=gym.make(
        eid,obs_mode="state",robot_uids="panda",num_envs=1,
        control_mode=("pd_joint_pos" if name=="source" else "pd_ee_delta_pose"),
        render_mode="rgb_array",sim_backend="physx_cpu",
        distract=False,max_episode_steps=MAX_STEPS)
    wrapper=RecordEpisode(base,output_dir=str(out),
        trajectory_name=name,save_video=True,save_on_reset=False,
        source_type="motionplanning" if name=="source" else "privileged_diagnostic",
        source_desc="M1B2 research DEV pilot, NOT published baseline",
        record_reward=False)
    wrapper.reset(seed=seed)
    return wrapper


class TracedEnv:
    """Record actual simulator state after every original solver action."""
    def __init__(self, env):
        self.env=env
        self.unwrapped=env.unwrapped
        self.trace=[]
    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        b=self.unwrapped
        gripper=float(np.asarray(action).reshape(-1)[-1])
        self.trace.append({
            "t":len(self.trace),"task_cube":_pose(b.obj).tolist(),
            "blocker":_pose(b.blocker).tolist(),
            "tcp":_pose(b.agent.tcp_pose).tolist(),
            "gripper_action":gripper,
            "is_grasping_task":bool(np1(b.agent.is_grasping(b.obj))[0]),
            "env_success":bool(np1(info.get("success",False))[0]),
        })
        return obs,reward,terminated,truncated,info


def actual_gate(trace, initial, final, goal, source):
    """Evidence thresholds use OBSERVED cube states, not planned waypoints."""
    if not trace:
        return {"valid":False,"reason":"no_physx_actions"}
    xyz=np.asarray([q["task_cube"] for q in trace],float)
    goal_error=float(np.linalg.norm(final[:2]-goal[:2]))
    peak_lift=float(xyz[:,2].max()-initial[2])
    grasped=any(q["is_grasping_task"] for q in trace)
    released=not trace[-1]["is_grasping_task"]
    grounded=abs(final[2]-initial[2])<0.009
    if source:
        over=any(abs(q["task_cube"][0]-BLOCKER_X_OFFSET_M)<0.04
                 and q["task_cube"][2]>(2*BARRIER_HALF_SIZE_M+0.02)
                 for q in trace)
        good=(goal_error<0.05 and grounded and grasped and released
              and peak_lift>0.07 and over)
        return dict(valid=bool(good),grasped=grasped,released=released,
                    peak_lift_m=peak_lift,over_obstacle=over,
                    goal_error_m=goal_error,grounded=bool(grounded))
    return dict(valid=bool(goal_error<0.05 and grounded),
                goal_error_m=goal_error,grounded=bool(grounded),
                peak_lift_m=peak_lift,ever_grasped=grasped)


def run_source(env):
    """Adapt OFFICIAL ManiSkill OBB grasp and MPLib solver to this goal."""
    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver)
    from mani_skill.examples.motionplanning.base_motionplanner.utils import (
        compute_grasp_info_by_obb,get_actor_obb)

    traced=TracedEnv(env)
    b=env.unwrapped
    init=_pose(b.obj)
    goal=np1(b.goal_pos)[:3].copy()
    planner=PandaArmMotionPlanningSolver(
        traced,debug=False,vis=False,
        base_pose=b.agent.robot.pose,
        visualize_target_grasp_pose=False,print_env_info=False)
    phases=[]
    err=None
    try:
        obb=get_actor_obb(b.obj)
        approaching=np.array([0.,0.,-1.])
        closing=b.agent.tcp.pose.to_transformation_matrix()[0,:3,1].cpu().numpy()
        grasp_info=compute_grasp_info_by_obb(
            obb,approaching=approaching,
            target_closing=closing,depth=0.025)
        grasp_pose=b.agent.build_grasp_pose(
            approaching,grasp_info["closing"],b.obj.pose.sp.p)
        # Exactly the official "reach -> grasp" structure, then
        # SAME grounded goal relocation and RELEASE (not LiftCube).
        stages=[
            ("approach",grasp_pose*sapien.Pose([0,0,-0.05])),
            ("grasp_pose",grasp_pose),
        ]
        for stage,pose in stages:
            result=planner.move_to_pose_with_screw(pose)
            phases.append({"name":stage,"planner_success":result!=-1})
            if result==-1:
                raise RuntimeError(f"official_mplib_{stage}_failed")
        planner.close_gripper()
        phases.append({"name":"close_gripper","planner_success":True})
        # Direct original grasp frame is preserved while translating
        # task cube relative to source start, not a new hand-coded arm
        # servo/controller.
        delta_xy=np.array([goal[0]-init[0],goal[1]-init[1],0.])
        offset=np.array([0.,0.,0.13])
        targets=[
            ("lift",sapien.Pose(grasp_pose.p+offset,grasp_pose.q)),
            ("across",sapien.Pose(grasp_pose.p+offset+delta_xy,grasp_pose.q)),
            ("lower",sapien.Pose(grasp_pose.p+delta_xy,grasp_pose.q)),
        ]
        for stage,pose in targets:
            result=planner.move_to_pose_with_screw(pose)
            phases.append({"name":stage,"planner_success":result!=-1})
            if result==-1:
                raise RuntimeError(f"official_mplib_{stage}_failed")
        planner.open_gripper(t=12)
        phases.append({"name":"release","planner_success":True})
    except Exception as exc:
        err=f"{type(exc).__name__}: {str(exc)[:350]}"
    finally:
        planner.close()
    final=_pose(b.obj)
    gate=actual_gate(traced.trace,init,final,goal,True)
    gate["valid"]=bool(gate["valid"] and err is None)
    return dict(kind="source_pick_to_same_goal",
                state_before=init.tolist(),goal=goal.tolist(),
                blocker_initial=[BLOCKER_X_OFFSET_M,0.,BLOCKER_HALF_SIZE_M],
                final=final.tolist(),gate=gate,error=err,
                steps=len(traced.trace),planner_phases=phases,
                trace=traced.trace)


class ActorPush(DiagnosticPushTeacher):
    """Reuse baseline physics push executor on selected actor, not new Ours."""
    def __init__(self,env,actor,target_xy):
        super().__init__(env,max_attempts=2)
        self.actor=actor
        self.target_xy=np.asarray(target_xy,float)
    def cube_xyz(self):
        return _pose(self.actor).astype(np.float32)
    def goal_xyz(self):
        return np.r_[self.target_xy,self.cube_xyz()[2]].astype(np.float32)


def run_push(env,mode,max_steps):
    traced=TracedEnv(env)
    b=env.unwrapped
    init=_pose(b.obj)
    goal=np1(b.goal_pos)[:3].copy()
    blocker_init=_pose(b.blocker)
    if mode=="target_clear":
        sequence=[
            ("clear_blocker",b.blocker,
                blocker_init[:2]+np.array([0.,BLOCKER_CLEARANCE_Y_M])),
            ("push_task_cube",b.obj,goal[:2]),
        ]
    else:
        sequence=[("push_task_cube",b.obj,goal[:2])]
    phases=[]
    for name,actor,dest in sequence:
        ctrl=ActorPush(traced,actor,dest)
        start_index=len(traced.trace)
        for _ in range(max_steps-len(traced.trace)):
            if float(np.linalg.norm(_pose(actor)[:2]-dest))<0.035:
                break
            action=np.asarray(ctrl.act(),dtype=np.float32)
            action.reshape(-1)[-1]=-1.0
            _,_,term,trunc,info=traced.step(action)
            if ctrl.failure or bool(np1(term)[0]) or bool(np1(trunc)[0]):
                break
        phases.append({
            "name":name,"steps":len(traced.trace)-start_index,
            "target_xy":np.asarray(dest).tolist(),
            "final_actor_xy":_pose(actor)[:2].tolist(),
            "distance_m":float(np.linalg.norm(_pose(actor)[:2]-dest)),
            "executor_failed":ctrl.failure,
        })
        if phases[-1]["distance_m"]>=0.035 or ctrl.failure:
            break
    task_final=_pose(b.obj)
    block_final=_pose(b.blocker)
    task=actual_gate(traced.trace,init,task_final,goal,False)
    clearance=float(np.linalg.norm(block_final[:2]-blocker_init[:2]))
    saw_gripper_close=any(q["gripper_action"]>0 for q in traced.trace)
    # A Push-only goal completion is invalid if it lifted/grasped.
    target_physically_valid=bool(task["valid"] and
        task["peak_lift_m"]<0.015 and not task["ever_grasped"]
        and not saw_gripper_close)
    clearance_real=bool(clearance>0.075)
    inserted=bool(mode=="target_clear" and
                  phases and phases[0]["name"]=="clear_blocker"
                  and clearance_real)
    return dict(kind=mode,state_before=init.tolist(),goal=goal.tolist(),
                blocker_before=blocker_init.tolist(),
                blocker_final=block_final.tolist(),
                final=task_final.tolist(),
                task_gate=task,
                push_only_success=target_physically_valid,
                blocker_displacement_m=clearance,
                blocker_effect_physically_inserted=inserted,
                observed_source_free_flow_copied=False,
                actual_rollout_steps=len(traced.trace),
                privileged_object_state=True,
                baseline_class="physical_oracle_diagnostic_NOT_FULL_TAMP",
                phases=phases,trace=traced.trace)


def run_case(official_root,seed,mode,out,max_steps):
    if mode not in ("source","target_direct","target_clear"):
        raise ValueError(mode)
    out=Path(out).resolve()
    root=out/mode
    if (root/"status.json").is_file():
        raise FileExistsError(f"Existing case; never overwrite evidence: {root}")
    root.mkdir(parents=True,exist_ok=True)
    env=None
    try:
        env=make_env(official_root,seed,root,mode)
        result=run_source(env) if mode=="source" else run_push(env,mode,max_steps)
    except Exception as e:
        result=dict(kind=mode,error=f"{type(e).__name__}: {str(e)[:1200]}",
                    actual_physics_success=False,trace=[],
                    status="M1B2_EXECUTION_ERROR_NOT_VALID_EVIDENCE")
    finally:
        if env is not None:
            env.close()
    trace=result.pop("trace",[])
    (root/"physical_trace.json").write_text(
        json.dumps(trace,indent=2,default=_bounded_summary)+"\n")
    result.update(seed=seed,scene=ENV_ID,
                  same_final_goal_config=True,
                  physically_verified_matched_success=(
                      bool(result.get("gate",{}).get("valid")) if mode=="source"
                      else bool(result.get("push_only_success",False))),
                  not_learned_method=True, paper_ready=False)
    (root/"status.json").write_text(
        json.dumps(result,indent=2,default=_bounded_summary)+"\n")
    print("M1B2_CASE_PHYSICS",json.dumps({
        "mode":mode,"success":result["physically_verified_matched_success"],
        "source_gate":result.get("gate"),"target_gate":result.get("task_gate"),
        "blocker_effect":result.get("blocker_effect_physically_inserted"),
        "steps":result.get("steps",result.get("actual_rollout_steps")),
        "error":result.get("error"),
        "path":str(root/"status.json")},default=_bounded_summary),flush=True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-root",default="third_party/roboreel_models")
    p.add_argument("--seed",type=int,default=120)
    p.add_argument("--mode",choices=("source","target_direct","target_clear"),required=True)
    p.add_argument("--max-steps",type=int,default=MAX_STEPS)
    p.add_argument("--out",default="outputs/m1b2_same_goal_dev")
    args=p.parse_args(argv)
    if not 1<=args.max_steps<=MAX_STEPS:
        p.error("max steps must stay within frozen fixed budget")
    run_case(args.official_root,args.seed,args.mode,args.out,args.max_steps)


if __name__=="__main__":
    main()
