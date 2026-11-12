"""M1 original scene v6: repair premature rear-contact descent stop in v5.

Engineering replacement only, not a method or TAMP. Use original
M1B3 "blocked_clear" task, unchanged actor geometry/seed/goal and
original proven actor-push blocker clearing, then prevent the earlier
unsafe lateral transition by measured overhead hover / behind descent.
Finally reuse the already successful M1B3 open_direct Cartesian
delta-pose +X push.  ONE trial, 450 absolute action cap and fast stall
stop; preserve every earlier H5/MP4/JSON unchanged.

WHY: v4 stage3 performed 210 PhysX actions with TCP X within ~0.6mm
and cube X constant at 0.16961m. More joint-space Screw steps cannot
establish why it is stuck. Existing open_direct achieved same goal in
35 steps with "pd_ee_delta_pose" on an open scene.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .m1b3_corridor_physics import (
    BLOCKER_X, BLOCKER_HALF, PARKING_Y, TASK_HALF, GOAL_RADIUS,
    SEED, MAX_STEPS, geometry_preflight, _actual_satisfied,
)
from .m1_current_closed_pusher_ablation import ContactTracer, observed_push_only
from .m1b2_same_goal_effect_physics import ActorPush, _pose, _bounded_summary
from .roboreel_bridge import delta_pose_action, np1

VERSION="m1_original_scene_cartesian_pusher_v6"
CASE="target_cartesian_contact_fixed"
OUT="outputs/m1_original_scene_cartesian_pusher_v6"

# Grounded in existing 35-step successful original "open_direct" trace,
# whose measured TCP when the cube first moves is z=.0373777m.
SAFE_HOVER_Z=.160
CONTACT_Z=.038
BEHIND_X_OFFSET=.060
MAX_APPROACH_ACTIONS_PER_STAGE=50
# v5 stopped descend at 18.89mm 3D residual because .022 was accepted,
# while push_to_goal REJECTED actual TCP Z=.05689 (> .052). 8mm total
# Cartesian tolerance is below the old 18.89mm residual and preserves
# original contact height =.038m, original underlying delta pose control.
DESCEND_REAR_TOLERANCE_M=.008
MAX_CLEAR_ACTIONS=110
MAX_PUSH_ACTIONS=120
MAX_STALL_ACTIONS=15
MIN_MEANINGFUL_TCP_PROGRESS=.00025
MIN_MEANINGFUL_CUBE_PROGRESS=.00035
MAX_UNEXPECTED_PRECONTACT_CUBE_MOVE=.008
MAX_LATERAL_CUBE_DRIFT=.010


def contract():
    geometry=geometry_preflight()
    if not geometry["pass"]:
        raise ValueError("Frozen original M1B3 geometry gate failed")
    return {
       "type":"M1_SAME_SCENE_EXISTING_CARTESIAN_PUSH_INTERFACE_REPLACEMENT",
       "scene":"AGCOFLM1B3Corridor-v0",
       "seed":SEED,"goal_x_m":.210,"budget_actions":MAX_STEPS,
       "source_control_mode":"pd_ee_delta_pose",
       "authentic_success_reference":"original M1B3 open_direct 35-step PhysX",
       "blocker_clearance":"original M1B3 ActorPush, already succeeded 32 steps",
       "after_clear_handoff":"safe hover and back, no low-height lateral robot travel",
       "pusher":"original DiagnosticPushTeacher delta_pose_action as used in open_direct",
       "closed_finger_command":-1.,
       "contact_tcp_target_z_m":CONTACT_Z,
       "never_change_original_task_or_geometry":True,
       "strict_stall_max_steps":MAX_STALL_ACTIONS,
       "no_repeated_baseline_or_source_trial":True,
       "strong_tamp_evaluated":False,"learned_method_evaluated":False,
    }


class PhaseTrace(ContactTracer):
    def __init__(self,env):
        super().__init__(env)
        self.phase="initialize"
    def step(self,action):
        result=super().step(action)
        row=self.trace[-1]
        row["execution_phase"]=self.phase
        return result


def safe_action(traced, target):
    tcp=_pose(traced.unwrapped.agent.tcp_pose)
    shape=tuple(traced.unwrapped.agent.action_space.shape)
    if shape!=(7,):
        raise ValueError(f"Original delta pose action must have 7 dimensions, got {shape}")
    action=delta_pose_action(tcp,np.asarray(target,float),shape)
    if float(action[-1]) != -1.:
        raise RuntimeError("Original closed-finger action semantics changed")
    return action


def check_push_only(traced,start_cube):
    row=traced.trace[-1]
    if row["is_grasping_task"] or row["is_grasping_blocker"]:
        raise RuntimeError("PHYSICAL_GRASP_FORBIDDEN_PUSH_ONLY")
    cube=np.asarray(row["task_cube"],float)
    if cube[2] > start_cube[2]+.015:
        raise RuntimeError("TASK_LIFT_FORBIDDEN_PUSH_ONLY")
    if abs(cube[1]) > MAX_LATERAL_CUBE_DRIFT:
        raise RuntimeError("CUBE_OFF_CENTER_LATERAL_FAIL")


def motion_to(traced,name,target,start_cube,tolerance=.015):
    """Existing reference Cartesian low-level interface; finite approach."""
    traced.phase=name
    initial=len(traced.trace)
    for _ in range(MAX_APPROACH_ACTIONS_PER_STAGE):
        tcp=_pose(traced.unwrapped.agent.tcp_pose)
        if np.linalg.norm(tcp-np.asarray(target))<=tolerance:
            break
        action=safe_action(traced,target)
        _,_,term,trunc,_=traced.step(action)
        check_push_only(traced,start_cube)
        if np.linalg.norm(_pose(traced.unwrapped.obj)[:2]-start_cube[:2]) > MAX_UNEXPECTED_PRECONTACT_CUBE_MOVE:
            raise RuntimeError(f"UNINTENDED_CUBE_MOVE_DURING_{name}")
        if bool(np1(term)[0]) or bool(np1(trunc)[0]):
            raise RuntimeError(f"PHYSICS_TERMINATED_DURING_{name}")
    actual=_pose(traced.unwrapped.agent.tcp_pose)
    if np.linalg.norm(actual-np.asarray(target))>tolerance:
        raise RuntimeError(f"NO_VALID_POSITION_AFTER_{name}")
    return {"phase":name,"actions":len(traced.trace)-initial,
            "final_tcp_xyz_m":actual.tolist(),
            "cube_xyz_m":_pose(traced.unwrapped.obj).tolist()}


def push_to_goal(traced,start_cube,goal):
    """One original +X delta-pose push; exits at REAL success or stall."""
    b=traced.unwrapped
    cube0=_pose(b.obj)
    if abs(cube0[1])>.004:
        raise RuntimeError("CUBE_NOT_CENTERED_BEFORE_NARROW_THROAT")
    tcp=_pose(b.agent.tcp_pose)
    if (not tcp[0] <= cube0[0]-.028
        or abs(tcp[1]-cube0[1])>.012
        or not .032<=tcp[2]<=.052):
        raise RuntimeError("ORIGINAL_PUSH_CONTACT_POSE_INVALID")
    # Original ActorPush target formula: goal + (half cube + 50mm)
    # for the TCP, while actual cube success is checked on each step.
    target=np.asarray([float(goal[0])+.070,float(goal[1]),CONTACT_Z])
    traced.phase="push_cartesian_original_reference"
    stall_tcp=0
    stall_cube=0
    last_tcp=_pose(b.agent.tcp_pose)
    last_cube=_pose(b.obj)
    started=len(traced.trace)
    stop_reason=None
    for _ in range(MAX_PUSH_ACTIONS):
        if _actual_satisfied(b) and np.linalg.norm(_pose(b.obj)[:2]-goal[:2])<GOAL_RADIUS:
            stop_reason="REAL_GOAL_REACHED"
            break
        _,_,terminated,truncated,_=traced.step(safe_action(traced,target))
        check_push_only(traced,start_cube)
        curr_tcp=_pose(b.agent.tcp_pose)
        curr_cube=_pose(b.obj)
        if curr_cube[0] < start_cube[0]-.006:
            raise RuntimeError("CUBE_MOVED_REVERSE_X")
        if curr_cube[0]>.095 and abs(curr_cube[1])>.006:
            raise RuntimeError("CUBE_OUTSIDE_NARROW_THROAT_BAND")
        # Monitor commanded +X movement and cube movement independently.
        stall_tcp=(stall_tcp+1 if curr_tcp[0]-last_tcp[0]<MIN_MEANINGFUL_TCP_PROGRESS else 0)
        stall_cube=(stall_cube+1 if curr_cube[0]-last_cube[0]<MIN_MEANINGFUL_CUBE_PROGRESS else 0)
        if stall_tcp>=MAX_STALL_ACTIONS and stall_cube>=MAX_STALL_ACTIONS:
            stop_reason="ROBOT_TCP_AND_CUBE_STALLED"
            break
        if bool(np1(terminated)[0]) or bool(np1(truncated)[0]):
            stop_reason="SIMULATOR_EPISODE_END"
            break
        last_tcp,last_cube=curr_tcp,curr_cube
    else:
        stop_reason="PUSH_SUBBUDGET_EXCEEDED"
    return {"phase":traced.phase,"actions":len(traced.trace)-started,
            "stop_reason":stop_reason,
            "goal_error_m":float(np.linalg.norm(_pose(b.obj)[:2]-goal[:2])),
            "cube_xyz_m":_pose(b.obj).tolist(),
            "tcp_xyz_m":_pose(b.agent.tcp_pose).tolist()}


def execute(env):
    traced=PhaseTrace(env)
    b=traced.unwrapped
    if b.control_mode!="pd_ee_delta_pose":
        raise ValueError("Expected proven original M1B3 delta-pose controller")
    start=_pose(b.obj)
    blocker_before=_pose(b.blocker)
    goal=np1(b.goal_pos)[:3].copy()
    phases=[]
    error=None
    clearance_valid=False
    try:
        # Replay only the already physically successful M1B3 blocker clear
        # primitive in a new candidate episode. Not a new tuning attempt.
        traced.phase="clear_blocker"
        clear=ActorPush(traced,b.blocker,np.asarray([BLOCKER_X,PARKING_Y]))
        t0=len(traced.trace)
        for _ in range(MAX_CLEAR_ACTIONS):
            if np.linalg.norm(_pose(b.blocker)[:2]-[BLOCKER_X,PARKING_Y])<.022:
                break
            _,_,terminated,truncated,_=traced.step(clear.act())
            check_push_only(traced,start)
            if clear.failure or bool(np1(terminated)[0]) or bool(np1(truncated)[0]):
                raise RuntimeError(f"PRIOR_CLEAR_EXECUTOR_FAILED: {clear.failure}")
        clear_error=float(np.linalg.norm(_pose(b.blocker)[:2]-[BLOCKER_X,PARKING_Y]))
        phases.append({"phase":"clear_blocker","actions":len(traced.trace)-t0,
                       "goal_error_m":clear_error,
                       "blocker_xyz_m":_pose(b.blocker).tolist()})
        if clear_error>=.022:
            raise RuntimeError("PREVIOUSLY_SUCCESSFUL_CLEAR_NO_LONGER_VALID")
        clearance_valid=True
        if np.linalg.norm(_pose(b.obj)[:2]-start[:2])>.008:
            raise RuntimeError("UNINTENDED_CUBE_MOVE_DURING_BLOCKER_CLEAR")
        # Unlike old M1B3 direct controller, never traverse sideways
        # across task cube at table height after block clearance.
        tcp=_pose(b.agent.tcp_pose)
        behind=np.asarray([float(start[0])-BEHIND_X_OFFSET,float(start[1]),CONTACT_Z])
        phases.append(motion_to(traced,"retreat_vertical",
                                [float(tcp[0]),float(tcp[1]),SAFE_HOVER_Z],start))
        phases.append(motion_to(traced,"hover_rear",
                                [float(behind[0]),float(behind[1]),SAFE_HOVER_Z],start))
        phases.append(motion_to(traced,"descend_rear",behind,start,
                                tolerance=DESCEND_REAR_TOLERANCE_M))
        phases.append(push_to_goal(traced,start,goal))
    except Exception as exc:
        error=f"{type(exc).__name__}: {str(exc)[:450]}"
    final=_pose(b.obj)
    blocker=_pose(b.blocker)
    peak_task=max([start[2]]+[q["task_cube"][2] for q in traced.trace])
    peak_blocker=max([blocker_before[2]]+[q["blocker"][2] for q in traced.trace])
    goal_error=float(np.linalg.norm(final[:2]-goal[:2]))
    push_only=observed_push_only(traced.trace)
    success=bool(clearance_valid and
        goal_error<GOAL_RADIUS and _actual_satisfied(b) and
        push_only["valid"] and error is None and
        abs(final[2]-TASK_HALF)<.009 and
        abs(blocker[2]-BLOCKER_HALF)<.012 and
        peak_task-start[2]<.015 and
        peak_blocker-blocker_before[2]<.015 and
        blocker[1]>PARKING_Y-.035 and
        np.linalg.norm(blocker[:2]-blocker_before[:2])>.075)
    return {"version":VERSION,"mode":CASE,"scene":"AGCOFLM1B3Corridor-v0",
        "seed":SEED,"control_mode":"pd_ee_delta_pose",
        "success":success,"error":error,"steps":len(traced.trace),
        "cube_before":start.tolist(),"cube_final":final.tolist(),
        "goal":goal.tolist(),"blocker_before":blocker_before.tolist(),
        "blocker_final":blocker.tolist(),"goal_error_m":goal_error,
        "clearance_valid":clearance_valid,"push_only":push_only,
        "phases":phases,"trace":traced.trace,
        "original_scene_unchanged":True,
        "source_same_goal_valid":False,"strong_tamp_evaluated":False,
        "learned_method_evaluated":False,"paper_ready":False}


def make_v5_env(official_root,out):
    """Original M1B3 PhysX scene, exact pd_ee_delta_pose configuration.

    New RecordEpisode identifier avoids mislabeling v5 as the old
    2026-11-11 blocked_clear baseline; old geometry remains identical.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.record import RecordEpisode
    from .m1b3_corridor_physics import ENV_ID, register_env

    register_env(official_root)
    raw=gym.make(ENV_ID,obs_mode="state",robot_uids="panda",
                 num_envs=1,control_mode="pd_ee_delta_pose",
                 render_mode="rgb_array",sim_backend="physx_cpu",
                 distract=False,max_episode_steps=MAX_STEPS)
    rec=RecordEpisode(
        raw,output_dir=str(out),trajectory_name=CASE,save_video=True,
        save_on_reset=False,record_reward=False,
        source_type="privileged_diagnostic",
        source_desc="same M1B3 geometry, v6 fixed contact descent acceptance; not learned or TAMP")
    rec.reset(seed=SEED,options={"m1b3_open":False})
    return rec


def run_once(out,official_root):
    contract()
    root=Path(out).resolve()
    case=root/CASE
    if (case/"status.json").exists() or any(case.glob("*.json")) or any(case.glob("*.h5")):
        raise FileExistsError(f"Refuse overwrite of frozen same-scene data: {case}")
    case.mkdir(parents=True,exist_ok=True)
    env=None
    artifact_error=None
    try:
        # Actual proven original M1B3 pd_ee_delta_pose controller,
        # original blocker scene, original actor placement, frozen seed.
        env=make_v5_env(official_root,case)
        data=execute(env)
    except Exception as exc:
        data={"version":VERSION,"mode":CASE,"scene":"AGCOFLM1B3Corridor-v0",
              "seed":SEED,"success":False,
              "steps":0,"trace":[],
              "error":f"{type(exc).__name__}: {str(exc)[:600]}",
              "status":"NO_VALID_PHYSICS"}
    finally:
        if env is not None:
            try:
                env.flush_trajectory()
                env.flush_video()
            except Exception as exc:
                artifact_error=f"{type(exc).__name__}: {str(exc)[:400]}"
            finally:
                env.close()
    trace=data.pop("trace",[])
    (case/"physical_trace.json").write_text(
        json.dumps(trace,indent=2,default=_bounded_summary)+"\n")
    data["artifact_error"]=artifact_error
    (case/"status.json").write_text(
        json.dumps(data,indent=2,default=_bounded_summary)+"\n")
    print("M1_CARTESIAN_PUSHER_RESULT",json.dumps({
        "success":data["success"],"steps":data.get("steps"),
        "error":data.get("error"),"phases":data.get("phases"),
        "goal_error_m":data.get("goal_error_m"),
        "cube_final":data.get("cube_final"),
        "clearance_valid":data.get("clearance_valid"),
        "push_only":data.get("push_only"),
        "artifact_error":artifact_error,
        "status":str(case/"status.json"),
    }),flush=True)
    return data


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--out",default=OUT)
    p.add_argument("--official-root",default="third_party/roboreel_models")
    args=p.parse_args(argv)
    if args.preflight:
        print("M1_CARTESIAN_PUSHER_PREFLIGHT",json.dumps(contract()),flush=True)
        return
    run_once(args.out,args.official_root)


if __name__=="__main__":
    main()
