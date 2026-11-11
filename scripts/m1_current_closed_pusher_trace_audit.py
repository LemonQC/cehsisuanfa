#!/usr/bin/env python3
"""Read-only M1 original-scene target-push failure audit. NO PhysX, GPU, model or rerun.

A completed previous *closed-gripper* MPLib source recorded 450 physical
actions, with 6 close, 82 successful blocker-clear, and an unfinished
post-clear cube movement attempt. The original status reports ONLY
completed phases; reconstruct the unfinished suffix from saved per-step
physical_trace.json rather than assuming nothing happened in that suffix.

Read-only reference: existing original M1B3 open_direct, which succeeded.
Does not identify collision forces or turn a single scripted failure into
necessity of an editor/TAMP/learning method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path("outputs/m1_current_closed_pusher_v2")
CASE="target_closed_mplib"
REF=Path("outputs/m1b3_corridor_dev/open_direct")
PRE_CLOSURE_STEPS=6
SCENE="AGCOFLM1B3Corridor-v0"
VERSION="m1_current_target_mplib_closed_pusher_v2"


def xyz(v,description):
    if not isinstance(v,list) or len(v)!=3 or any(
        type(q) not in (int,float) or not math.isfinite(q) for q in v):
        raise ValueError(f"{description}: invalid measured physical XYZ")
    return [float(q) for q in v]


def dxy(a,b):
    return math.hypot(a[0]-b[0],a[1]-b[1])


def checksum(path):
    h=hashlib.sha256()
    with path.open("rb") as fd:
        for data in iter(lambda: fd.read(1<<20),b""):
            h.update(data)
    return h.hexdigest()


def load_pair(base,identity):
    a=base/"status.json"
    b=base/"physical_trace.json"
    if not a.is_file() or not b.is_file():
        raise FileNotFoundError(f"{identity} raw trace/status not found: {base}")
    status=json.loads(a.read_text())
    trace=json.loads(b.read_text())
    if not isinstance(trace,list) or not trace:
        raise ValueError(f"{identity} has no recorded PhysX trajectory")
    if (type(status.get("steps")) is not int or
            status["steps"]!=len(trace)):
        raise ValueError(f"{identity} incomplete or mismatched actual step count")
    for k,r in enumerate(trace):
        if not isinstance(r,dict) or r.get("t")!=k:
            raise ValueError(f"{identity} bad physical step index {k}")
        for key in ("task_cube","blocker","tcp"):
            xyz(r.get(key),f"{identity} step {k} {key}")
        if type(r.get("gripper_action")) not in (float,int):
            raise ValueError(f"{identity} missing gripper command at {k}")
    return status,trace,{
        "status_json":str(a),"status_sha256":checksum(a),
        "trace_json":str(b),"trace_sha256":checksum(b),
    }


def stage(trace,goal,pre_cube=None,pre_tcp=None,pre_blocker=None):
    """Measured motion, relative TCP, progress and end-of-episode stall."""
    if not trace:
        return {"steps":0,"contains_actual_motion":False}
    first=trace[0]
    last=trace[-1]
    cube=[pre_cube or first["task_cube"]]+[r["task_cube"] for r in trace]
    tcp=[pre_tcp or first["tcp"]]+[r["tcp"] for r in trace]
    blocker=[pre_blocker or first["blocker"]]+[r["blocker"] for r in trace]
    xy_path=lambda pts:sum(dxy(a,b) for a,b in zip(pts,pts[1:]))
    cube_goal=[dxy(q,goal) for q in cube]
    progress=[cube_goal[0]-e for e in cube_goal]
    min_tcp_cube=min(math.dist(r["tcp"],r["task_cube"]) for r in trace)
    tail=trace[-15:]
    peak_gripper=max(float(r["finger_joint_qpos_m"][0]) for r in trace
                     if r.get("finger_joint_qpos_m") is not None) if any(
                     r.get("finger_joint_qpos_m") is not None for r in trace) else None
    return {
        "steps":len(trace),
        "first_cube_xyz_m":cube[0],
        "last_cube_xyz_m":cube[-1],
        "cube_net_xy_m":dxy(cube[0],cube[-1]),
        "cube_total_xy_m":xy_path(cube),
        "cube_last_15_steps_xy_m":xy_path([r["task_cube"] for r in tail]),
        "cube_start_goal_error_m":cube_goal[0],
        "cube_closest_goal_error_m":min(cube_goal),
        "cube_final_goal_error_m":cube_goal[-1],
        "maximum_approach_to_goal_m":max(progress),
        "cube_progress_goal_m":cube_goal[0]-cube_goal[-1],
        "cube_x_change_m":cube[-1][0]-cube[0][0],
        "cube_y_change_m":cube[-1][1]-cube[0][1],
        "cube_max_z_m":max(r["task_cube"][2] for r in trace),
        "cube_ever_at_throat_front_x_m":any(
            abs(r["task_cube"][0]-.126)<.006 for r in trace),
        "cube_outside_5mm_throat_center_band_steps":sum(
            abs(r["task_cube"][1])>.005 and r["task_cube"][0]>.095
            for r in trace),
        "tcp_last_xyz_m":tcp[-1],
        "tcp_total_xy_m":xy_path(tcp),
        "tcp_nearest_cube_center_m":min_tcp_cube,
        "tcp_closest_cube_xyz_m":min(
            trace,key=lambda r:math.dist(r["tcp"],r["task_cube"]))["tcp"],
        "blocker_final_xyz_m":blocker[-1],
        "blocker_total_xy_m":xy_path(blocker),
        "any_target_grasped":any(r["is_grasping_task"] for r in trace),
        "any_blocker_grasped":any(
            bool(r.get("is_grasping_blocker")) for r in trace),
        "closed_gripper_action_steps":sum(
            r["gripper_action"]<0 for r in trace),
        "first_finger_joint_qpos_m":first.get("finger_joint_qpos_m"),
        "last_finger_joint_qpos_m":last.get("finger_joint_qpos_m"),
        "max_first_finger_joint_qpos_m":peak_gripper,
    }


def checkpoint(trace,step,goal):
    if not 0<=step<len(trace):
        raise ValueError(f"Invalid checkpoint {step}")
    r=trace[step]
    return {
        "t":step,
        "cube_xyz_m":r["task_cube"],"tcp_xyz_m":r["tcp"],
        "blocker_xyz_m":r["blocker"],
        "cube_goal_error_xy_m":dxy(r["task_cube"],goal),
        "tcp_cube_center_distance_m":math.dist(r["tcp"],r["task_cube"]),
        "gripper_command":r["gripper_action"],
        "finger_joint_qpos_m":r.get("finger_joint_qpos_m"),
        "grasped_cube":r.get("is_grasping_task"),
        "grasped_blocker":r.get("is_grasping_blocker"),
    }


def analyse(case,reference):
    status,trace,case_provenance=load_pair(case,"closed_mplib")
    old,ref_trace,ref_provenance=load_pair(reference,"open_direct")
    if (status.get("kind")!=CASE or
            status.get("version")!=VERSION or
            status.get("scene")!=SCENE or
            status.get("seed")!=120 or
            status.get("success") is not False or
            status.get("steps")!=450 or
            "STEP_BUDGET_EXCEEDED" not in str(status.get("error"))):
        raise ValueError("Not the reported 450-step actual closed-pusher budget failure")
    if (old.get("mode")!="open_direct" or
            old.get("scene")!=SCENE or
            old.get("seed")!=120 or
            old.get("gate",{}).get("valid") is not True):
        raise ValueError("Original open_direct physical success reference invalid")
    phases=status.get("phases",[])
    if (not isinstance(phases,list) or len(phases)!=1 or
            phases[0].get("name")!="clear_blocker" or
            type(phases[0].get("actual_steps")) is not int or
            phases[0]["actual_steps"]<=0 or
            phases[0].get("actual_error_m",float("inf"))>=.022):
        raise ValueError("No measured completed real clear-blocker phase")
    goal=xyz(status.get("goal"),"new same-goal")
    if dxy(goal,xyz(old.get("goal"),"old goal"))>1e-4:
        raise ValueError("Open-direct reference has a DIFFERENT final goal")
    init=xyz(status.get("state_before"),"new cube start")
    if dxy(init,xyz(old.get("state_before"),"old start"))>1e-4:
        raise ValueError("Open-direct and blocked runs use different cube starts")
    boundary=PRE_CLOSURE_STEPS+phases[0]["actual_steps"]
    if boundary>=len(trace):
        raise ValueError("No post-clearance actual actions in physical trace")
    closed=trace[:PRE_CLOSURE_STEPS]
    clear=trace[PRE_CLOSURE_STEPS:boundary]
    post=trace[boundary:]
    if not all(r["gripper_action"]<0 for r in trace):
        raise ValueError("Closed-tool experimental factor changed mid-run")
    if any(bool(r.get("is_grasping_task")) or
           bool(r.get("is_grasping_blocker")) for r in trace):
        raise ValueError("Object physically grasped in claimed push-only trial")

    prior=trace[boundary-1]
    post_metric=stage(post,goal,pre_cube=prior["task_cube"],
                      pre_tcp=prior["tcp"],pre_blocker=prior["blocker"])
    ref_metric=stage(ref_trace,goal,pre_cube=old["state_before"],
                     pre_tcp=ref_trace[0]["tcp"],
                     pre_blocker=old["blocker_before"])
    guide_steps=sorted(set(
        x for x in (0,PRE_CLOSURE_STEPS-1,PRE_CLOSURE_STEPS,
                    boundary-1,boundary,min(boundary+25,449),
                    min(boundary+100,449),min(boundary+200,449),
                    449) if x<len(trace)))
    important={
        "clear_completed_step_index":boundary-1,
        "first_post_clear_action_index":boundary,
        "post_clear_min_cube_goal_error_step":min(
            range(boundary,len(trace)),
            key=lambda k:dxy(trace[k]["task_cube"],goal)),
        "post_clear_min_tcp_cube_distance_step":min(
            range(boundary,len(trace)),
            key=lambda k:math.dist(trace[k]["tcp"],trace[k]["task_cube"])),
    }
    report={
        "type":"M1_CLOSED_MPLIB_PUSH_450_STEP_OFFLINE_AUDIT",
        "source_scene":SCENE,"same_start_goal_as_open_direct":True,
        "no_new_physics":True,"no_new_controller":True,
        "raw_450_physics_steps":len(trace),
        "initial_close_steps":PRE_CLOSURE_STEPS,
        "completed_clear_phase_steps":phases[0]["actual_steps"],
        "uncompleted_post_clear_attempt_steps":len(post),
        "observed_clearance_error_m":phases[0]["actual_error_m"],
        "full_result_error":status["error"],
        "segments":{
            "close_gripper":stage(closed,goal,pre_cube=init,
                                  pre_blocker=status["blocker_before"]),
            "clear_blocker":stage(clear,goal,
                                  pre_cube=closed[-1]["task_cube"],
                                  pre_tcp=closed[-1]["tcp"],
                                  pre_blocker=closed[-1]["blocker"]),
            "after_clearance_attempt":post_metric,
            "old_open_direct_success_reference":ref_metric,
        },
        "actual_timeline_checkpoints":[checkpoint(trace,i,goal)
                                       for i in guide_steps],
        "critical_timeline_indices":important,
        "input_sha256":{"closed_mplib":case_provenance,
                        "original_open_direct":ref_provenance},
        "limits":[
            "A 450-step planner budget exception only means the unfinished target stage was interrupted; do not infer no cube motion.",
            "Distance to TCP center does not establish object contact forces or finger link collisions.",
            "Open-direct was a different controller and pre-parked-blocker state; it is a reference for physical traversability, not a causal policy comparison.",
            "No original source grasp was repaired. No fair TAMP, learned transfer or source-video value was tested."
        ],
    }
    if post_metric["cube_progress_goal_m"]<-.02:
        report["observed_failure_signature"]="CUBE_MOVED_FARTHER_FROM_GOAL_AFTER_CLEARANCE"
    elif (post_metric["cube_ever_at_throat_front_x_m"]
          and post_metric["cube_last_15_steps_xy_m"]<.002
          and post_metric["cube_outside_5mm_throat_center_band_steps"]>0):
        report["observed_failure_signature"]="THROAT_FRONT_OFF_CENTER_STALL_CANDIDATE"
    elif (post_metric["cube_total_xy_m"]<.005):
        report["observed_failure_signature"]="NO_MEASURABLE_CUBE_MOTION_AFTER_CLEARANCE"
    else:
        report["observed_failure_signature"]="INCONCLUSIVE_TRACE_REQUIRES_TCP_OBJECT_POSE_REVIEW"
    return report


def write_once(path,report):
    render=json.dumps(report,indent=2,ensure_ascii=False)+"\n"
    if path.is_file():
        if path.read_text()!=render:
            raise FileExistsError(f"Existing immutable diagnostic differs: {path}")
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(render)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",type=Path,default=ROOT/CASE)
    p.add_argument("--reference",type=Path,default=REF)
    p.add_argument("--out",type=Path,
        default=ROOT/"closed_pusher_450_step_offline_audit.json")
    args=p.parse_args(argv)
    result=analyse(args.case,args.reference)
    write_once(args.out,result)
    metric=result["segments"]["after_clearance_attempt"]
    print("M1_CLOSED_PUSHER_450_STEP_TRACE_DIAG",json.dumps({
        "no_new_physics":True,
        "source_physics_unchanged":True,
        "closed_tool_clearance_success":True,
        "clearance_steps":result["completed_clear_phase_steps"],
        "after_clearance_actual_steps":result["uncompleted_post_clear_attempt_steps"],
        "after_clearance_cube_start_xyz_m":metric["first_cube_xyz_m"],
        "after_clearance_cube_final_xyz_m":metric["last_cube_xyz_m"],
        "after_clearance_cube_goal_progress_m":metric["cube_progress_goal_m"],
        "after_clearance_cube_min_goal_error_m":metric["cube_closest_goal_error_m"],
        "after_clearance_cube_xy_total_motion_m":metric["cube_total_xy_m"],
        "after_clearance_tcp_min_distance_m":metric["tcp_nearest_cube_center_m"],
        "after_clearance_last15_cube_motion_m":metric["cube_last_15_steps_xy_m"],
        "observed_failure_signature":result["observed_failure_signature"],
        "report":str(args.out),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
