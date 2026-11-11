#!/usr/bin/env python3
"""Offline comparison of failed current-source contact vs proven earlier M1B2 source.

Reads LOCAL original JSONs only: no ManiSkill imports, no GPU, no new
trajectory or simulator. Avoids swapping grasp approach blindly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def xyz(value, key):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{key}: expected 3-vector")
    if not all(type(v) in (int,float) and math.isfinite(v) for v in value):
        raise ValueError(f"{key}: non-finite position")
    return value


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_case(root, *, kind):
    p=root/"status.json"
    q=root/"physical_trace.json"
    if not p.is_file() or not q.is_file():
        raise FileNotFoundError(f"Existing source evidence missing: {p} or {q}")
    status=json.loads(p.read_text())
    trace=json.loads(q.read_text())
    expected="source_side_grasp" if kind=="new" else "source_pick_to_same_goal"
    if status.get("kind")!=expected:
        raise ValueError(f"{kind}: unexpected saved source kind")
    if kind=="new" and (
        status.get("version")!="m1_current_task_official_executor_swap_v1"
        or status.get("original_geometry") is not True
        or status.get("seed")!=120):
        raise ValueError("new: wrong frozen source/scene version")
    if not isinstance(trace,list) or not trace or status.get("steps")!=len(trace):
        raise ValueError(f"{kind}: physical trace missing or action count mismatch")
    for i,t in enumerate(trace):
        if not isinstance(t,dict) or t.get("t")!=i:
            raise ValueError(f"{kind}: trace step {i} inconsistent")
        for k in ("task_cube","tcp","blocker"):
            xyz(t.get(k),f"{kind}/{i}/{k}")
        if (type(t.get("gripper_action")) not in (int,float)
                or not math.isfinite(t["gripper_action"])):
            raise ValueError(f"{kind}: gripper action missing")
    return status,trace,{"status_sha256":sha256(p),
                         "physical_trace_sha256":sha256(q),
                         "status_file":str(p),"trace_file":str(q)}


def distance(a,b):
    return math.dist(a,b)


def summarise_source(status, trace, kind):
    first=trace[0]
    actual=[i for i,x in enumerate(trace) if bool(x.get("is_grasping_task"))]
    close=[i for i,x in enumerate(trace) if x["gripper_action"] < -0.5]
    opens=[i for i,x in enumerate(trace) if x["gripper_action"] > 0.5]
    # Any close command with no actual grasp: avoid miscalling "controller
    # omitted closure". A gripper command is not measured finger closure.
    close_start=close[0] if close else None
    grasp_start=actual[0] if actual else None
    min_distance=min(distance(r["tcp"],r["task_cube"]) for r in trace)
    samples={
        "initial":0,
        "first_close":close_start,
        "six_steps_after_close":min(close_start+5,len(trace)-1) if close else None,
        "last":len(trace)-1,
        "first_physically_grasping":grasp_start,
    }
    checkpoints={}
    for label,idx in samples.items():
        if idx is None: continue
        row=trace[idx]
        checkpoints[label]={
            "step":idx,"tcp_xyz_m":row["tcp"],
            "cube_xyz_m":row["task_cube"],
            "tcp_to_cube_distance_m":distance(row["tcp"],row["task_cube"]),
            "gripper_command":row["gripper_action"],
            "is_grasping":bool(row.get("is_grasping_task")),
        }
    phases=status.get("planner_phases",[])
    staged=[r for r in phases if r.get("stage")=="physical_grasp_check"]
    if kind=="new" and len(staged)!=1:
        raise ValueError("new: no original physical_grasp_check stage evidence")
    if kind=="new":
        stage=staged[0]
        if stage.get("actual_steps")!=len(trace):
            raise ValueError("new: saved actual contact gate not at final trace step")
        if stage.get("actual_is_grasping") is not False:
            raise ValueError("new: not the reported negative contact case")
    return {
        "kind":kind,
        "actions_observed":len(trace),
        "actual_grasp_steps":len(actual),
        "gripper_close_command_steps":len(close),
        "gripper_open_command_steps":len(opens),
        "first_close_command_index":close_start,
        "first_actual_grasp_index":grasp_start,
        "minimum_tcp_object_center_distance_m":min_distance,
        "cube_net_motion_m":distance(trace[0]["task_cube"],trace[-1]["task_cube"]),
        "cube_peak_height_m":max(r["task_cube"][2] for r in trace),
        "tcp_net_motion_m":distance(trace[0]["tcp"],trace[-1]["tcp"]),
        "checkpoints":checkpoints,
        "physical_contact_stage":staged[0] if staged else None,
        "solver_stages":phases,
    }


def compare(new_root, reference_root):
    new_status,new_trace,new_hash=read_case(new_root,kind="new")
    old_status,old_trace,old_hash=read_case(reference_root,kind="old")
    new=summarise_source(new_status,new_trace,"new")
    old=summarise_source(old_status,old_trace,"old")
    if old_status.get("gate",{}).get("valid") is not True:
        raise ValueError("Prior M1B2 source not proven successful; stop reference comparison")
    if new_status.get("success") is not False:
        raise ValueError("Current swap is not the frozen failed grasp run")
    findings=[]
    if not new["gripper_close_command_steps"]:
        findings.append("CURRENT: No closed-gripper command recorded; diagnose action/control mapping first.")
    elif new["actual_grasp_steps"]==0:
        findings.append("CURRENT: Close-gripper commands WERE sent but no grasp registered; inspect TCP/object pose and finger-wall clearance.")
    if new["minimum_tcp_object_center_distance_m"]>.055:
        findings.append("CURRENT: TCP never within 55mm of cube center; check Panda pose execution/collision interference before changing grasp goals.")
    elif new["actual_grasp_steps"]==0:
        findings.append("CURRENT: TCP approached cube center but grasp absent; suspect finger orientation/contact, static rail clearance, or gripper actuation; no force-level proof.")
    if old["actual_grasp_steps"]>0:
        findings.append("REFERENCE: M1B2 real source demonstrated contact. Compare first-close TCP relative cube against failed scene; same goal not assumed.")
    findings.append("No automatic rerun, world-frame registration, new planner, or object attachment is performed.")
    return {
        "type":"M1_CURRENT_SOURCE_CONTACT_FAILURE_COMPARISON",
        "original_scene_untouched":True,
        "source_and_target_final_goals_not_claimed_identical":True,
        "current_failed_scene_goal_x":new_status.get("goal",[None])[0],
        "old_success_scene_goal_x":old_status.get("goal",[None])[0],
        "new_failed_source":new,
        "previously_successful_reference":old,
        "input_files":{"current":new_hash,"reference":old_hash},
        "findings":findings,
        "physical_simulator_runs":0,
        "learned_methods_trained":0,
    }


def write_once(path, content):
    if path.exists() and path.read_text()!=content:
        raise FileExistsError(f"Existing offline audit differs: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failed",type=Path,default=Path("outputs/m1_current_executor_swap_smoke/source_side_grasp"))
    p.add_argument("--reference",type=Path,default=Path("outputs/m1b2_same_goal_dev/source"))
    p.add_argument("--out",type=Path,default=Path("outputs/m1_current_executor_swap_smoke/contact_comparison.json"))
    a=p.parse_args(argv)
    result=compare(a.failed,a.reference)
    write_once(a.out,json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    compact={
        "type":result["type"],
        "current_grasp_steps":result["new_failed_source"]["actual_grasp_steps"],
        "current_close_command_steps":result["new_failed_source"]["gripper_close_command_steps"],
        "current_closest_tcp_cube_distance_m":result["new_failed_source"]["minimum_tcp_object_center_distance_m"],
        "current_first_close":result["new_failed_source"]["checkpoints"].get("first_close"),
        "current_six_steps_after_close":result["new_failed_source"]["checkpoints"].get("six_steps_after_close"),
        "old_reference_grasp_steps":result["previously_successful_reference"]["actual_grasp_steps"],
        "old_reference_first_close":result["previously_successful_reference"]["checkpoints"].get("first_close"),
        "findings":result["findings"],
        "out":str(a.out),
    }
    print("M1_SOURCE_CONTACT_ROOT_DIAG",json.dumps(compact,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
