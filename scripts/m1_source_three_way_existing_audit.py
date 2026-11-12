#!/usr/bin/env python3
"""CPU-only three-way audit of immutable CURRENT M1 source grasp evidence.

Current original scene has TWO failed physical source Pick attempts:
- original M1B3 blocked_source top-down grasp/pick: output from 2026-11 pilot
- M1B3 side-grasp source swap: 106 real steps, 0 grasp, TCP>=102mm away
Reference prior M1B2 source: 170 real steps, 84 grasp; DIFFERENT goal x=.150.

Do NOT auto-retry PhysX or treat prior M1B2 success as current same-goal.
Print physically measured first close / closest TCP / cube lift and source
phase plan metadata. This selects next SOURCE repair based on actual evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SCENE="AGCOFLM1B3Corridor-v0"
SEED=120
DEFAULT_B3=Path("outputs/m1b3_corridor_dev/blocked_source")
DEFAULT_SIDE=Path("outputs/m1_current_executor_swap_smoke/source_side_grasp")
DEFAULT_OLD=Path("outputs/m1b2_same_goal_dev/source")
DEFAULT_OUT=Path("outputs/m1_current_executor_swap_smoke/source_three_way_existing_audit.json")


def pos3(v,key):
    if (not isinstance(v,list) or len(v)!=3 or
            any(type(x) not in (int,float) or not math.isfinite(x) for x in v)):
        raise ValueError(f"{key}: malformed position")
    return [float(x) for x in v]


def read(root,label):
    p=root/"status.json"
    q=root/"physical_trace.json"
    if not p.is_file() or not q.is_file():
        raise FileNotFoundError(f"{label}: missing prior PHYSICS records at {root}")
    st=json.loads(p.read_text())
    trace=json.loads(q.read_text())
    if not isinstance(trace,list) or not trace or st.get("steps")!=len(trace):
        raise ValueError(f"{label}: missing real steps / lengths inconsistent")
    for i,r in enumerate(trace):
        if not isinstance(r,dict) or r.get("t")!=i:
            raise ValueError(f"{label}: invalid actual step {i}")
        for k in ("tcp","task_cube","blocker"):
            pos3(r.get(k),f"{label} step {i} {k}")
        if (type(r.get("gripper_action")) not in (float,int)
                or type(r.get("is_grasping_task")) is not bool):
            raise ValueError(f"{label}: no physical grasp / gripper action evidence at {i}")
    return st,trace,{
        "status_path":str(p),"trace_path":str(q),
        "status_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
        "trace_sha256":hashlib.sha256(q.read_bytes()).hexdigest(),
    }


def checkpoint(trace,index):
    r=trace[index]
    return {
       "step_index":index,"tcp_xyz_m":r["tcp"],
       "cube_xyz_m":r["task_cube"],
       "tcp_cube_center_distance_m":math.dist(r["tcp"],r["task_cube"]),
       "gripper_action":r["gripper_action"],
       "physically_grasping":r["is_grasping_task"],
    }


def metrics(st,trace,label):
    distances=[math.dist(r["tcp"],r["task_cube"]) for r in trace]
    nearest=min(range(len(trace)),key=lambda i:distances[i])
    closed=[i for i,r in enumerate(trace) if r["gripper_action"]<-.5]
    actual=[i for i,r in enumerate(trace) if r["is_grasping_task"]]
    init=pos3(trace[0]["task_cube"],f"{label} initial cube")
    full={
      "label":label,
      "physical_actions":len(trace),
      "close_command_step_count":len(closed),
      "actual_grasping_step_count":len(actual),
      "first_closed_step":checkpoint(trace,closed[0]) if closed else None,
      "sixth_closed_step":checkpoint(trace,closed[min(5,len(closed)-1)]) if closed else None,
      "closest_tcp_step":checkpoint(trace,nearest),
      "minimum_tcp_cube_center_distance_m":distances[nearest],
      "lowest_tcp_z_m":min(r["tcp"][2] for r in trace),
      "max_cube_height_m":max(r["task_cube"][2] for r in trace),
      "initial_cube_height_m":init[2],
      "maximum_cube_lift_m":max(r["task_cube"][2] for r in trace)-init[2],
      "cube_final_xyz_m":trace[-1]["task_cube"],
      "first_true_grasp_step":checkpoint(trace,actual[0]) if actual else None,
      "preclose_min_tcp_cube_distance_m":min(distances[:closed[0]+1]) if closed else min(distances),
      "planner_phases_original":st.get("planner_phases"),
      "recorded_error":st.get("error"),
    }
    return full


def analyse(b3_root,side_root,old_root):
    b3,b3_trace,b3_sha=read(b3_root,"original_b3")
    side,side_trace,side_sha=read(side_root,"new_b3_side")
    old,old_trace,old_sha=read(old_root,"b2_reference")
    if not (b3.get("mode")=="blocked_source" and b3.get("scene")==SCENE
            and b3.get("seed")==SEED and b3.get("kind")=="source_pick_to_same_goal"
            and b3.get("gate",{}).get("valid") is False):
        raise ValueError("Original B3 blocked_source is not the genuine failed same-scene source")
    if not (side.get("kind")=="source_side_grasp"
            and side.get("version")=="m1_current_task_official_executor_swap_v1"
            and side.get("original_geometry") is True
            and side.get("seed")==SEED and side.get("success") is False):
        raise ValueError("Side source does not match genuine failed same-scene candidate")
    if not (old.get("kind")=="source_pick_to_same_goal"
            and old.get("gate",{}).get("valid") is True):
        raise ValueError("M1B2 reference was not a physically valid source")
    first_b3=pos3(b3.get("state_before"),"b3 initial")
    first_side=pos3(side.get("initial_cube"),"side initial")
    first_old=pos3(old.get("state_before"),"old initial")
    b3_goal=pos3(b3.get("goal"),"b3 goal")
    side_goal=pos3(side.get("goal"),"side goal")
    old_goal=pos3(old.get("goal"),"old goal")
    if math.dist(first_b3,first_side)>.001 or math.dist(first_b3,first_old)>.001:
        raise ValueError("Initial cube differs; cannot compare real contact")
    if math.dist(b3_goal,side_goal)>.001 or abs(b3_goal[0]-.21)>.001:
        raise ValueError("Two CURRENT source attempts do not share real target goal")
    if abs(old_goal[0]-.15)>.001:
        raise ValueError("Old M1B2 SOURCE reference goal is NOT the known .150m")
    cases={
        "b3_original_topdown":metrics(b3,b3_trace,"b3_original_topdown"),
        "b3_side_grasp":metrics(side,side_trace,"b3_side_grasp"),
        "b2_success_reference":metrics(old,old_trace,"b2_success_reference"),
    }
    side_case=cases["b3_side_grasp"]
    top=cases["b3_original_topdown"]
    reference=cases["b2_success_reference"]
    if side_case["actual_grasping_step_count"]!=0:
        raise ValueError("Side source no longer matches frozen zero-grasp failure")
    if reference["actual_grasping_step_count"]<=0:
        raise ValueError("Old B2 not physically gripping in saved trace")
    flags=[]
    if side_case["preclose_min_tcp_cube_distance_m"]>.055:
        flags.append("M1B3 SIDE: physically never approached within 55mm of cube before closure; side-gripper-close gain cannot solve non-contact.")
    if top["close_command_step_count"]==0:
        flags.append("M1B3 TOPDOWN: old solver never commanded finger closure; failure is pre-grasp planning/control, not grip-force failure.")
    elif top["first_closed_step"]["tcp_cube_center_distance_m"]>.055:
        flags.append("M1B3 TOPDOWN: original grasp closure happened >55mm from cube center; both approaches suffer real approach tracking.")
    elif top["actual_grasping_step_count"]==0:
        flags.append("M1B3 TOPDOWN: TCP reached cube vicinity before closure but Panda never registered grip; physical finger/tool orientation or interference needs inspection.")
    else:
        flags.append("M1B3 TOPDOWN: old physical trace HAS real grasp steps. Recover observed object lift/carry stage before retrying any approach.")
    if top["maximum_cube_lift_m"]<.01:
        flags.append("M1B3 TOPDOWN: no physical cube lift observed; cannot call this a successful source.")
    flags.append("Reference B2 x=.150 different goal; its measured first closure/grasp is a CONTACT comparison only, not a matched-source demonstration.")
    flags.append("No new simulation, pretrained model, source try, geometry change, or training is performed.")
    return {
        "type":"M1_SOURCE_ALL_PRIOR_GRASP_EVIDENCE_OFFLINE",
        "scene":SCENE,"seed":SEED,"same_source_target_goal_x_m":.21,
        "old_reference_different_goal_x_m":old_goal[0],
        "actual_physx_new_runs":0,
        "cases":cases,"findings":flags,
        "inputs_sha256":{
            "original_b3":b3_sha,"side_current":side_sha,
            "old_b2_success_reference":old_sha
        },
        "source_pair_success_claimed":False,
        "learning_or_tamp_evaluated":False,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-b3",type=Path,default=DEFAULT_B3)
    p.add_argument("--side",type=Path,default=DEFAULT_SIDE)
    p.add_argument("--reference",type=Path,default=DEFAULT_OLD)
    p.add_argument("--out",type=Path,default=DEFAULT_OUT)
    a=p.parse_args(argv)
    result=analyse(a.original_b3,a.side,a.reference)
    payload=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    if a.out.exists() and a.out.read_text()!=payload:
        raise FileExistsError(f"Immutable source read-only report differs: {a.out}")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if not a.out.exists():
        a.out.write_text(payload)
    def pick(k):
        c=result["cases"][k]
        return {
          "actions":c["physical_actions"],
          "closed_steps":c["close_command_step_count"],
          "actual_grasp_steps":c["actual_grasping_step_count"],
          "actual_lift_m":c["maximum_cube_lift_m"],
          "closest_tcp_cube_m":c["minimum_tcp_cube_center_distance_m"],
          "first_close":c["first_closed_step"],
          "first_true_grasp":c["first_true_grasp_step"],
          "error":c["recorded_error"],
          "planner_phases":c["planner_phases_original"],
        }
    print("M1_SOURCE_THREE_WAY_REAL_TRACE_GATE",json.dumps({
      "zero_new_physx":True,
      "same_goal_current_source_x_m":.21,
      "original_b3_topdown":pick("b3_original_topdown"),
      "current_b3_side":pick("b3_side_grasp"),
      "previous_b2_different_goal":pick("b2_success_reference"),
      "findings":result["findings"],
      "audit_file":str(a.out),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
