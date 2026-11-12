#!/usr/bin/env python3
"""Inspect frozen 65-action M1B3 source failure, NO PhysX/replay/GPU.

Separate measured hand/finger versus STATIC-WALL force from MPLib's
failure to plan final grasp_pose; these are NOT synonymous.  The old
physical trace lacks whole-arm contact forces and command qpos, so
zero hand/finger wall force cannot prove robot collision-free.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path("outputs/m1_source_geom_aligned_topdown_v1/source_aligned_topdown")
OUT=Path("outputs/m1_source_geom_aligned_topdown_v1/real65_step_contact_audit.json")


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(v,w):
    return abs(v-w)<.002


def measured(r,key,size):
    v=r.get(key)
    if not isinstance(v,list) or len(v)!=size or any(
            type(x) not in (int,float) or not math.isfinite(x) for x in v):
        raise ValueError(f"Malformed real {key} at action {r.get('t')}")
    return [float(x) for x in v]


def distance(a,b):
    return math.dist(a,b)


def local_y_in_world(q):
    """Rotate local TCP +Y (Panda grasp closing axis) by measured wxyz."""
    w,x,y,z=measured({"tcp_quaternion_wxyz":q,"t":"q"},
                      "tcp_quaternion_wxyz",4)
    if abs(w*w+x*x+y*y+z*z-1)>.04:
        raise ValueError("Measured TCP quaternion not unit length")
    return [2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)]


def analyse(base):
    p=base/"status.json"
    t=base/"physical_trace.json"
    if not p.is_file() or not t.is_file():
        raise FileNotFoundError(f"Original source v1 physical JSON not found: {base}")
    st,rows=load(p),load(t)
    if (st.get("version")!="m1_source_geom_aligned_topdown_v1"
            or st.get("kind")!="source_aligned_topdown"
            or st.get("scene")!="AGCOFLM1B3Corridor-v0"
            or st.get("seed")!=120 or st.get("success") is not False
            or st.get("actual_steps")!=65 or not isinstance(rows,list)
            or len(rows)!=65 or
            "SOURCE_UPSTREAM_grasp_pose_PLANNING_FAILED"
                not in str(st.get("error"))):
        raise ValueError("Not the operator-reported genuine 65-step v1 source failure")
    for i,r in enumerate(rows):
        if r.get("t")!=i:
            raise ValueError(f"Noncontiguous actual physical action {i}")
        for k in ("tcp","task_cube"):
            measured(r,k,3)
        measured(r,"finger_joint_qpos_m",2)
        measured(r,"tcp_quaternion_wxyz",4)
        if r.get("execution_phase") not in ("set_partial_open","pregrasp","grasp_pose"):
            raise ValueError(f"Unexpected source execution phase at action {i}")
        if type(r.get("is_grasping_task")) is not bool:
            raise ValueError("No actual Panda grasp state in old trace")
    phases=st.get("phases",[])
    pre=next((x for x in phases if x.get("phase")=="pregrasp"),None)
    grasp=next((x for x in phases if x.get("phase")=="grasp_pose"),None)
    if (not pre or not grasp or pre.get("actual_steps")!=59
            or pre.get("planner_result") is not True
            or grasp.get("actual_steps")!=0
            or grasp.get("planner_result") is not False):
        raise ValueError("Saved planning phase facts are not the observed failure")
    if any(r["is_grasping_task"] for r in rows):
        raise ValueError("Unexpected real grasp contradicts v1 failed contact gate")
    force_rows=[]
    missing=0
    for r in rows:
        f=r.get("tool_static_wall_contact_force_max_n")
        if f is None:
            missing+=1
        elif type(f) not in (int,float) or not math.isfinite(f) or f<0:
            raise ValueError("Invalid physical contact force")
        else:
            force_rows.append((r["t"],float(f),r["execution_phase"]))
    last=rows[-1]
    cube=measured(last,"task_cube",3)
    tcp=measured(last,"tcp",3)
    pretarget=[cube[0],cube[1],cube[2]+.060]
    actual_force=max((f for _,f,_ in force_rows),default=None)
    significant=[{"t":i,"force_n":f,"phase":phase}
                 for i,f,phase in force_rows if f>.5]
    closing=local_y_in_world(last["tcp_quaternion_wxyz"])
    result={
       "type":"M1_SOURCE_V1_65_REAL_ACTIONS_CONTACT_VS_PLANNER_AUDIT",
       "original_scene":"AGCOFLM1B3Corridor-v0",
       "source_goal_x_m":.210,"actual_physx_new_runs":0,
       "physical_actions":65,"pregrasp_actual_actions":59,
       "grasp_pose_executed_actions":0,
       "grasp_pose_planner_success":False,
       "real_grasp_steps":0,
       "pregrasp_actual_tcp_xyz_m":tcp,
       "pregrasp_target_xyz_approx_m":pretarget,
       "pregrasp_tcp_target_position_error_m":distance(tcp,pretarget),
       "tcp_cube_distance_m":distance(tcp,cube),
       "tcp_measured_world_closing_axis":closing,
       "tcp_world_x_alignment_abs":abs(closing[0]),
       "finger_final_qpos_m":last["finger_joint_qpos_m"],
       "tool_static_wall_contact_force_sample_count":len(force_rows),
       "tool_static_wall_contact_force_missing_count":missing,
       "tool_static_wall_force_max_n":actual_force,
       "tool_static_wall_force_gt_0p5N_count":len(significant),
       "first_significant_tool_wall_contact":significant[0] if significant else None,
       "last_significant_tool_wall_contact":significant[-1] if significant else None,
       "status_sha256":digest(p),"physical_trace_sha256":digest(t),
       "failure_classification":(
          "OBSERVED_HAND_OR_FINGER_STATIC_WALL_CONTACT"
          if significant else
          "HAND_FINGER_WALL_CONTACT_NOT_OBSERVED_IN_RECORDED_FORCE" if force_rows
          else "HAND_FINGER_WALL_CONTACT_FORCES_UNAVAILABLE"),
       "limits":[
          "MPLib screw plan failed BEFORE grasp_pose actions; no physical penetration attempt at its target.",
          "Only hand and both fingers vs STATIC walls were logged; whole-arm links, table and blocker contact forces were not logged.",
          "Zero measured tool force does NOT prove collision-free access or IK feasibility.",
          "No source grasp, lift, place, same-goal pair, learned transfer or equal-info TAMP evaluated."
       ],
    }
    return result


def save_once(out,r):
    data=json.dumps(r,indent=2,ensure_ascii=False)+"\n"
    if out.is_file():
        if out.read_text()!=data:
            raise FileExistsError(f"Existing immutable audit differs: {out}")
    else:
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(data)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",type=Path,default=ROOT)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args(argv)
    r=analyse(a.case)
    save_once(a.out,r)
    print("M1_SOURCE_65_STEP_FORCE_VS_PLANNER_DIAG",json.dumps({
       k:r[k] for k in (
          "actual_physx_new_runs",
          "pregrasp_actual_tcp_xyz_m",
          "pregrasp_tcp_target_position_error_m",
          "tcp_cube_distance_m",
          "tcp_world_x_alignment_abs",
          "tool_static_wall_contact_force_sample_count",
          "tool_static_wall_contact_force_missing_count",
          "tool_static_wall_force_max_n",
          "tool_static_wall_force_gt_0p5N_count",
          "first_significant_tool_wall_contact",
          "failure_classification"
       )} | {"audit":str(a.out)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
