"""M1 original corridor: one versioned, measured-direction-safe TARGET handoff.

Use the established v2 closed-finger Panda/MPLib blocker CLEARANCE unchanged.
Change ONLY the target-cube approach (lift, hover over cube, descend from -X)
and guarded +X push. No new scene, goal, low-level gains, dataset or Method.

A successful planner return is NOT a successful contact action. Every
simulator action is monitored for wrong-sign cube motion or lateral jam.
This privileged-state physical executor is NOT a learned policy or TAMP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .m1b2_same_goal_effect_physics import _pose, _bounded_summary
from .m1b3_corridor_physics import (
    BLOCKER_HALF, BLOCKER_X, PARKING_Y, GOAL_RADIUS, TASK_HALF,
    MAX_STEPS, SEED, THROAT_X, _actual_satisfied, geometry_preflight,
)
from .m1_current_closed_pusher_ablation import (
    ContactTracer, make_joint_push_env, observed_push_only,
)
from .m1_current_executor_swap import get_planner
from .roboreel_bridge import np1

VERSION="m1_current_target_safe_approach_v3"
CASE="target_safe_approach"
OUT="outputs/m1_current_target_safe_approach_v3"
HOVER_Z=.160
BACK_OFFSET_M=.065
UNEXPECTED_REVERSE_M=.006
LATERAL_TOL_M=.010
THROAT_CENTER_BAND_M=.006
PUSH_SEGMENT_TCP_X=(.110,.180,.260)


def safe_approach_waypoints(cube, tcp, q, final_goal_x=.210):
    """Measured, separate hover/rear/descend targets, no low-level gains."""
    cube=np.asarray(cube,dtype=float)
    tcp=np.asarray(tcp,dtype=float)
    q=np.asarray(q,dtype=float)
    if (cube.shape!=(3,) or tcp.shape!=(3,) or q.shape!=(4,)
            or not np.isfinite(cube).all() or not np.isfinite(tcp).all()
            or not np.isfinite(q).all()):
        raise ValueError("Actual cube/TCP/quaternion has invalid shape or numeric state")
    if abs(cube[0])>.010 or abs(cube[1])>.010:
        raise ValueError("Existing cube moved before rear-approach handoff")
    if not np.isclose(final_goal_x,.210,atol=1e-4):
        raise ValueError("Current M1 original final goal must remain x=.210")
    back_x=float(cube[0]-BACK_OFFSET_M)
    return [
      ("retreat_vertical", np.array([tcp[0],tcp[1],HOVER_Z])),
      ("overhead_to_back",np.array([back_x,float(cube[1]),HOVER_Z])),
      ("descend_behind_cube",np.array([back_x,float(cube[1]),float(cube[2])])),
    ]


def phase_violation(phase, cube, start, max_positive_x):
    """Single-step observable hard safety gates for target object."""
    cube=np.asarray(cube,dtype=float)
    start=np.asarray(start,dtype=float)
    if (cube.shape!=(3,) or start.shape!=(3,)
            or not np.isfinite(cube).all() or not np.isfinite(start).all()):
        return "NONFINITE_PHYSICAL_CUBE_STATE"
    if cube[2]-start[2]>.015:
        return "TASK_CUBE_LIFTED_DISALLOWED_PUSH_ONLY"
    if phase in ("retreat_vertical","overhead_to_back","descend_behind_cube"):
        if cube[0]<start[0]-UNEXPECTED_REVERSE_M:
            return "APPROACH_MOVED_CUBE_BACKWARDS_BEFORE_PUSH"
        if np.linalg.norm(cube[:2]-start[:2])>.010:
            return "APPROACH_DISTURBED_TASK_BEFORE_CONTACT"
    if phase.startswith("push_"):
        if cube[0]<start[0]-UNEXPECTED_REVERSE_M:
            return "TASK_PUSH_MOVED_REVERSE_X"
        if cube[0]<max_positive_x-.009:
            return "TASK_PUSH_LOST_FORWARD_PROGRESS"
        if abs(cube[1])>LATERAL_TOL_M:
            return "TASK_PUSH_DEVIATED_LATERALLY"
        if cube[0]>.095 and abs(cube[1])>THROAT_CENTER_BAND_M:
            return "TASK_OFF_CENTER_BEFORE_NARROW_THROAT"
    return None


def direction_contract():
    g=geometry_preflight()
    if not g["pass"]:
        raise ValueError("Old frozen corridor C-space check failed")
    previous_precontact_x=0.-.05
    actual_old_cube_x=-.050044044852256775
    return {
        "type":"M1_ORIGINAL_SCENE_TARGET_REAR_APPROACH_PREFLIGHT",
        "scene":"AGCOFLM1B3Corridor-v0",
        "seed":SEED,
        "goal_x_m":.210,
        "old_failed_planned_precontact_x_m":previous_precontact_x,
        "old_observed_cube_final_x_m":actual_old_cube_x,
        "precontact_matches_old_wrong_way_cube_x_within_1mm":
            abs(previous_precontact_x-actual_old_cube_x)<.001,
        "hover_height_m":HOVER_Z,
        "rear_offset_m":BACK_OFFSET_M,
        "new_action_sequence":["existing_closed_mplib_clear",
                               "retreat_vertical",
                               "overhead_to_back",
                               "descend_behind_cube",
                               "push_tcp_x_.110",
                               "push_tcp_x_.180",
                               "push_tcp_x_.260"],
        "original_geometry_unchanged":True,
        "original_gripper_closed":True,
        "old_saved_results_never_modified":True,
        "learned_method":False,
        "equal_info_tamp":False,
    }


class SafeTargetTracer(ContactTracer):
    def __init__(self,env,origin):
        super().__init__(env)
        self.phase="gripper_close"
        self.origin=np.asarray(origin,float)
        self.max_positive_x=float(self.origin[0])
        self.violation=None

    def step(self,action):
        result=super().step(action)
        record=self.trace[-1]
        record["execution_phase"]=self.phase
        cube=np.asarray(record["task_cube"])
        if self.phase.startswith("push_"):
            if float(cube[0])>self.max_positive_x:
                self.max_positive_x=float(cube[0])
        if self.phase!="clear_blocker":
            reason=phase_violation(self.phase,cube,
                                    self.origin,self.max_positive_x)
            if reason is not None:
                self.violation={"phase":self.phase,
                                "step_index":record["t"],
                                "reason":reason,
                                "cube_xyz_m":cube.tolist(),
                                "tcp_xyz_m":record["tcp"]}
                raise RuntimeError("M1_TARGET_DIRECTION_GUARD_"+reason)
        if record["is_grasping_task"] or record.get("is_grasping_blocker"):
            raise RuntimeError("M1_TARGET_OBJECT_GRASP_DISALLOWED")
        return result


def upstream_move(planner,phase,position,q,steps,rrt=False):
    import sapien
    goal=sapien.Pose(p=position,q=q)
    began=len(steps.trace)
    steps.phase=phase
    if rrt:
        rv=planner.move_to_pose_with_RRTConnect(goal)
    else:
        rv=planner.move_to_pose_with_screw(goal)
    item={
      "phase":phase,"method":"official_RRTConnect" if rrt else "official_screw",
      "actions":len(steps.trace)-began,"planner_success":rv!=-1,
      "cube_final_xyz_m":_pose(steps.unwrapped.obj).tolist(),
      "tcp_final_xyz_m":_pose(steps.unwrapped.agent.tcp_pose).tolist(),
    }
    if rv==-1:
        raise RuntimeError("UPSTREAM_MPLIB_"+phase+"_PLANNING_FAILED")
    return item


def execute(env):
    traced=SafeTargetTracer(env,_pose(env.unwrapped.obj))
    b=traced.unwrapped
    start=_pose(b.obj)
    blocker_start=_pose(b.blocker)
    goal=np1(b.goal_pos)[:3].copy()
    planner=None
    phases=[]
    error=None
    initial_clearance_valid=False
    try:
        planner=get_planner(traced)
        traced.phase="gripper_close"
        planner.close_gripper(t=6)
        # IDENTICAL v2 closed-tool MPLib blocker clearance: do not change.
        target=np.asarray([BLOCKER_X,PARKING_Y])
        initial=_pose(b.blocker)
        delta=target-initial[:2]
        direction=delta/np.linalg.norm(delta)
        pre=np.r_[initial[:2]-direction*.05,initial[2]]
        post=np.r_[target+direction*.05,initial[2]]
        q=np1(b.agent.tcp.pose.q)[:4]
        t0=len(traced.trace)
        phases.append(upstream_move(planner,"clear_blocker",
                     pre,q,traced,rrt=True))
        phases.append(upstream_move(planner,"clear_blocker",
                     post,q,traced,rrt=False))
        final_blocker=_pose(b.blocker)
        clearance_error=float(np.linalg.norm(final_blocker[:2]-target))
        phases.append({
            "phase":"clearance_measured",
            "actions":len(traced.trace)-t0,
            "error_m":clearance_error,
            "blocker_final_xyz_m":final_blocker.tolist(),
            "task_cube_xyz_m":_pose(b.obj).tolist(),
        })
        if clearance_error>=.022 or final_blocker[1]<PARKING_Y-.035:
            raise RuntimeError("M1_TARGET_CLEARANCE_FAILED_NOT_A_TASK_PUSH_RESULT")
        initial_clearance_valid=True

        if np.linalg.norm(_pose(b.obj)[:2]-start[:2])>.010:
            raise RuntimeError("M1_TARGET_CUBE_MOVED_DURING_CLEARANCE_NO_HANDOFF")
        cube=_pose(b.obj)
        current_tcp=_pose(b.agent.tcp_pose)
        q=np1(b.agent.tcp.pose.q)[:4]
        seq=safe_approach_waypoints(cube,current_tcp,q,final_goal_x=float(goal[0]))
        for name,xyz in seq:
            phases.append(upstream_move(planner,name,xyz,q,traced,
                                        rrt=name!="descend_behind_cube"))
        tcp=_pose(b.agent.tcp_pose)
        cube=_pose(b.obj)
        if not (tcp[0] <= cube[0]-.038
                and abs(tcp[1]-cube[1])<.012
                and abs(tcp[2]-cube[2])<.018):
            raise RuntimeError(
                "M1_TARGET_NO_VALID_REAR_CONTACT_POSE_BEFORE_PUSH")
        for i,x in enumerate(PUSH_SEGMENT_TCP_X):
            if i>0 and _pose(b.obj)[0]<(.025 if i==1 else .085):
                raise RuntimeError("M1_TARGET_FORWARD_PROGRESS_INSUFFICIENT")
            xyz=np.asarray([x,float(start[1]),float(start[2])])
            phases.append(upstream_move(planner,f"push_{i+1}_x_{x:.3f}",
                                        xyz,q,traced,rrt=False))
    except Exception as exc:
        error=f"{type(exc).__name__}: {str(exc)[:520]}"
    finally:
        if planner is not None:
            planner.close()
    final_cube=_pose(b.obj)
    final_blocker=_pose(b.blocker)
    err=float(np.linalg.norm(final_cube[:2]-goal[:2]))
    push_only=observed_push_only(traced.trace)
    task_peak=max([start[2]]+[x["task_cube"][2] for x in traced.trace])
    blocker_peak=max([blocker_start[2]]+[x["blocker"][2] for x in traced.trace])
    success=bool(
       initial_clearance_valid and error is None and push_only["valid"]
       and err<GOAL_RADIUS and _actual_satisfied(b)
       and abs(final_cube[2]-TASK_HALF)<.009
       and abs(final_blocker[2]-BLOCKER_HALF)<.012
       and task_peak-start[2]<.015
       and blocker_peak-blocker_start[2]<.015
       and np.linalg.norm(final_blocker[:2]-blocker_start[:2])>.075
       and final_blocker[1]>PARKING_Y-.035
    )
    return {
      "kind":"target_safe_rear_approach","success":success,"error":error,
      "version":VERSION,"scene":"AGCOFLM1B3Corridor-v0",
      "seed":SEED,"original_scene_unchanged":True,
      "cube_initial":start.tolist(),"goal":goal.tolist(),
      "cube_final":final_cube.tolist(),"blocker_initial":blocker_start.tolist(),
      "blocker_final":final_blocker.tolist(),
      "closed_tool_clearance_valid":initial_clearance_valid,
      "cube_goal_error_m":err,
      "guard_violation":traced.violation,
      "physical_push_only":push_only,"phases":phases,
      "action_count":len(traced.trace),"trace":traced.trace,
      "source_physically_valid":False,
      "strong_tamp_evaluated":False,"learned_method_evaluated":False,
      "paper_ready":False,
    }


def run_once(root,official_root):
    if not direction_contract()["precontact_matches_old_wrong_way_cube_x_within_1mm"]:
        raise RuntimeError("Old failure not consistent with stated approach hypothesis")
    base=Path(root).resolve()
    case=base/CASE
    if (case/"status.json").is_file() or any(case.glob("*.h5")) or any(case.glob("*.json")):
        raise FileExistsError(f"Frozen v3 evidence must not be overwritten: {case}")
    case.mkdir(parents=True,exist_ok=True)
    env=None
    artifact_error=None
    try:
        # Same old seed/scene/control_mode and 450-action budget.
        # The RecordEpisode filename inside a NEW output folder is
        # historical v2, but the result itself is explicitly v3.
        env=make_joint_push_env(official_root,case)
        data=execute(env)
    except Exception as exc:
        data={
          "kind":"target_safe_rear_approach",
          "success":False,"action_count":0,"trace":[],
          "error":f"{type(exc).__name__}: {str(exc)[:550]}",
          "status":"EXECUTION_ERROR_NOT_PHYSICAL_SUCCESS",
        }
    finally:
        if env is not None:
            try:
                env.flush_trajectory()
                env.flush_video()
            except Exception as exc:
                artifact_error=f"{type(exc).__name__}: {str(exc)[:250]}"
            finally:
                env.close()
    trace=data.pop("trace",[])
    (case/"physical_trace.json").write_text(
        json.dumps(trace,indent=2,default=_bounded_summary)+"\n")
    data.update(version=VERSION,scene="AGCOFLM1B3Corridor-v0",
                seed=SEED,artifact_error=artifact_error,
                paper_ready=False)
    (case/"status.json").write_text(
        json.dumps(data,indent=2,default=_bounded_summary)+"\n")
    print("M1_TARGET_SAFE_APPROACH_RESULT",json.dumps({
        "success":data["success"],"actions":data.get("action_count"),
        "clearance_valid":data.get("closed_tool_clearance_valid"),
        "goal_error_m":data.get("cube_goal_error_m"),
        "final_cube_xyz_m":data.get("cube_final"),
        "guard_violation":data.get("guard_violation"),
        "error":data.get("error"),
        "phases":data.get("phases"),
        "status_file":str(case/"status.json"),
    }),flush=True)
    return data


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--out",default=OUT)
    p.add_argument("--official-root",default="third_party/roboreel_models")
    args=p.parse_args(argv)
    if args.preflight:
        print("M1_TARGET_SAFE_APPROACH_PREFLIGHT",json.dumps(direction_contract()),flush=True)
        return
    run_once(args.out,args.official_root)


if __name__=="__main__":
    main()
