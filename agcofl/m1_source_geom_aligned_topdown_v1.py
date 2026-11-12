"""M1 same-goal Pick SOURCE — ONE original-corridor geometry-aware contact probe.

Prior actual source TOP grasp at z~87mm (cube 20mm) and SIDE grasp at
z~121mm, both zero actual bilateral grasp. Existing M1B3 corridor's
inner y-wall clearance is +/-52mm below wall height 75mm. Exact Panda
URDF finger screw mount y envelope (with closing axis Y) is q + 26mm;
fully open q=40mm reaches +/-66mm and cannot fit between rails.
This one candidate uses official TOPDOWN OBB grasp with closing axis
parallel corridor longitudinal X and an intermediate 46mm fingertip
opening (q=23mm each), also reducing collision near blocker x=100mm.

This is original M1B3 physical feasibility diagnosis, NOT learned method.
Actually detect TCP reach before closing; abort without lift when no
contact. Only continue lift/carry/release if REAL Panda is_grasping.
All original M1 target v6, source B3/B2 raw remain unmodified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .m1b2_same_goal_effect_physics import TracedEnv, _pose, _bounded_summary
from .m1b3_corridor_physics import (
    BLOCKER_HALF, GOAL_RADIUS, MAX_STEPS, PARKING_Y,
    SEED, TASK_HALF, THROAT_X, WALL_HEIGHT, BLOCKER_X,
    _actual_satisfied, geometry_preflight, register_env, static_walls,
)
from .m1_current_executor_swap import get_planner
from .roboreel_bridge import np1

VERSION="m1_source_geom_aligned_topdown_v1"
CASE="source_aligned_topdown"
OUT="outputs/m1_source_geom_aligned_topdown_v1"
PRECONTACT_ACTION_MAX=155
FULL_ACTION_MAX=MAX_STEPS
OPEN_FINGER_Q_M=.023
# Panda PDJointPosMimic config lower=-.01 upper=.04,
# normalized gripper action = 2*(q+.01)/.05 - 1
PREGRASP_ACTION=2*(OPEN_FINGER_Q_M+.01)/.05-1.
MAX_TCP_CUBE_CENTER_DIST_M=.013
MAX_PRELIFT_BLOCKER_SHIFT_M=.015


def geometry_contract():
    g=geometry_preflight()
    if not g["pass"]:
        raise RuntimeError("Original M1B3 corridor preflight changed")
    inside_rail=.068-.016   # original wall closest inner face
    screw_y_outer=.0185+.015/2  # PANDA panda_v2.urdf first finger collision box
    return {
        "scene":"AGCOFLM1B3Corridor-v0",
        "seed":SEED,"goal_x_m":.210,
        "rail_inner_y_m":inside_rail,
        "wall_height_m":WALL_HEIGHT,
        "source_topdown_approaching":[0.,0.,-1.],
        "closing_axis_parallel_corridor_longitudinal_x":[1.,0.,0.],
        "urdf_reference":"mani_skill/assets/robots/panda/panda_v2.urdf",
        "fully_open_gripper_finger_q_m":.040,
        "fully_open_screw_mount_outer_radius_m":.040+screw_y_outer,
        "fully_open_if_y_aligned_rail_intrusion_m":.040+screw_y_outer-inside_rail,
        "candidate_gripper_finger_q_m":OPEN_FINGER_Q_M,
        "candidate_screw_mount_outer_radius_m":OPEN_FINGER_Q_M+screw_y_outer,
        "candidate_if_y_aligned_rail_clearance_m":inside_rail-(OPEN_FINGER_Q_M+screw_y_outer),
        "candidate_tip_opening_m":2*OPEN_FINGER_Q_M,
        "cube_width_m":2*TASK_HALF,
        "normalized_panda_gripper_action":PREGRASP_ACTION,
        "contact_measured_before_close":True,
        "original_scenes_and_results_unchanged":True,
        "no_same_goal_source_physics_claimed":True,
    }


def collision_box_surface(cx,cy,cz,hx,hy,hz,rng,n_each=12):
    """Deterministically sample world-frame box faces for upstream MPLib."""
    points=[]
    c=np.array([cx,cy,cz],float)
    h=np.array([hx,hy,hz],float)
    for axis in range(3):
        other=[a for a in range(3) if a!=axis]
        for sign in (-1.,1.):
            xy=rng.uniform(-1.,1.,size=(n_each,2))
            p=np.repeat(c[None,:],n_each,axis=0)
            p[:,axis]+=sign*h[axis]
            p[:,other[0]]+=xy[:,0]*h[other[0]]
            p[:,other[1]]+=xy[:,1]*h[other[1]]
            points.extend(p.tolist())
    return points


def original_obstacle_points():
    """Actual original static walls + original dynamic blocker's reset pose."""
    rng=np.random.default_rng(SEED)
    points=[]
    for w in static_walls():
        points.extend(collision_box_surface(
            w["cx"],w["cy"],WALL_HEIGHT/2,
            w["hx"],w["hy"],WALL_HEIGHT/2,rng))
    points.extend(collision_box_surface(
        BLOCKER_X,0,BLOCKER_HALF,BLOCKER_HALF,BLOCKER_HALF,BLOCKER_HALF,rng))
    return np.asarray(points,dtype=np.float64)


class SourceTrace(TracedEnv):
    def __init__(self,env):
        super().__init__(env)
        self.phase="initialize"
        self.wall_force_measured_count=0

    def step(self,action):
        if self.phase in ("pregrasp","grasp_pose") and len(self.trace)>=PRECONTACT_ACTION_MAX:
            raise RuntimeError("SOURCE_PRECONTACT_REAL_ACTION_CAP_NOT_AT_CUBE")
        res=super().step(action)
        b=self.unwrapped
        current=self.trace[-1]
        current["execution_phase"]=self.phase
        q=np1(b.agent.robot.get_qpos())
        current["finger_joint_qpos_m"]=[float(q[-2]),float(q[-1])]
        current["tcp_quaternion_wxyz"]=np1(b.agent.tcp.pose.q)[:4].tolist()
        # Contact force may not be available in some local SAPIEN versions.
        # Never infer absence of collision when this measurement is missing.
        maximum=0.
        try:
            for link_name in ("panda_hand","panda_leftfinger","panda_rightfinger"):
                link=b.agent.robot.links_map[link_name]
                for wall in b.corridor_walls:
                    val=np1(b.scene.get_pairwise_contact_forces(link,wall))
                    maximum=max(maximum,float(np.linalg.norm(val)))
            current["tool_static_wall_contact_force_max_n"]=maximum
            self.wall_force_measured_count+=1
        except Exception:
            current["tool_static_wall_contact_force_max_n"]=None
        return res


def make_env(official_root,out):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.record import RecordEpisode

    eid=register_env(official_root)
    raw=gym.make(
        eid,obs_mode="state",robot_uids="panda",num_envs=1,
        control_mode="pd_joint_pos",render_mode="rgb_array",
        sim_backend="physx_cpu",distract=False,max_episode_steps=MAX_STEPS)
    rec=RecordEpisode(
        raw,output_dir=str(out),trajectory_name=CASE,
        save_video=True,save_on_reset=False,record_reward=False,
        source_type="privileged_diagnostic",
        source_desc="Original M1B3 exact same-goal Pick aligned topdown contact/transfer feasibility; NOT method/TAMP")
    rec.reset(seed=SEED,options={"m1b3_open":False})
    return rec


def execute(env):
    import sapien
    from mani_skill.examples.motionplanning.base_motionplanner.utils import (
        compute_grasp_info_by_obb,get_actor_obb)

    traced=SourceTrace(env)
    b=traced.unwrapped
    cube=b.obj
    initial=_pose(cube)
    goal=np1(b.goal_pos)[:3].copy()
    blocker_initial=_pose(b.blocker)
    planner=None
    phases=[]
    error=None
    actual_contact=False
    try:
        planner=get_planner(traced)
        pts=original_obstacle_points()
        planner.add_collision_pts(pts)
        # Upstream add_collision_pts() updates the cloud but does NOT
        # automatically enable it for plan_screw/plan_qpos_to_pose.
        planner.use_point_cloud=True
        phases.append({"phase":"original_static_and_blocker_collision_points",
                       "point_count":len(pts),
                       "pointcloud_not_exact_mesh_proof":True})

        traced.phase="set_partial_open"
        planner.open_gripper(t=6,gripper_state=PREGRASP_ACTION)
        measured_finger=np1(b.agent.robot.get_qpos())[-2:]
        phases.append({"phase":"set_partial_open","steps":6,
                       "actual_finger_joints_m":measured_finger.tolist()})
        if not np.all(np.abs(measured_finger-OPEN_FINGER_Q_M)<.004):
            raise RuntimeError("SOURCE_PARTIAL_OPEN_FINGER_NOT_AT_PHYSICAL_WIDTH")

        approaching=np.asarray([0.,0.,-1.],float)
        target_closing=np.asarray([1.,0.,0.],float)
        gi=compute_grasp_info_by_obb(get_actor_obb(cube),
                  approaching=approaching,target_closing=target_closing,
                  depth=.025)
        closing=np.asarray(gi["closing"],float)
        if abs(np.dot(closing,target_closing))<.98:
            raise RuntimeError("SOURCE_CLOSING_AXIS_NOT_ALIGNED_WITH_CORRIDOR")
        grasp=b.agent.build_grasp_pose(approaching,closing,cube.pose.sp.p)

        for phase,pose in (
            ("pregrasp",grasp*sapien.Pose([0,0,-.060])),
            ("grasp_pose",grasp),
        ):
            traced.phase=phase
            before=len(traced.trace)
            result=planner.move_to_pose_with_screw(pose)
            after=_pose(b.agent.tcp_pose)
            dist=float(np.linalg.norm(after-_pose(cube)))
            phases.append({"phase":phase,
                           "actual_steps":len(traced.trace)-before,
                           "planner_result":result!=-1,
                           "tcp_cube_distance_m":dist,
                           "tcp_xyz_m":after.tolist(),
                           "physical_finger_qpos_m":
                              np1(b.agent.robot.get_qpos())[-2:].tolist()})
            if result==-1:
                raise RuntimeError("SOURCE_UPSTREAM_"+phase+"_PLANNING_FAILED")
            if phase=="grasp_pose" and dist>=MAX_TCP_CUBE_CENTER_DIST_M:
                raise RuntimeError(
                   f"SOURCE_ACTUAL_TCP_NOT_AT_GRASP_BEFORE_CLOSE: {dist:.4f}m")

        if np.linalg.norm(_pose(b.blocker)[:2]-blocker_initial[:2])>=MAX_PRELIFT_BLOCKER_SHIFT_M:
            raise RuntimeError("SOURCE_MOVED_BLOCKER_DURING_PREGRASP")
        traced.phase="close_gripper"
        before=len(traced.trace)
        planner.close_gripper(t=8)
        actual_contact=bool(np1(b.agent.is_grasping(cube))[0])
        phases.append({"phase":"physical_grasp_gate",
                       "actual_steps":len(traced.trace)-before,
                       "actual_bilateral_grasp":actual_contact,
                       "tcp_cube_m":float(np.linalg.norm(
                           _pose(b.agent.tcp_pose)-_pose(cube)))})
        if not actual_contact:
            raise RuntimeError("SOURCE_NO_REAL_BILATERAL_GRASP_ABORT_BEFORE_LIFT")

        delta=np.array([goal[0]-initial[0],goal[1]-initial[1],0.])
        up=np.array([0.,0.,.13])
        for phase,pose in (
            ("lift",sapien.Pose(grasp.p+up,grasp.q)),
            ("carry",sapien.Pose(grasp.p+up+delta,grasp.q)),
            ("lower",sapien.Pose(grasp.p+delta,grasp.q)),
        ):
            traced.phase=phase
            before=len(traced.trace)
            result=planner.move_to_pose_with_screw(pose)
            phases.append({"phase":phase,
                           "actual_steps":len(traced.trace)-before,
                           "planner_result":result!=-1,
                           "actual_cube_xyz_m":_pose(cube).tolist()})
            if result==-1:
                raise RuntimeError("SOURCE_UPSTREAM_"+phase+"_PLANNING_FAILED")
            if not bool(np1(b.agent.is_grasping(cube))[0]):
                raise RuntimeError("SOURCE_LOST_REAL_GRASP_DURING_"+phase)
            if phase=="lift" and _pose(cube)[2]<WALL_HEIGHT+TASK_HALF+.005:
                raise RuntimeError("SOURCE_CUBE_NOT_LIFTED_ABOVE_ORIGINAL_WALLS")
            if phase=="carry" and np.linalg.norm(_pose(cube)[:2]-goal[:2])>.035:
                raise RuntimeError("SOURCE_CUBE_NOT_CARRIED_TO_SAME_GOAL")

        traced.phase="release"
        # Same partial opening (46mm gap >40mm cube) avoids widening
        # finger screw mounts beside the right 285mm static fence.
        planner.open_gripper(t=8,gripper_state=PREGRASP_ACTION)
    except Exception as exc:
        error=f"{type(exc).__name__}: {str(exc)[:480]}"
    finally:
        if planner is not None:
            planner.close()

    final=_pose(cube)
    blocker_final=_pose(b.blocker)
    peak=float(max([initial[2]]+[r["task_cube"][2] for r in traced.trace]))
    blocker_shift=float(np.linalg.norm(
        blocker_final[:2]-blocker_initial[:2]))
    over=any(abs(r["task_cube"][0]-THROAT_X)<.022
             and r["task_cube"][2]>WALL_HEIGHT+TASK_HALF+.005
             for r in traced.trace)
    actual_grasp_steps=sum(bool(r["is_grasping_task"]) for r in traced.trace)
    goal_error=float(np.linalg.norm(final[:2]-goal[:2]))
    success=bool(error is None and actual_contact and
       actual_grasp_steps>0 and peak-initial[2]>.08 and over and
       blocker_shift<.025 and goal_error<GOAL_RADIUS and
       abs(final[2]-TASK_HALF)<.009 and
       not bool(np1(b.agent.is_grasping(cube))[0]) and
       _actual_satisfied(b))
    return {
        "version":VERSION,"kind":CASE,
        "scene":"AGCOFLM1B3Corridor-v0","seed":SEED,
        "success":success,"error":error,
        "actual_steps":len(traced.trace),
        "actual_contact_confirmed":actual_contact,
        "actual_grasp_steps":actual_grasp_steps,
        "max_cube_lift_m":peak-initial[2],
        "over_original_throat_physically":over,
        "blocker_net_xy_m":blocker_shift,
        "blocker_initial_xyz_m":blocker_initial.tolist(),
        "blocker_final_xyz_m":blocker_final.tolist(),
        "cube_initial_xyz_m":initial.tolist(),
        "cube_final_xyz_m":final.tolist(),
        "goal_xyz_m":goal.tolist(),
        "goal_error_m":goal_error,
        "phases":phases,
        "wall_force_measured_steps":traced.wall_force_measured_count,
        "not_a_learned_method":True,"not_an_equal_info_tamp":True,
        "paper_ready":False,"trace":traced.trace,
    }


def run_once(out,official_root):
    geometry_contract()
    root=Path(out).resolve()
    case=root/CASE
    if ((case/"status.json").exists()
            or any(case.glob("*.json"))
            or any(case.glob("*.h5"))):
        raise FileExistsError(f"Original M1 SOURCE aligned contact case immutable: {case}")
    case.mkdir(parents=True,exist_ok=True)
    env=None
    artifact_error=None
    try:
        env=make_env(official_root,case)
        data=execute(env)
    except Exception as exc:
        data={"version":VERSION,"kind":CASE,"success":False,
              "actual_steps":0,"trace":[],
              "error":f"{type(exc).__name__}: {str(exc)[:500]}",
              "status":"INVALID_EXECUTION_NO_SUCCESS"}
    finally:
        if env is not None:
            try:
                env.flush_trajectory()
                env.flush_video()
            except Exception as exc:
                artifact_error=f"{type(exc).__name__}: {str(exc)[:500]}"
            finally:
                env.close()
    trace=data.pop("trace",[])
    (case/"physical_trace.json").write_text(
        json.dumps(trace,indent=2,default=_bounded_summary)+"\n")
    data["artifact_error"]=artifact_error
    (case/"status.json").write_text(
        json.dumps(data,indent=2,default=_bounded_summary)+"\n")
    print("M1_SOURCE_GEOMETRY_ALIGNED_REAL_GATE",json.dumps({
      "success":data["success"],
      "steps":data.get("actual_steps"),
      "physical_grasp_reached":data.get("actual_contact_confirmed"),
      "actual_grasp_steps":data.get("actual_grasp_steps"),
      "cube_lift_m":data.get("max_cube_lift_m"),
      "cube_final_xyz_m":data.get("cube_final_xyz_m"),
      "goal_error_m":data.get("goal_error_m"),
      "blocker_xy_shift_m":data.get("blocker_net_xy_m"),
      "contact_force_measured_steps":data.get("wall_force_measured_steps"),
      "phases":data.get("phases"),"error":data.get("error"),
      "status_file":str(case/"status.json"),
    }),flush=True)
    return data


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--out",default=OUT)
    p.add_argument("--official-root",default="third_party/roboreel_models")
    a=p.parse_args(argv)
    if a.preflight:
        print("M1_SOURCE_GEOMETRY_ALIGNED_PREFLIGHT",
              json.dumps(geometry_contract()),flush=True)
        return
    run_once(a.out,a.official_root)


if __name__=="__main__":
    main()
