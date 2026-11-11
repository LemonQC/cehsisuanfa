"""M1-B3 candidate corridor — NEW SCENE, untouched M1-B2 negative pilot.

Geometric C-space checks are necessary preflight, NOT proof about actual
dynamics. Four fixed PhysX episodes test blocked source Pick+Place, blocked
direct Push, blocked clear-then-Push, and open-passage direct Push.
This is still oracle-state diagnostic execution (NO learned method/TAMP).
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys

import numpy as np

from .roboreel_bridge import np1
from .m1b2_same_goal_effect_physics import (
    ActorPush, TracedEnv, _pose, _bounded_summary,
    run_source as original_mplib_pick_place,
)

ENV_ID = "AGCOFLM1B3Corridor-v0"
SEED = 120
MAX_STEPS = 450
TASK_HALF = 0.020
BLOCKER_HALF = 0.029
BLOCKER_X = 0.100
PARKING_Y = 0.125
GOAL_X = 0.210
GOAL_RADIUS = 0.020
WALL_HEIGHT = 0.075
THROAT_X = 0.158
THROAT_HALF_GAP_Y = 0.025
DOOR_X0, DOOR_X1 = 0.065, 0.135
MODES = ("blocked_source", "blocked_direct", "blocked_clear", "open_direct")


def static_walls():
    """Fixed 3D solid boxes (x,y center + half sizes, z from table to height)."""
    w = []
    for sign in (-1, 1):
        for name, low, high in (("left", -0.065, DOOR_X0),
                                ("right", DOOR_X1, 0.245)):
            w.append(dict(name=f"rail_{sign}_{name}",
                          cx=(low+high)/2, cy=sign*.068,
                          hx=(high-low)/2, hy=.016))
        # Size-selective throat: 0.05m clear width, task width 0.04m
        # fits whereas blocker width 0.058m cannot pass it.
        w.append(dict(name=f"throat_{sign}",cx=THROAT_X,
                      cy=sign*.0385,hx=.012,hy=.0135))
    w.extend([
        dict(name="fence_left", cx=-.095, cy=0, hx=.012, hy=.170),
        dict(name="fence_right",cx=.285,cy=0,hx=.012,hy=.170),
        dict(name="fence_top",cx=.095,cy=.170,hx=.190,hy=.012),
        dict(name="fence_bottom",cx=.095,cy=-.170,hx=.190,hy=.012),
    ])
    return w


def _blocked_xy(x,y,blocked):
    walls=static_walls()
    walls.append(dict(name="blocker",cx=BLOCKER_X,
                      cy=0.0 if blocked else PARKING_Y,
                      hx=BLOCKER_HALF,hy=BLOCKER_HALF))
    for w in walls:
        if (abs(x-w["cx"]) < w["hx"]+TASK_HALF
                and abs(y-w["cy"]) < w["hy"]+TASK_HALF):
            return True
    return False


def cspace_reachable(blocked, grid_step=.003):
    """Conservative 2D square-footprint C-space grid (not a robot proof)."""
    origin=(-.090,-.165)
    n_x=127
    n_y=111
    start=(round((0-origin[0])/grid_step),
           round((0-origin[1])/grid_step))
    goal=(round((GOAL_X-origin[0])/grid_step),
          round((0-origin[1])/grid_step))
    def free(cell):
        ix,iy=cell
        return (0 <= ix < n_x and 0 <= iy < n_y
                and not _blocked_xy(origin[0]+ix*grid_step,
                                    origin[1]+iy*grid_step,blocked))
    if not free(start) or not free(goal):
        raise ValueError("Scene seed start/goal overlaps geometry")
    queue=deque([start])
    visited={start}
    while queue:
        x,y=queue.popleft()
        if (x,y)==goal:
            return dict(reachable=True, visited_cells=len(visited))
        for cell in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if cell not in visited and free(cell):
                visited.add(cell)
                queue.append(cell)
    return dict(reachable=False,visited_cells=len(visited))


def geometry_preflight():
    blocked=cspace_reachable(True)
    opened=cspace_reachable(False)
    throat_width=2*THROAT_HALF_GAP_Y
    door_width=DOOR_X1-DOOR_X0
    checks={
        "task_fits_throat":2*TASK_HALF < throat_width,
        "blocker_cannot_fit_throat":2*BLOCKER_HALF > throat_width,
        "blocker_fits_side_door":2*BLOCKER_HALF < door_width,
        "blocker_can_park_within_outer_fence":
            PARKING_Y+BLOCKER_HALF < .170-.012,
        "blocked_static_cube_path_disconnected":not blocked["reachable"],
        "open_static_cube_path_connected":opened["reachable"],
    }
    return {
        "type":"M1B3_GEO_CSPACE_PREFLIGHT",
        "scene":ENV_ID,
        "checks":checks,
        "pass":all(checks.values()),
        "blocked_cspace":blocked,"open_cspace":opened,
        "grid_step_m":.003,
        "task_half_m":TASK_HALF,
        "blocker_half_m":BLOCKER_HALF,
        "throat_width_m":throat_width,
        "door_width_m":door_width,
        "caution":"Finite grid with stationary blocker, NOT dynamic feasibility or proof of universal controller failure.",
    }


def register_env(official_root):
    import gymnasium as gym
    import sapien
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.building import actors
    from mani_skill.utils.structs import Pose

    upstream=Path(official_root).resolve()/"roboreel_env/src"
    if not (upstream/"envs/push_cube.py").is_file():
        raise FileNotFoundError(f"Pinned RoboReel submodule missing: {upstream}")
    if str(upstream) not in sys.path:
        sys.path.insert(0,str(upstream))
    from envs.push_cube import PushCubeEnvV2

    if ENV_ID in gym.registry:
        return ENV_ID

    class M1B3Corridor(PushCubeEnvV2):
        goal_radius=GOAL_RADIUS

        def _load_scene(self, options):
            super()._load_scene(options)
            self.blocker=actors.build_cube(
                self.scene,half_size=BLOCKER_HALF,
                name="m1b3_dynamic_blocker",body_type="dynamic",
                color=[.9,.55,.12,1],
                initial_pose=sapien.Pose(p=[BLOCKER_X,0,BLOCKER_HALF]))
            self.corridor_walls=[]
            for wall in static_walls():
                wall_actor=actors.build_box(
                    self.scene,
                    half_sizes=[wall["hx"],wall["hy"],WALL_HEIGHT/2],
                    body_type="static",
                    color=[.7,.18,.2,1],
                    name=f"m1b3_{wall['name']}",
                    initial_pose=sapien.Pose(
                        p=[wall["cx"],wall["cy"],WALL_HEIGHT/2]))
                self.corridor_walls.append(wall_actor)

        def _initialize_episode(self, env_idx, options):
            super()._initialize_episode(env_idx, options)
            self.obj.set_pose(Pose.create_from_pq(
                p=[[0,0,TASK_HALF]],q=[1,0,0,0]))
            self.goal_pos=self.obj.pose.p.clone()
            self.goal_pos[...,0]=GOAL_X
            self.goal_pos[...,2]=1e-3
            start_y=PARKING_Y if options.get("m1b3_open",False) else 0.0
            self.blocker.set_pose(Pose.create_from_pq(
                p=[[BLOCKER_X,start_y,BLOCKER_HALF]],q=[1,0,0,0]))

        def _get_obs_extra(self, info):
            obs=super()._get_obs_extra(info)
            if "state" in self.obs_mode:
                obs["blocker_pose"]=self.blocker.pose.raw_pose
            return obs

    gym.register(id=ENV_ID,entry_point=M1B3Corridor,
                 max_episode_steps=MAX_STEPS)
    return ENV_ID


def make_env(root,seed,out,mode):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.record import RecordEpisode
    register_env(root)
    env=gym.make(ENV_ID,obs_mode="state",robot_uids="panda",
                 num_envs=1,
                 control_mode=("pd_joint_pos" if mode=="blocked_source"
                               else "pd_ee_delta_pose"),
                 render_mode="rgb_array",sim_backend="physx_cpu",
                 distract=False,max_episode_steps=MAX_STEPS)
    record=RecordEpisode(
        env,output_dir=str(out),trajectory_name=mode,save_video=True,
        save_on_reset=False,record_reward=False,
        source_type=("motionplanning" if mode=="blocked_source"
                     else "privileged_diagnostic"),
        source_desc="M1B3 fixed corridor DEV, not a learned policy or paper TAMP")
    record.reset(seed=seed,options={"m1b3_open":mode=="open_direct"})
    return record


def _actual_satisfied(b):
    return bool(np1(b.evaluate()["success"])[0])


def run_paired_source(env):
    b=env.unwrapped
    initial_blocker=_pose(b.blocker)
    result=original_mplib_pick_place(env)
    trace=result["trace"]
    # Existing M1B2 source includes an old scene-specific overflight
    # heuristic. We ALSO require crossing this NEW throat above its wall.
    over_new_throat=any(
        abs(p["task_cube"][0]-THROAT_X)<.022
        and p["task_cube"][2]>WALL_HEIGHT+TASK_HALF+.005
        for p in trace)
    error=float(np.linalg.norm(_pose(b.obj)[:2]-np1(b.goal_pos)[:2]))
    valid=bool(result["gate"]["valid"] and
               over_new_throat and error<GOAL_RADIUS
               and _actual_satisfied(b))
    result["gate"]["valid"]=valid
    result["gate"]["m1b3_over_throat"]=over_new_throat
    result["gate"]["m1b3_goal_error_m"]=error
    result["gate"]["m1b3_sim_success"]=_actual_satisfied(b)
    result["blocker_before"]=initial_blocker.tolist()
    return result


def run_target(env,mode,max_steps):
    traced=TracedEnv(env)
    b=env.unwrapped
    start_cube=_pose(b.obj)
    goal=np1(b.goal_pos)[:3].copy()
    blocker_start=_pose(b.blocker)
    if mode=="blocked_clear":
        route=[
            ("clear_blocker",b.blocker,
             np.array([BLOCKER_X,PARKING_Y],float),.022),
            ("push_cube",b.obj,goal[:2],GOAL_RADIUS*.75),
        ]
    else:
        route=[("push_cube",b.obj,goal[:2],GOAL_RADIUS*.75)]
    phases=[]
    for label,actor,destination,tolerance in route:
        controller=ActorPush(traced,actor,destination)
        t0=len(traced.trace)
        hit_budget=False
        for _ in range(max_steps-len(traced.trace)):
            if np.linalg.norm(_pose(actor)[:2]-destination)<tolerance:
                break
            action=np.asarray(controller.act(),dtype=np.float32)
            action.reshape(-1)[-1]=-1
            _,_,terminated,truncated,_=traced.step(action)
            if controller.failure or bool(np1(terminated)[0]) or bool(np1(truncated)[0]):
                break
        else:
            hit_budget=True
        distance=float(np.linalg.norm(_pose(actor)[:2]-destination))
        phases.append(dict(name=label,steps=len(traced.trace)-t0,
                           distance_to_subgoal_m=distance,
                           success=distance<tolerance,
                           failure=controller.failure,
                           exhausted=hit_budget))
        if not phases[-1]["success"]:
            break

    final_cube=_pose(b.obj)
    final_blocker=_pose(b.blocker)
    steps=traced.trace
    goal_error=float(np.linalg.norm(final_cube[:2]-goal[:2]))
    peak=float(max((x["task_cube"][2] for x in steps),
                   default=start_cube[2])-start_cube[2])
    gripper_never_closed=all(r["gripper_action"]<=0 for r in steps)
    never_grasped=all(not r["is_grasping_task"] for r in steps)
    grounded=abs(final_cube[2]-TASK_HALF)<.009
    actual_success=bool(_actual_satisfied(b) and
                        goal_error<GOAL_RADIUS and peak<.015
                        and grounded and gripper_never_closed and never_grasped
                        and bool(steps))
    clear_observed=(mode=="blocked_clear" and
                    final_blocker[1]>PARKING_Y-.035
                    and np.linalg.norm(final_blocker[:2]-blocker_start[:2])>.075
                    and len(phases)>=2 and phases[0]["success"])
    success=bool(actual_success and (
        clear_observed if mode=="blocked_clear" else True))
    return {
        "kind":mode,"state_before":start_cube.tolist(),"goal":goal.tolist(),
        "blocker_before":blocker_start.tolist(),
        "blocker_final":final_blocker.tolist(),
        "final":final_cube.tolist(),
        "gate":{
            "valid":success,"task_goal_success":actual_success,
            "goal_error_m":goal_error,"grounded":grounded,
            "peak_lift_m":peak,"gripper_never_closed":gripper_never_closed,
            "never_grasped":never_grasped,
            "blocker_moved_m":float(np.linalg.norm(
                final_blocker[:2]-blocker_start[:2])),
            "blocker_cleared_to_side_pocket":bool(clear_observed),
            "sim_goal_success":_actual_satisfied(b),
        },
        "phases":phases,"trace":steps,"steps":len(steps),
        "is_privileged_scripted_diagnostic":True,
        "is_equal_information_tamp":False,
    }


def run_case(official_root,seed,mode,out,max_steps=MAX_STEPS):
    if mode not in MODES or seed!=SEED or not 1<=max_steps<=MAX_STEPS:
        raise ValueError("Frozen M1B3 case/seed/step budget violated")
    out=Path(out).resolve()
    case=out/mode
    if (case/"status.json").exists():
        raise FileExistsError(f"Preserve original data; no case overwrite: {case}")
    for suffix in (".h5",".json"):
        if (case/(mode+suffix)).exists():
            raise FileExistsError(f"Partial raw evidence exists: {case/(mode+suffix)}")
    case.mkdir(parents=True,exist_ok=True)
    preflight=geometry_preflight()
    if not preflight["pass"]:
        raise RuntimeError("Geometry C-space preflight failed")
    env=None
    artifact_error=None
    try:
        env=make_env(official_root,seed,case,mode)
        result=(run_paired_source(env) if mode=="blocked_source"
                else run_target(env,mode,max_steps))
    except Exception as e:
        result={
            "kind":mode,
            "status":"M1B3_EXECUTION_ERROR_INVALID_PHYSICS",
            "error":f"{type(e).__name__}: {str(e)[:1200]}",
            "trace":[],"gate":{"valid":False},"steps":0,
        }
    finally:
        if env is not None:
            try:
                env.flush_trajectory()
                env.flush_video()
            except Exception as e:
                artifact_error=f"{type(e).__name__}: {str(e)[:500]}"
            finally:
                env.close()
    trace=result.pop("trace",[])
    (case/"physical_trace.json").write_text(json.dumps(
        trace,indent=2,default=_bounded_summary)+"\n")
    result.update(scene=ENV_ID,seed=seed,mode=mode,
                  same_goal_scene_version="m1b3_corridor_v0",
                  blocked_condition=mode!="open_direct",
                  artifact_error=artifact_error,
                  paper_ready=False,
                  learned_method=False)
    (case/"status.json").write_text(json.dumps(
        result,indent=2,default=_bounded_summary)+"\n")
    print("M1B3_CASE_PHYSICS",json.dumps({
        "mode":mode,"success":result.get("gate",{}).get("valid",False),
        "steps":result.get("steps"),
        "goal_error":result.get("gate",{}).get("goal_error_m"),
        "blocker_cleared":result.get("gate",{}).get("blocker_cleared_to_side_pocket"),
        "error":result.get("error"),"artifact_error":artifact_error,
        "out":str(case/"status.json"),
    },default=_bounded_summary),flush=True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=MODES,required=True)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--max-steps",type=int,default=MAX_STEPS)
    p.add_argument("--official-root",default="third_party/roboreel_models")
    p.add_argument("--out",default="outputs/m1b3_corridor_dev")
    args=p.parse_args(argv)
    run_case(args.official_root,args.seed,args.mode,args.out,args.max_steps)


if __name__=="__main__":
    main()
