"""M1-B4 ONE-TIME execution-feasibility rescue; same frozen M1-B3 geometry.

Only source lower-planning fallback (official MPLib RRTConnect) and target
pre-throat lateral alignment are altered. All previous M1B3 outputs
remain immutable. Not learned transfer, not a TAMP baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .roboreel_bridge import np1
from .m1b2_same_goal_effect_physics import (
    ActorPush, TracedEnv, _pose, _bounded_summary, actual_gate,
)
from .m1b3_corridor_physics import (
    ENV_ID, SEED, MAX_STEPS, TASK_HALF, BLOCKER_HALF, BLOCKER_X,
    PARKING_Y, GOAL_X, GOAL_RADIUS, WALL_HEIGHT, THROAT_X,
    THROAT_HALF_GAP_Y, MODES as PREVIOUS_MODES,
    make_env, geometry_preflight, _actual_satisfied,
)

MODES = ("source_lower_fallback", "target_align_after_clear")
VERSION = "m1b4_same_geometry_v1_execution_rescue"
PRE_THROAT_X = 0.103
PRE_THROAT_Y_TOL = 0.004


def measured_mechanism_gate(old_root: Path):
    """No simulator; accepts ONLY an existing frozen user's M1-B3 diagnosis."""
    file = old_root / "diagnosis.json"
    if not file.is_file():
        raise FileNotFoundError(f"Original immutable M1B3 diagnosis missing: {file}")
    diagnosis = json.loads(file.read_text())
    if (diagnosis.get("type") != "M1B3_EXISTING_PHYSICS_READ_ONLY_DIAGNOSIS"
            or diagnosis.get("scene") != ENV_ID or diagnosis.get("seed") != SEED
            or diagnosis.get("ran_simulator") is not False):
        raise ValueError("M1B3 origin diagnostic provenance invalid")
    source = diagnosis["cases"]["blocked_source"]
    clear = diagnosis["cases"]["blocked_clear"]
    if "official_mplib_lower_failed" not in (source.get("error") or ""):
        raise ValueError("Source failed for an unanticipated reason; no planner rescue")
    phases = clear["physical_phase_segments"]
    if (len(phases) != 2 or phases[0]["phase"] != "clear_blocker"
            or phases[0]["phase_success"] is not True
            or phases[1]["phase"] != "push_cube"
            or phases[1]["phase_success"] is not False):
        raise ValueError("Prior actual physics did not isolate clear-success/push-failure")
    final = clear["measured"]["task_final_xy_m"]
    # Throat wall starts x=THROAT_X-.012; task's half-width
    # prevents its center advancing beyond x=THROAT_X-.012-TASK_HALF
    # unless its center passes inside +/- (0.025-0.020) m.
    wall_front_center_limit = THROAT_X - .012 - TASK_HALF
    free_half_center_band = THROAT_HALF_GAP_Y - TASK_HALF
    spatial_match = (abs(final[0] - wall_front_center_limit) < .005
                     and abs(final[1]) > free_half_center_band + .010)
    if not spatial_match:
        raise ValueError("Saved stuck cube is NOT close to expected throat collision geometry")
    source_grasp = int(source["measured"]["task_grasped_steps"])
    source_lift = float(source["measured"]["task_max_z_m"])
    source_motion = float(source["measured"]["task_net_xy_m"])
    source_total_motion = float(source["measured"]["task_total_xy_m"])
    # DO NOT blindly rerun an identical original grasp controller when
    # the old real source video shows no held/lifted/transported cube.
    source_lower_fallback_relevant = bool(
        source_grasp > 0
        and source_lift > WALL_HEIGHT+TASK_HALF+.005
        and source_total_motion > .08
    )
    return {
        "old_scene": ENV_ID,
        "old_source_error": source["error"],
        "source_actual_grasp_steps": source_grasp,
        "source_actual_max_task_z_m": source_lift,
        "source_actual_task_net_xy_m": source_motion,
        "source_actual_task_total_xy_m": source_total_motion,
        "source_lower_planner_fallback_is_justified": source_lower_fallback_relevant,
        "old_push_stall_xy_m": final,
        "wall_front_cube_center_x_limit_m": wall_front_center_limit,
        "free_cube_center_half_band_y_m": free_half_center_band,
        "stuck_xy_consistent_with_throat_collision": spatial_match,
        "geometry_unchanged": True,
        "prior_trial_rerun": False,
    }


def run_source_once(env):
    """Official Panda OBB/MPLib; no new grasp PID, one RRTConnect on lower."""
    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )
    from mani_skill.examples.motionplanning.base_motionplanner.utils import (
        compute_grasp_info_by_obb, get_actor_obb,
    )

    traced = TracedEnv(env)
    b = env.unwrapped
    start = _pose(b.obj)
    goal = np1(b.goal_pos)[:3].copy()
    blocker_start = _pose(b.blocker)
    stages = []
    failure = None
    planner = PandaArmMotionPlanningSolver(
        traced, debug=False, vis=False, base_pose=b.agent.robot.pose,
        visualize_target_grasp_pose=False, print_env_info=False,
    )
    try:
        obb = get_actor_obb(b.obj)
        approach = np.asarray([0., 0., -1.])
        closing = b.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        g = compute_grasp_info_by_obb(
            obb, approaching=approach, target_closing=closing, depth=.025)
        grasp = b.agent.build_grasp_pose(
            approach, g["closing"], b.obj.pose.sp.p)

        def move(name, pose, lower=False):
            status = planner.move_to_pose_with_screw(pose)
            method = "original_screw"
            if status == -1 and lower:
                status = planner.move_to_pose_with_RRTConnect(pose)
                method = "original_rrtconnect_after_screw_failure"
            stages.append(dict(name=name, method=method, success=bool(status != -1)))
            if status == -1:
                raise RuntimeError(f"m1b4_{name}_planner_failed")

        move("approach", grasp * sapien.Pose([0, 0, -.05]))
        move("grasp", grasp)
        planner.close_gripper()
        actual_grasp = bool(np1(b.agent.is_grasping(b.obj))[0])
        stages.append(dict(name="actual_grasp_check", success=actual_grasp))
        if not actual_grasp:
            raise RuntimeError("m1b4_no_physical_grasp_after_close")

        translation = np.asarray([goal[0]-start[0], goal[1]-start[1], 0.])
        raise_offset = np.asarray([0.,0.,.13])
        move("lift", sapien.Pose(grasp.p+raise_offset, grasp.q))
        if _pose(b.obj)[2] < WALL_HEIGHT+TASK_HALF+.005:
            raise RuntimeError("m1b4_object_not_physically_lifted_above_wall")
        move("across", sapien.Pose(grasp.p+raise_offset+translation, grasp.q))
        if float(np.linalg.norm(_pose(b.obj)[:2]-goal[:2])) > .055:
            raise RuntimeError("m1b4_object_did_not_follow_across_planner")
        move("lower", sapien.Pose(grasp.p+translation, grasp.q), lower=True)
        planner.open_gripper(t=12)
        stages.append(dict(name="release", success=True))
    except Exception as exc:
        failure=f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        planner.close()

    final = _pose(b.obj)
    over = any(abs(row["task_cube"][0]-THROAT_X) < .022 and
               row["task_cube"][2] > WALL_HEIGHT+TASK_HALF+.005
               for row in traced.trace)
    obstruction_displacement = float(np.linalg.norm(
        _pose(b.blocker)[:2]-blocker_start[:2]))
    source_gate = actual_gate(traced.trace,start,final,goal,True)
    goal_error = float(np.linalg.norm(final[:2]-goal[:2]))
    valid = bool(
        source_gate["valid"] and failure is None and over
        and obstruction_displacement < .025
        and goal_error < GOAL_RADIUS and _actual_satisfied(b)
    )
    source_gate.update(
        valid=valid, over_new_throat=bool(over),
        blocker_displacement_m=obstruction_displacement,
        goal_error_m=goal_error,
        sim_success=_actual_satisfied(b),
    )
    return {
        "kind":"m1b4_source_fallback", "state_before":start.tolist(),
        "goal":goal.tolist(), "final":final.tolist(),
        "blocker_before":blocker_start.tolist(),
        "gate":source_gate,"error":failure,
        "planner_phases":stages,"steps":len(traced.trace),"trace":traced.trace,
    }


def pre_throat_lateral_alignment_needed(cube_xyz):
    """Classify cube lateral alignment just before the original 50mm throat.

    The throat's upstream wall front is at THROAT_X - 0.012, so an
    axis-aligned 40mm cube contacts it at center x=THROAT_X-.012-TASK_HALF
    (0.126m). Flag misalignment within 3mm BEFORE that face; the old
    '+.003' only flagged cubes already 3mm BEYOND the collision limit.
    Keep this as a geometry preflight helper, not physics/contact proof.
    """
    x, y = float(cube_xyz[0]), float(cube_xyz[1])
    wall_front_cube_center_limit = THROAT_X - .012 - TASK_HALF
    if (x >= wall_front_cube_center_limit - .003
            and abs(y) > PRE_THROAT_Y_TOL):
        return "MISALIGNED_AT_WALL_FRONT"
    return abs(y) > PRE_THROAT_Y_TOL


def run_target_once(env, max_steps=MAX_STEPS):
    traced=TracedEnv(env)
    b=env.unwrapped
    start=_pose(b.obj)
    goal=np1(b.goal_pos)[:3].copy()
    blocker_start=_pose(b.blocker)
    phases=[]

    def do_push(name, actor, destination, tolerance):
        if len(traced.trace) >= max_steps:
            return False
        ctrl=ActorPush(traced,actor,np.asarray(destination,dtype=float))
        began=len(traced.trace)
        ended_reason=None
        for _ in range(max_steps-len(traced.trace)):
            if np.linalg.norm(_pose(actor)[:2]-destination)<tolerance:
                break
            action=np.asarray(ctrl.act(),dtype=np.float32)
            action.reshape(-1)[-1]=-1.
            _,_,term,trunc,_=traced.step(action)
            if ctrl.failure or bool(np1(term)[0]) or bool(np1(trunc)[0]):
                ended_reason=ctrl.failure or "environment_end"
                break
        error=float(np.linalg.norm(_pose(actor)[:2]-destination))
        success=bool(error<tolerance)
        phases.append({
            "name":name,"steps":len(traced.trace)-began,
            "success":success,"distance_to_subgoal_m":error,
            "reason":ended_reason,"cube_final":_pose(b.obj).tolist(),
            "blocker_final":_pose(b.blocker).tolist(),
        })
        return success

    cleared=do_push(
        "clear_blocker", b.blocker,
        np.asarray([BLOCKER_X,PARKING_Y]), .022)
    precenter=False
    centered=False
    crossed=False
    delivered=False
    if cleared:
        precenter=do_push(
            "push_to_prethroat",b.obj,
            np.asarray([PRE_THROAT_X,0.0]),.012)
    if precenter:
        cube=_pose(b.obj)
        if cube[0] >= THROAT_X-.012-TASK_HALF and abs(cube[1]) > PRE_THROAT_Y_TOL:
            phases.append(dict(
                name="pre_throat_alignment_guard",
                success=False,
                reason="cube_already_against_throat_outside_safe_center_band"))
        else:
            if abs(cube[1]) > PRE_THROAT_Y_TOL:
                # One ordinary existing pushing primitive, along +y or
                # -y with x held at current center. No custom contact PID.
                aligned_x=float(cube[0])
                centered=do_push(
                    "align_y_before_throat",b.obj,
                    np.asarray([aligned_x,0.0]),PRE_THROAT_Y_TOL)
            else:
                centered=True
                phases.append(dict(name="align_y_before_throat",
                                   success=True,steps=0,reason="already_centered"))
            if centered:
                crossed=do_push(
                    "cross_throat",b.obj,
                    np.asarray([THROAT_X+.024,0.0]),.012)
            if crossed:
                delivered=do_push(
                    "push_to_goal",b.obj,goal[:2],GOAL_RADIUS*.75)

    final=_pose(b.obj)
    blocker_final=_pose(b.blocker)
    goal_error=float(np.linalg.norm(final[:2]-goal[:2]))
    peak_z=max([q["task_cube"][2] for q in traced.trace]+[start[2]])
    blocker_peak=max([q["blocker"][2] for q in traced.trace]+[blocker_start[2]])
    valid=bool(
        cleared and precenter and centered and crossed and delivered
        and traced.trace and goal_error<GOAL_RADIUS and _actual_satisfied(b)
        and peak_z-start[2]<.015 and blocker_peak-BLOCKER_HALF<.015
        and abs(final[2]-TASK_HALF)<.009
        and not any(q["is_grasping_task"] for q in traced.trace)
        and all(q["gripper_action"]<=0 for q in traced.trace)
        and blocker_final[1]>PARKING_Y-.035
    )
    return {
        "kind":"m1b4_target_centering",
        "state_before":start.tolist(),"goal":goal.tolist(),
        "final":final.tolist(),"blocker_before":blocker_start.tolist(),
        "blocker_final":blocker_final.tolist(),"error":None,
        "gate":{
            "valid":valid,"clear_success":cleared,
            "prethroat_success":precenter,"centered_success":centered,
            "cross_throat_success":crossed,"delivered_success":delivered,
            "goal_error_m":goal_error,"actual_sim_success":_actual_satisfied(b),
            "blocker_displacement_m":float(np.linalg.norm(
                blocker_final[:2]-blocker_start[:2])),
            "peak_task_z_m":float(peak_z),
            "peak_blocker_z_m":float(blocker_peak),
        },
        "steps":len(traced.trace),"phases":phases,"trace":traced.trace,
    }


def run_case(mode,out,old_root,official_root,max_steps=MAX_STEPS):
    if mode not in MODES or not (1<=max_steps<=MAX_STEPS):
        raise ValueError("Frozen one-time M1B4 rescue parameters invalid")
    old=measured_mechanism_gate(Path(old_root))
    geometry=geometry_preflight()
    if not geometry["pass"]:
        raise RuntimeError("Geometry changed or preflight failed")
    root=Path(out).resolve()/mode
    if (root/"status.json").exists() or any(root.glob("*.h5")):
        raise FileExistsError(f"M1B4 output is immutable: {root}")
    root.mkdir(parents=True,exist_ok=True)
    env=None
    artifact_error=None
    try:
        env=make_env(official_root,SEED,root,
                     "blocked_source" if mode==MODES[0] else "blocked_clear")
        result=(run_source_once(env) if mode==MODES[0]
                else run_target_once(env,max_steps))
    except Exception as exc:
        result=dict(kind=mode,status="M1B4_EXECUTION_ERROR",
                    gate={"valid":False},
                    error=f"{type(exc).__name__}: {str(exc)[:500]}",
                    steps=0,trace=[])
    finally:
        if env is not None:
            try:
                env.flush_trajectory()
                env.flush_video()
            except Exception as exc:
                artifact_error=f"{type(exc).__name__}: {str(exc)[:300]}"
            finally:
                env.close()
    trace=result.pop("trace",[])
    (root/"physical_trace.json").write_text(json.dumps(
        trace,indent=2,default=_bounded_summary)+"\n")
    result.update(mode=mode,version=VERSION,seed=SEED,
                  original_scene=ENV_ID,geometry_unchanged=True,
                  prior_immutable_source=old,artifact_error=artifact_error,
                  strong_tamp_evaluated=False,learned_method=False,
                  paper_ready=False)
    (root/"status.json").write_text(json.dumps(
        result,indent=2,default=_bounded_summary)+"\n")
    print("M1B4_CASE",json.dumps({
        "mode":mode,"success":bool(result["gate"]["valid"]),
        "steps":result["steps"],"gate":result["gate"],
        "phases":result.get("phases",result.get("planner_phases")),
        "error":result.get("error"),"artifact_error":artifact_error,
        "raw_status":str(root/"status.json"),
    }),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=MODES)
    p.add_argument("--offline-preflight",action="store_true")
    p.add_argument("--require-physical-source-evidence",action="store_true",
                   help="Preflight only; exit 6 if original trace never held/lifted/transported the source cube")
    p.add_argument("--old-root",default="outputs/m1b3_corridor_dev")
    p.add_argument("--out",default="outputs/m1b4_one_time_feasibility")
    p.add_argument("--official-root",default="third_party/roboreel_models")
    args=p.parse_args(argv)
    old=measured_mechanism_gate(Path(args.old_root))
    if args.offline_preflight or args.require_physical_source_evidence:
        print("M1B4_EVIDENCE_PREFLIGHT",json.dumps(old),flush=True)
        if (args.require_physical_source_evidence
                and not old["source_lower_planner_fallback_is_justified"]):
            print("M1B4_NO_GO_SOURCE_DID_NOT_PHYSICALLY_CARRY_CUBE_IN_FROZEN_RUN",
                  flush=True)
            raise SystemExit(6)
        return
    if not args.mode:
        p.error("--mode is required unless --offline-preflight")
    if not old["source_lower_planner_fallback_is_justified"]:
        raise SystemExit(
            "NO_GO: prior source lacked observed grasp/lift/translation. "
            "A lower RRT fallback cannot rescue an object it did not carry. "
            "Do not spend physical trials with the unchanged grasp controller."
        )
    run_case(args.mode,args.out,args.old_root,args.official_root)


if __name__=="__main__":
    main()
