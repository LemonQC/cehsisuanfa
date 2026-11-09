"""Headless Panda physically-controlled teacher without MPLib.

The native ManiSkill pd_ee_delta_pos controller supplies the IK internally.
Failed or incomplete trajectories are recorded as failures.
"""
from __future__ import annotations
import numpy as np
import torch
from .representation import future_masks, route_points, contact_waypoints, mask_route_side


def np1(x):
    if torch.is_tensor(x): return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x).reshape(-1)


def robot_state(base):
    q=np1(base.agent.robot.get_qpos())
    qv=np1(base.agent.robot.get_qvel())
    tcp=np1(base.agent.tcp.pose.p)
    obj=np1(base.obj.pose.p)
    goal=np1(base.goal_region.pose.p)
    # 9+9+3+3+3=27 for official Panda; length recorded in dataset.
    return np.concatenate([q,qv,tcp,obj,goal],axis=0).astype(np.float32)


class RecorderEnv:
    """Drop-in env adapter: intercept solver actions, record pre-step state."""
    def __init__(self, env, side, video=False):
        self.env=env
        self.unwrapped=env.unwrapped
        self.side=int(side)
        self.records=[]
        self.video=False  # camera renderer disabled for safe headless operation
        self.frames=[]
    def step(self,action):
        b=self.unwrapped
        obj=np1(b.obj.pose.p)[:2].copy()
        before=robot_state(b)
        ref=future_masks(obj,self.side)
        result=self.env.step(action)
        after=np1(b.obj.pose.p)[:3].copy()
        info=result[4]
        visited=bool(np1(info.get('check_visited',False))[0])
        self.records.append({'state':before,'action':np.asarray(action,dtype=np.float32).reshape(-1),
                             'flow':ref,'cube_before':obj,'cube_after':after,'visited':visited})
        return result
    def __getattr__(self,k): return getattr(self.env,k)


def cartesian_action(actual, target, shape=(4,)):
    """Normalized Panda XYZ delta control plus closed gripper; no MPLib."""
    shape = tuple(shape)
    if shape not in ((4,), (1, 4)):
        raise ValueError(f"pd_ee_delta_pos action should have 4 entries, got {shape}")
    a = np.asarray(actual, dtype=np.float32).reshape(3)
    b = np.asarray(target, dtype=np.float32).reshape(3)
    delta = np.clip(0.75 * (b - a) / 0.1, -0.65, 0.65)
    return np.r_[delta, -1.0].astype(np.float32).reshape(shape)


def _servo_tcp(recorder, target, *, max_steps=45, tolerance=0.018,
               stop_when=None, abort_when=None):
    """Servo a TCP pose; optionally finish early when the cube reaches its waypoint.

    In v0.3 the TCP was sent all the way past the waypoint even when the cube
    made less progress than the TCP. The end-effector target is not the object
    state: we must measure the object on every physics step.
    """
    b = recorder.unwrapped
    target = np.asarray(target, dtype=np.float32).reshape(3)
    for _ in range(max_steps):
        if stop_when is not None and stop_when():
            return True
        xyz = np1(b.agent.tcp.pose.p)[:3]
        if np.linalg.norm(xyz - target) < tolerance:
            return True
        if len(recorder.records) >= 530:
            raise RuntimeError("Reached episode step budget (530) before TCP target")
        recorder.step(cartesian_action(xyz, target, recorder.env.action_space.shape))
        if abort_when is not None and abort_when():
            return False
    if stop_when is not None and stop_when():
        return True
    return bool(np.linalg.norm(np1(b.agent.tcp.pose.p)[:3] - target) < tolerance)


def _push_leg(recorder, waypoint_xy, *, tolerance=0.030, contact_gap=0.060):
    """Re-contacting physics push, with one explicit tolerance per waypoint.

    The old implementation accepted a corner at 4.8 cm residual error, then
    attempted the final turn from a poorly placed cube. For turns, use 3 cm;
    final-goal approach uses its own (unchanged) evaluation rule.
    """
    b = recorder.unwrapped
    cube = np1(b.obj.pose.p)[:3].copy()
    goal = np.asarray(waypoint_xy, dtype=np.float32).reshape(2)
    dxy = goal - cube[:2]
    distance = float(np.linalg.norm(dxy))
    recorder.last_push_telemetry = {}
    if distance < tolerance:
        return True
    direction = dxy / distance
    behind = cube[:2] - float(contact_gap) * direction
    tcp = np1(b.agent.tcp.pose.p)[:3]
    for pose, budget, phase in (
        (np.r_[tcp[:2], 0.16], 35, "raise"),
        (np.r_[behind, 0.16], 55, "hover"),
        (np.r_[behind, 0.028], 45, "descend"),
    ):
        if not _servo_tcp(recorder, pose, max_steps=budget):
            raise RuntimeError(
                f"Failed TCP {phase}; target={pose.tolist()}, "
                f"actual={np1(b.agent.tcp.pose.p)[:3].tolist()}"
            )

    contact_tcp = np1(b.agent.tcp.pose.p)[:3].copy()
    cube_at_contact = np1(b.obj.pose.p)[:2].copy()
    final_tcp = np.r_[behind + direction * (distance + 0.090), 0.028]
    push_first_step = len(recorder.records)

    def cube_at_waypoint():
        return np.linalg.norm(np1(b.obj.pose.p)[:2] - goal) < tolerance

    def push_stalled():
        # Do not consume 85 actions once contact is demonstrably lost.
        # Require at least 24 push actions, then test the last 12 *measured*
        # physics frames; avoid guessing from commanded TCP movements.
        steps = len(recorder.records) - push_first_step
        if steps < 24 or len(recorder.records) < 12:
            return False
        recent = np.asarray([r["cube_after"][:2]
                             for r in recorder.records[-12:]], dtype=float)
        return float(np.max(np.linalg.norm(recent - recent[0], axis=1))) < 0.0015

    reached = _servo_tcp(
        recorder, final_tcp, max_steps=85, tolerance=0.024,
        stop_when=cube_at_waypoint, abort_when=push_stalled
    )
    actual_tcp = np1(b.agent.tcp.pose.p)[:3].copy()
    cube_after_push = np1(b.obj.pose.p)[:2].copy()
    recorder.last_push_telemetry = {
        "contact_gap_m": float(contact_gap),
        "waypoint_tolerance_m": float(tolerance),
        "cube_at_contact": cube_at_contact.tolist(),
        "cube_after_push": cube_after_push.tolist(),
        "cube_push_displacement_m": round(float(np.linalg.norm(
            cube_after_push - cube_at_contact)), 5),
        "tcp_at_contact": contact_tcp.tolist(),
        "tcp_target": np.asarray(final_tcp).tolist(),
        "tcp_after_push": actual_tcp.tolist(),
        "tcp_target_error_m": round(float(np.linalg.norm(actual_tcp - final_tcp)), 5),
        "push_steps": len(recorder.records) - push_first_step,
        "servo_reached": bool(reached),
    }
    if not reached:
        print("PUSH_SERVO_INCOMPLETE",
              __import__("json").dumps(recorder.last_push_telemetry), flush=True)

    # A missed TCP servo target is not proof of global infeasibility.
    # Return the measured cube result and let the bounded outer loop try
    # a different approach distance, provided safe retraction is possible.
    retract_tcp = np1(b.agent.tcp.pose.p)[:3]
    if not _servo_tcp(recorder, np.r_[retract_tcp[:2], 0.16], max_steps=40):
        raise RuntimeError(
            "Failed retract from push; "
            f"tcp={np1(b.agent.tcp.pose.p)[:3].tolist()}"
        )
    return bool(np.linalg.norm(np1(b.obj.pose.p)[:2] - goal) < tolerance)

def scripted_rollout(env, side=1, seed=0, save_video=False, max_retries=2):
    """Closed-loop, re-contacting physical push teacher; never calls MPLib.

    Retries re-measure the cube and re-plan contact from its *actual* location.
    They are not duplicate replay of a fixed TCP trajectory. Per-attempt error
    and observed progress are recorded even when all attempts fail.
    """
    env.reset(seed=int(seed))
    b = env.unwrapped
    rec = RecorderEnv(env, side, video=False)
    error = None
    success = False
    attempts = []
    # Consecutive collinear segments share ONE physical contact stroke.
    # The original middle waypoint remains in the reference masks and in
    # the simulator's check_visited evaluation; we just do not force a
    # needless retract/descend exactly at the intermediate checkpoint.
    strokes = contact_waypoints(side)
    print("EXPERT_CONTACT_STROKES", __import__("json").dumps({
        "side": int(side), "strokes": strokes.tolist(),
        "flow_route": route_points(side).tolist()
    }), flush=True)
    try:
        if tuple(env.action_space.shape) not in ((4,), (1,4)):
            raise RuntimeError(f"Wrong controller, action shape={env.action_space.shape}; use pd_ee_delta_pos")
        close = np.array([0., 0., 0., -1.], dtype=np.float32).reshape(env.action_space.shape)
        for _ in range(8):
            rec.step(close)
        for leg, point in enumerate(strokes, start=1):
            good = False
            point = np.asarray(point, dtype=np.float32)
            for attempt in range(int(max_retries) + 1):
                before = np1(b.obj.pose.p)[:2].copy()
                start_distance = float(np.linalg.norm(before - point))
                step_before = len(rec.records)
                tcp_before = np1(b.agent.tcp.pose.p)[:3].tolist()
                # Turns must be reached more precisely than the old 0.055 m
                # tolerance; final success evaluation itself is unchanged.
                tolerance = 0.030 if leg < len(strokes) else 0.045
                # Fallback contact distances are only used on the final turn,
                # after the first strategy actually fails in simulation.
                gap = ([0.060, 0.045, 0.075][min(attempt, 2)]
                       if leg == len(strokes) else 0.060)
                rec.last_push_telemetry = {}
                try:
                    good = _push_leg(rec, point, tolerance=tolerance,
                                     contact_gap=gap)
                except Exception as exc:
                    attempts.append(dict(leg=leg, attempt=attempt+1,
                        target=point.tolist(), before=before.tolist(),
                        after=np1(b.obj.pose.p)[:2].tolist(),
                        error=f"{type(exc).__name__}: {exc}",
                        tcp_before=tcp_before,
                        tcp_after=np1(b.agent.tcp.pose.p)[:3].tolist(),
                        push_telemetry=dict(rec.last_push_telemetry),
                        steps=len(rec.records)-step_before))
                    raise
                after = np1(b.obj.pose.p)[:2].copy()
                end_distance = float(np.linalg.norm(after - point))
                diagnostic = dict(
                    leg=leg, attempt=attempt+1, target=point.tolist(),
                    before=before.tolist(), after=after.tolist(),
                    start_error_m=round(start_distance, 5),
                    end_error_m=round(end_distance, 5),
                    progress_m=round(start_distance-end_distance, 5),
                    steps=len(rec.records)-step_before,
                    tcp_before=tcp_before,
                    tcp_after=np1(b.agent.tcp.pose.p)[:3].tolist(),
                    push_telemetry=dict(rec.last_push_telemetry),
                    reached=bool(good),
                )
                attempts.append(diagnostic)
                print("PUSH_ATTEMPT", __import__("json").dumps(diagnostic), flush=True)
                if good:
                    break
                # Next iteration re-computes the contact side and push vector
                # from the measured cube position; do NOT reuse the old target.
            if not good:
                raise RuntimeError(
                    f"Cube missed waypoint {point.tolist()} after "
                    f"{int(max_retries)+1} measured attempts; "
                    f"measured={np1(b.obj.pose.p)[:2].tolist()}, "
                    f"steps={len(rec.records)}"
                )
        success = bool(np1(b.evaluate()["success"])[0])
        if not success:
            error = "Final physics task evaluation failed"
    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}"
    traj = {}
    if rec.records:
        for key in ("state","action","flow","cube_before","cube_after","visited"):
            traj[key] = np.stack([r[key] for r in rec.records])
    return dict(success=success, error=error, side=int(side), seed=int(seed),
                controller="pd_ee_delta_pos", trajectory=traj, frames=[],
                push_attempts=attempts, max_retries=int(max_retries),
                contact_strokes=strokes.tolist(),
                end_cube=np1(b.obj.pose.p)[:3].tolist(),
                check_visited=bool(np1(b.evaluate()["check_visited"])[0]))


def save_episode(result,path,trajectory_video=False):
    from pathlib import Path
    import json
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=result['trajectory']
    if data:
        np.savez_compressed(path, **data, controller=np.asarray('pd_ee_delta_pos'), side=np.array(result['side'],dtype=np.int8),
                            seed=np.array(result['seed'],dtype=np.int32),
                            success=np.array(result['success'],dtype=np.bool_))
    meta={k:v for k,v in result.items() if k not in ('trajectory','frames')}
    path.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    if trajectory_video and data and 'cube_after' in data:
        from .plotting import trajectory_video as write_trajectory_video
        try:
            write_trajectory_video(data['cube_after'],path.with_suffix('.mp4'),
                                   title=f"Expert seed={result['seed']} side={result['side']} (actual cube XY, not RGB)")
        except Exception as exc:
            print('trajectory video unavailable:',repr(exc),flush=True)
