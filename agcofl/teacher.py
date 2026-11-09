"""Headless Panda physically-controlled teacher without MPLib.

The native ManiSkill pd_ee_delta_pos controller supplies the IK internally.
Failed or incomplete trajectories are recorded as failures.
"""
from __future__ import annotations
import numpy as np
import torch
from .representation import future_masks, route_points, mask_route_side


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


def _servo_tcp(recorder, target, *, max_steps=45, tolerance=0.018):
    b = recorder.unwrapped
    target = np.asarray(target, dtype=np.float32).reshape(3)
    for _ in range(max_steps):
        xyz = np1(b.agent.tcp.pose.p)[:3]
        if np.linalg.norm(xyz - target) < tolerance:
            return True
        if len(recorder.records) >= 530:
            raise RuntimeError("Reached episode step budget before TCP target")
        recorder.step(cartesian_action(xyz, target, recorder.env.action_space.shape))
    return bool(np.linalg.norm(np1(b.agent.tcp.pose.p)[:3] - target) < tolerance)


def _push_leg(recorder, waypoint_xy):
    b = recorder.unwrapped
    cube = np1(b.obj.pose.p)[:3]
    goal = np.asarray(waypoint_xy, dtype=np.float32).reshape(2)
    dxy = goal - cube[:2]
    distance = float(np.linalg.norm(dxy))
    if distance < 0.040:
        return True
    direction = dxy / distance
    behind = cube[:2] - 0.060 * direction
    tcp = np1(b.agent.tcp.pose.p)[:3]
    for pose, budget, phase in (
        (np.r_[tcp[:2], 0.16], 35, "raise"),
        (np.r_[behind, 0.16], 55, "hover"),
        (np.r_[behind, 0.028], 45, "descend"),
    ):
        if not _servo_tcp(recorder, pose, max_steps=budget):
            raise RuntimeError(f"Failed TCP {phase}; target={pose.tolist()}, actual={np1(b.agent.tcp.pose.p)[:3].tolist()}")
    final_tcp = np.r_[behind + direction * (distance + 0.090), 0.028]
    if not _servo_tcp(recorder, final_tcp, max_steps=85, tolerance=0.024):
        raise RuntimeError(f"Failed contact push towards waypoint={goal.tolist()}")
    tcp = np1(b.agent.tcp.pose.p)[:3]
    if not _servo_tcp(recorder, np.r_[tcp[:2], 0.16], max_steps=40):
        raise RuntimeError("Failed retract from push")
    return bool(np.linalg.norm(np1(b.obj.pose.p)[:2] - goal) < 0.055)


def scripted_rollout(env, side=1, seed=0, save_video=False, max_retries=0):
    """Real physics Cartesian-IK push teacher; never calls mplib.Planner."""
    env.reset(seed=int(seed))
    b = env.unwrapped
    rec = RecorderEnv(env, side, video=False)
    error = None
    success = False
    try:
        if tuple(env.action_space.shape) not in ((4,), (1,4)):
            raise RuntimeError(f"Wrong controller, action shape={env.action_space.shape}; use pd_ee_delta_pos")
        close = np.array([0., 0., 0., -1.], dtype=np.float32).reshape(env.action_space.shape)
        for _ in range(8):
            rec.step(close)
        for point in route_points(side)[1:]:
            good = False
            for _ in range(int(max_retries) + 1):
                good = _push_leg(rec, point)
                if good:
                    break
            if not good:
                raise RuntimeError(f"Cube missed waypoint {np.asarray(point).tolist()}; measured={np1(b.obj.pose.p)[:2].tolist()}")
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
