"""Physical Panda scripted rollout with official ManiSkill/mplib solver.

The planner is used only to collect expert demonstrations; learned controller
gets joint actions. Failed or incomplete trajectories are recorded as failures.
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


def _planner_move(planner, target, quat, approach=False):
    import sapien
    p=sapien.Pose(p=np.asarray(target,dtype=np.float64),q=np.asarray(quat,dtype=np.float64))
    if approach:
        res=planner.move_to_pose_with_RRTConnect(p)
        if isinstance(res,int) and res == -1:
            res=planner.move_to_pose_with_screw(p)
    else:
        res=planner.move_to_pose_with_screw(p)
    if isinstance(res,int) and res == -1:
        raise RuntimeError(f'motion planning failed for TCP target {target}')
    return res


def scripted_rollout(env, side=1, seed=0, save_video=False, max_retries=1):
    """Four physical pushing legs via PandaArmMotionPlanningSolver.

    Does NOT use env routing labels as input to learned policy: side is expert-only.
    Uses RRT for free-space waypoints, screw motion during contact pushing.
    """
    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    env.reset(seed=int(seed))
    base=env.unwrapped
    recorder=RecorderEnv(env,side,video=save_video)
    planner=None
    error=None
    try:
        planner=PandaArmMotionPlanningSolver(recorder, debug=False,vis=False,
                    base_pose=base.agent.robot.pose,visualize_target_grasp_pose=False,
                    print_env_info=False,joint_vel_limits=.65,joint_acc_limits=.65)
        # Collision cloud for the central barrier, used by free-space motion planning.
        # It can be omitted if the installed mplib rejects this obstacle.
        try:
            planner.add_box_collision(np.array([.10,.154,.09]),
                                      sapien.Pose(p=[0,0,.045]))
        except Exception as exc:
            print('warning: obstacle cloud rejected:',repr(exc))
        planner.close_gripper(t=6)
        quat=np1(base.agent.tcp.pose.q)[:4]
        for target in route_points(side)[1:]:
            for attempt in range(max_retries+1):
                cube=np1(base.obj.pose.p)[:3].copy()
                diff=np.asarray(target)-cube[:2]
                dist=np.linalg.norm(diff)
                if dist < .045: break
                direction=diff/max(dist,1e-8)
                contact_xy=cube[:2]-direction*.055
                above=np.r_[contact_xy,.125]
                at_contact=np.r_[contact_xy,.025]
                # Put gripper behind the cube, keep jaws closed and on tabletop.
                _planner_move(planner,above,quat,approach=True)
                _planner_move(planner,at_contact,quat)
                # Push slightly beyond object goal to compensate gripper/cube offset.
                push_target=at_contact+np.r_[direction*(dist+.025),0]
                _planner_move(planner,push_target,quat)
                _planner_move(planner,np.r_[push_target[:2],.125],quat)
                actual=np1(base.obj.pose.p)[:2]
                if np.linalg.norm(actual-target)<.055: break
        info=base.evaluate()
        complete=bool(np1(info['success'])[0])
    except Exception as exc:
        complete=False
        error=f'{type(exc).__name__}: {exc}'
    finally:
        if planner is not None:
            try: planner.close()
            except Exception: pass
    trajectory={}
    if recorder.records:
        for k in ('state','action','flow','cube_before','cube_after','visited'):
            trajectory[k]=np.stack([r[k] for r in recorder.records])
    return {'success':complete,'error':error,'side':int(side),'seed':int(seed),
            'trajectory':trajectory,'frames':recorder.frames,
            'end_cube':np1(base.obj.pose.p)[:3].tolist(),
            'check_visited':bool(np1(base.evaluate()['check_visited'])[0])}


def save_episode(result,path,trajectory_video=False):
    from pathlib import Path
    import json
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=result['trajectory']
    if data:
        np.savez_compressed(path, **data, side=np.array(result['side'],dtype=np.int8),
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
