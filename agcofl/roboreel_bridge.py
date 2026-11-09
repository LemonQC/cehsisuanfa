"""Official RoboReel real-video + physical Panda PushCube-v2 diagnostic.

This is a privileged-state scripted controller, NOT learned Full/Anchor/End
training or human-to-robot transfer. The video_motion mode changes only speed
using measured human-video statistics. Headless CPU PhysX avoids GLX.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from .real_video_flow import extract_video_flow


def np1(x):
    if hasattr(x,'detach'): x=x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def import_benchmark(root):
    root=Path(root).resolve()
    if not (root/'src'/'benchmark.py').is_file():
        raise FileNotFoundError(f'Clone official RoboReel into {root} first')
    sys.path.insert(0,str(root/'src'))
    import benchmark
    return benchmark


def delta_pose_action(actual,target,shape,gain=1.0):
    if tuple(shape) not in ((7,),(1,7)):
        raise ValueError(f'Official pd_ee_delta_pose needs 7D actions, got {shape}')
    actual=np.asarray(actual,np.float32).reshape(3)
    target=np.asarray(target,np.float32).reshape(3)
    xyz=np.clip((target-actual)/.1 * .75 * float(gain),-.55,.55)
    return np.r_[xyz,0.,0.,0.,-1.].astype(np.float32).reshape(shape)


class DiagnosticPushTeacher:
    """Oracle state-based pushing baseline to validate the simulation bridge."""
    def __init__(self,env,gain=1.,max_attempts=2):
        self.env=env
        self.b=env.unwrapped
        self.shape=env.action_space.shape
        self.gain=float(gain)
        self.phase='above_contact'
        self.attempt=0
        self.phase_steps=0
        self.max_attempts=int(max_attempts)
        self.trace=[]
        self.last_cube=None
        self.stale_steps=0
        self.failure=None

    def robot_xyz(self):
        agent=self.b.agent
        tcp=getattr(agent,'tcp_pose',None)
        if tcp is None: tcp=agent.tcp.pose
        return np1(tcp.p)[:3].astype(np.float32)

    def cube_xyz(self): return np1(self.b.obj.pose.p)[:3].astype(np.float32)
    def goal_xyz(self): return np1(self.b.goal_pos)[:3].astype(np.float32)

    def _phase(self,name):
        self.phase=name
        self.phase_steps=0

    def act(self):
        cube=self.cube_xyz();goal=self.goal_xyz();tcp=self.robot_xyz()
        direction=goal[:2]-cube[:2]
        distance=float(np.linalg.norm(direction))
        unit=direction/max(distance,1e-7)
        half=float(getattr(self.b,'cube_half_size',.02))
        behind=cube[:2]-unit*(half+.04)
        z_contact=float(cube[2]+half*.4)
        hover=max(.13,float(cube[2]+.10))
        if self.phase=='above_contact':
            target=np.r_[behind,hover]
            if np.linalg.norm(tcp-target)<.03:self._phase('descend')
        elif self.phase=='descend':
            target=np.r_[behind,z_contact]
            if np.linalg.norm(tcp-target)<.020:self._phase('push')
        elif self.phase=='push':
            target=np.r_[goal[:2]+unit*(half+.05),z_contact]
            if self.last_cube is not None:
                movement=float(np.linalg.norm(cube[:2]-self.last_cube[:2]))
                self.stale_steps=self.stale_steps+1 if movement<.00035 else 0
            if self.stale_steps>=18 and self.phase_steps>=24:
                self._phase('recover');self.attempt+=1
            if self.attempt>=self.max_attempts:
                self.failure='max_no_progress_attempts';self._phase('failed')
        elif self.phase=='recover':
            target=np.r_[tcp[:2],hover]
            if np.linalg.norm(tcp-target)<.03:
                self._phase('above_contact');self.stale_steps=0
        elif self.phase=='failed':target=tcp
        else:raise AssertionError(self.phase)
        self.phase_steps+=1
        self.last_cube=cube.copy()
        action=delta_pose_action(tcp,target,self.shape,gain=self.gain)
        self.trace.append(dict(phase=self.phase,cube=cube.tolist(),tcp=tcp.tolist(),
            goal_distance_m=distance,attempt=self.attempt,
            action=np.asarray(action).reshape(-1).tolist()))
        return action


def evaluate(root,episodes,out,method='fixed',max_steps=250,min_coverage=.5,no_video=False):
    # This switch affects only whether official *human MP4* is required.
    # The physics scene and controller are still actual ManiSkill execution.
    if no_video and method!='fixed':
        raise ValueError('--no-video requires --mode fixed; no human video statistics exist')
    benchmark=import_benchmark(root)
    import gymnasium as gym
    import mani_skill.envs
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    choices=benchmark.load_episodes('PushCube-v2',episode_ids=list(episodes))
    rows=[]
    for ep in choices:
        epout=out/f'episode_{ep.episode_id:04d}'
        epout.mkdir(parents=True,exist_ok=True)
        if no_video:
            demo_id=int(benchmark.HUMAN_DEMO_IDS[
                ep.episode_id % benchmark.EPISODES_PER_RUN])
            video_meta=dict(
                accepted=False,coverage=0.,timing_gain=1.,
                missing_video=True,reason='explicit --no-video physics-only diagnostic')
            print('OFFICIAL_PHYSICS_ONLY_NO_HUMAN_VIDEO',
                json.dumps(dict(episode=ep.episode_id,expected_demo_id=demo_id)),
                flush=True)
        else:
            demo_id,vids=benchmark.human_demo_for_episode('PushCube-v2',ep.episode_id)
            if 'front' not in vids:
                raise RuntimeError('Missing authentic front human MP4; use single-file mirror downloader or --no-video.')
            _,video_meta=extract_video_flow(vids['front'],epout,min_coverage=min_coverage)
            if method=='video_motion' and not video_meta['accepted']:
                raise RuntimeError(
                    f'Blue-object candidate coverage {video_meta["coverage"]:.2%} inadequate. '
                    f'Inspect {epout}/human_mask_overlay.png')
            print('REAL_DEMO_LOADED',
                json.dumps(dict(episode=ep.episode_id,demo_id=demo_id,
                    mode=method,coverage=video_meta['coverage'])),
                flush=True)
        gain=video_meta['timing_gain'] if method=='video_motion' else 1.
        env=None
        try:
            env=gym.make('PushCube-v2',obs_mode='state',num_envs=1,
                robot_uids=benchmark.ROBOT_UID,control_mode='pd_ee_delta_pose',
                sim_backend='physx_cpu',render_mode=None,render_backend='none',
                max_episode_steps=1000,distract=False)
            benchmark.reset_to_episode(env,ep)
            policy=DiagnosticPushTeacher(env,gain=gain)
            success=False
            for step in range(max_steps):
                action=policy.act()
                _,_,terminated,truncated,info=env.step(action)
                success=bool(np1(info['success'])[0])
                if success or bool(np1(terminated)[0]) or bool(np1(truncated)[0]):
                    break
            result=dict(episode_id=ep.episode_id,human_demo_id=demo_id,success=success,
                steps=len(policy.trace),policy='diagnostic_privileged_state_push',
                method=method,video_gain=gain,video_meta=video_meta,
                final_cube=policy.cube_xyz().tolist(),final_goal=policy.goal_xyz().tolist(),
                final_goal_error_m=float(np.linalg.norm(policy.cube_xyz()[:2]-policy.goal_xyz()[:2])),
                failure=policy.failure,control_mode='pd_ee_delta_pose',
                obs_mode='state',render_backend='none')
            (epout/'physical_trace.json').write_text(json.dumps(policy.trace))
            np.save(epout/'actions.npy',np.asarray([p['action'] for p in policy.trace],np.float32))
            (epout/'episode_result.json').write_text(json.dumps(result,indent=2))
            rows.append(result)
            print('PHYSICAL_RESULT',json.dumps(dict(episode=ep.episode_id,success=success,
                steps=len(policy.trace),goal_error=result['final_goal_error_m'],
                failure=policy.failure)),flush=True)
        finally:
            if env is not None:env.close()
    report=dict(
        type='OFFICIAL_ROBOREEL_PHYSICS_ONLY_NO_HUMAN_VIDEO'
        if no_video else 'REAL_VIDEO_REAL_SIM_DIAGNOSTIC_NOT_PAPER',
        task='PushCube-v2',
        mode=method,episodes=len(rows),successes=sum(r['success'] for r in rows),
        success_rate=float(np.mean([r['success'] for r in rows])) if rows else None,
        results=rows,warning='Privileged-state scripted controller; no learned cross-behavior policy. '
        'In physics-only mode NO HUMAN VIDEO is used. Otherwise human video changes only scalar timing '
        'in video_motion. Full/Anchor/End is NOT evaluated here.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('REAL_PHYSICS_DIAGNOSTIC_COMPLETE',
        json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description='RoboReel real human video + Panda physics diagnostic')
    sub=p.add_subparsers(dest='cmd',required=True)
    ins=sub.add_parser('inspect-video')
    ins.add_argument('--video',required=True)
    ins.add_argument('--out',default='outputs/roboreel_video_inspect')
    ev=sub.add_parser('evaluate')
    ev.add_argument('--roboreel-root',default='third_party/roboreel')
    ev.add_argument('--episodes',type=int,nargs='+',default=[0])
    ev.add_argument('--out',default='outputs/roboreel_diagnostic')
    ev.add_argument('--mode',choices=['fixed','video_motion'],default='fixed')
    ev.add_argument('--max-steps',type=int,default=250)
    ev.add_argument('--no-video',action='store_true',
                    help='Verify saved RoboReel physical scene WITHOUT requiring human MP4; not LfO evidence')
    args=p.parse_args(argv)
    if args.cmd=='inspect-video':
        _,metadata=extract_video_flow(args.video,args.out)
        print('VIDEO_FLOW_DIAGNOSTIC',json.dumps(metadata),flush=True)
    else:evaluate(args.roboreel_root,args.episodes,args.out,args.mode,
                  args.max_steps,no_video=args.no_video)


if __name__=='__main__':main()
