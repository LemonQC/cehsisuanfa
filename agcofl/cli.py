from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def make_env(side,video=False):
    # Never initialize Vulkan even when --video is supplied: --video means
    # matplotlib Agg trajectory animation, NOT a camera RGB recording.
    # This avoids hard native segfaults in SSH X11 forwarding sessions.
    import gymnasium as gym
    import mani_skill.envs
    from . import env as _register  # noqa: F401
    kwargs={'obs_mode':'state_dict','control_mode':'pd_ee_delta_pos','sim_backend':'physx_cpu',
            'render_backend':'none',  # ALWAYS no Vulkan; video is offline 2D trajectory movie
            'route_side':int(side),'reward_mode':'dense','num_envs':1}
    kwargs['render_mode'] = None
    return gym.make('AGCOFL-TwoRoute-v1',**kwargs)


def run_smoke(args):
    from .teacher import robot_state
    from .representation import future_masks,mask_route_side
    for side in [-1,1]:
        print(f'SMOKE_CREATE side={side}',flush=True)
        env=make_env(side)
        try:
            print('SMOKE_RESET',flush=True)
            obs,info=env.reset(seed=args.seed)
            base=env.unwrapped
            state=robot_state(base)
            masks=future_masks(np.asarray(base.obj.pose.p.cpu().numpy()).ravel()[:2],side)
            label=mask_route_side(masks)
            if label!=side:raise AssertionError(f'Flow mask lost route label: {label} != {side}')
            action=np.zeros(env.action_space.shape,dtype=np.float32)
            # A neutral action confirms end-to-end env.step. Not a valid push.
            print('SMOKE_STEP',flush=True)
            nextobs,rew,done,trunc,info=env.step(action)
            print(json.dumps({'side':side,'state_dim':len(state),'action_shape':env.action_space.shape,
                              'mask_shape':masks.shape,'detected_flow_side':label,
                              'reward':float(np.asarray(rew).reshape(-1)[0])}))
        finally:env.close()
    print('SMOKE_OK: physics stepping and reference masks (render_backend=none)',flush=True)


def run_collect(args):
    from .teacher import scripted_rollout,save_episode
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.max_retries < 0 or args.max_retries > 5:
        raise ValueError('--max-retries must be between 0 and 5')
    status=[]
    for side in (-1,1):
        env=make_env(side,video=args.video)
        try:
            print('EXPERT_CONTROLLER pd_ee_delta_pos, no MPLib',flush=True)
            for i in range(args.episodes):
                seed=args.seed+i
                result=scripted_rollout(env,side=side,seed=seed,save_video=False,max_retries=args.max_retries)
                save_episode(result,out/f'route_{"upper" if side==1 else "lower"}_{seed:06d}.npz', trajectory_video=args.video)
                rec={'side':side,'seed':seed,'success':result['success'],
                     'steps':len(result['trajectory'].get('state',[])),
                     'error':result['error'],'visited':result['check_visited'],
                     'push_attempts':len(result.get('push_attempts',[]))}
                print(json.dumps(rec)); status.append(rec)
        finally:env.close()
    summary={'attempted':len(status),'success':sum(r['success'] for r in status),
             'failures':[r for r in status if not r['success']]}
    (out/'collection_summary.json').write_text(json.dumps(summary,indent=2))
    print('EXPERT_COLLECTION:',json.dumps({k:v for k,v in summary.items() if k!='failures'}))
    if not any(r['success'] for r in status):
        raise RuntimeError('All physical teacher rollouts failed. See JSON error logs; do NOT train on failures.')


def run_train(args):
    from .learning import train_bc
    train_bc(args.data,args.ckpt,args.epochs,batch=args.batch)


def run_gate(args):
    from .learning import load_bc,rollout_bc
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    net,device=load_bc(args.ckpt)
    rows=[]
    # Same seed and start state; only route reference masks change.
    for seed in range(args.seed,args.seed+args.episodes):
        for side in [-1,1]:
            env=make_env(side,args.video)
            try:
                result=rollout_bc(env,net,device,side,seed,max_steps=args.max_steps,save_video=args.video)
            finally:env.close()
            if args.video:
                from .plotting import trajectory_video
                try:
                    trajectory_video(np.asarray([r['cube'] for r in result['history']]),
                                     out/f'bc_seed{seed}_side{side}.mp4',
                                     title=f'Gate A seed={seed} side={side} (actual cube XY, not RGB)')
                except Exception as exc:
                    print('trajectory video unavailable:',repr(exc),flush=True)
            row={k:v for k,v in result.items() if k not in ('history','frames')}
            row['trajectory']=str(out/f'bc_seed{seed}_side{side}.json')
            (out/f'bc_seed{seed}_side{side}.json').write_text(json.dumps(result['history'],indent=1))
            print(json.dumps(row));rows.append(row)
    paired_initial_states_ok=all(np.allclose(
        next(r['initial_state'] for r in rows if r['seed']==s and r['side']==-1),
        next(r['initial_state'] for r in rows if r['seed']==s and r['side']==1), atol=1e-6)
        for s in range(args.seed,args.seed+args.episodes))
    report={'episodes':len(rows),'paired_initial_states_ok':paired_initial_states_ok,
            'success_rate':np.mean([r['success'] for r in rows]).item(),
            'check_visit_rate':np.mean([r['visited'] for r in rows]).item(),
            'successful_upper':sum(r['success'] for r in rows if r['side']==1),
            'successful_lower':sum(r['success'] for r in rows if r['side']==-1),
            'gateA_passed':paired_initial_states_ok and all(any(r['success'] and r['side']==side for r in rows) for side in (-1,1)),
            'per_episode':rows}
    (out/'gateA_report.json').write_text(json.dumps(report,indent=2))
    try:
        from .plotting import plot_gate_a
        print('trajectory figure:',plot_gate_a(out))
    except Exception as exc:
        print('figure could not be drawn:',repr(exc))
    print('GATE_A_REPORT',json.dumps({k:v for k,v in report.items() if k!='per_episode'}))
    if not report['gateA_passed']:
        print('GATE_A_NOT_PASSED: Do not proceed to high-level supervision training.',file=sys.stderr)


def run_doctor(args):
    """First test official physics, then custom environment. No rendering."""
    import platform
    print('PYTHON',sys.version.split()[0], 'OS',platform.platform(),flush=True)
    import torch, gymnasium as gym
    import sapien, mani_skill.envs
    print('TORCH',torch.__version__,'CUDA',torch.cuda.is_available(),flush=True)
    print('STOCK_CREATE',flush=True)
    env=gym.make('PushCube-v1',obs_mode='state_dict',control_mode='pd_ee_delta_pos',
                 sim_backend='physx_cpu',render_backend='none',render_mode=None,num_envs=1)
    try:
        print('STOCK_RESET',flush=True)
        env.reset(seed=args.seed)
        action=np.zeros(env.action_space.shape,dtype=np.float32)
        print('STOCK_STEP',flush=True)
        env.step(action)
    finally:
        env.close()
    print('STOCK_PHYSICS_OK',flush=True)
    run_smoke(args)


def run_losscheck(_):
    import torch
    from .partial_supervision import torch_segment_loss
    logits=torch.zeros((8,32,32),requires_grad=True)
    masks=torch.zeros((8,32,32)); masks[:,13:18,13:18]=1
    for mode in ('full','anchor','end'):
        if logits.grad is not None:logits.grad.zero_()
        loss=torch_segment_loss(logits,masks,mode)
        loss.backward(retain_graph=True)
        grad=logits.grad.detach().abs().sum(dim=(1,2)).cpu().numpy()
        print(mode,'loss=',float(loss.item()),'direct_gradient_frames=',np.flatnonzero(grad>1e-9).tolist())
    print('LOSS_CHECK_OK')


def main():
    p=argparse.ArgumentParser(description='AG-COFL feasibility suite (ManiSkill3 + GPU PyTorch)')
    sub=p.add_subparsers(dest='cmd',required=True)
    dr=sub.add_parser('doctor');dr.add_argument('--seed',type=int,default=42);dr.set_defaults(func=run_doctor)
    sm=sub.add_parser('smoke');sm.add_argument('--seed',type=int,default=42);sm.set_defaults(func=run_smoke)
    co=sub.add_parser('collect');co.add_argument('--episodes',type=int,default=10);co.add_argument('--seed',type=int,default=100);
    co.add_argument('--out',default='outputs/ee_v040_episodes');co.add_argument('--video',action='store_true',help='Save offline 2D cube-trajectory MP4; does not enable GPU/GLX rendering');
    co.add_argument('--max-retries',type=int,default=2,help='Additional measured re-contact attempts for each missed intermediate waypoint (default 2)');co.set_defaults(func=run_collect)
    tr=sub.add_parser('train-bc');tr.add_argument('--data',default='outputs/ee_v030_episodes');tr.add_argument('--ckpt',default='outputs/models/flow_bc_ee_v030.pt');
    tr.add_argument('--epochs',type=int,default=80);tr.add_argument('--batch',type=int,default=256);tr.set_defaults(func=run_train)
    ev=sub.add_parser('gate-a');ev.add_argument('--ckpt',default='outputs/models/flow_bc_ee_v030.pt');ev.add_argument('--out',default='outputs/gate_a');
    ev.add_argument('--episodes',type=int,default=5);ev.add_argument('--seed',type=int,default=9000);ev.add_argument('--max-steps',type=int,default=500);
    ev.add_argument('--video',action='store_true',help='Save offline 2D cube-trajectory MP4; does not enable GPU/GLX rendering');ev.set_defaults(func=run_gate)
    los=sub.add_parser('test-loss');los.set_defaults(func=run_losscheck)
    args=p.parse_args();args.func(args)

if __name__=='__main__':main()
