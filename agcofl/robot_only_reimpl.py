"""B01 Transformer BC / B02 DDPM: RoboReel RGB robot-only REIMPLEMENTATIONS.

Strict split/provenance, common RGB/proprio/actions, explicit DEV vs formal gates.
These are not original-paper checkpoints, nor an official RoboReel source release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'outputs/v120_baselines/official_dataset/robot_trajectories/no_distraction/push_cube'
CACHE_ROOT = ROOT / 'outputs/b01_b02_robot_only_cache'
RUN_ROOT = ROOT / 'outputs/research_runs/b01_b02_robot_only'
ART_QPOS_OFFSET = 13
ACTION_DIM = 8
CAMERA = 'base_camera'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_new_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def pack_state(art):
    """Upstream process_training_robot_trajectories: qpos columns 13:22."""
    q = np.asarray(art, dtype=np.float32)[..., ART_QPOS_OFFSET:ART_QPOS_OFFSET+9]
    if q.shape[-1] != 9 or not np.isfinite(q).all():
        raise ValueError('INVALID_PANDA_ARTICULATION_QPOS')
    return np.concatenate([q[..., :7], q[..., 7:9].mean(-1, keepdims=True)], -1)


def live_state(obs):
    q = obs['agent']['qpos']
    if hasattr(q, 'detach'):
        q = q.detach().cpu().numpy()
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if q.size < 9:
        raise ValueError('LIVE_ROBOT_QPOS_SHORT')
    return np.r_[q[:7], q[7:9].mean()].astype(np.float32)


def live_rgb(obs):
    entry = obs['sensor_data'][CAMERA]
    img = entry.get('Color', entry.get('rgb'))
    if img is None:
        raise KeyError('ROBOREEL_BASE_CAMERA_RGB_MISSING')
    if hasattr(img, 'detach'):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    img = img[..., :3]
    if img.dtype != np.uint8:
        img = (img * (255 if img.max() <= 1 else 1)).clip(0, 255).astype(np.uint8)
    return img


def resize_rgb(frame, size):
    from PIL import Image
    return np.asarray(Image.fromarray(frame).resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def _real_file(path):
    path = Path(path)
    return path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))


def _decode(path, side):
    import imageio.v2 as imageio
    reader = imageio.get_reader(str(path), format='ffmpeg')
    try:
        frames = [resize_rgb(np.asarray(f, dtype=np.uint8), side) for f in reader]
    finally:
        reader.close()
    return np.stack(frames) if frames else np.empty((0, side, side, 3), dtype=np.uint8)


def preprocess(data_root=DATA_ROOT, cache_root=CACHE_ROOT, side=96, seed=42,
               min_demos=64, allow_small_dev=False):
    """Train IDs only; persistent non-overwriting arrays, split by demonstration."""
    import h5py
    data_root, cache_root = Path(data_root).absolute(), Path(cache_root).absolute()
    if cache_root.exists() and any(cache_root.iterdir()):
        raise FileExistsError('CACHE_NOT_EMPTY_REFUSE_OVERWRITE: choose a new --cache-root')
    if not data_root.is_dir() or any(p.is_symlink() for p in (data_root, *data_root.parents)):
        raise FileNotFoundError('AUTHENTIC_ROBOT_TRAIN_ROOT_MISSING_OR_SYMLINK')
    if side < 32 or side > 256 or min_demos < 2:
        raise ValueError('INVALID_PREPROCESS_SETTINGS')
    # Stage, reject invalid data before publishing a valid manifest.
    rows = []
    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir() or folder.is_symlink() or not folder.name.isascii() or not folder.name.isdigit() or len(folder.name) != 4 or int(folder.name) >= 80:
            continue  # never read 0080..0099 EVAL
        hs = sorted(x for x in folder.glob('*.h5') if '.state.' not in x.name)
        if len(hs) != 1 or not all(_real_file(p) for p in (hs[0], folder/'base_camera.mp4', folder/'hand_camera.mp4')):
            continue
        with h5py.File(hs[0], 'r') as h:
            if 'traj_0/actions' not in h or 'traj_0/env_states/articulations' not in h:
                continue
            a = np.asarray(h['traj_0/actions'], np.float32)
            art_group = h['traj_0/env_states/articulations']
            if 'panda_wristcam_hd' not in art_group:
                raise ValueError('PANDA_WRISTCAM_ARTICULATION_NOT_FOUND: '+folder.name)
            s = pack_state(np.asarray(art_group['panda_wristcam_hd']))
        if a.ndim != 2 or a.shape[1] != ACTION_DIM or len(a) < 5 or len(s) < len(a) or not np.isfinite(a).all():
            raise ValueError('ACTION_STATE_SCHEMA_OR_LENGTH_MISMATCH: '+folder.name)
        frames = _decode(folder/'base_camera.mp4', side)
        if len(frames) not in (len(a), len(a)+1):
            raise ValueError('VIDEO_ACTION_FRAME_COUNT_MISMATCH: '+folder.name+f' frames={len(frames)} actions={len(a)}')
        cache_root.mkdir(parents=True, exist_ok=True)
        out = cache_root / folder.name
        out.mkdir(exist_ok=False)
        np.save(out/'rgb.npy', frames[:len(a)])
        np.save(out/'state.npy', s[:len(a)])
        np.save(out/'action.npy', a)
        rows.append({'demo_id':folder.name, 'steps':len(a), 'h5_sha256':sha256(hs[0]),
                     'base_camera_sha256':sha256(folder/'base_camera.mp4'),
                     'hand_camera_sha256':sha256(folder/'hand_camera.mp4')})
    ids = [r['demo_id'] for r in rows]
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    pivot = int(0.8 * len(ids))
    train_ids, val_ids = sorted(shuffled[:pivot]), sorted(shuffled[pivot:])
    if len(ids) < min_demos and not allow_small_dev:
        raise ValueError(f'AUTHENTIC_TRAIN_DATA_SHORTFALL_{len(ids)}_OF_{min_demos}; use --allow-small-dev for engineering-only smoke')
    if not train_ids or not val_ids:
        raise ValueError('NEED_TWO_OR_MORE_VALID_DEMONSTRATIONS')
    result = {'schema':'TRO1_ROBOT_ONLY_CACHE_V1', 'task':'push_cube', 'camera':CAMERA,
              'control_mode':'pd_joint_pos', 'articulation_offset':ART_QPOS_OFFSET,
              'seed':seed, 'side':side, 'train':train_ids, 'validation':val_ids,
              'demos':rows, 'required_demos':min_demos,
              'label':'SMALL_DEV_DATA_ONLY' if len(ids)<min_demos else 'TRAIN_DATA_READY_NOT_TRAINED',
              'not_verified':['UPSTREAM_PROVENANCE_EXTERNAL', 'LIVE_ENV_COMPATIBILITY', 'POLICY_PHYSX_SUCCESS']}
    write_new_json(cache_root/'manifest.json',result)
    return result


class RobotDataset(Dataset):
    def __init__(self, root, split, horizon, stats=None):
        self.root=Path(root)
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.horizon=horizon
        self.data={}
        self.indices=[]
        for did in self.manifest[split]:
            folder=self.root/did
            data=[np.load(folder/(n+'.npy'), mmap_mode='r') for n in ('rgb','state','action')]
            if len({len(v) for v in data})!=1 or data[2].shape[1]!=ACTION_DIM:
                raise ValueError('CORRUPTED_PREPROCESSED_TRAIN_TRIPLET')
            self.data[did]=data
            self.indices.extend((did,t) for t in range(len(data[2])-horizon+1))
        if not self.indices:
            raise ValueError('NO_VALID_ACTION_CHUNKS')
        self.stats=stats

    def __len__(self):return len(self.indices)

    def __getitem__(self,i):
        did,t=self.indices[i]
        frames,s,a=self.data[did]
        x=np.stack([frames[max(0,t-1)],frames[t]],0).copy()
        state=np.array(s[t],dtype=np.float32)
        actions=np.array(a[t:t+self.horizon],dtype=np.float32)
        if self.stats:
            state=(state-self.stats['state_mean'])/self.stats['state_std']
            actions=(actions-self.stats['action_mean'])/self.stats['action_std']
        return (torch.from_numpy(x).permute(0,3,1,2).float()/255.0,
                torch.from_numpy(state),torch.from_numpy(actions))


def train_stats(ds):
    # Only demo IDs in ds.manifest['train']; no frame-level leakage.
    if any(did not in ds.manifest['train'] for did in ds.data):
        raise ValueError('NORMALIZATION_MUST_BE_TRAIN_ONLY')
    states=np.concatenate([np.asarray(v[1]) for v in ds.data.values()],0)
    actions=np.concatenate([np.asarray(v[2]) for v in ds.data.values()],0)
    def calc(x):return x.mean(0).astype(np.float32),np.maximum(x.std(0),1e-3).astype(np.float32)
    sm,ss=calc(states);am,ass=calc(actions)
    return dict(state_mean=sm,state_std=ss,action_mean=am,action_std=ass)


class Encoder(nn.Module):
    def __init__(self,dim=128):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),
                               nn.Conv2d(64,128,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,dim))
        self.state=nn.Sequential(nn.Linear(8,dim),nn.LayerNorm(dim))
        self.pos=nn.Parameter(torch.zeros(1,2,dim))
        self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,4,dim*2,batch_first=True),num_layers=1,enable_nested_tensor=False)
    def forward(self,rgb,state):
        b,t,c,h,w=rgb.shape
        z=self.cnn(rgb.reshape(b*t,c,h,w)).reshape(b,t,-1)
        z=self.temporal(z+self.pos).mean(1)
        return z+self.state(state)


class TransformerBC(nn.Module):
    def __init__(self,horizon=4,dim=128):
        super().__init__()
        self.encoder=Encoder(dim)
        self.horizon=horizon
        self.query=nn.Parameter(torch.randn(1,horizon,dim)*0.02)
        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(dim,4,dim*2,batch_first=True),num_layers=2)
        self.head=nn.Linear(dim,ACTION_DIM)
    def forward(self,rgb,state):
        z=self.encoder(rgb,state).unsqueeze(1)
        return self.head(self.decoder(self.query.expand(len(z),-1,-1),z))


class DenoiseMLP(nn.Module):
    def __init__(self,horizon=4,dim=128):
        super().__init__()
        self.encoder=Encoder(dim)
        self.t_embed=nn.Sequential(nn.Linear(1,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.net=nn.Sequential(nn.Linear(horizon*ACTION_DIM+dim*2,512),nn.SiLU(),
                               nn.Linear(512,512),nn.SiLU(),nn.Linear(512,horizon*ACTION_DIM))
        self.horizon=horizon
    def forward(self,noisy,t,rgb,state):
        z=self.encoder(rgb,state)
        t=self.t_embed(t.float().reshape(-1,1))
        return self.net(torch.cat((noisy.flatten(1),z,t),1)).reshape_as(noisy)


class DDPM(nn.Module):
    def __init__(self,horizon=4,steps=40,dim=128):
        super().__init__()
        self.net=DenoiseMLP(horizon,dim)
        self.horizon=horizon
        self.steps=steps
        beta=torch.linspace(1e-4,0.02,steps)
        self.register_buffer('beta',beta)
        self.register_buffer('alpha',1-beta)
        self.register_buffer('abar',torch.cumprod(1-beta,0))
    def loss(self,rgb,state,actions):
        t=torch.randint(self.steps,(len(actions),),device=actions.device)
        eps=torch.randn_like(actions)
        abar=self.abar[t].reshape(-1,1,1)
        noisy=abar.sqrt()*actions+(1-abar).sqrt()*eps
        pred=self.net(noisy,t.float()/self.steps,rgb,state)
        return nn.functional.mse_loss(pred,eps)
    @torch.no_grad()
    def sample(self,rgb,state,generator=None):
        z=torch.randn((len(rgb),self.horizon,ACTION_DIM),device=rgb.device,generator=generator)
        for i in reversed(range(self.steps)):
            beta=self.beta[i];alpha=self.alpha[i];abar=self.abar[i]
            pred=self.net(z,torch.full((len(rgb),),i/self.steps,device=z.device),rgb,state)
            z=(z-beta/torch.sqrt(1-abar)*pred)/torch.sqrt(alpha)
            if i>0:z=z+torch.sqrt(beta)*torch.randn(z.shape,device=z.device,generator=generator)
        return z


def new_model(method,horizon,steps=40):
    if method=='b01':return TransformerBC(horizon)
    if method=='b02':return DDPM(horizon,steps)
    raise ValueError('METHOD_MUST_BE_B01_OR_B02')


def train(method,cache_root=CACHE_ROOT,out=RUN_ROOT,horizon=4,steps=2000,
          batch=32,seed=42,lr=2e-4,allow_small_dev=False,device='auto'):
    if not 1<=steps<=1000000 or not 1<=batch<=256 or not 1<=horizon<=32:
        raise ValueError('INVALID_TRAIN_HYPERPARAMETERS')
    cache_root=Path(cache_root);out=Path(out)
    manifest=json.loads((cache_root/'manifest.json').read_text())
    if manifest['label']=='SMALL_DEV_DATA_ONLY' and not allow_small_dev:
        raise ValueError('SMALL_DEV_DATA_REQUIRES_EXPLICIT_OPT_IN')
    if out.exists():
        raise FileExistsError('TRAIN_OUTPUT_EXISTS_DO_NOT_OVERWRITE')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    actual_device='cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device
    ds=RobotDataset(cache_root,'train',horizon)
    stats=train_stats(ds)
    ds.stats=stats
    val=RobotDataset(cache_root,'validation',horizon,stats)
    loader=DataLoader(ds,batch_size=min(batch,len(ds)),shuffle=True,num_workers=0)
    model=new_model(method,horizon).to(actual_device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    it=iter(loader);history=[];best=float('inf')
    out.mkdir(parents=True,exist_ok=False)
    for step in range(1,steps+1):
        try:x,s,a=next(it)
        except StopIteration:
            it=iter(loader);x,s,a=next(it)
        x,s,a=x.to(actual_device),s.to(actual_device),a.to(actual_device)
        model.train();opt.zero_grad(set_to_none=True)
        loss=nn.functional.mse_loss(model(x,s),a) if method=='b01' else model.loss(x,s,a)
        if not torch.isfinite(loss):raise RuntimeError('NONFINITE_TRAIN_LOSS')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if step==1 or step==steps or step%max(1,steps//10)==0:
            model.eval();vals=[]
            with torch.no_grad():
                for j in range(min(16,len(val))):
                    xx,ss,aa=val[j];xx,ss,aa=xx[None].to(actual_device),ss[None].to(actual_device),aa[None].to(actual_device)
                    v=nn.functional.mse_loss(model(xx,ss),aa) if method=='b01' else model.loss(xx,ss,aa)
                    vals.append(float(v.item()))
            vloss=float(np.mean(vals))
            rec={'step':step,'train_loss':float(loss.item()),'validation_loss':vloss}
            history.append(rec)
            print('TRO1_ROBOT_ONLY_TRAIN',json.dumps({'method':method,**rec}),flush=True)
            if vloss<best:
                best=vloss
                torch.save({'method':method,'horizon':horizon,'diffusion_steps':40,
                            'model':model.state_dict(),'stats':stats,'side':manifest['side'],
                            'data_label':manifest['label'],'demos':len(manifest['demos']),
                            'dataset_manifest_sha256':sha256(cache_root/'manifest.json'),
                            'seed':seed,'train_step':step},out/'best.pt')
    write_new_json(out/'train_summary.json',{'status':'TRAINED_NOT_PHYSX_EVALUATED',
            'model':method,'provenance':'REIMPL_ON_ROBOREEL','data_label':manifest['label'],
            'demos':len(manifest['demos']),'best_val_loss':best,'history':history,
            'horizon':horizon,'steps':steps,'seed':seed})
    return out/'best.pt'


def load_policy(checkpoint,device='cpu'):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['method'] not in ('b01','b02') or ck['data_label'] not in ('SMALL_DEV_DATA_ONLY','TRAIN_DATA_READY_NOT_TRAINED'):
        raise ValueError('UNKNOWN_REIMPL_CHECKPOINT')
    model=new_model(ck['method'],ck['horizon'],ck['diffusion_steps']).to(device)
    model.load_state_dict(ck['model'],strict=True);model.eval()
    return model,ck


@torch.no_grad()
def infer(model,ck,history,state,device='cpu',generator=None):
    side=ck['side'];stats=ck['stats']
    if len(history)==0:raise ValueError('EMPTY_RGB_HISTORY')
    rgb=np.stack([resize_rgb(history[max(0,len(history)-2)],side),resize_rgb(history[-1],side)],axis=0)
    x=torch.tensor(rgb,device=device).permute(0,3,1,2)[None].float()/255.
    s=(state-stats['state_mean'])/stats['state_std']
    s=torch.tensor(s[None],device=device,dtype=torch.float32)
    a=model(x,s) if ck['method']=='b01' else model.sample(x,s,generator=generator)
    action=a[0].cpu().numpy()*stats['action_std']+stats['action_mean']
    if action.shape!=(ck['horizon'],ACTION_DIM) or not np.isfinite(action).all():
        raise RuntimeError('NONFINITE_OR_WRONG_SHAPE_PREDICTED_ACTION')
    return action.astype(np.float32)


def eval_bank(checkpoint,upstream,out,mode='dev',pass_index=0,episodes=2,max_steps=800,seed=42,save_video=False):
    """Run frozen env-config bank, no access to held-out human videos or privileged model state."""
    if mode not in ('dev','formal') or pass_index not in (0,1,2):raise ValueError('INVALID_EVAL_MODE')
    if mode=='dev' and not 1<=episodes<=2:raise ValueError('DEV_MAX_TWO_ROLLOUTS')
    if mode=='formal' and episodes!=20:raise ValueError('FORMAL_REQUIRES_20_SCENES_PER_PASS')
    out=Path(out)
    if out.exists():raise FileExistsError('EVAL_OUTPUT_EXISTS_DO_NOT_OVERWRITE')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,ck=load_policy(checkpoint,device)
    if mode=='formal' and (ck['demos']<64 or ck['data_label']=='SMALL_DEV_DATA_ONLY'):
        raise ValueError('FORMAL_REQUIRES_64_AUTHENTIC_TRAIN_DEMOS')
    upstream=Path(upstream).resolve()
    if not (upstream/'src/evaluation/utils.py').is_file():
        raise FileNotFoundError('PINNED_ROBOREEL_MODELS_MISSING')
    # Pinned official env/submodule registration (do not change environment).
    sys.path.insert(0,str(upstream/'src'))
    sys.path.insert(0,str(upstream/'roboreel_env/src'))
    import mani_skill.envs  # noqa: F401
    import envs  # noqa: F401
    from evaluation.utils import load_eval_env_bank,make_eval_env
    bank=load_eval_env_bank('push_cube',False) if mode=='formal' else None
    offset=20*pass_index if mode=='formal' else 0
    if bank is not None and len(bank)<offset+episodes:raise ValueError('OFFICIAL_EVAL_BANK_TOO_SHORT')
    env,env_id=make_eval_env('push_cube',False,max_episode_steps=max_steps)
    def truth(value):
        if hasattr(value, 'detach'):return bool(value.reshape(-1)[0].item())
        return bool(np.asarray(value).reshape(-1)[0])
    out.mkdir(parents=True,exist_ok=False)
    results=[]
    try:
        for k in range(episodes):
            idx=offset+k if mode=='formal' else None
            env_seed=int(bank[idx]['seed']) if bank is not None else 310000+k
            obs,_=env.reset(seed=env_seed)
            if bank is not None:
                env.unwrapped.set_state_dict(bank[idx]['state'])
                obs=env.unwrapped.get_obs()
            hist=[];queue=[];success=False;used=0
            video=None
            if save_video:
                import imageio.v2 as imageio
                video=imageio.get_writer(str(out/f'rollout_{k:03d}.mp4'),fps=10,quality=6)
            generator=torch.Generator(device=device).manual_seed(seed+(idx if idx is not None else k))
            try:
                for t in range(max_steps):
                    frame=live_rgb(obs)
                    if video is not None and t%2==0:video.append_data(resize_rgb(frame,256))
                    hist.append(frame);hist=hist[-2:]
                    if not queue:queue=list(infer(model,ck,hist,live_state(obs),device,generator))
                    a=queue.pop(0)
                    obs,_,terminated,truncated,info=env.step(torch.from_numpy(a)[None].float())
                    success=truth(info.get('success',False))
                    used=t+1
                    if success or truth(terminated) or truth(truncated):break
            finally:
                if video is not None:video.close()
            results.append({'bank_index':idx,'env_seed':env_seed,'success':success,'steps':used})
            print('TRO1_ROBOT_ONLY_PHYSX',json.dumps({'method':ck['method'],'mode':mode,**results[-1]}),flush=True)
    finally:env.close()
    summary={'status':'FORMAL_SINGLE_PASS' if mode=='formal' else 'DEV_PHYSX_ONLY',
             'reproduction':'REIMPL_ON_ROBOREEL_NOT_AUTHOR_RELEASE',
             'task':'push_cube','bucket':'nd_clean_env','model':ck['method'],
             'checkpoint_sha256':sha256(checkpoint),'official_env_id':env_id,
             'pass_index':pass_index,'successes':sum(int(x['success']) for x in results),
             'trials':len(results),'results':results,'human_eval_videos_read':False,'dev_uses_frozen_eval_bank':False if mode=='dev' else None,'camera':CAMERA,'control_mode':'pd_joint_pos'}
    write_new_json(out/'results.json',summary)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--data-root',type=Path,default=DATA_ROOT)
    prep.add_argument('--cache-root',type=Path,default=CACHE_ROOT);prep.add_argument('--side',type=int,default=96)
    prep.add_argument('--min-demos',type=int,default=64);prep.add_argument('--allow-small-dev',action='store_true')
    tr=sub.add_parser('train');tr.add_argument('--method',choices=('b01','b02'),required=True)
    tr.add_argument('--cache-root',type=Path,default=CACHE_ROOT);tr.add_argument('--out',type=Path,required=True)
    tr.add_argument('--steps',type=int,default=2000);tr.add_argument('--batch',type=int,default=32)
    tr.add_argument('--horizon',type=int,default=4);tr.add_argument('--allow-small-dev',action='store_true')
    ev=sub.add_parser('eval');ev.add_argument('--checkpoint',type=Path,required=True)
    ev.add_argument('--upstream',type=Path,default=ROOT/'third_party/roboreel_models')
    ev.add_argument('--out',type=Path,required=True);ev.add_argument('--mode',choices=('dev','formal'),default='dev')
    ev.add_argument('--pass-index',type=int,default=0);ev.add_argument('--episodes',type=int,default=2)
    ev.add_argument('--max-steps',type=int,default=800);ev.add_argument('--save-video',action='store_true')
    args=p.parse_args(argv)
    if args.cmd=='prepare':print('TRO1_B01_B02_PREPARE',json.dumps(preprocess(data_root=args.data_root,cache_root=args.cache_root,side=args.side,min_demos=args.min_demos,allow_small_dev=args.allow_small_dev)['label']))
    if args.cmd=='train':train(args.method,args.cache_root,args.out,horizon=args.horizon,steps=args.steps,batch=args.batch,allow_small_dev=args.allow_small_dev)
    if args.cmd=='eval':print('TRO1_B01_B02_EVAL_COMPLETE',json.dumps(eval_bank(args.checkpoint,args.upstream,args.out,args.mode,args.pass_index,args.episodes,args.max_steps,save_video=args.save_video)))


if __name__=='__main__':main()
