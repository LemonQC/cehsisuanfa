"""Flow-conditioned low-level behavioral cloning with real physics test."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .representation import future_masks
from .teacher import robot_state


class FlowBC(nn.Module):
    def __init__(self,state_dim,action_dim,channels=4):
        super().__init__()
        self.mask_net=nn.Sequential(
            nn.Conv2d(channels,16,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(32,32,3,stride=2,padding=1),nn.ReLU(),nn.Flatten())
        self.state_net=nn.Sequential(nn.Linear(state_dim,96),nn.ReLU(),nn.Linear(96,96),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(32*4*4+96,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,action_dim))
    def forward(self,state,flow):
        return self.head(torch.cat([self.state_net(state),self.mask_net(flow)],dim=-1))


class Demonstrations(Dataset):
    def __init__(self,folder):
        paths=sorted(Path(folder).glob('*.npz'))
        if not paths: raise RuntimeError('No .npz episodes. Run collect first.')
        xs=[];fs=[];ys=[]
        for path in paths:
            with np.load(path) as d:
                if not bool(d['success']): continue
                xs.append(d['state'].astype(np.float32))
                fs.append(d['flow'].astype(np.float32))
                ys.append(d['action'].astype(np.float32))
        if not xs: raise RuntimeError('No successful physical episodes. Fix the Panda teacher before training.')
        self.x=np.concatenate(xs); self.f=np.concatenate(fs); self.y=np.concatenate(ys)
        self.state_dim=self.x.shape[-1]; self.action_dim=self.y.shape[-1]
    def __len__(self): return len(self.x)
    def __getitem__(self,idx): return self.x[idx],self.f[idx],self.y[idx]


def train_bc(data_dir,checkpoint,epochs=80,device=None,batch=256,seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ds=Demonstrations(data_dir)
    net=FlowBC(ds.state_dim,ds.action_dim).to(device)
    loader=DataLoader(ds,batch_size=batch,shuffle=True,num_workers=0)
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-5)
    history=[]
    for epoch in range(epochs):
        net.train(); totals=[]
        for x,f,y in loader:
            x=x.to(device);f=f.to(device);y=y.to(device)
            pred=net(x,f)
            loss=(pred-y).square().mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step();totals.append(loss.item())
        history.append(float(np.mean(totals)))
        if epoch==0 or (epoch+1)%10==0:print(f'epoch={epoch+1:03d} mse={history[-1]:.5f}')
    checkpoint=Path(checkpoint);checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':net.state_dict(),'state_dim':ds.state_dim,'action_dim':ds.action_dim,
                'epochs':epochs,'num_transitions':len(ds),'loss':history},checkpoint)
    checkpoint.with_suffix('.json').write_text(json.dumps({'last_mse':history[-1],'steps':len(ds),'device':device},indent=2))
    return checkpoint


def load_bc(checkpoint,device=None):
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    d=torch.load(checkpoint,map_location=device,weights_only=False)
    model=FlowBC(d['state_dim'],d['action_dim']).to(device)
    model.load_state_dict(d['model']);model.eval()
    return model,device


def rollout_bc(env,model,device,side,seed,max_steps=500,save_video=False):
    env.reset(seed=int(seed))
    base=env.unwrapped
    frames=[];history=[]
    from .teacher import np1
    initial_state=robot_state(base).tolist()
    for t in range(max_steps):
        state=robot_state(base)
        ref=future_masks(np1(base.obj.pose.p)[:2],side)
        with torch.no_grad():
            action=model(torch.from_numpy(state).to(device)[None],
                         torch.from_numpy(ref).to(device)[None]).cpu().numpy()[0]
        _,_,terminated,truncated,info=env.step(action)
        history.append({'step':t,'cube':np1(base.obj.pose.p)[:3].tolist(),
                        'visited':bool(np1(info['check_visited'])[0]),
                        'success':bool(np1(info['success'])[0])})
        # --video only renders a trajectory animation after the rollout,
        # never use render_rgb_array() or initialize the Vulkan renderer.
        if bool(np1(info['success'])[0]) or bool(np1(truncated)[0]) or bool(np1(terminated)[0]):break
    return {'side':int(side),'seed':int(seed),'initial_state':initial_state,
            'success':bool(np1(base.evaluate()['success'])[0]),
            'visited':bool(np1(base.evaluate()['check_visited'])[0]),
            'num_steps':len(history),'history':history,'frames':frames}
