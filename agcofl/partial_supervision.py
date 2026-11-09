"""Full / anchor / end mask supervision, paper-facing standalone building block.

Input logits & labels [T,H,W] with T>=3. Unselected prediction frames have no
*direct* loss. High-level predictor integration is intentionally separate.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def mask_distance(logits, target, gamma=2.):
    target=target.float()
    prob=torch.sigmoid(logits)
    ce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    pt=prob*target+(1-prob)*(1-target)
    focal=((1-pt)**gamma * ce).mean()
    dims=tuple(range(1,logits.ndim))
    intersect=(prob*target).sum(dim=dims)
    dice=1-(2*intersect+1)/((prob+target).sum(dim=dims)+1)
    return focal+dice.mean()


def _half_arclength_anchor(target):
    # Find the observed mask center closest to 50% accumulated object displacement.
    with torch.no_grad():
        m=target.detach().float()
        n,h,w=m.shape
        ys=torch.arange(h,device=m.device).float()[None,:,None]
        xs=torch.arange(w,device=m.device).float()[None,None,:]
        mass=m.sum((1,2)).clamp_min(1e-6)
        centers=torch.stack([(m*xs).sum((1,2))/mass,(m*ys).sum((1,2))/mass],dim=1)
        ds=torch.linalg.norm(centers[1:]-centers[:-1],dim=1)
        cumul=torch.cat([torch.zeros(1,device=m.device),ds.cumsum(0)])
        if float(cumul[-1])<1e-6:return (n-1)//2
        return int(torch.argmin((cumul-cumul[-1]*.5).abs()).clamp(0,n-2))


def select_matches(pred,target,mode,anchor_label=None):
    """Return (pred_frame,label_frame) pairs. Anchor matched in central 50% of window."""
    n=len(pred)
    if n!=len(target):raise ValueError('pred and target must share T')
    if n<3: raise ValueError('at least 3 frames needed')
    if mode=='full':return [(i,i) for i in range(n)]
    if mode=='end':return [(n-1,n-1)]
    if mode!='anchor':raise ValueError('mode must be full, anchor or end')
    j=_half_arclength_anchor(target) if anchor_label is None else int(anchor_label)
    if not 0<=j<n-1:raise ValueError('anchor index must precede end')
    lower=max(0,int(n*.25));upper=max(lower+1,int(n*.75))
    upper=min(upper,n-1)
    costs=[float(mask_distance(pred[i:i+1].detach(),target[j:j+1]).item()) for i in range(lower,upper)]
    k=lower+int(torch.tensor(costs).argmin())
    return [(k,j),(n-1,n-1)]


def torch_segment_loss(logits,masks,mode,anchor_label=None,union_weight=.15):
    if logits.ndim!=3 or masks.shape!=logits.shape:
        raise ValueError('expected matching [T,H,W] tensors')
    pairs=select_matches(logits,masks,mode,anchor_label)
    pred=torch.stack([logits[i] for i,j in pairs])
    gold=torch.stack([masks[j] for i,j in pairs])
    indiv=mask_distance(pred,gold)
    punion=torch.sigmoid(pred).amax(dim=0,keepdim=True)
    union_logits=torch.logit(punion.clamp(1e-6,1-1e-6))
    target_union=gold.amax(dim=0,keepdim=True)
    return indiv+union_weight*mask_distance(union_logits,target_union)
