"""Top-down geometric **binary occupancy mask sequence** used as future reference.

Not dense optical flow. For a single square cube, each future pose becomes a mask.
This matches the mask-sequence *type* of the paper, but is a controlled synthetic
oracle reference rather than a learned camera segmentation output.
"""
from __future__ import annotations
import numpy as np
from .env_constants import START, GOAL, CHECK_Y

RES=32
XY_MIN=-0.30
XY_MAX=0.30
HALF_CUBE=0.02


def route_points(side):
    assert side in (-1, 1)
    return np.array([START,[-.105,side*CHECK_Y],[0.,side*CHECK_Y],[.105,side*CHECK_Y],GOAL], np.float32)


def advance_on_path(polyline, current, distances=(.045,.09,.14,.20)):
    polyline=np.asarray(polyline,dtype=np.float32)
    current=np.asarray(current,dtype=np.float32)
    lens=np.linalg.norm(np.diff(polyline,axis=0),axis=1)
    cum=np.r_[0.,np.cumsum(lens)]
    best_cost=float('inf'); best_s=0.
    for i,(a,b) in enumerate(zip(polyline[:-1],polyline[1:])):
        v=b-a; den=float(v@v)
        alpha=float(np.clip(((current-a)@v)/max(den,1e-10),0,1))
        projection=a+alpha*v
        cost=float(np.linalg.norm(current-projection))
        if cost<best_cost:
            best_cost=cost
            best_s=float(cum[i]+alpha*lens[i])
    out=[]
    for d in distances:
        s=min(float(cum[-1]), best_s+d)
        i=int(np.clip(np.searchsorted(cum, s, side='right')-1, 0, len(lens)-1))
        alpha=(s-cum[i])/max(lens[i],1e-9)
        out.append(polyline[i]*(1-alpha)+polyline[i+1]*alpha)
    return np.asarray(out,dtype=np.float32)


def rasterize_square_mask(xy,res=RES,half=HALF_CUBE):
    xy=np.asarray(xy,dtype=np.float32)
    a=np.linspace(XY_MIN,XY_MAX,res,endpoint=False,dtype=np.float32)+(XY_MAX-XY_MIN)/(res*2)
    xx,yy=np.meshgrid(a,a)
    return ((np.abs(xx-xy[0])<=half)&(np.abs(yy-xy[1])<=half)).astype(np.float32)


def future_masks(current,side,res=RES):
    points=advance_on_path(route_points(side), current)
    return np.stack([rasterize_square_mask(p,res=res) for p in points],axis=0)


def mask_route_side(masks):
    """Only for teacher diagnostics: infer the side from reference mask center of mass."""
    m=np.asarray(masks)
    yy=np.linspace(XY_MIN,XY_MAX,m.shape[-2],endpoint=False)+(XY_MAX-XY_MIN)/(2*m.shape[-2])
    mass=m.sum(axis=(-1,-2))
    ys=(m * yy.reshape(1,-1,1)).sum(axis=(-1,-2))/np.maximum(mass,1e-6)
    ind=np.flatnonzero(np.abs(ys)>.05)
    if len(ind)==0: raise ValueError('reference mask does not disambiguate left/right side')
    return 1 if ys[ind[0]]>0 else -1
