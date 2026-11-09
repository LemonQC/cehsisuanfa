"""Same-data Full / Anchor / End supervision for future object mask prediction.

Offline training only: no robot policy evaluation or human-robot transfer claim.
Synthetic proxy data are marked and guarded as pipeline tests, not paper results.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .partial_supervision import torch_segment_loss

MODES = ("full", "anchor", "end")
SIZE, HORIZON = 32, 8
GRID = np.linspace(-0.3, 0.3, SIZE, endpoint=False, dtype=np.float32)+0.3/SIZE
XGRID, YGRID = np.meshgrid(GRID, GRID)


def rasterize(xy, half=0.025):
    x, y = np.asarray(xy, np.float32)[:2]
    return ((np.abs(XGRID-x)<=half)&(np.abs(YGRID-y)<=half)).astype(np.float32)


def resample_polyline(points, n=HORIZON):
    p=np.asarray(points,np.float32)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<2:
        raise ValueError("Expected polyline [N,2] with N>=2")
    lengths=np.linalg.norm(np.diff(p,axis=0),axis=1)
    cum=np.r_[0.,np.cumsum(lengths)]
    if cum[-1]<1e-7: return np.repeat(p[:1],n,axis=0)
    samples=np.linspace(0,cum[-1],n)
    return np.stack([np.interp(samples,cum,p[:,i]) for i in range(2)],axis=1).astype(np.float32)


def masks_from_points(points):
    return np.stack([rasterize(x) for x in resample_polyline(points)])


def context_of(start, goal, phase, scene):
    """Only initial observed pose, task-specified goal, phase and scene context.

    DO NOT construct goal from held-out future frames for real data.
    """
    return [float(start[0]),float(start[1]),float(goal[0]),float(goal[1]),
            float(phase),*[float(x) for x in scene]]


def split_groups(records, seed):
    groups=sorted(set(x["layout_id"] for x in records))
    if len(groups)<6:
        raise ValueError("Require >=6 independent layouts; new seeds in the same static scene are NOT new layouts")
    ids=np.random.default_rng(seed).permutation(len(groups))
    nt=max(1,round(len(groups)*.2));nv=max(1,round(len(groups)*.2))
    test={groups[i] for i in ids[:nt]}
    val={groups[i] for i in ids[nt:nt+nv]}
    train=set(groups)-test-val
    return dict(train=sorted(train),val=sorted(val),test=sorted(test))


def make_proxy(output, layouts=24, seed=42):
    """Synthetic geometry PROXY for code checking. Not real human/robot data."""
    root=Path(output)
    (root/"masks").mkdir(parents=True,exist_ok=True)
    rows=[]
    rng=np.random.default_rng(seed)
    for layout in range(layouts):
        side=1 if layout%2 else -1
        jitter=float(rng.uniform(-.018,.018))
        corridor=float(rng.uniform(.12,.18))
        src=np.array([-.20,jitter],np.float32)
        check=np.array([0,side*corridor],np.float32)
        dest=np.array([.20,jitter],np.float32)
        for condition in ("useful","habit"):
            # Synthetic *scene* property (corridor required vs open), not a
            # privileged future path. No real obstacle physics is simulated.
            scene=[side,corridor,jitter,1. if condition=="useful" else 0.]
            for phase,(start,end) in enumerate(((src,check),(check,dest))):
                direct=[start,end]
                helpful=[start,np.array([(start[0]+end[0])/2,side*corridor],np.float32),end]
                detour=[start,np.array([(start[0]+end[0])/2,-side*.22],np.float32),end]
                rp=helpful if condition=="useful" else direct
                hp=helpful if condition=="useful" else detour
                for source,points in (("robot",rp),("human_proxy",hp)):
                    uid=f"L{layout:03d}_{condition}_P{phase}_{source}"
                    fname=f"masks/{uid}.npy"
                    np.save(root/fname,masks_from_points(points))
                    rows.append(dict(id=uid,layout_id=f"L{layout:03d}",
                        condition=condition,phase=phase,source=source,
                        mask_path=fname,context=context_of(start,end,phase,scene)))
    path=root/"manifest.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    (root/"PROXY_NOT_A_RESULT.txt").write_text(
        "SYNTHETIC PROXY ONLY. NO human video, actual robot actions or policy success.\n")
    print(f"PROXY_ONLY manifest={path} clips={len(rows)} layouts={layouts}",flush=True)
    return path


def read_manifest(path):
    path=Path(path)
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not rows: raise ValueError("empty manifest")
    required={"id","layout_id","condition","phase","source","mask_path","context"}
    if any(not required.issubset(r) for r in rows):
        raise ValueError(f"Every row needs {sorted(required)}")
    if len(set(len(r["context"]) for r in rows))!=1:
        raise ValueError("context dimension mismatch")
    sources={r["source"] for r in rows}
    if "robot" not in sources or not (sources&{"human","human_proxy"}):
        raise ValueError("Need robot and human or human_proxy samples")
    if {"human","human_proxy"}.issubset(sources):
        raise ValueError("Cannot mix human recordings with synthetic human_proxy")
    if len(set(r["id"] for r in rows))!=len(rows):
        raise ValueError("duplicate sample IDs")
    for r in rows:
        if r["source"] not in {"robot","human","human_proxy"}:
            raise ValueError("source must be robot, human or human_proxy")
        f=(path.parent/r["mask_path"]).resolve()
        if not f.is_relative_to(path.parent.resolve()):
            raise ValueError("mask path escapes dataset root")
        if np.load(f,mmap_mode="r").shape!=(HORIZON,SIZE,SIZE):
            raise ValueError(f"{f}: expected [8,32,32] masks")
    return rows


class FlowPredictor(nn.Module):
    """Identical network across all human supervision modes."""
    def __init__(self,context_dim):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(context_dim,128),nn.ReLU(),
            nn.Linear(128,256),nn.ReLU(),nn.Linear(256,HORIZON*SIZE*SIZE))
    def forward(self,x):return self.layers(x).reshape(-1,HORIZON,SIZE,SIZE)


def iou(pred,truth):
    a=pred>.5;b=truth>.5
    return float((a&b).sum()/max(1,(a|b).sum()))


def write_csv(path,rows):
    if not rows:return
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run_training(manifest,out,epochs=20,seed=42,human_weight=.5,
                 device="auto",allow_proxy=False,split_seed=42):
    manifest=Path(manifest);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=read_manifest(manifest)
    proxy=any(r["source"]=="human_proxy" for r in rows)
    if proxy and not allow_proxy:
        raise RuntimeError("Synthetic proxy requires explicit --allow-proxy; NEVER use as paper evidence")
    if device=="auto": device="cuda" if torch.cuda.is_available() else "cpu"
    groups=split_groups(rows,split_seed)
    (out/"split.json").write_text(json.dumps(groups,indent=2))
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    data=[(r,np.asarray(r["context"],np.float32),
           np.load(manifest.parent/r["mask_path"]).astype(np.float32)) for r in rows]
    metrics=[];history=[];summary=[]
    for mode in MODES:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        model=FlowPredictor(len(rows[0]["context"])).to(device)
        optim=torch.optim.Adam(model.parameters(),lr=.001)
        train=[d for d in data if d[0]["layout_id"] in groups["train"]]
        rdata=[d for d in train if d[0]["source"]=="robot"]
        hdata=[d for d in train if d[0]["source"]!="robot"]
        if not rdata or not hdata: raise RuntimeError("Empty robot or human training group")
        order=np.random.default_rng(seed)

        for epoch in range(epochs):
            model.train()
            ri=order.permutation(len(rdata));hi=order.permutation(len(hdata))
            values=[]
            for step in range(int(np.ceil(max(len(rdata),len(hdata))/8))):
                rb=[rdata[int(ri[(8*step+i)%len(ri)])] for i in range(8)]
                hb=[hdata[int(hi[(8*step+i)%len(hi)])] for i in range(8)]
                def group_loss(batch,supervision):
                    terms=[]
                    for _,ctx,target in batch:
                        p=model(torch.as_tensor(ctx,device=device)[None])[0]
                        t=torch.as_tensor(target,device=device)
                        terms.append(torch_segment_loss(p,t,supervision))
                    return torch.stack(terms).mean()
                # Exactly fixed source group weights; robot always full.
                loss=group_loss(rb,"full")+human_weight*group_loss(hb,mode)
                optim.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optim.step();values.append(float(loss.detach().cpu()))
            history.append(dict(mode=mode,epoch=epoch+1,
                                train_loss=float(np.mean(values))))

        torch.save(dict(state_dict=model.state_dict(),
             context_dim=len(rows[0]["context"]),mode=mode,seed=seed,
             manifest_sha256=digest),out/f"{mode}.pt")
        model.eval()
        for part in ("val","test"):
            for rec,ctx,target in data:
                if rec["layout_id"] not in groups[part]:continue
                with torch.no_grad():
                    pred=torch.sigmoid(model(
                       torch.as_tensor(ctx,device=device)[None])[0]).cpu().numpy()
                metrics.append(dict(mode=mode,split=part,layout_id=rec["layout_id"],
                    id=rec["id"],source=rec["source"],condition=rec["condition"],
                    phase=rec["phase"],full_iou=iou(pred,target),
                    end_iou=iou(pred[-1],target[-1]),
                    middle_iou=iou(pred[HORIZON//2],target[HORIZON//2]),
                    predicted_union_area=float((pred.max(0)>.5).sum()),
                    target_union_area=float((target.max(0)>.5).sum())))
        for condition in sorted({m["condition"] for m in metrics
                                 if m["mode"]==mode and m["source"]=="robot"}):
            subset=[m for m in metrics if m["mode"]==mode and m["split"]=="test"
                    and m["source"]=="robot" and m["condition"]==condition]
            if subset:summary.append(dict(mode=mode,condition=condition,n=len(subset),
                mean_full_iou=float(np.mean([m["full_iou"] for m in subset])),
                mean_end_iou=float(np.mean([m["end_iou"] for m in subset]))))
    write_csv(out/"training_history.csv",history)
    write_csv(out/"predictions.csv",metrics)
    write_csv(out/"robot_test_summary.csv",summary)
    paired=defaultdict(dict)
    for m in metrics:
        if m["split"]=="test" and m["source"]=="robot":paired[m["id"]][m["mode"]]=m
    diffs=[]
    for uid,v in paired.items():
        if set(v)!=set(MODES):continue
        diffs.append(dict(id=uid,layout_id=v["full"]["layout_id"],
            condition=v["full"]["condition"],phase=v["full"]["phase"],
            anchor_minus_full_iou=v["anchor"]["full_iou"]-v["full"]["full_iou"],
            end_minus_full_iou=v["end"]["full_iou"]-v["full"]["full_iou"],
            anchor_minus_full_endpoint=v["anchor"]["end_iou"]-v["full"]["end_iou"],
            end_minus_full_endpoint=v["end"]["end_iou"]-v["full"]["end_iou"]))
    write_csv(out/"paired_differences.csv",diffs)
    report=dict(type="SYNTHETIC_PROXY_DEBUG_NOT_PAPER" if proxy else "OFFLINE_MASK_PREDICTION_ONLY",
        device=device,seed=seed,split_seed=split_seed,epochs=epochs,human_weight=human_weight,
        manifest_sha256=digest,layouts={k:len(v) for k,v in groups.items()},
        samples=len(data),trained_modes=list(MODES),summary=summary,
        warning="No robot actions executed. Offline IoU is NOT closed-loop task success. "
                "Actual human/robot recordings and physical testing are mandatory.")
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("STUDY_COMPLETE",json.dumps({k:v for k,v in report.items() if k!="summary"}),flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description="Offline paired Full/Anchor/End study")
    sub=p.add_subparsers(dest="cmd",required=True)
    sm=sub.add_parser("make-proxy",help="SYNTHETIC pipeline test, no research evidence")
    sm.add_argument("--out",default="outputs/proxy_dataset")
    sm.add_argument("--layouts",type=int,default=24)
    sm.add_argument("--seed",type=int,default=42)
    tr=sub.add_parser("train",help="Train from aligned real 8x32x32 mask segments")
    tr.add_argument("--manifest",required=True)
    tr.add_argument("--out",default="outputs/partial_study")
    tr.add_argument("--epochs",type=int,default=20)
    tr.add_argument("--seed",type=int,default=42)
    tr.add_argument("--human-weight",type=float,default=.5)
    tr.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    tr.add_argument("--allow-proxy",action="store_true")
    tr.add_argument("--split-seed",type=int,default=42,help="Held-out layout split stays fixed across model training seeds")
    args=p.parse_args(argv)
    if args.cmd=="make-proxy":make_proxy(args.out,args.layouts,args.seed)
    else:run_training(args.manifest,args.out,args.epochs,args.seed,
                      args.human_weight,args.device,args.allow_proxy,args.split_seed)


if __name__=="__main__":main()
