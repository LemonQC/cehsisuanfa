"""Real MP4 mask-prediction Full/Anchor/End *engineering pilot*.

Input: ONLY observed first two blue-candidate masks; target: next 8 masks
from authentic human demonstration video. No synthetic video is generated.
This is NOT human-to-robot training: no robot trajectories or actions used.
RoboReel evaluation videos are development-only; never cite this as official
RoboReel benchmark performance or claim camera mask identity without QA.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from .partial_supervision import torch_segment_loss
from .real_video_flow import blue_candidate, select_video_frames

FRAMES = 64
PAST = 2
FUTURE = 8
HW = 32
MODES = ("full", "anchor", "end")


def discovered_videos(root, max_videos=0):
    root=Path(root)
    candidates=sorted(root.glob("*/front.mp4"))
    if max_videos:
        candidates=candidates[:int(max_videos)]
    if not candidates:
        raise FileNotFoundError(f"No local front.mp4 under {root}; supply real videos, do not synthesize data")
    return candidates


def video_masks(path, frames=FRAMES):
    import cv2
    sampled,total=select_video_frames(path, target=frames)
    occupancy=[];coverage=[];centers=[];areas=[]
    for frame_id,rgb in sampled:
        candidate,center=blue_candidate(rgb)
        # Area-resize rather than nearest-neighbor to preserve small masks.
        small=(cv2.resize(candidate.astype(np.float32),(HW,HW),
                          interpolation=cv2.INTER_AREA) > .06).astype(np.float32)
        occupancy.append(small)
        centers.append(center)
        coverage.append(center is not None and small.sum()>=1)
        areas.append(float(small.sum()))
    masks=np.stack(occupancy)
    report=dict(source=str(Path(path).resolve()),source_sha256=hashlib.sha256(
        Path(path).read_bytes()).hexdigest(),
        source_frames=total,sampled_frames=len(sampled),
        mask_coverage=float(np.mean(coverage)),
        minimum_mask_area_pixels=float(np.min(areas)),
        maximum_mask_area_pixels=float(np.max(areas)),
        candidate_segmentation="Unverified HSV blue largest component; inspect original MP4/overlay",
        camera="front",source_is_roboreel_eval="/eval/" in str(Path(path).resolve()))
    return masks,report


def make_windows(masks, source_id, stride=4):
    """No future mask is included in the input; future starts at t+2."""
    masks=np.asarray(masks,dtype=np.float32)
    if masks.ndim!=3 or masks.shape[1:]!=(HW,HW):
        raise ValueError("Expected [N,32,32] real video masks")
    if len(masks)<PAST+FUTURE:
        raise ValueError("Video shorter than 10 sampled mask frames")
    examples=[]
    for idx in range(0,len(masks)-(PAST+FUTURE)+1,stride):
        past=masks[idx:idx+PAST]
        future=masks[idx+PAST:idx+PAST+FUTURE]
        # Reject missing target or observation tracking, not all-zero labels.
        if np.any(past.sum((1,2))<1) or np.any(future.sum((1,2))<1):
            continue
        examples.append(dict(source_id=source_id,index=idx,
                             past=past.copy(),future=future.copy()))
    return examples


def split_video_ids(ids, seed=42):
    ids=sorted(set(ids))
    if len(ids)<5:
        return dict(train=ids,val=[],test=[]), "TRAIN_ONLY_NOT_GENERALIZATION"
    rng=np.random.default_rng(seed)
    shuffled=[ids[i] for i in rng.permutation(len(ids))]
    nt=max(1,round(len(ids)*.2))
    nv=max(1,round(len(ids)*.2))
    train=shuffled[nt+nv:]
    return dict(train=sorted(train),val=sorted(shuffled[nt:nt+nv]),
                test=sorted(shuffled[:nt])), "HELD_OUT_HUMAN_VIDEOS_INTERNAL_ONLY"


class HumanMaskPredictor(nn.Module):
    """Same input, same architecture, same init for all three modes.

    Two OBSERVED past masks -> eight future masks; no future label leakage.
    """
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(2,32,3,padding=1),nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(64,96,3,stride=2,padding=1),nn.ReLU(),
            nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(96,48,3,padding=1),nn.ReLU(),
            nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(48,24,3,padding=1),nn.ReLU(),
            nn.Conv2d(24,FUTURE,1))
    def forward(self,past):
        if past.ndim!=4 or past.shape[1:]!=(2,HW,HW):
            raise ValueError(f"Expected [B,2,32,32], got {tuple(past.shape)}")
        return self.net(past)


def _iou(pred,true):
    a=pred>.5;b=true>.5
    return float(np.logical_and(a,b).sum()/
                 max(1,int(np.logical_or(a,b).sum())))


def _write_csv(path,rows):
    if not rows:return
    with open(path,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def _preview(example,preds,out):
    import cv2
    tile=84
    labels=[("Observed past",example["past"]),
            ("Human future GT",example["future"])]
    for mode in MODES:labels.append((f"{mode.upper()} prediction",preds[mode]))
    width=FUTURE*tile
    canvas=np.full((len(labels)*(tile+29),width,3),246,dtype=np.uint8)
    for row,(title,frames) in enumerate(labels):
        y=row*(tile+29)
        cv2.putText(canvas,title,(5,y+19),cv2.FONT_HERSHEY_SIMPLEX,.51,(55,55,55),1)
        for j,mask in enumerate(frames):
            if j>=FUTURE:break
            rendered=cv2.resize(np.uint8(mask>.5)*255,(tile,tile),
                                interpolation=cv2.INTER_NEAREST)
            canvas[y+29:y+29+tile,j*tile:(j+1)*tile]=cv2.cvtColor(rendered,cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out),canvas)


def train_real_videos(video_root,out,epochs=20,seed=42,
                      max_videos=1,device="auto",frames=64,batch=8):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    video_paths=discovered_videos(video_root,max_videos=max_videos)
    sources=[];examples=[]
    for video in video_paths:
        sid=video.parent.name
        masks,meta=video_masks(video,frames)
        (out/f"observed_masks_{sid}.npy").parent.mkdir(exist_ok=True,parents=True)
        np.save(out/f"observed_masks_{sid}.npy",masks)
        windows=make_windows(masks,sid)
        meta["usable_windows"]=len(windows)
        meta["min_required_mask_coverage"]=.75
        meta["accepted"]=bool(meta["mask_coverage"]>=.75 and windows)
        sources.append(meta)
        if not meta["accepted"]:
            print("VIDEO_MASK_REJECTED",json.dumps(meta),flush=True)
            continue
        examples.extend(windows)
        print("REAL_HUMAN_VIDEO_READY",
              json.dumps(dict(video=sid,frames=len(masks),windows=len(windows),
                              coverage=meta["mask_coverage"])),flush=True)
    (out/"source_audit.json").write_text(json.dumps(sources,indent=2))
    if not examples:
        raise RuntimeError("No usable real MP4 mask windows. Review blue-mask overlays; do not fabricate examples.")
    if device=="auto":device="cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but not available")
    split,status=split_video_ids([x["source_id"] for x in examples],seed)
    (out/"video_split.json").write_text(json.dumps(split,indent=2))
    by_group={key:[ex for ex in examples if ex["source_id"] in vids]
              for key,vids in split.items()}
    if not by_group["train"]:raise RuntimeError("No train windows")
    arrays=lambda cases:(
        np.stack([ex["past"] for ex in cases]).astype(np.float32),
        np.stack([ex["future"] for ex in cases]).astype(np.float32)
    )
    all_models={};history=[];metrics=[];detailed=[]
    for mode in MODES:
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        net=HumanMaskPredictor().to(device)
        opt=torch.optim.Adam(net.parameters(),lr=3e-4)
        xtrain,ytrain=arrays(by_group["train"])
        rng=np.random.default_rng(seed)
        for epoch in range(epochs):
            net.train()
            values=[]
            for indices in np.array_split(rng.permutation(len(xtrain)),
                                          max(1,int(np.ceil(len(xtrain)/batch)))):
                x=torch.as_tensor(xtrain[indices],device=device)
                y=torch.as_tensor(ytrain[indices],device=device)
                logits=net(x)
                terms=[torch_segment_loss(logits[j],y[j],mode) for j in range(len(indices))]
                loss=torch.stack(terms).mean()
                opt.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
                values.append(float(loss.detach().cpu()))
            history.append(dict(mode=mode,epoch=epoch+1,train_objective=float(np.mean(values))))
        ckpt=out/f"{mode}_real_video.pt"
        torch.save(dict(model=net.state_dict(),mode=mode,seed=seed,
                        input="2 observed HSV candidate masks",
                        output="8 future HSV candidate masks",
                        source_ids=split["train"],data_status=status),ckpt)
        net.eval()
        all_models[mode]=net
        # When n<5 videos, ONLY training fit is shown; there is no held-out test.
        for part in ("train","val","test"):
            cases=by_group[part]
            if not cases:continue
            x,y=arrays(cases)
            with torch.no_grad():
                probs=torch.sigmoid(net(torch.as_tensor(x,device=device))).cpu().numpy()
            for k,ex in enumerate(cases):
                row=dict(mode=mode,split=part,video_id=ex["source_id"],
                         window_start=ex["index"],
                         full_iou=_iou(probs[k],y[k]),
                         end_iou=_iou(probs[k,-1],y[k,-1]),
                         prediction_coverage=float((probs[k]>.5).mean()))
                detailed.append(row)
            rows=[r for r in detailed if r["mode"]==mode and r["split"]==part]
            metrics.append(dict(mode=mode,split=part,windows=len(rows),
                mean_full_iou=float(np.mean([r["full_iou"] for r in rows])),
                mean_end_iou=float(np.mean([r["end_iou"] for r in rows])),
                prediction_coverage=float(np.mean([r["prediction_coverage"] for r in rows]))))
        print("MODE_TRAINED",json.dumps(dict(mode=mode,epochs=epochs,
            train_windows=len(xtrain),last_loss=history[-1]["train_objective"])),flush=True)
    preview_part="test" if by_group["test"] else "train"
    sample=by_group[preview_part][0]
    with torch.no_grad():
        sample_input=torch.as_tensor(sample["past"][None],device=device)
        predictions={mode:torch.sigmoid(model(sample_input))[0].cpu().numpy()
                     for mode,model in all_models.items()}
    _preview(sample,predictions,out/"three_mode_predictions.png")
    _write_csv(out/"training_history.csv",history)
    _write_csv(out/"all_windows.csv",detailed)
    _write_csv(out/"mode_summary.csv",metrics)
    report=dict(type="REAL_HUMAN_VIDEO_MASK_PREDICTION_ONLY_NOT_ROBOT_TRANSFER",
         validity=status,warning=(
             "All source videos are RoboReel official EVALUATION demonstrations; "
             "use only for pipeline development, not official benchmark training/results. "
             "The mask detector is HSV, not verified instance segmentation. "
             "No robot actions or policy rollouts use the predicted masks. "
             "For one video, reported training-fit IoUs are NOT generalization."),
         video_root=str(Path(video_root).resolve()),
         source_videos=len(sources),accepted_videos=len(split["train"])+len(split["val"])+len(split["test"]),
         split=split,train_windows=len(by_group["train"]),val_windows=len(by_group["val"]),
         test_windows=len(by_group["test"]),device=device,epochs=epochs,seed=seed,
         modes=list(MODES),summary=metrics,preview_split=preview_part)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("REAL_VIDEO_SUPERVISION_COMPLETE",
          json.dumps({k:v for k,v in report.items() if k not in ("summary","warning")}),flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description="Real RoboReel human-video Full/Anchor/End mask prediction pilot")
    p.add_argument("--video-root",
       default="third_party/roboreel/human_demos/eval/no_distraction/push_cube")
    p.add_argument("--out",default="outputs/real_human_supervision_v090")
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-videos",type=int,default=1)
    p.add_argument("--frames",type=int,default=64)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    args=p.parse_args(argv)
    train_real_videos(args.video_root,args.out,args.epochs,args.seed,
                      args.max_videos,args.device,args.frames,args.batch)


if __name__=="__main__":
    main()
