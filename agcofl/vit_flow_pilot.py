"""REAL human TRAIN videos: pretrained DINOv2 ViT + temporal Transformer pilot.

Frozen *pretrained* DINOv2, not a tiny CNN, no 32x32 segmentation loss.
Per-frame native-resolution HSV center candidates are LABELS/observed inputs,
NOT verified 3D object poses. They require RGB overlay identity inspection.

This script is a GPU / representation sanity experiment, NOT the final
cross-behavior method or a learned RoboReel control policy.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .real_video_flow import blue_candidate, select_video_frames

MODEL_ID="facebook/dinov2-small"
PAST=2
FUTURE=8
MODES=("full","anchor","end")


class TemporalViTFlow(nn.Module):
    """Two PRETRAINED ViT frame embeddings + observed centers -> future XY.

    Strictly observational inputs only, identical architecture across modes.
    """
    def __init__(self,input_dim=768,d_model=192,horizon=FUTURE):
        super().__init__()
        self.horizon=horizon
        self.project=nn.Sequential(nn.Linear(input_dim+2,d_model),nn.LayerNorm(d_model))
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=d_model,nhead=4,dim_feedforward=512,
            dropout=0.0,batch_first=True,activation="gelu")
        decoder_layer=nn.TransformerDecoderLayer(
            d_model=d_model,nhead=4,dim_feedforward=512,
            dropout=0.0,batch_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(encoder_layer,num_layers=2,enable_nested_tensor=False)
        self.decoder=nn.TransformerDecoder(decoder_layer,num_layers=2)
        self.queries=nn.Parameter(torch.randn(horizon,d_model)*.02)
        self.output=nn.Linear(d_model,2)
        # Training starts from the persistence baseline, not a giant
        # incorrect foreground mask.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,features,observed_xy):
        if features.ndim!=3 or features.shape[1]!=PAST:
            raise ValueError("Features must be [B,2,D] observed timestamps only")
        if observed_xy.shape!=(len(features),PAST,2):
            raise ValueError("Observed XY must be [B,2,2]")
        state=self.project(torch.cat([features,observed_xy],dim=-1))
        memory=self.encoder(state)
        q=self.queries[None].expand(len(features),-1,-1)
        decoded=self.decoder(q,memory)
        # Normalized XY residual. No access to future ground truth.
        return observed_xy[:,-1:, :] + .5*torch.tanh(self.output(decoded))


def select_loss(pred,target,mode):
    if pred.shape!=target.shape or pred.shape[-2:]!=(FUTURE,2):
        raise ValueError("Expected matching [B,8,2]")
    if mode=="full": idx=list(range(FUTURE))
    elif mode=="anchor":idx=[(FUTURE-1)//2,FUTURE-1]
    elif mode=="end":idx=[FUTURE-1]
    else:raise ValueError(mode)
    return F.smooth_l1_loss(pred[:,idx],target[:,idx],beta=.02)


def video_split(ids,seed=42):
    ids=sorted(set(ids))
    if len(ids)<5:raise ValueError("Need at least FIVE distinct real train MP4s for video-level split")
    shuffled=[ids[int(i)] for i in np.random.default_rng(seed).permutation(len(ids))]
    return dict(train=shuffled[2:],val=[shuffled[0]],test=[shuffled[1]])


def build_windows(records,split,strict=False):
    """Feature frames never include the eight future output timestamps."""
    out={k:[] for k in split}
    for rec in records:
        xy=np.asarray(rec["centers"],np.float32)
        z=np.asarray(rec["features"],np.float32)
        if xy.ndim!=2 or xy.shape[1]!=2 or z.ndim!=2 or len(xy)!=len(z):
            raise ValueError("Invalid feature+center recording")
        for start in range(0,len(xy)-PAST-FUTURE+1,4):
            ox=xy[start:start+PAST]
            future=xy[start+PAST:start+PAST+FUTURE]
            if not(np.isfinite(ox).all() and np.isfinite(future).all()):
                continue
            sample=dict(video=rec["video"],start=int(start),
                        z=z[start:start+PAST].copy(),xy=ox.copy(),
                        target=future.copy(),native_wh=rec["native_wh"])
            for group,ids in split.items():
                if rec["video"] in ids:out[group].append(sample)
    # Unit tests may inspect a single training video to prove no future-frame
    # leakage; the full experiment must have disjoint train/val/test windows.
    if not out.get("train"):
        raise RuntimeError("Not enough valid real TRAIN video windows: empty training partition")
    if strict and (not out.get("val") or not out.get("test")):
        raise RuntimeError("Strict training requires valid, independent train/val/test video windows")
    return out


def baseline_predict(sample,mode):
    past=sample["xy"]
    if mode=="persistence":return np.repeat(past[-1:],FUTURE,axis=0)
    if mode=="constant_velocity":
        steps=np.arange(1,FUTURE+1,dtype=np.float32)[:,None]
        return past[-1:]+steps*np.clip(past[-1:]-past[-2:-1],-.02,.02)
    raise ValueError(mode)


def pixel_error(pred,target,wh):
    scale=np.asarray(wh,np.float32)
    return np.linalg.norm((np.asarray(pred)-np.asarray(target))*scale,axis=-1)


def _batch(cases,device):
    z=np.stack([s["z"] for s in cases])
    xy=np.stack([s["xy"] for s in cases])
    tgt=np.stack([s["target"] for s in cases])
    return (torch.as_tensor(z,device=device),
            torch.as_tensor(xy,device=device),
            torch.as_tensor(tgt,device=device))


def _eval(model,group,device,mode):
    rows=[]
    if not group:return rows
    if model is not None:model.eval()
    with torch.no_grad():
        for sample in group:
            if model is None:
                pred=baseline_predict(sample,mode)
            else:
                z,xy,_=_batch([sample],device)
                pred=model(z,xy)[0].detach().cpu().numpy()
            target=sample["target"]
            err=pixel_error(pred,target,sample["native_wh"])
            movement=float(np.linalg.norm(
                (target[-1]-sample["xy"][-1])*np.asarray(sample["native_wh"])))
            rows.append(dict(mode=mode,video=sample["video"],start=sample["start"],
                 mean_error_px=float(err.mean()),end_error_px=float(err[-1]),
                 motion_px=movement,moving=bool(movement>=6),
                 pred_end=pred[-1].tolist(),target_end=target[-1].tolist()))
    return rows


def _extract(index_path,out,frames,model_id,device,batch_size=8):
    from transformers import AutoImageProcessor, AutoModel
    items=json.loads(Path(index_path).read_text())
    videos=[x for x in items if x.get("path","").startswith("human_demos/train/") and
            Path(x.get("local_video","")).is_file()]
    if len(videos)<5:
        raise RuntimeError("Need at least five genuine TRAIN MP4s, not eval IDs 0080-0099")
    processor=AutoImageProcessor.from_pretrained(model_id)
    backbone=AutoModel.from_pretrained(model_id).to(device).eval()
    for p in backbone.parameters():p.requires_grad_(False)
    loaded_name=getattr(backbone.config,"_name_or_path","")
    hidden=int(getattr(backbone.config,"hidden_size",0))
    if hidden<=0:raise RuntimeError("ViT pretrained backbone hidden_size not available")
    print("PRETRAINED_VIT_LOADED",json.dumps(dict(
        model=model_id,reported_checkpoint=loaded_name,hidden=hidden,
        params=sum(p.numel() for p in backbone.parameters()),
        device=str(device), pretrained=True)),flush=True)
    records=[]
    for entry in videos:
        path=Path(entry["local_video"])
        sampled,count=select_video_frames(path,target=frames)
        rgbs=[image for _,image in sampled]
        center=[];native_wh=[]
        for img in rgbs:
            mask,c=blue_candidate(img)
            center.append(c if c is not None else [float("nan"),float("nan")])
            native_wh=[int(img.shape[1]),int(img.shape[0])]
        tokens=[]
        with torch.inference_mode():
            for start in range(0,len(rgbs),batch_size):
                batch=processor(images=rgbs[start:start+batch_size],
                                return_tensors="pt")
                pixel_values=batch["pixel_values"].to(device)
                if pixel_values.shape[-2:]!=(224,224):
                    raise RuntimeError("Expected DINOv2 input 224x224")
                outputs=backbone(pixel_values=pixel_values)
                hidden_state=outputs.last_hidden_state
                if hidden_state.ndim!=3 or hidden_state.shape[1]<2:
                    raise RuntimeError("Pretrained backbone returned no patch tokens")
                # BOTH CLS and dense patch aggregate, no mask inputs to ViT.
                feats=torch.cat([hidden_state[:,0,:],
                                 hidden_state[:,1:,:].mean(dim=1)],dim=-1)
                tokens.append(feats.cpu().float().numpy())
        features=np.concatenate(tokens)
        if not np.isfinite(features).all():
            raise RuntimeError("ViT produced invalid features")
        records.append(dict(video=entry["path"],sha256=entry.get("source_sha256"),
             features=features,centers=np.asarray(center,np.float32),
             native_wh=native_wh,original_frames=count))
        print("VIT_REAL_VIDEO_ENCODED",json.dumps(dict(
            video=entry["path"],frames=len(rgbs),
            valid_centers=int(np.isfinite(center).all(axis=1).sum()),
            pretrained_features_shape=list(features.shape))),flush=True)
    del backbone
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return records,hidden*2


def run(index_path,out,epochs=15,seed=42,frames=48,batch_size=16,
        model_id=MODEL_ID,device="cuda"):
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("v1.1 ViT GPU pilot REQUIRES CUDA; no silent CPU fallback")
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    records,dim=_extract(index_path,out,frames,model_id,device)
    split=video_split([r["video"] for r in records],seed)
    samples=build_windows(records,split,strict=True)
    (out/"split.json").write_text(json.dumps(split,indent=2))
    baseline_rows=[]
    for name in ("persistence","constant_velocity"):
        for part in ("val","test"):
            for r in _eval(None,samples[part],device,name):
                baseline_rows.append(dict(split=part,**r))
    all_rows=list(baseline_rows);hist=[];reports=[]
    for mode in MODES:
        torch.manual_seed(seed)
        net=TemporalViTFlow(input_dim=dim).to(device)
        optimizer=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-3)
        rng=np.random.default_rng(seed)
        best=float("inf");best_state=None
        for epoch in range(epochs):
            net.train()
            order=rng.permutation(len(samples["train"]))
            losses=[]
            for start in range(0,len(order),batch_size):
                cases=[samples["train"][int(k)] for k in order[start:start+batch_size]]
                z,xy,target=_batch(cases,device)
                pred=net(z,xy)
                loss=select_loss(pred,target,mode)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            val_rows=_eval(net,samples["val"],device,mode)
            val_mean=float(np.mean([r["mean_error_px"] for r in val_rows]))
            hist.append(dict(mode=mode,epoch=epoch+1,
                 train_loss=float(np.mean(losses)),val_mean_error_px=val_mean))
            if val_mean<best:
                best=val_mean
                best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if best_state is None:raise RuntimeError("No valid validation results")
        net.load_state_dict(best_state)
        for part in ("val","test"):
            for row in _eval(net,samples[part],device,mode):
                all_rows.append(dict(split=part,**row))
        # Only the lightweight head/Transformer, not pretrained backbone.
        torch.save(dict(state=best_state,mode=mode,seed=seed,vit_model_id=model_id,
                        vit_pretrained=True,features_dim=dim,
                        split=split,valid_selection="val mean native-pixel error"),
                   out/f"{mode}_temporal_vit_head.pt")
        print("VIT_MODE_TRAINED",json.dumps(dict(
            mode=mode,epochs=epochs,best_val_px=best,
            test_windows=len(samples["test"]),actual_cuda=device)),flush=True)
    with (out/"per_window_errors.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        w.writeheader();w.writerows(all_rows)
    with (out/"training_history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(hist[0]))
        w.writeheader();w.writerows(hist)
    metrics=[]
    for mode in ("persistence","constant_velocity")+MODES:
        for part in ("val","test"):
            rows=[x for x in all_rows if x["mode"]==mode and x["split"]==part]
            if not rows:continue
            moving=[x for x in rows if x["moving"]]
            metrics.append(dict(mode=mode,split=part,windows=len(rows),
                  independent_videos=len(set(r["video"] for r in rows)),
                  mean_error_px=float(np.mean([r["mean_error_px"] for r in rows])),
                  end_error_px=float(np.mean([r["end_error_px"] for r in rows])),
                  moving_windows=len(moving),
                  moving_mean_error_px=(float(np.mean([r["mean_error_px"] for r in moving]))
                                        if moving else None)))
    with (out/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]))
        w.writeheader();w.writerows(metrics)
    winner={m["mode"]:m for m in metrics if m["split"]=="test"}
    report=dict(type="PRETRAINED_VIT_REAL_HUMAN_VIDEO_DIAGNOSTIC_NOT_ROBOT_POLICY",
        backbone=model_id,vit_pretrained=True,backbone_frozen=True,
        raw_rgb_input=True,rgb_resolution=[224,224],
        source_cameras="original front RGB, native-resolution HSV center labels",
        source_split="OFFICIAL_ROBOREEL_TRAIN_ONLY",
        data_sha256=hashlib.sha256(Path(index_path).read_bytes()).hexdigest(),
        seed=seed,epochs=epochs,cuda_device=torch.cuda.get_device_name(0),
        split=split,independent_test_videos=len(split["test"]),
        summary=metrics,learned_full_under_persistence=(
          winner.get("full",{}).get("mean_error_px",float("inf")) >
          winner.get("persistence",{}).get("mean_error_px",float("inf"))),
        hard_note=("HSV still needs manual ID validation. Six videos / one heldout "
          "source is a pipeline sanity test, not a paper result or an argument "
          "to abandon the research. The model predicts camera-plane 2D "
          "candidate centroids, NOT 3D effect, not object flow masks, "
          "not robot action, and no simulator success was measured. "
          "Do not use official RoboReel EVAL videos for checkpoint selection."))
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("VIT_GPU_PILOT_COMPLETE",json.dumps(dict(
       backbone=model_id,cuda=report["cuda_device"],
       independent_test_videos=len(split["test"]),results={
          name:float(v["mean_error_px"]) for name,v in winner.items()},
       report=str(out/"report.json"))),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description="Pretrained DINOv2-S ViT 224 RGB + temporal Transformer GPU diagnostic")
    p.add_argument("--index",default="outputs/next_gate_v100/train_data/downloaded_train_index.json")
    p.add_argument("--out",default="outputs/v110_baselines/vit_pilot")
    p.add_argument("--epochs",type=int,default=15)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--frames",type=int,default=48)
    p.add_argument("--batch",type=int,default=16)
    p.add_argument("--backbone",default=MODEL_ID)
    p.add_argument("--device",default="cuda")
    a=p.parse_args()
    run(a.index,a.out,a.epochs,a.seed,a.frames,a.batch,a.backbone,a.device)


if __name__=="__main__":main()
