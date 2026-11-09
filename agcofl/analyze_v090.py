"""Compare v0.9 trained-mask results against no-learning motion baselines.

Uses ACTUAL previously extracted MP4 candidate masks, not generated data.
With one video all metrics are TRAIN-ONLY, and incorrect blue color tracking
must be visually inspected before publication.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .real_human_supervision import make_windows


def mask_center(mask):
    yy,xx=np.nonzero(np.asarray(mask)>.5)
    if len(xx)==0:return None
    return np.array([float(np.mean(xx)),float(np.mean(yy))])


def move_mask(mask,dx,dy):
    """Translate binary object mask with zero padding (no toroidal wrap)."""
    a=np.asarray(mask,np.float32)
    h,w=a.shape
    x=int(round(dx));y=int(round(dy))
    out=np.zeros_like(a)
    x0=max(0,-x);x1=min(w,w-x)
    y0=max(0,-y);y1=min(h,h-y)
    if x1>x0 and y1>y0:
        out[y0+y:y1+y,x0+x:x1+x]=a[y0:y1,x0:x1]
    return out


def generate(pred_past,mode):
    if pred_past.shape!=(2,32,32):
        raise ValueError("Need two REAL observed 32x32 masks")
    if mode=="persistence":
        return np.repeat(pred_past[-1:,:,:],8,axis=0)
    if mode!="constant_velocity":
        raise ValueError(mode)
    c0=mask_center(pred_past[0]);c1=mask_center(pred_past[1])
    if c0 is None or c1 is None:
        return generate(pred_past,"persistence")
    v=np.clip(c1-c0,-8.,8.)
    return np.stack([move_mask(pred_past[-1],*(v*i)) for i in range(1,9)])


def mask_iou(a,b):
    a=np.asarray(a)>.5;b=np.asarray(b)>.5
    inter=int(np.logical_and(a,b).sum())
    union=int(np.logical_or(a,b).sum())
    return float(inter/union) if union else 1.


def run(folder,out):
    folder=Path(folder)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report=json.loads((folder/"report.json").read_text())
    status=report["validity"]
    if report["type"]!="REAL_HUMAN_VIDEO_MASK_PREDICTION_ONLY_NOT_ROBOT_TRANSFER":
        raise ValueError("Wrong source experiment")
    results=[]
    for file in sorted(folder.glob("observed_masks_*.npy")):
        uid=file.stem.removeprefix("observed_masks_")
        masks=np.load(file).astype(np.float32)
        windows=make_windows(masks,uid)
        for ex in windows:
            target=ex["future"]
            target_area=float(target.sum((1,2)).mean())
            for mode in ("persistence","constant_velocity"):
                pred=generate(ex["past"],mode)
                results.append(dict(video_id=uid,window_start=ex["index"],
                    mode=mode,full_iou=mask_iou(pred,target),
                    end_iou=mask_iou(pred[-1],target[-1]),
                    target_mean_area=target_area,
                    pred_mean_area=float(pred.sum((1,2)).mean())))
    if not results:
        raise RuntimeError("No observed real mask frames found; do not invent results")
    with (out/"motion_baseline_windows.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    summary=[]
    for mode in ("persistence","constant_velocity"):
        rows=[x for x in results if x["mode"]==mode]
        summary.append(dict(mode=mode,windows=len(rows),
            mean_full_iou=float(np.mean([r["full_iou"] for r in rows])),
            mean_end_iou=float(np.mean([r["end_iou"] for r in rows])),
            target_area=float(np.mean([r["target_mean_area"] for r in rows])),
            predicted_area=float(np.mean([r["pred_mean_area"] for r in rows]))))
    for r in report.get("summary",[]):
        if r["split"]=="train":
            summary.append(dict(mode=r["mode"],windows=r["windows"],
                mean_full_iou=r["mean_full_iou"],
                mean_end_iou=r["mean_end_iou"],
                target_area=None,predicted_area=
                   round(r["prediction_coverage"]*32*32,3)))
    with (out/"learning_vs_motion_baselines.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]))
        w.writeheader();w.writerows(summary)
    persistence=summary[0]["mean_full_iou"]
    full=next((x["mean_full_iou"] for x in summary if x["mode"]=="full"),None)
    diagnostics=dict(
        type="REAL_HUMAN_VIDEO_MASK_LEARNABILITY_GATE",
        input_report=str((folder/"report.json").resolve()),
        validity=status,
        windows=len({(x["video_id"],x["window_start"]) for x in results}),
        baseline_comparison=summary,
        learned_full_under_persistence=bool(full is not None and full<persistence),
        source_has_official_roboreel_eval_video=any("eval/" in x["source"] for x
            in json.loads((folder/"source_audit.json").read_text())),
        scientific_status="NO_VALID_GENERALIZATION_TEST" if status=="TRAIN_ONLY_NOT_GENERALIZATION" else "VIDEO_HELDOUT_ONLY_NOT_ROBOT_TASK",
        recommendation=("Inspect real MP4 overlays, correct segmentation or target motion modeling "
           "before robot-policy training; do not increase epochs blindly."),
        note="No real-robot policy was tested; learned IoUs from original report are TRAIN metrics for 0080."
    )
    (out/"learnability_gate.json").write_text(json.dumps(diagnostics,indent=2))
    print("REAL_LEARNABILITY_GATE",json.dumps({
        "validity":status,"persistence_full_iou":persistence,
        "learned_full_iou":full,
        "learned_under_persistence":diagnostics["learned_full_under_persistence"],
        "report":str(out/"learnability_gate.json")
    }),flush=True)
    return diagnostics


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="outputs/real_human_supervision_v090")
    p.add_argument("--out",default="outputs/next_gate_v100")
    args=p.parse_args()
    run(args.source,args.out)


if __name__=="__main__":main()
