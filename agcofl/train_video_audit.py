"""Audit genuine RoboReel TRAIN-split video mask candidates, without training.

HSV tracking coverage is NOT semantic accuracy. Produce original-frame overlays
for human inspection and quantitative consistency metrics.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .real_human_supervision import video_masks
from .real_video_flow import extract_video_flow


def audit(index_path,out,frames=48):
    index_path=Path(index_path)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not index_path.exists():
        result=dict(type="REAL_TRAIN_MASK_AUDIT",status="NO_REAL_TRAIN_VIDEOS",
                    videos=0,items=[],reason="No authenticated TRAIN-split videos acquired")
        (out/"train_video_audit.json").write_text(json.dumps(result,indent=2))
        print("TRAIN_MASK_GATE",json.dumps({"status":result["status"]}),flush=True)
        return result
    items=json.loads(index_path.read_text())
    report=[]
    for r in items:
        video=Path(r["local_video"])
        if not video.is_file():
            raise FileNotFoundError(video)
        # No new or modified source video pixels are generated.
        masks,meta=video_masks(video,frames)
        overlay=out/"video_overlays"/str(r["source_group"])
        extract_video_flow(video,overlay,frames=24)
        area=masks.sum((1,2))
        centers=[]
        for mask in masks:
            ys,xs=np.nonzero(mask>.5)
            centers.append(None if len(xs)==0 else [float(xs.mean()),float(ys.mean())])
        valid=np.asarray([x is not None for x in centers],bool)
        track=np.array([c for c in centers if c is not None],float)
        moves=np.linalg.norm(np.diff(track,axis=0),axis=-1) if len(track)>1 else np.zeros(0)
        entry=dict(
            uid=r["source_group"],
            original_train_path=r["path"],
            bytes=r["bytes"],sha256=r["source_sha256"],
            frames=int(len(masks)),mask_coverage=float(valid.mean()),
            mean_object_pixels=float(area.mean()),
            p95_object_pixels=float(np.percentile(area,95)),
            median_centroid_step_px=float(np.median(moves)) if len(moves) else 0.,
            p95_centroid_step_px=float(np.percentile(moves,95)) if len(moves) else 0.,
            masked_sequence_sha256=hashlib.sha256(masks.tobytes()).hexdigest(),
            overlay_png=str(overlay/"human_mask_overlay.png"),
            needs_manual_object_identity_check=True,
            camera="front")
        report.append(entry)
    result=dict(type="REAL_TRAIN_MASK_AUDIT",
        status="TRAINING_VIDEO_MASK_CANDIDATES_NEED_VISUAL_REVIEW",
        videos=len(report),items=report,
        note="No training until the RGB overlay is confirmed to track the intended object. "
             "Multiple front videos do not imply useful/dispensable intermediate events. "
             "No camera-to-robot calibration is assumed.")
    (out/"train_video_audit.json").write_text(json.dumps(result,indent=2))
    print("TRAIN_MASK_GATE",json.dumps(dict(status=result["status"],
          videos=len(report),report=str(out/"train_video_audit.json"))),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--index",default="outputs/next_gate_v100/train_data/downloaded_train_index.json")
    p.add_argument("--out",default="outputs/next_gate_v100")
    p.add_argument("--frames",type=int,default=48)
    args=p.parse_args()
    audit(args.index,args.out,args.frames)


if __name__=="__main__":main()
