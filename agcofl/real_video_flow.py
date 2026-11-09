"""Diagnostic blue-object occupancy masks from actual demonstration MP4s.

This deliberately does NOT claim instance segmentation; an HSV-selected blue
component may be wrong. Inspect overlays and acceptance metrics before any study.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def blue_candidate(rgb):
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # A conservative BLUE color candidate, not learned semantic segmentation.
    binary = cv2.inRange(hsv, (92, 45, 35), (135, 255, 255))
    h,w = binary.shape
    mask = np.zeros_like(binary)
    # Exclude image borders; many videos have overlays outside active work area.
    mask[int(.08*h):int(.94*h),int(.08*w):int(.94*w)] = binary[int(.08*h):int(.94*h),int(.08*w):int(.94*w)]
    n,labels,stats,cent = cv2.connectedComponentsWithStats(mask, 8)
    area_min = max(12, int(h*w*0.00004))
    area_max = int(h*w*.12)
    choices = [i for i in range(1,n) if area_min<=stats[i,cv2.CC_STAT_AREA]<=area_max]
    if not choices:
        return np.zeros((h,w),dtype=np.uint8),None
    # Prefer the largest coherent candidate. This can be confounded by blue distractors.
    i=max(choices,key=lambda k:int(stats[k,cv2.CC_STAT_AREA]))
    return np.uint8(labels==i), (float(cent[i,0])/w, float(cent[i,1])/h)


def select_video_frames(video_path, target=24):
    import cv2
    video_path=Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot decode {video_path}')
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count<=0:
        cap.release();raise RuntimeError('Video has no indexed frames')
    indexes=np.unique(np.linspace(0,count-1,min(target,count)).astype(int))
    frames=[]
    for idx in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx))
        ok,bgr=cap.read()
        if ok:frames.append((int(idx),cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
    cap.release()
    if len(frames)<2:raise RuntimeError('Too few decodable video frames')
    return frames,count


def extract_video_flow(video_path,out_dir=None,min_coverage=.5,frames=24):
    import cv2
    frames_rgb,count=select_video_frames(video_path,target=frames)
    masks=[];centers=[];valid=[]
    for _,rgb in frames_rgb:
        m,c=blue_candidate(rgb)
        masks.append(cv2.resize(m.astype(np.uint8),(32,32),interpolation=cv2.INTER_NEAREST))
        centers.append([float('nan'),float('nan')] if c is None else list(c))
        valid.append(c is not None)
    arr=np.stack(masks).astype(np.float32)
    # Fixed eight equally spaced sequence elements, preserving temporal order.
    loc=np.linspace(0,len(arr)-1,8).astype(int)
    flow=arr[loc]
    xy=np.asarray(centers,dtype=float)
    observed=xy[np.array(valid,dtype=bool)]
    coverage=float(np.mean(valid))
    travel=float(np.sum(np.linalg.norm(np.diff(observed,axis=0),axis=-1))) if len(observed)>1 else 0.
    # Motion magnitude is merely a robot-timing DIAGNOSTIC, NOT a world trajectory.
    gain=float(np.clip(.75+travel, .75, 1.25))
    sha=hashlib.sha256(Path(video_path).read_bytes()).hexdigest()
    report=dict(video=str(Path(video_path).resolve()),sha256=sha,frames_decoded=len(frames_rgb),
        video_frame_count=count,valid_blue_candidate_frames=int(sum(valid)),
        coverage=coverage,normalized_motion_path=travel,
        timing_gain=gain,accepted=bool(coverage>=min_coverage and np.count_nonzero(flow)>=4),
        limitation='HSV blue connected-component CANDIDATE, not verified segmentation; human screen pixel motion is not robot world XY')
    if out_dir is not None:
        out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
        np.save(out/'human_blue_candidate_masks_8x32.npy',flow)
        (out/'human_video_diagnostic.json').write_text(json.dumps(report,indent=2))
        canvas=np.hstack([cv2.resize((m*255).astype(np.uint8),(128,128),interpolation=cv2.INTER_NEAREST) for m in flow])
        cv2.imwrite(str(out/'mask_contact_sheet.png'),canvas)
    return flow,report
