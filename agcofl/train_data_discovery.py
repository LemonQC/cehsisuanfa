"""Discover official RoboReel *TRAINING* human/robot files via HF mirror.

Never download evaluation videos as training data. Metadata-first; download is
opt-in and size bounded. Network failures become explicit JSON blockers.
No hardcoded guess about the dataset's training directory layout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

REPO="LogosRoboticsLab/roboreel"
DEFAULT_ENDPOINT="https://hf-mirror.com"


def split_kind(path):
    parts=[p.lower() for p in str(path).replace("\\","/").split("/")]
    if any(x in parts for x in ("eval","evaluation","test","testing","val","validation")):
        return "evaluation"
    if not any(x in parts for x in ("train","training")):
        return "unknown"
    return "training"


def classify(path):
    p=str(path).lower()
    if split_kind(p)!="training":
        return None
    task=any(x in p for x in ("push_cube","pushcube","push-cube"))
    camera=any(x in p for x in ("front.mp4","front_camera.mp4"))
    if task and camera:
        return "human_video"
    if task and p.endswith((".h5",".hdf5",".npz",".pkl",".parquet",".jsonl")):
        return "train_trajectory_or_annotation"
    if task and p.endswith(".mp4"):
        return "other_camera_video"
    return None


def discover(endpoint=DEFAULT_ENDPOINT,max_entries=15000):
    os.environ["HF_ENDPOINT"]=endpoint.rstrip("/")
    os.environ.setdefault("HF_HUB_DISABLE_XET","1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT","10")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT","30")
    from huggingface_hub import HfApi
    api=HfApi(endpoint=endpoint.rstrip("/"))
    first=list(api.list_repo_tree(REPO,repo_type="dataset",recursive=False))
    roots=[dict(path=x.path,kind=x.__class__.__name__,
                size=int(getattr(x,"size",0) or 0)) for x in first]
    # Explore only directories likely to contain train data, not the
    # eval video hierarchy. If no train root exists, record this blocker.
    dirs=[x.path for x in first
          if "folder" in x.__class__.__name__.lower()
          and ("train" in x.path.lower() or
               x.path.lower() in ("human_demos","robot_demos","data","dataset"))]
    if not dirs:
        dirs=[x.path for x in first if "folder" in x.__class__.__name__.lower()]
    entries=[];limit_hit=False
    for prefix in dirs:
        # Avoid accidental scan of large known evaluation folders.
        if split_kind(prefix)=="evaluation":
            continue
        try:
            iterator=api.list_repo_tree(REPO,repo_type="dataset",
                                        path_in_repo=prefix,recursive=True)
            for entry in iterator:
                if "file" not in entry.__class__.__name__.lower():
                    continue
                kind=classify(entry.path)
                if kind is not None:
                    entries.append(dict(path=entry.path,kind=kind,
                                        bytes=int(getattr(entry,"size",0) or 0)))
                if len(entries)>=max_entries:
                    limit_hit=True
                    break
        except Exception as ex:
            entries.append(dict(error=f"{prefix}: {type(ex).__name__}: {str(ex)[:240]}",
                                kind="scan_error"))
        if limit_hit: break
    entries.sort(key=lambda x:(x.get("kind",""),x.get("path","")))
    return dict(dataset=REPO,endpoint=endpoint,
                roots=roots,candidates=entries,limit_hit=limit_hit,
                human_training_front_videos=sum(x["kind"]=="human_video" for x in entries),
                warning="Only genuine TRAINING paths accepted; eval/test paths rejected. "
                        "This is dataset file discovery, NOT a trained model.")


def download_selected(report,root,count=0,max_mb=300):
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    video_paths=[x for x in report["candidates"] if x["kind"]=="human_video"]
    selected=video_paths[:max(0,int(count))]
    if not selected:
        return dict(requested=int(count),downloaded=[],message="No selected real TRAIN human videos")
    for item in selected:
        if split_kind(item["path"])!="training":
            raise RuntimeError("Attempt to import evaluation sample into training: "+item["path"])
    size=sum(int(x["bytes"]) for x in selected)
    if any(x["bytes"]<=0 for x in selected):
        raise RuntimeError("Remote MP4 sizes unknown; refusing unbounded transfer")
    if size>int(max_mb)*1024*1024:
        raise RuntimeError(f"Selected files total {size/1e6:.1f} MB > budget {max_mb} MB")
    from huggingface_hub import hf_hub_download
    indexed=[]
    for item in selected:
        path=item["path"]
        file=Path(hf_hub_download(repo_id=REPO,repo_type="dataset",
             filename=path,local_dir=str(root/"hf_cache")))
        if file.stat().st_size<512:
            raise RuntimeError(f"Incomplete downloaded training video {file}")
        # Verify full MP4 decode rather than accepting an HTTP error page.
        from .download_roboreel_cn import valid_video
        if not valid_video(file):
            raise RuntimeError(f"Not a fully decodable MP4 {file}")
        uid=hashlib.sha256(path.encode()).hexdigest()[:12]
        link=root/"front_videos"/uid/"front.mp4"
        link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():
            link.symlink_to(file.resolve())
        indexed.append(dict(path=path,source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                            bytes=file.stat().st_size,local_video=str(link),
                            source_group=uid))
        print("REAL_TRAIN_VIDEO_READY",json.dumps(indexed[-1]),flush=True)
    (root/"downloaded_train_index.json").write_text(json.dumps(indexed,indent=2))
    return dict(requested=count,downloaded=indexed,
                real_video_root=str(root/"front_videos"))


def main(argv=None):
    p=argparse.ArgumentParser(description="Discover genuine RoboReel train split, no eval leakage")
    p.add_argument("--endpoint",default=os.getenv("HF_ENDPOINT",DEFAULT_ENDPOINT))
    p.add_argument("--out",default="outputs/roboreel_real_train_gate")
    p.add_argument("--download",type=int,default=0,
                   help="Opt in to download this many verified TRAIN front MP4s")
    p.add_argument("--max-mb",type=int,default=300)
    args=p.parse_args(argv)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    try:
        report=discover(args.endpoint)
        report["downloads"]=download_selected(report,out,args.download,args.max_mb)
        report["status"]=("TRAIN_VIDEOS_ACQUIRED"
                          if report["downloads"].get("downloaded")
                          else "DATA_DISCOVERY_ONLY")
    except Exception as ex:
        report=dict(dataset=REPO,endpoint=args.endpoint,status="DATA_ACCESS_BLOCKED",
                    error=f"{type(ex).__name__}: {str(ex)[:700]}",
                    next_step="Do not use eval videos as training; obtain official train split or record new real demos")
    path=out/"train_discovery.json"
    path.write_text(json.dumps(report,indent=2))
    print("TRAIN_DATA_GATE",json.dumps({k:v for k,v in report.items()
            if k not in ("roots","candidates","downloads")}),flush=True)
    print("TRAIN_DATA_REPORT",str(path),flush=True)
    return report


if __name__=="__main__":main()
