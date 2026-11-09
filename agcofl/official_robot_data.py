"""Download a BOUNDED official RoboReel robot TRAIN subset for baseline smoke.

This uses the upstream published raw state-mode H5 and robot camera videos,
with their original task/demonstration folders. It does NOT generate actions.
Default task is push_cube and first two independent official robot demos.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

DATASET="LogosRoboticsLab/roboreel"
# Published Hugging Face dataset uses no_distraction; official model code
# translates the internal alias without_distraction at the raw-data boundary.
BASE="robot_trajectories/no_distraction"
ENDPOINT="https://hf-mirror.com"


def plan(entries,task="push_cube",max_demos=2,max_mb=300):
    root=f"{BASE}/{task}/"
    groups=defaultdict(dict)
    for item in entries:
        p=item["path"];parts=p.split("/")
        if not p.startswith(root) or len(parts)<5:
            continue
        demo=parts[3]
        name=parts[-1]
        if name.endswith(".h5") and ".state." not in name:
            groups[demo]["h5"]=item
        elif name=="base_camera.mp4":
            groups[demo]["base_camera"]=item
        elif name=="hand_camera.mp4":
            groups[demo]["hand_camera"]=item
    selected=[]
    for demo in sorted(groups):
        g=groups[demo]
        if all(x in g for x in ("h5","base_camera","hand_camera")):
            selected.append(dict(demo=demo,files=[g[k] for k in ("h5","base_camera","hand_camera")]))
        if len(selected)>=max_demos:break
    total=sum(int(v["bytes"]) for d in selected for v in d["files"])
    if any(int(v["bytes"])<=0 for d in selected for v in d["files"]):
        raise ValueError("Some upstream robot file sizes unknown; refusing unbounded downloads")
    if total>int(max_mb)*1024*1024:
        raise ValueError(f"Selected robot subset {total/1e6:.1f} MB > {max_mb} MB budget")
    return dict(task=task,demos=selected,bytes=total,
                budget_mb=max_mb,status="ROBOT_TRAIN_SUBSET_PLANNED" if selected else "ROBOT_H5_OR_REQUIRED_CAMERAS_MISSING")


def discover(task="push_cube",endpoint=ENDPOINT,max_entries=8000):
    os.environ["HF_ENDPOINT"]=endpoint
    os.environ.setdefault("HF_HUB_DISABLE_XET","1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT","10")
    from huggingface_hub import HfApi
    api=HfApi(endpoint=endpoint)
    root=f"{BASE}/{task}"
    files=[]
    for obj in api.list_repo_tree(DATASET,repo_type="dataset",
                                  path_in_repo=root,recursive=True):
        if "file" in obj.__class__.__name__.lower():
            files.append(dict(path=obj.path,bytes=int(getattr(obj,"size",0) or 0)))
        if len(files)>=max_entries:break
    return files


def download(plan_data,output_root,endpoint=ENDPOINT):
    from huggingface_hub import hf_hub_download
    output=Path(output_root);output.mkdir(parents=True,exist_ok=True)
    found=[]
    for demo in plan_data["demos"]:
        paths=[]
        for item in demo["files"]:
            file=Path(hf_hub_download(repo_id=DATASET,repo_type="dataset",
                  filename=item["path"],local_dir=str(output)))
            if file.stat().st_size!=int(item["bytes"]):
                raise RuntimeError(f"Size mismatch on {file}")
            paths.append(dict(relative_path=item["path"],
                 bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        found.append(dict(demo=demo["demo"],files=paths))
        print("REAL_ROBOT_TRAIN_DEMO_READY",json.dumps(dict(
             demo=demo["demo"],files=len(paths))),flush=True)
    return found


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",default="push_cube")
    p.add_argument("--max-demos",type=int,default=2)
    p.add_argument("--max-mb",type=int,default=300)
    p.add_argument("--endpoint",default=os.getenv("HF_ENDPOINT",ENDPOINT))
    p.add_argument("--out",default="outputs/v110_baselines")
    p.add_argument("--data-root",default="outputs/v110_baselines/official_dataset")
    p.add_argument("--download",action="store_true")
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    try:
        entries=discover(args.task,args.endpoint)
        d=plan(entries,args.task,args.max_demos,args.max_mb)
        d["candidate_file_count"]=len(entries)
        d["endpoint"]=args.endpoint
        if args.download and d["demos"]:
            d["downloaded"]=download(d,args.data_root,args.endpoint)
            d["status"]="REAL_ROBOT_TRAIN_DATA_ACQUIRED"
        else:d["downloaded"]=[]
    except Exception as exc:
        d=dict(task=args.task,status="ROBOT_DATA_BLOCKED",
               error=f"{type(exc).__name__}: {str(exc)[:600]}",
               downloaded=[],next_step="Never manufacture robot actions")
    (out/"robot_data_manifest.json").write_text(json.dumps(d,indent=2))
    print("OFFICIAL_ROBOT_DATA_GATE",json.dumps({
        k:v for k,v in d.items() if k not in ("demos","downloaded")}),flush=True)
    return d


if __name__=="__main__":main()
