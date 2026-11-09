"""Official RoboReel TRAIN human 3D pose/keypoint annotation availability audit.

Never assume raw MP4 blue-color centroids are 3D object pose. Inspect metadata
for actual published annotations and calibration files before selecting a
pose-based cross-behavior experiment. No binary downloads by default.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

DATASET="LogosRoboticsLab/roboreel"
ENDPOINT="https://hf-mirror.com"
FOLDERS=[
    "human_demos/train/no_distraction/push_cube/0000",
    "human_demos/train/no_distraction/pick_cube/0000",
    "human_demos/train/no_distraction/pick_cube",
    "human_demos/train/no_distraction/push_cube",
]

def _category(path):
    name=Path(path).name.lower()
    if any(x in name for x in ("keypoint","pose","calib","intrinsic","extrinsic","annotation","tracking")):
        return "POTENTIAL_3D_OR_CALIBRATION"
    if name.endswith(".mp4"):
        return "RGB_VIDEO"
    if "depth" in name:
        return "DEPTH_NOT_NEEDED_FOR_THIS_AUDIT"
    return "UNKNOWN_TYPE"

def scan(endpoint=ENDPOINT,max_files=120):
    from huggingface_hub import HfApi
    api=HfApi(endpoint=endpoint)
    data=[]
    for folder in FOLDERS:
        entries=[]
        try:
            # This is ONLY file listing, not actual downloading or decoding.
            for item in api.list_repo_tree(DATASET,repo_type="dataset",
                                           path_in_repo=folder,recursive=False):
                entries.append(dict(path=item.path,type=item.__class__.__name__,
                                    bytes=int(getattr(item,"size",0) or 0),
                                    category=_category(item.path)))
                if len(entries)>=max_files:break
            data.append(dict(folder=folder,status="LISTED",entries=entries))
        except Exception as ex:
            data.append(dict(folder=folder,status="SCANNING_FAILED",
                             error=f"{type(ex).__name__}: {str(ex)[:400]}",entries=[]))
    found=[x for d in data for x in d["entries"]
           if x["category"]=="POTENTIAL_3D_OR_CALIBRATION"]
    return dict(type="REAL_TRAIN_ANNOTATION_AVAILABILITY",
      source="OFFICIAL_ROBOREEL_HF_DATASET",endpoint=endpoint,
      tasks=["pick_cube","push_cube"],folders=data,possible_annotation_files=found,
      status=("POSE_FILE_CANDIDATES_FOUND_NEEDS_FORMAT_VALIDATION" if found
              else "NO_VERIFIED_3D_FILE_AT_INSPECTED_LEVEL"),
      warning=("A file name is not proof of calibrated SE(3) object trajectories. "
               "Do not treat front-camera HSV centers as 3D pose. "
               "Different PickCube and PushCube tasks have different official goals."))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--endpoint",default=os.getenv("HF_ENDPOINT",ENDPOINT))
    p.add_argument("--out",default="outputs/v110_baselines/human_pose_annotation_audit.json")
    args=p.parse_args()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    try:r=scan(args.endpoint)
    except Exception as ex:
        r=dict(type="REAL_TRAIN_ANNOTATION_AVAILABILITY",status="ANNOTATION_METADATA_BLOCKED",
          error=f"{type(ex).__name__}: {str(ex)[:700]}")
    out.write_text(json.dumps(r,indent=2))
    print("POSE_SOURCE_AVAILABILITY",json.dumps(
       {k:v for k,v in r.items() if k not in ("folders","possible_annotation_files")}),flush=True)

if __name__=="__main__":main()
