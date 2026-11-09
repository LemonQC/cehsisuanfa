"""Inspect AUTHENTIC downloaded RoboReel TRAIN robot H5 actions (never synthesize).

Aligned with official process_training_robot_trajectories.py at pinned commit:
- Raw H5 groups: traj_0/actions and traj_0/env_states/articulations.
- qpos follows 13 root-pose+velocity cols, nine Panda DOFs.
- Sibling base_camera.mp4 and hand_camera.mp4 (robot RGB observations).

Only reads shape/value metadata and first/last sample; no privileged-model
claims or policy metrics. Any discrepancy is explicitly reported as blocked.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def audit_one(entry,root):
    import h5py,cv2
    files=entry["files"]
    paths={Path(p["relative_path"]).name:Path(root)/p["relative_path"]
           for p in files}
    h5_paths=[p for p in paths.values() if p.suffix==".h5" and ".state." not in p.name]
    if len(h5_paths)!=1:raise ValueError(f"Expected ONE raw state H5, got {h5_paths}")
    fpath=h5_paths[0]
    with h5py.File(fpath,"r") as h5:
        if "traj_0/actions" not in h5:
            raise ValueError("Official traj_0/actions absent")
        a=h5["traj_0/actions"]
        shp=tuple(a.shape)
        if len(shp)!=2 or shp[0]<2 or shp[1]<1:
            raise ValueError(f"Unexpected action tensor shape: {shp}")
        if "traj_0/env_states/articulations" not in h5:
            raise ValueError("Official articulation metadata absent")
        group=h5["traj_0/env_states/articulations"]
        names=list(group.keys())
        key="panda_wristcam_hd" if "panda_wristcam_hd" in names else names[0]
        art=group[key]
        if art.ndim!=2 or art.shape[1]<22:
            raise ValueError(f"Unexpected articulation layout {art.shape}")
        qpos=art[:min(10,art.shape[0]),13:22]
        start=np.asarray(a[:min(10,a.shape[0])],np.float64)
        finish=np.asarray(a[max(0,a.shape[0]-10):],np.float64)
        if not np.isfinite(start).all() or not np.isfinite(finish).all() or not np.isfinite(qpos).all():
            raise ValueError("Non-finite actions or joint states")
        action_preview=np.concatenate([start,finish],axis=0)
    videos={}
    for name in ("base_camera.mp4","hand_camera.mp4"):
        v=paths.get(name)
        if v is None or not v.is_file():raise FileNotFoundError(f"Missing official robot camera: {name}")
        cap=cv2.VideoCapture(str(v))
        ok,frame=cap.read()
        meta=dict(opened=bool(cap.isOpened()),read_frame=bool(ok),
                  frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                  width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        if not ok or meta["frame_count"]<2:raise ValueError(f"Invalid real RGB stream {v}")
        videos[name]=meta
    return dict(demo=entry["demo"],h5=str(fpath),action_shape=list(shp),
        articulation_name=key,articulation_shape=list(art.shape),
        qpos_9d_is_finite=True,action_min_preview=float(action_preview.min()),
        action_max_preview=float(action_preview.max()),
        camera_videos=videos,real_robot_actions=True,
        status="REAL_ROBOREEL_ACTION_LABELS_VALIDATED")

def run(manifest_path,data_root,out):
    m=json.loads(Path(manifest_path).read_text())
    if m.get("status")!="REAL_ROBOT_TRAIN_DATA_ACQUIRED":
        report=dict(status="BLOCKED_NO_REAL_ROBOT_TRAIN_ACTIONS",
          upstream_status=m.get("status"),error=m.get("error"))
    else:
        rows=[]
        for entry in m.get("downloaded",[]):
            try:rows.append(audit_one(entry,data_root))
            except Exception as e:
                rows.append(dict(demo=entry["demo"],status="INVALID_REAL_ROBOT_TRAJECTORY",
                                 error=f"{type(e).__name__}: {str(e)[:450]}"))
        report=dict(type="REAL_ROBOREEL_TRAIN_ACTION_H5_AUDIT",
                    status="REAL_ACTION_DATA_VALIDATED" if rows and all(
                    x["status"]=="REAL_ROBOREEL_ACTION_LABELS_VALIDATED" for x in rows)
                    else "REAL_ACTION_DATA_REQUIRES_REPAIR",
                    rows=rows,valid_demos=sum(
                    r["status"]=="REAL_ROBOREEL_ACTION_LABELS_VALIDATED" for r in rows),
                    contains_train_only=True,
                    note="This is real OFFICIAL ROBOT TRAIN DATA, not a policy success score.")
    out=Path(out);out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(report,indent=2))
    print("ROBOT_H5_AUDIT",json.dumps(dict(
       status=report["status"],valid_demos=report.get("valid_demos",0),
       report=str(out))),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",default="outputs/v110_baselines/robot_data_manifest.json")
    p.add_argument("--data-root",default="outputs/v110_baselines/official_dataset")
    p.add_argument("--out",default="outputs/v120_baselines/real_robot_h5_audit.json")
    args=p.parse_args()
    run(args.manifest,args.data_root,args.out)

if __name__=="__main__":main()
