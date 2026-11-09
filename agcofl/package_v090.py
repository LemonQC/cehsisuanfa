"""Package authentic human MP4, measured physics replay and all v0.9 diagnostics.

This does not rerun simulation or download anything. Zip intended for one
single next-turn upload (source authenticity is preserved with SHA256).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def package(root=".",out="outputs/real_human_supervision_v090/review_bundle.zip"):
    root=Path(root).resolve()
    target=(root/out).resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    entries=[
      ("human_video/0080_front_original.mp4",
       "third_party/roboreel/human_demos/eval/no_distraction/push_cube/0080/front.mp4"),
      ("physics/physical_xy_replay.mp4",
       "outputs/roboreel_cn/fixed/episode_0000/physical_xy_replay.mp4"),
      ("physics/physical_xy_replay.png",
       "outputs/roboreel_cn/fixed/episode_0000/physical_xy_replay.png"),
      ("physics/episode_result.json",
       "outputs/roboreel_cn/fixed/episode_0000/episode_result.json"),
      ("masks/human_mask_overlay.png",
       "outputs/real_human_supervision_v090/video_0080/human_mask_overlay.png"),
      ("masks/three_mode_predictions.png",
       "outputs/real_human_supervision_v090/three_mode_predictions.png"),
      ("metrics/report.json",
       "outputs/real_human_supervision_v090/report.json"),
      ("metrics/source_audit.json",
       "outputs/real_human_supervision_v090/source_audit.json"),
      ("metrics/video_split.json",
       "outputs/real_human_supervision_v090/video_split.json"),
      ("metrics/mode_summary.csv",
       "outputs/real_human_supervision_v090/mode_summary.csv"),
      ("metrics/training_history.csv",
       "outputs/real_human_supervision_v090/training_history.csv")
    ]
    entries_found=[];missing=[]
    for arc,local in entries:
        path=root/local
        if not path.is_file():
            missing.append(local)
        else:
            entries_found.append((arc,path))
    if not (root/"outputs/real_human_supervision_v090/report.json").exists():
        raise FileNotFoundError("v0.9 training report missing; run the pilot script first")
    inventory=[]
    with zipfile.ZipFile(target,"w") as writer:
        for arc,path in entries_found:
            is_mp4=path.suffix.lower()==".mp4"
            writer.write(path,arcname=arc,
                         compress_type=zipfile.ZIP_STORED if is_mp4 else zipfile.ZIP_DEFLATED)
            h=hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda:f.read(1024*1024),b""):
                    h.update(chunk)
            inventory.append(dict(zip_path=arc,source=str(path.relative_to(root)),
                                  bytes=path.stat().st_size,sha256=h.hexdigest()))
        info=dict(type="REAL_HUMAN_VIDEO_PLUS_MEASURED_PHYSICS_DEVELOPMENT_ONLY",
                  included=inventory,missing=missing,
                  notes="The human MP4 is authentic RoboReel evaluation data. "
                        "Physical XY MP4 is NOT rendered robot RGB. "
                        "Training metrics are same-video reconstruction, NOT generalization, "
                        "and NO learned policy was run in RoboReel.")
        writer.writestr("BUNDLE_MANIFEST.json",json.dumps(info,indent=2))
    print("REVIEW_BUNDLE_READY",json.dumps(dict(path=str(target),
          total_files=len(entries_found),missing=missing,
          bytes=target.stat().st_size)),flush=True)
    return target


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default=".")
    p.add_argument("--out",default="outputs/real_human_supervision_v090/review_bundle.zip")
    a=p.parse_args()
    package(a.root,a.out)


if __name__=="__main__":
    main()
