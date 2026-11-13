#!/usr/bin/env python3
"""Offline ingester for TWO ORIGINAL human counterfactual goal videos.

No downloads, synthetic videos, dataset mirrors, inference, training or PhysX.
Produces an immutable draft manifest and 3-frame/clip review strip. Every
human attestation is FALSE by default. The G0 auditor remains fail-closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

DEFAULT_REL = Path("outputs/g0_human_goal_pairs")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for buf in iter(lambda: stream.read(1024*1024), b""):
            h.update(buf)
    return h.hexdigest()


def project_file(project: Path, supplied: Path) -> Path:
    project = project.resolve()
    full = (supplied if supplied.is_absolute() else project / supplied).expanduser().resolve()
    if not full.is_relative_to(project):
        raise ValueError("OUT_OF_PROJECT_INPUT_NOT_ALLOWED")
    # Never use official EVAL media as research development input.
    if "/human_demos/eval/" in full.as_posix():
        raise ValueError("HELDOUT_EVAL_MEDIA_NOT_ALLOWED")
    return full


def read_clips(project: Path, left: Path, right: Path, probe=None) -> list[dict[str, Any]]:
    """Use the existing G0 ffmpeg validator; only cryptographically distinct originals."""
    if probe is None:
        def probe(path):
            from scripts.audit_goal_contrast_g0 import probe_video
            return probe_video(path)
    files=[project_file(project,left),project_file(project,right)]
    if files[0]==files[1]:
        raise ValueError("SAME_VIDEO_PATH_CANNOT_BE_OPPOSING_GOALS")
    records=[]
    for label,path in zip(("-x","+x"),files):
        if not path.is_file() or path.suffix.lower() not in (".mp4", ".mov", ".webm"):
            raise ValueError("MISSING_OR_UNSUPPORTED_REAL_HUMAN_VIDEO: "+str(path))
        size=path.stat().st_size
        if not 1024<size<=1024**3:
            raise ValueError("REAL_VIDEO_BYTES_OUT_OF_BOUNDS")
        info=probe(path)
        if not info.get("ok"):
            raise ValueError("REAL_VIDEO_DECODE_FAILED: "+str(info.get("reason")))
        records.append({"goal_robot_x":label,"path":str(path.relative_to(project)),
                        "sha256":file_sha256(path),"bytes":size,"video_probe":info,
                        "source_type":"self_recorded",
                        # These are intentionally NOT declared true by the machine.
                        "rights":"REQUIRES_HUMAN_REVIEW",
                        "human_hand_demonstration":False,
                        "initial_scene_equivalence_attested":False,
                        "camera_to_robot_x_mapping_documented":False,
                        "visible_target_region":""})
    if records[0]["sha256"]==records[1]["sha256"]:
        raise ValueError("IDENTICAL_HUMAN_VIDEO_BYTES_REUSED_AS_COUNTERFACTUAL")
    return records


def review_strip(project: Path, records: list[dict[str,Any]], out: Path) -> str:
    """A visual helper, not a human task-semantics verifier."""
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as exc:
        return "OPTIONAL_IMAGE_STACK_UNAVAILABLE: "+type(exc).__name__
    tile_w,tile_h=320,210
    canvas=Image.new("RGB", (3*tile_w,2*(tile_h+35)),"white")
    drawer=ImageDraw.Draw(canvas)
    for row,item in enumerate(records):
        cap=cv2.VideoCapture(str(project/item["path"]))
        if not cap.isOpened():
            return "VIDEO_CAPTURE_NOT_OPENED"
        try:
            frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frames<3:return "NOT_ENOUGH_DECODED_FRAMES_FOR_VISUAL_REVIEW"
            indices=(0,frames//2,frames-1)
            for col,frame_id in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
                ok,bgr=cap.read()
                if not ok:return "FRAME_DECODE_FAILED_AT_INDEX_"+str(frame_id)
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                img=Image.fromarray(rgb)
                img.thumbnail((tile_w,tile_h))
                left=col*tile_w+(tile_w-img.width)//2
                top=row*(tile_h+35)+(tile_h-img.height)//2
                canvas.paste(img,(left,top))
                drawer.text((col*tile_w+8,row*(tile_h+35)+tile_h+5),
                           f"{item['goal_robot_x']} video frame {frame_id} / {frames}",fill="black")
        finally:
            cap.release()
    canvas.save(out,quality=90)
    return "REVIEW_STRIP_CREATED_NOT_SEMANTICS_CERTIFIED"


def write_immutable(out: Path, obj:dict[str,Any],prefix:str,suffix:str) -> Path:
    payload=json.dumps(obj,sort_keys=True,ensure_ascii=False,indent=2)+"\n"
    digest=hashlib.sha256(payload.encode()).hexdigest()[:16]
    path=out/(prefix+"_"+digest+suffix)
    out.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf8")!=payload:
            raise FileExistsError("PREVIOUS_DRAFT_CANNOT_BE_OVERWRITTEN")
    else:
        path.write_text(payload,encoding="utf8")
    return path


def build_draft(records:list[dict[str,Any]]) -> dict[str,Any]:
    return {"target_task":"video_goal_push","clips":[{
        key:value for key,value in rec.items() if key not in ("video_probe","bytes")
    } for rec in records],"status":"G0_CANDIDATE_DRAFT_UNREVIEWED",
    "manual_review_required":True,
    "instructions":"Inspect review strip & original clips; fill only factual rights/scene/mapping attestations. Original SHA256 must not change."}


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--left",type=Path,default=DEFAULT_REL/"raw/left.mp4")
    parser.add_argument("--right",type=Path,default=DEFAULT_REL/"raw/right.mp4")
    parser.add_argument("--out",type=Path,default=DEFAULT_REL/"review")
    args=parser.parse_args(argv)
    project=args.project.resolve()
    out=project_file(project,args.out)
    try:
        records=read_clips(project,args.left,args.right)
    except (ValueError,OSError) as exc:
        print("TRO1_G0_RECORDING_REQUIRED",json.dumps({
            "status":"G0_NO_VERIFIED_HUMAN_PAIR", "reason":str(exc)[:350],
            "expected_original_recordings":[str(args.left),str(args.right)],
            "new_physx_actions":0,"new_model_training":0},ensure_ascii=False),flush=True)
        return 2
    draft=build_draft(records)
    saved=write_immutable(out,draft,"pair_draft",".json")
    image_path=out/(saved.stem+"_review.jpg")
    if image_path.exists():
        image_status="ALREADY_EXISTS_UNTOUCHED"
    else:
        image_status=review_strip(project,records,image_path)
    print("TRO1_G0_PAIR_PREPARED",json.dumps({
        "status":"MANUAL_REVIEW_REQUIRED_NOT_G0_PASSED",
        "manifest_draft":str(saved),
        "review_strip":str(image_path) if image_path.is_file() else None,
        "image_status":image_status,"source_video_sha256":[x["sha256"] for x in records],
        "new_physx_actions":0},ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
