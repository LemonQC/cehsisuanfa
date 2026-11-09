"""Download exactly one RoboReel evaluation MP4 through an optional China-accessible mirror.

No install/upgrade of torch, ManiSkill or NumPy. Uses existing huggingface_hub.
HF_ENDPOINT must be configured BEFORE importing huggingface_hub.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


DATASET_ID = "LogosRoboticsLab/roboreel"


def video_relative_path(episode_id=80, camera="front"):
    if camera not in ("front","left","right","egocentric"):
        raise ValueError("unsupported camera")
    if not 80 <= int(episode_id) <= 99:
        raise ValueError("RoboReel evaluation human demo IDs are 0080-0099")
    return "human_demos/eval/no_distraction/push_cube/%04d/%s.mp4" % (episode_id, camera)


def valid_video(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 512:
        return False
    with path.open("rb") as handle:
        header = handle.read(64)
    if b"ftyp" not in header[:24]:
        return False
    # A partial timed-out download can have a valid MP4 header. Require
    # actual decodable frames before treating it as ready for experiments.
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 2:
            return False
        ok_first, _ = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count - 1))
        ok_last, _ = cap.read()
        return bool(ok_first and ok_last)
    finally:
        cap.release()


def ensure_video(root, episode_id=80, camera="front", endpoint="https://hf-mirror.com"):
    root = Path(root).resolve()
    relative = video_relative_path(episode_id, camera)
    target = root/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid_video(target):
        print("LOCAL_VIDEO_READY", target, target.stat().st_size, flush=True)
        return target
    if target.exists():
        raise RuntimeError("Invalid existing MP4 at %s; check partial download; will not overwrite" % target)

    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "25")
    try:
        from huggingface_hub import hf_hub_download
        print("DOWNLOAD_ONE_MP4", json.dumps({
            "endpoint": os.environ.get("HF_ENDPOINT"),
            "repo": DATASET_ID, "relative_path":relative
        }), flush=True)
        resolved = hf_hub_download(
            repo_id=DATASET_ID, repo_type="dataset", filename=relative,
            local_dir=str(root), etag_timeout=10
        )
    except Exception as exc:
        raise RuntimeError(
            "Single-file mirror download failed: %s: %s. "
            "Mirror reachability is NOT guaranteed on this server. "
            "You may copy the genuine RoboReel MP4 to %s. "
            "For physics-only validation, run the bridge with --no-video."
            % (type(exc).__name__,str(exc), target)
        ) from exc
    if not valid_video(target):
        raise RuntimeError("Downloaded file is not a valid MP4: %s (hub resolved %s)" %
                           (target, resolved))
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    meta={"repo":DATASET_ID,"filename":relative,
          "endpoint":os.environ.get("HF_ENDPOINT","upstream"),
          "sha256":digest,"bytes":target.stat().st_size}
    target.with_suffix(".download.json").write_text(json.dumps(meta,indent=2))
    print("VIDEO_READY",json.dumps(meta),flush=True)
    return target


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="third_party/roboreel")
    p.add_argument("--demo-id",type=int,default=80)
    p.add_argument("--camera",default="front")
    p.add_argument("--endpoint",default=os.environ.get("HF_ENDPOINT","https://hf-mirror.com"))
    args=p.parse_args(argv)
    ensure_video(args.root,args.demo_id,args.camera,args.endpoint)


if __name__=="__main__":
    main()
