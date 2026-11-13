#!/usr/bin/env python3
"""G0: strictly read-only audit of genuine human-video goal contrast.

No downloads, physics engines, pretrained model, simulator, or training.
An absent/insufficient pair is BLOCKED. Two operator-supplied authentic videos
remain CANDIDATE_REVIEW_REQUIRED until independently visually reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

TASK_GOALS = {"-x", "+x"}
SOURCE_TYPES = {"self_recorded", "licensed_external"}
ALLOWED_EXT = {".mp4", ".webm", ".mov"}
MAX_CLIP = 1024 * 1024 * 1024
CATALOG = [
    {
        "dataset": "RoboReel PushCube-v2",
        "url": "https://github.com/gupta-anmol99/roboreel/blob/afaf4a03df0be1e2347becde96da4f34e0de83fc/src/envs/push_cube.py",
        "actual_human_video_pair_verified": False,
        "opposing_goals_verified": False,
        "reason": "Policy task's target is fixed at cube initial X +0.15m. Different demo IDs retain the same goal.",
        "disposition": "NOT_G0"
    },
    {
        "dataset": "RoboReel StackCups-v1",
        "url": "https://github.com/gupta-anmol99/roboreel/blob/afaf4a03df0be1e2347becde96da4f34e0de83fc/src/envs/stack_cups.py",
        "actual_human_video_pair_verified": False,
        "opposing_goals_verified": False,
        "reason": "Stack sequence dark_blue -> red -> light_blue is fixed, all authentic TRAIN demos share task goal.",
        "disposition": "NOT_G0"
    },
    {
        "dataset": "Something-Something V2",
        "url": "https://www.qualcomm.com/developer/software/something-something-v-2-dataset",
        "license_url": "https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/jester_something_something_exercise_research_license_final_qti_28jul2022.pdf",
        "actual_human_video_pair_verified": False,
        "opposing_goals_verified": "class_labels_only",
        "classes": ["Pushing something from left to right", "Pushing something from right to left"],
        "reason": "Official directional HUMAN action categories exist. No licensed clips or same-object/frame-matched pair checked. ~19.4GB corpus with research-use agreement; never download blindly.",
        "disposition": "CANDIDATE_METADATA_ONLY"
    },
    {
        "dataset": "H2RBench (CoRL 2026)",
        "url": "https://h2rbench.github.io/",
        "actual_human_video_pair_verified": False,
        "opposing_goals_verified": False,
        "reason": "Human/robot demos exist for four fixed-goal manipulation families, but no verified same-scene opposing-goal pair on the project page.",
        "disposition": "NO_VERIFIED_PAIR"
    },
    {
        "dataset": "EgoDex",
        "url": "https://machinelearning.apple.com/research/egodex-learning-dexterous-manipulation",
        "actual_human_video_pair_verified": False,
        "opposing_goals_verified": False,
        "reason": "Large diverse real-human videos; no locally certified same-scene, opposite-goal pair or embedding-to-ManiSkill mapping. Dataset scope and license must be checked before transfer.",
        "disposition": "NO_VERIFIED_PAIR"
    }
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def check_authored_task(project: Path) -> dict[str, Any]:
    root = project / "third_party/roboreel_models/roboreel_env/src/envs"
    push = root / "push_cube.py"
    cups = root / "stack_cups.py"
    r = {"upstream_root": str(root), "push_source_verified": False,
         "cups_source_verified": False, "target_contrast_provided": False}
    if push.is_file():
        text = push.read_text(encoding="utf-8")
        r["push_source_verified"] = bool(re.search(r"PUSH_DISTANCE\s*=\s*0\.15\b", text)
                                         and re.search(r"target_xyz.*?\[\.\.\.,\s*0\].*?\+=\s*self\.PUSH_DISTANCE", text))
        r["push_source_sha256"] = sha256(push)
    if cups.is_file():
        text = cups.read_text(encoding="utf-8")
        r["cups_source_verified"] = (all(s in text for s in
            ("dark_blue_cup", "red_cup", "light_blue_cup", "StackCups-v1")))
        r["cups_source_sha256"] = sha256(cups)
    return r


def inspect_local_prepared_media(project: Path) -> list[dict[str, Any]]:
    root = project / "outputs/v120_baselines/official_eval_data/human_demos/train/no_distraction"
    result = []
    for task in ("push_cube", "stack_cups", "toys_in_drawer"):
        found = []
        for identifier in (0, 1):
            video = root / task / f"{identifier:04d}/front.mp4"
            if video.is_file() and video.stat().st_size > 1024:
                found.append({"id": identifier, "bytes": video.stat().st_size})
        result.append({"task": task, "valid_existing_media_files": found,
            "goal_contrast_verified": False,
            "reason": "Source goal is fixed within task; physical media presence is not opposing goal evidence."})
    return result


def within_project(path: Path, project: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(project.resolve()):
        raise ValueError("SOURCE_MUST_BE_WITHIN_PROJECT_TO_PREVENT_UNCONTROLLED_INPUTS")
    return resolved


def probe_video(path: Path) -> dict[str, Any]:
    """Use installed ffprobe; no codec inference or fake signatures from extensions."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,duration,nb_frames,codec_name",
           "-of", "json", str(path)]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=12,
                            check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "reason": type(e).__name__}
    if cp.returncode != 0:
        return {"ok": False, "reason": "FFPROBE_FAILED"}
    # Prove at least one actual decoded frame, not just container metadata.
    try:
        decoded = subprocess.run(["ffmpeg", "-hide_banner", "-v", "error",
            "-i", str(path), "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=20, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "reason": "DECODER_" + type(e).__name__}
    if decoded.returncode != 0:
        return {"ok": False, "reason": "FIRST_FRAME_DECODE_FAILED"}
    try:
        stream = json.loads(cp.stdout)["streams"][0]
        w = int(stream["width"]); h = int(stream["height"])
        dur = float(stream.get("duration", "nan"))
        # Some container formats do not supply stream.duration.
        frames = stream.get("nb_frames")
        if w < 64 or h < 64:
            return {"ok": False, "reason": "FRAME_TOO_SMALL"}
        return {"ok": True, "width": w, "height": h,
                "duration_sec": dur if dur == dur and dur > 0 else None,
                "frames_reported": frames, "codec_name": str(stream.get("codec_name", "unknown"))}
    except (ValueError, IndexError, KeyError, TypeError):
        return {"ok": False, "reason": "INVALID_FFPROBE_STREAM"}


def audit_pair(manifest_path: Path, project: Path) -> dict[str, Any]:
    """Audits files and provenance declarations, never human task semantics."""
    try:
        manifest_path = within_project(manifest_path, project)
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as e:
        return {"status": "G0_BLOCKED", "reason": "BAD_INPUT_MANIFEST:" + type(e).__name__}
    clips = raw.get("clips")
    if not isinstance(clips, list) or len(clips) != 2:
        return {"status": "G0_BLOCKED", "reason": "EXACTLY_TWO_GENUINE_HUMAN_CLIPS_REQUIRED"}
    if str(raw.get("target_task")) != "video_goal_push":
        return {"status": "G0_BLOCKED", "reason": "TASK_NOT_ISOLATED_VIDEO_GOAL_PUSH"}
    reasons=[]; inspected=[]; goals=[]
    for entry in clips:
        if not isinstance(entry, dict):
            reasons.append("BAD_CLIP_RECORD");continue
        goal=entry.get("goal_robot_x")
        if not isinstance(goal, str) or goal not in TASK_GOALS:
            reasons.append("INVALID_TARGET_SIGN");continue
        goals.append(goal)
        if entry.get("source_type") not in SOURCE_TYPES:
            reasons.append("SOURCE_PROVENANCE_MISSING")
        if entry.get("rights") not in ("self_recorded_consent_confirmed", "external_license_verified"):
            reasons.append("RIGHTS_UNVERIFIED")
        if entry.get("human_hand_demonstration") is not True:
            reasons.append("HUMAN_SOURCE_NOT_ATTESTED")
        if entry.get("initial_scene_equivalence_attested") is not True:
            reasons.append("SOURCE_SCENE_EQUIVALENCE_NOT_ATTESTED")
        if entry.get("camera_to_robot_x_mapping_documented") is not True:
            reasons.append("NO_CAMERA_TO_ROBOT_MAPPING")
        if not isinstance(entry.get("visible_target_region"), str) or not entry["visible_target_region"].strip():
            reasons.append("TARGET_REGION_UNDOCUMENTED")
        rel=entry.get("path")
        if not isinstance(rel, str):
            reasons.append("MISSING_SOURCE_PATH");continue
        candidate = project / rel
        try:
            file=within_project(candidate, project)
        except ValueError:
            reasons.append("INVALID_PATH_OUTSIDE_PROJECT");continue
        if "human_demos/eval/" in file.as_posix() or "0080" in file.parts:
            reasons.append("HELDOUT_EVAL_SOURCE_FORBIDDEN");continue
        if file.suffix.lower() not in ALLOWED_EXT or not file.is_file():
            reasons.append("VIDEO_NOT_PRESENT_OR_BAD_EXTENSION");continue
        length=file.stat().st_size
        if not 1024 < length <= MAX_CLIP:
            reasons.append("VIDEO_SIZE_INVALID");continue
        probe=probe_video(file)
        if not probe["ok"]:
            reasons.append("VIDEO_CANNOT_DECODE:"+str(probe["reason"]));continue
        actual_sha=sha256(file)
        if actual_sha != entry.get("sha256"):
            reasons.append("HASH_NOT_EQUAL_DECLARED");continue
        inspected.append({"goal_robot_x": goal,"relative_file":str(file.relative_to(project)),
                          "file_sha256":actual_sha,"video_bytes":length,
                          "video_probe":probe,"source_type":entry["source_type"]})
    if set(goals) != TASK_GOALS:
        reasons.append("OPPOSITE_GOALS_NOT_BOTH_PRESENT")
    if len(inspected)==2 and inspected[0]["file_sha256"]==inspected[1]["file_sha256"]:
        reasons.append("SAME_VIDEO_FILE_REUSED")
    # Images and annotations not actually interpreted by machine. Even two
    # well-formed videos are NOT causally verified without visual review.
    if reasons:
        return {"status":"G0_BLOCKED", "reason_codes":sorted(set(reasons)),
                "files_verified":inspected,"video_semantics_independently_verified":False}
    return {"status":"G0_CANDIDATE_REQUIRES_MANUAL_VISUAL_REVIEW",
            "files_verified":inspected,"goal_signs":sorted(set(goals)),
            "video_semantics_independently_verified":False,
            "next_step":"Review 0/middle/end frames, ground-plane axis, same initial objects and target regions before any G1 physics."}


def make_report(project: Path, manifest: Path | None) -> dict[str, Any]:
    p=project.resolve()
    local=inspect_local_prepared_media(p)
    task=check_authored_task(p)
    pair={"status":"G0_BLOCKED","reason":"NO_GENUINE_OPPOSING_GOAL_HUMAN_VIDEO_PAIR_VERIFIED"}
    if manifest is not None:
        pair=audit_pair(manifest,p)
    return {"protocol":"TRO1_G0_V1","physical_rollouts":0,"training_runs":0,
            "remote_downloads":0,"source_catalog":CATALOG,
            "official_task_source_audit":task,"existing_train_data_inventory":local,
            "human_goal_pair_audit":pair,
            "decision": "G0_NOT_PASSED" if pair["status"]!="G0_CANDIDATE_REQUIRES_MANUAL_VISUAL_REVIEW" else "G0_MANUAL_REVIEW_PENDING",
            "research_no_go_for_video_goal_dependence_until_verified":True}


def save_immutable(out: Path, value: dict[str, Any]) -> Path:
    payload=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n"
    digest=hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    file=out/f"g0_audit_{digest}.json"
    out.mkdir(parents=True,exist_ok=True)
    if file.exists():
        if file.read_text(encoding="utf-8")!=payload:
            raise FileExistsError("NEVER_OVERWRITE_CHANGED_G0_EVIDENCE")
    else:
        file.write_text(payload,encoding="utf-8")
    return file


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--manifest",type=Path,default=None,
                   help="Optional project-local JSON: two genuine human clips with opposing, manually annotated goals.")
    p.add_argument("--out",type=Path,default=None)
    args=p.parse_args(argv)
    project=args.project.expanduser().resolve()
    report=make_report(project,args.manifest)
    target=args.out if args.out is not None else project/"outputs/g0_goal_pair_audit/v1"
    out=within_project(target,project)
    written=save_immutable(out,report)
    summary={"decision":report["decision"],"human_goal_pair_status":report["human_goal_pair_audit"]["status"],
             "candidate_dataset":"Something-Something V2 (labels only, not approved source pairs)",
             "report":str(written),"real_physx_actions":0}
    print("TRO1_G0_AUDIT",json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if report["decision"]=="G0_MANUAL_REVIEW_PENDING" else 2

if __name__=="__main__":
    raise SystemExit(main())
