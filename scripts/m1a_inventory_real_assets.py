#!/usr/bin/env python3
"""M1-A one-shot local asset and evidence gate for same-goal capability transfer.

STD-LIB ONLY, read-only on source files; no GPU, checkpoints or network.
Never treats original RoboReel PickCube/PushCube as matched-goal tasks.
Reports what is *really* on this machine, not remote listed candidates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _safe_json(file: Path):
    if not file.is_file():
        return None
    try:
        return json.loads(file.read_text())
    except (ValueError, OSError):
        return None


def local_videos(project: Path, kind: str) -> list[dict]:
    """Only TRAIN data. Never enumerate held-out EVAL videos."""
    patterns = [
        f"outputs/v110_baselines/official_dataset/human_demos/train/no_distraction/{kind}/*/front.mp4",
        f"outputs/v120_baselines/official_dataset/human_demos/train/no_distraction/{kind}/*/front.mp4",
        f"outputs/next_gate_v100/train_data/hf_cache/human_demos/train/no_distraction/{kind}/*/front.mp4",
        f"outputs/roboreel_real_train_gate/hf_cache/human_demos/train/no_distraction/{kind}/*/front.mp4",
    ]
    result = {}
    for pattern in patterns:
        for file in project.glob(pattern):
            if not file.is_file() or file.stat().st_size < 512:
                continue
            parts = file.relative_to(project).parts
            if "eval" in parts or "test" in parts or "train" not in parts:
                continue
            result[str(file.relative_to(project))] = {
                "path":str(file.relative_to(project)),
                "bytes":file.stat().st_size,
                "status":"LOCAL_TRAIN_VIDEO_FILE_EXISTS_NOT_SAME_GOAL_VERIFIED",
            }
    return [result[k] for k in sorted(result)]


def local_robot_actions(project: Path) -> list[dict]:
    """Review real downloaded TRAIN robot H5+camera triples, no fake labels."""
    # Source-of-truth: run_v120_real_robot_and_official_baseline.sh
    # called official_robot_data with --data-root outputs/v120_baselines/robot_train.
    # The original v1 M1 scanner incorrectly looked only under
    # official_dataset (human EVAL storage), so it reported 0 despite
    # the independent v120 H5 audit validating demos 0000 and 0001.
    roots = [
        project/"outputs/v120_baselines/robot_train",
        project/"outputs/v110_baselines/official_dataset",
        project/"outputs/v120_baselines/official_dataset",
    ]
    manifests=[
        project/"outputs/v110_baselines/robot_data_manifest.json",
        project/"outputs/v120_baselines/robot_data_manifest.json",
    ]
    found={}
    for manifest in manifests:
        data = _safe_json(manifest)
        if not isinstance(data,dict):
            continue
        for entry in data.get("downloaded", []):
            if not isinstance(entry,dict) or "demo" not in entry:
                continue
            file_infos=entry.get("files",[])
            names=[x.get("relative_path","") for x in file_infos if isinstance(x,dict)]
            # Only the published TRAIN PushCube subset; never EVAL/test
            # or an unrelated task. Do not treat one H5 as an action
            # dataset until its two camera files are also present.
            if (not names or any(".." in Path(n).parts or Path(n).is_absolute()
                                 or not n.startswith("robot_trajectories/no_distraction/push_cube/")
                                 for n in names)
                    or not any(name.endswith(".h5") for name in names)):
                continue
            for root in roots:
                paths=[root/name for name in names]
                if not all(p.is_file() and p.stat().st_size>0 for p in paths):
                    continue
                have_h5=any(p.suffix==".h5" for p in paths)
                have_cameras={"base_camera.mp4","hand_camera.mp4"}.issubset(
                    {p.name for p in paths})
                if not (have_h5 and have_cameras):
                    continue
                key=(entry["demo"],str(root))
                found[key]={
                    "demo":str(entry["demo"]), "data_root":str(root.relative_to(project)),
                    "files":[str(p.relative_to(project)) for p in paths],
                    "source_manifest":str(manifest.relative_to(project)),
                    "status":"ACTUAL_LOCAL_TRAIN_H5_AND_CAMERAS_PRESENT_PRIOR_V120_CONTENT_AUDIT",
                }
    return [found[k] for k in sorted(found)]


def audit(project: Path):
    project=project.resolve()
    historical=_safe_json(project/"research_reports/2026-11-09_v120/real_robot_h5_audit.json") or {}
    original_m0=_safe_json(
        project/"research_reports/2026-11-11_m0_pi05_uniform/verified_summary.json") or {}
    human_pick=local_videos(project,"pick_cube")
    human_push=local_videos(project,"push_cube")
    robot=local_robot_actions(project)
    # Historical executed H5 audit is evidence, but local files must be
    # independently present before any new training is scheduled.
    historical_valid=(historical.get("status")=="REAL_ACTION_DATA_VALIDATED"
                      and historical.get("valid_demos",0)>=2)
    m0_real=(original_m0.get("type")==
             "M0_AUTHOR_PI05_UNIFORM_ND_PUSHCUBE_VERIFIED_60_TRIALS"
             and original_m0.get("pooled_successes")==46
             and original_m0.get("pooled_trials")==60)
    report={
        "type":"M1A_LOCAL_EVIDENCE_INVENTORY_NO_PHYSICS",
        "m0_official_pi05_archived_and_verified":m0_real,
        "m0_pi05_46_of_60_is_not_m1_same_goal_evidence":True,
        "historical_real_robot_h5_audit_validated":historical_valid,
        "historical_real_robot_h5_valid_demos":historical.get("valid_demos"),
        "historical_robot_audit_not_equal_local_disk":True,
        "actual_v120_robot_train_data_root":"outputs/v120_baselines/robot_train",
        "local_train_push_robot_demos_with_h5_and_two_cameras":robot,
        "local_train_pick_human_front_mp4":human_pick,
        "local_train_push_human_front_mp4":human_push,
        "official_pickcube_and_pushcube_have_different_goals":True,
        "source_pick_to_same_final_goal_physically_verified":False,
        "push_only_target_blocker_insertion_physically_verified":False,
        "matched_goal_policy_baselines_executed":False,
        "status":"M1_SAME_GOAL_PAIR_AND_INSERTION_STILL_NOT_VERIFIED",
        "next_scientific_gate":(
            "Use validated source demonstrations to build an ACTUAL matched-final-goal "
            "Pick versus Push-only pair with a movable blocker; measure target-only "
            "clearance and run equal-information Goal-only, flow and TAMP controls. "
            "Do not infer M1 success from old RoboReel PushCube 46/60."
        ),
        "no_network_no_gpu_no_checkpoint_download":True,
    }
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--out",type=Path,default=
                   Path("outputs/m1a_evidence_gate/local_inventory.json"))
    args=p.parse_args(argv)
    report=audit(args.project)
    output=args.out
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("M1A_LOCAL_EVIDENCE_GATE",json.dumps({
        "m0_verified":report["m0_official_pi05_archived_and_verified"],
        "historical_robot_h5_valid":report["historical_real_robot_h5_valid_demos"],
        "local_robot_action_demos":len(report["local_train_push_robot_demos_with_h5_and_two_cameras"]),
        "local_source_pick_train_videos":len(report["local_train_pick_human_front_mp4"]),
        "local_source_push_train_videos":len(report["local_train_push_human_front_mp4"]),
        "same_goal_pair_verified":report["source_pick_to_same_final_goal_physically_verified"],
        "target_insert_verified":report["push_only_target_blocker_insertion_physically_verified"],
        "status":report["status"],"out":str(output),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
