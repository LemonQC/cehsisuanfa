#!/usr/bin/env python3
"""M1-B: bounded, authentic ManiSkill Panda PickCube expert *reference*.

Runs the installed ManiSkill motionplanning solution (not our v140
hand-coded grasp script) for 3 fixed seeds. Saves the author's actual
PhysX H5 action trajectories, JSON episode outcomes, and RGB videos in
project-local outputs. Never claims PickCube's original goal equals the
new Pick->Push same-goal task. No model download, no training and no
unbounded 'only-count-success' loops.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def success_scalar(value):
    """Strict bool conversion for JSON encoded torch success scalar."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, list) and len(value)==1:
        return success_scalar(value[0])
    return None


def extract_episodes(meta:dict):
    """Accept only actual official success metadata, never infer from frames."""
    # Upstream ManiSkill RecordEpisode stores the environment under
    # meta["env_info"]["env_id"], not meta["env_id"]. This is the
    # official published schema used by its trajectory replay utility.
    env_info=meta.get("env_info") if isinstance(meta,dict) else None
    if (not isinstance(env_info,dict)
            or env_info.get("env_id")!="PickCube-v1"):
        raise ValueError("Not the original ManiSkill PickCube-v1 demo metadata")
    if meta.get("source_type")!="motionplanning":
        raise ValueError("Reference must be produced by official motionplanning")
    episodes=meta.get("episodes")
    if not isinstance(episodes,list) or not episodes:
        raise ValueError("No recorded episodes in official ManiSkill metadata")
    results=[]
    for ep in episodes:
        if not isinstance(ep,dict) or type(ep.get("episode_id")) is not int:
            raise ValueError("Invalid official episode ID")
        # Upstream RecordEpisode.flush_trajectory adds final success as
        # episode_info["success"] (top-level), not episode_info["info"].
        success=success_scalar(ep.get("success"))
        if success is None:
            raise ValueError(f"Official episode {ep.get('episode_id')} missing success flag")
        result={
            "episode_id":ep["episode_id"], "success":success,
            "elapsed_steps":ep.get("elapsed_steps"),
            "control_mode":ep.get("control_mode"),
        }
        results.append(result)
    if len({r["episode_id"] for r in results})!=len(results):
        raise ValueError("Duplicate official episode_id in metadata")
    return results


def validate_record(base:Path, required_attempts:int):
    """Validate actual H5 actions plus success metadata (not video appearance)."""
    json_path=base.with_suffix(".json")
    h5_path=base.with_suffix(".h5")
    if not json_path.is_file() or not h5_path.is_file():
        raise FileNotFoundError(f"Missing official H5/JSON: {base}")
    meta=json.loads(json_path.read_text())
    rows=extract_episodes(meta)
    # RecordEpisode flush_trajectory(prune_empty_action=True) may discard
    # an attempted motion planner episode containing ZERO actions, so
    # JSON/H5 may contain fewer rows than -n attempts. Never invent
    # those missing outcomes. The original solver.log remains evidence.
    if not 1 <= len(rows) <= required_attempts:
        raise ValueError(
            f"Recorded episode count {len(rows)} cannot exceed "
            f"requested attempts {required_attempts}")
    import h5py
    with h5py.File(h5_path,"r") as h5:
        for row in rows:
            key=f"traj_{row['episode_id']}"
            if key not in h5 or "actions" not in h5[key]:
                raise ValueError(f"Official motionplanning H5 lacks {key}/actions")
            actions=h5[key]["actions"]
            if actions.ndim!=2 or actions.shape[0]<=0 or actions.shape[1]<=0:
                raise ValueError(f"{key}: invalid action array {actions.shape}")
            row["action_shape"]=list(actions.shape)
            if int(actions.shape[0])!=int(row["elapsed_steps"]):
                raise ValueError(f"{key}: H5 steps inconsistent with JSON elapsed_steps")
            if "success" in h5[key]:
                success_samples=h5[key]["success"]
                if success_samples.ndim<1 or len(success_samples)==0:
                    raise ValueError(f"{key}: invalid H5 success samples")
                actual=bool(success_samples[-1].reshape(-1)[0])
                if actual!=row["success"]:
                    raise ValueError(f"{key}: H5 success != official JSON success")
    # Do not claim a specific camera or successful-episode RGB unless
    # its filename is matched; this only checks at least one recorded mp4.
    videos=sorted(p for p in base.parent.rglob("*.mp4") if p.is_file()
                  and p.stat().st_size>512)
    return {
        "type":"M1B_OFFICIAL_MANISKILL_PICK_REFERENCE_ONLY",
        "official_env_id":"PickCube-v1",
        "source_type":"motionplanning",
        "attempts":len(rows),  # Legacy: number of nonempty SAVED episodes.
        "attempts_requested":required_attempts,
        "saved_nonempty_episodes":len(rows),
        "unrecorded_or_pruned_attempts":required_attempts-len(rows),
        "successful_reference_episodes":sum(r["success"] for r in rows),
        "episodes":rows,
        "h5":str(h5_path),
        "json":str(json_path),
        "rgb_video_candidates":[str(p) for p in videos],
        "at_least_one_rgb_video_saved":bool(videos),
        "usable_successful_pick_reference":bool(videos) and any(r["success"] for r in rows),
        "video_episode_alignment_validated":False,
        "physically_executed_original_pick_task":True,
        "same_goal_pick_push_pair_validated":False,
        "is_human_source_video":False,
        "is_m1_physical_capability_result":False,
        "next_action":"Reuse validated grasp motion solver in custom same-final-goal task; do NOT relabel these original PickCube trajectories as matched-goal examples.",
    }


def run(out:Path, attempts:int=3, timeout_seconds:int=480,
        verify_only:bool=False):
    if not (1<=attempts<=5 and 30<=timeout_seconds<=1800):
        raise ValueError("Bounded references only: 1..5 attempts and 30..1800 seconds")
    out=out.resolve()
    target=out/"PickCube-v1/motionplanning/m1b_official_pick"
    report_path=out/"reference_verification.json"
    existing_h5=target.with_suffix(".h5")
    existing_json=target.with_suffix(".json")

    # REUSE actual completed official run even when our old wrapper's
    # overly restrictive JSON validation failed. No re-execution.
    if existing_h5.is_file() and existing_json.is_file():
        result=validate_record(target,attempts)
        if report_path.is_file():
            old=json.loads(report_path.read_text())
            if old!=result:
                raise RuntimeError(
                    "Existing reference verification differs; preserve original "
                    "record and investigate before overwriting")
        else:
            report_path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
        print("M1B_OFFICIAL_PICK_EXISTING_VERIFIED",json.dumps({
            "requested_attempts":attempts,
            "recorded_episodes":result["saved_nonempty_episodes"],
            "successes":result["successful_reference_episodes"],
            "any_video":result["at_least_one_rgb_video_saved"],
            "usable_reference":result["usable_successful_pick_reference"],
            "same_goal_pair_validated":False,
            "out":str(report_path)}),flush=True)
        return result

    if existing_h5.exists() or existing_json.exists() or report_path.exists():
        raise RuntimeError("Incomplete or inconsistent existing official source files. "
                           "Do not overwrite/re-execute; inspect original files.")
    if verify_only:
        raise FileNotFoundError(
            f"--verify-only: no complete original H5/JSON pair at {target}; "
            "refusing any new physics execution")

    env=os.environ.copy()
    env.update({
        "PYTHONFAULTHANDLER":"1","MPLBACKEND":"Agg",
        "OMP_NUM_THREADS":"4",
        "XLA_PYTHON_CLIENT_PREALLOCATE":"false",
    })
    env.pop("DISPLAY",None)
    env.pop("WAYLAND_DISPLAY",None)
    command=[
        sys.executable,"-u","-m","mani_skill.examples.motionplanning.panda.run",
        "-e","PickCube-v1","-n",str(attempts),
        "-b","cpu","--save-video","--record-dir",str(out),
        "--traj-name","m1b_official_pick","--num-procs","1",
    ]
    out.mkdir(parents=True,exist_ok=True)
    print("M1B_OFFICIAL_PICK_RUN",json.dumps({
        "cmd":command,"timeout_seconds":timeout_seconds,
        "original_task_not_same_goal":True}),flush=True)
    with (out/"official_solver.log").open("w") as log:
        try:
            completed=subprocess.run(
                command,env=env,stdout=log,stderr=subprocess.STDOUT,
                timeout=timeout_seconds,check=False)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Official motionplanner exceeded fixed timeout; "
                               "do not repeatedly increase timeout. See official_solver.log")
    if completed.returncode!=0:
        raise RuntimeError(
            f"Official motionplanner exited {completed.returncode}; "
            "see official_solver.log. No matched-goal result is claimed.")
    result=validate_record(target,attempts)
    report_path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("M1B_OFFICIAL_PICK_REFERENCE_VERIFIED",json.dumps({
        "requested_attempts":attempts,
        "recorded_episodes":result["saved_nonempty_episodes"],
        "successes":result["successful_reference_episodes"],
        "any_video":result["at_least_one_rgb_video_saved"],
        "usable_reference":result["usable_successful_pick_reference"],
        "same_goal_pair_validated":False,
        "out":str(report_path)}),flush=True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("outputs/m1b_official_pick_reference"))
    p.add_argument("--attempts",type=int,default=3)
    p.add_argument("--timeout",type=int,default=480)
    p.add_argument("--verify-only",action="store_true",
                   help="NO simulation: verify and preserve existing original H5/JSON/videos")
    args=p.parse_args(argv)
    run(args.out,args.attempts,args.timeout,args.verify_only)


if __name__=="__main__":
    main()
