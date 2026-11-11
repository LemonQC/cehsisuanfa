#!/usr/bin/env python3
"""B01 robot-only TRAIN data: a bounded actual download + H5/video integrity gate.

This is a NEW RoboReel baseline-foundation workstream after HARD STOP
of the custom Pick->Push M1 hypothesis. Does NOT revive M1, train Ours,
use EVAL data, or claim an evaluated policy.

Existing official downloader + real_robot_h5_audit remain the sources
of truth. Keeps all bytes in the original v120 robot_train directory;
outputs only new compact B01 metadata under outputs/b01_robot_only.
No checkpoint download. Must have all 64 authentic TRAIN triplets.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

TARGET = 64
BUDGET_MB = 300
TRAIN_UPPER_BOUND_EXCLUSIVE = 80
MANIFEST_DIR = Path("outputs/b01_robot_only")
ROBOT_DATA_ROOT = Path("outputs/v120_baselines/robot_train")
SPLIT_SEED = 42


def selected_train_guard(plan):
    """Prevent official 0080..0099 heldout data leakage in any B01 manifest."""
    chosen = plan.get("demos", [])
    if len(chosen) != TARGET:
        raise ValueError(
            f"B01 requires {TARGET} complete, authentic robot TRAIN triplets; "
            f"discovered {len(chosen)}. No fabricated or repeated demos."
        )
    demo_ids=[]
    for entry in chosen:
        code=entry.get("demo")
        if (not isinstance(code,str) or len(code)!=4
                or not code.isdecimal()
                or not 0 <= int(code) < TRAIN_UPPER_BOUND_EXCLUSIVE):
            raise ValueError(f"B01 TRAIN/EVAL split invalid: {code!r}")
        paths=[v.get("path") for v in entry.get("files",[])]
        root=f"robot_trajectories/no_distraction/push_cube/{code}/"
        suffixes={p.removeprefix(root) for p in paths if isinstance(p,str)}
        if (len(paths)!=3 or any(not isinstance(p,str) or
                                not p.startswith(root) for p in paths)
                or len(suffixes)!=3
                or not any(p.endswith(".h5") for p in suffixes)
                or not {"base_camera.mp4","hand_camera.mp4"}.issubset(suffixes)):
            raise ValueError(f"B01 demo {code}: incorrect official raw file triple")
        demo_ids.append(code)
    if len(set(demo_ids))!=TARGET:
        raise ValueError("B01 repeated robot TRAIN IDs")
    if int(plan.get("bytes",0))>BUDGET_MB*1024*1024:
        raise ValueError("B01 authentic robot TRAIN data exceeds 300MB budget")
    return demo_ids


def frozen_split(demo_ids):
    import random
    if len(demo_ids)!=TARGET or len(set(demo_ids))!=TARGET:
        raise ValueError("B01 split must be 64 distinct real TRAIN demo IDs")
    ids=sorted(demo_ids)
    random.Random(SPLIT_SEED).shuffle(ids)
    n_val=13
    train=sorted(ids[n_val:])
    val=sorted(ids[:n_val])
    if set(train)&set(val):
        raise ValueError("B01 train/validation leakage")
    return {
        "type":"B01_AUTHENTIC_ROBOREEL_ROBOT_TRAIN_SPLIT",
        "task":"push_cube","split_seed":SPLIT_SEED,
        "train_demo_ids":train,"validation_demo_ids":val,
        "test_demo_ids":"OFFICIAL_HELD_OUT_0080_THROUGH_0099_NEVER_USED_FOR_TRAINING",
        "source":"LogosRoboticsLab/roboreel robot_trajectories/no_distraction/push_cube",
        "condition":"TRAIN_ONLY","count_train":len(train),
        "count_val":len(val),"count_all":TARGET,
        "no_model_training_performed":True,
        "no_physics_evaluation_performed":True,
        "custom_pick_push_m1_result_not_rescued":True,
    }


def write_once(file,content):
    file.parent.mkdir(parents=True,exist_ok=True)
    rendered=json.dumps(content,indent=2,ensure_ascii=False)+"\n"
    if file.exists():
        previous=json.loads(file.read_text())
        if previous != content:
            raise FileExistsError(f"Immutable B01 evidence differs; do not overwrite: {file}")
        return
    file.write_text(rendered)


def make_local_report(manifest, audit):
    downloaded=manifest.get("downloaded")
    if (manifest.get("status")!="REAL_ROBOT_TRAIN_DATA_ACQUIRED"
            or not isinstance(downloaded,list) or len(downloaded)!=TARGET):
        raise ValueError("B01 missing genuine completed download manifest")
    ids=[x.get("demo") for x in downloaded]
    if len(set(ids))!=TARGET or any(not str(i).isdecimal() or int(i)>=80 for i in ids):
        raise ValueError("B01 downloaded robot data contain invalid/EVAL IDs")
    rows=audit.get("rows")
    if (audit.get("status")!="REAL_ACTION_DATA_VALIDATED"
            or audit.get("valid_demos")!=TARGET
            or not isinstance(rows,list) or len(rows)!=TARGET):
        raise ValueError("B01 official H5/camera content audit failed")
    if sorted(x.get("demo") for x in rows)!=sorted(ids):
        raise ValueError("B01 audited robot demo IDs differ from downloaded IDs")
    action_shapes={tuple(x.get("action_shape",[])[1:]) for x in rows}
    if action_shapes!={(8,)}:
        raise ValueError(f"Unexpected B01 raw policy action dimension: {action_shapes}")
    data=frozen_split(ids)
    data["authentic_robot_h5_audited"]=True
    data["observed_action_dim"]=8
    data["total_real_robot_action_steps"]=sum(int(r["action_shape"][0]) for r in rows)
    data["original_robot_train_root"]=str(ROBOT_DATA_ROOT)
    data["current_download_count"]=len(ids)
    data["upstream_checkpoint_training_budget_comparable"]=False
    data["research_note"]="A newly trained 64-demo robot-only BC is not a matched-training-data ablation of a released pretrained pi05 checkpoint."
    return data


def run(download_requested=False, verify_only=False):
    from agcofl.official_robot_data import discover,plan,download as get_real_data
    from agcofl.real_robot_h5_audit import run as audit_h5

    root=MANIFEST_DIR
    manifest_path=root/"robot_data_manifest.json"
    audit_path=root/"real_robot_h5_audit.json"
    final_path=root/"frozen_train_val_split.json"
    if final_path.exists() and manifest_path.exists() and audit_path.exists():
        prior=json.loads(final_path.read_text())
        if (prior.get("authentic_robot_h5_audited") is True
                and prior.get("count_all")==TARGET):
            print("B01_TRAIN_DATA_ALREADY_PREPARED_NO_DOWNLOAD",
                  json.dumps({"train":prior.get("count_train"),"validation":prior.get("count_val")}),
                  flush=True)
            return prior
    if verify_only:
        if not manifest_path.exists():
            raise FileNotFoundError("No B01 manifest; run one bounded --download first")
        source=json.loads(manifest_path.read_text())
    else:
        if manifest_path.exists():
            raise FileExistsError("B01 partial manifest exists: do not overwrite; inspect before resume")
        os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
        entries=discover(task="push_cube",endpoint=os.environ["HF_ENDPOINT"])
        selection=plan(entries,task="push_cube",max_demos=TARGET,max_mb=BUDGET_MB)
        ids=selected_train_guard(selection)
        print("B01_REAL_TRAIN_PLAN",json.dumps({
            "demos":len(ids),"total_bytes":selection["bytes"],
            "download_requested":download_requested,"train_only":True,
            "first_ids":ids[:5],"last_ids":ids[-5:]}),flush=True)
        if not download_requested:
            return {"status":"PLAN_ONLY_NO_DOWNLOAD","demos":len(ids)}
        # Acquire only authentic robot TRAIN H5 plus both RGB MP4s,
        # retaining the original 2 v120 demos in the same directory.
        selection["downloaded"]=get_real_data(
            selection,ROBOT_DATA_ROOT,endpoint=os.environ["HF_ENDPOINT"])
        selection["status"]="REAL_ROBOT_TRAIN_DATA_ACQUIRED"
        root.mkdir(parents=True,exist_ok=True)
        write_once(manifest_path,selection)
        source=selection

    audit=audit_h5(manifest_path,ROBOT_DATA_ROOT,audit_path)
    report=make_local_report(source,audit)
    write_once(final_path,report)
    print("B01_AUTHENTIC_ROBOT_TRAIN_READY",json.dumps({
        "action_dim":report["observed_action_dim"],
        "actual_demos":report["count_all"],
        "train":report["count_train"],"validation":report["count_val"],
        "total_action_steps":report["total_real_robot_action_steps"],
        "split_file":str(final_path),
        "policy_training_executed":False,
        "physics_rollouts":0,
    }),flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    group.add_argument("--download",action="store_true",
                       help="Explicit network opt-in, at most 300 MB / 64 TRAIN demos")
    group.add_argument("--verify-only",action="store_true",
                       help="No internet, audit an existing saved manifest")
    args=p.parse_args(argv)
    run(download_requested=args.download,verify_only=args.verify_only)


if __name__=="__main__":
    main()
