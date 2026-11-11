#!/usr/bin/env python3
"""M1-C replacement: run ONE original RoboReel long-horizon TRAIN development episode.

Preserves original π0.5 Uniform checkpoint and pinned author evaluator. Uses
published task env, ORIGINAL human TRAIN 0000 video/SigLIP and existing weights.
No new source controller, no custom scene, no M1B5, no claimed paper results.

Stages:
  gate           read-only local checkpoint/tasks/asset scan
  get-one-train  explicitly download only ONE authentic HF TRAIN source video
  encode-one     original author SigLIP on the one source video, fresh process
  pilot-one      original author official evaluator scope="one", train-ID DEV
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from agcofl.v140_official_pi05 import pinned_sha, PIN

TASKS=("toys_in_drawer","toys_in_basket")
RELEASED="vid_cond_pi05_uniform"
BUCKET="without_distraction"
SOURCE_DATASET="LogosRoboticsLab/roboreel"
TRAIN_ID=0
MAX_VIDEO_BYTES=100*1024*1024


def rel_video(task):
    if task not in TASKS:
        raise ValueError(f"Unknown pretrained long-horizon task: {task}")
    return f"human_demos/train/no_distraction/{task}/{TRAIN_ID:04d}/front.mp4"


def paths(assets,task):
    root=Path(assets).resolve()
    video=root/"official_eval_data"/rel_video(task)
    features=root/"official_eval_data"/f"video_demonstration_features/{BUCKET}/{task}/{TRAIN_ID:04d}/front.npy"
    ckpt=root/"released_checkpoints/nd"/RELEASED/"ckpt"
    return root,video,features,ckpt


def read_train_cfg(ckpt):
    for path in (ckpt/"train_config.json",ckpt.parent/"train_config.json"):
        if path.is_file():
            d=json.loads(path.read_text())
            return d,path
    return {},None


def frozen_gate(task,official_root,assets):
    root,video,features,ckpt=paths(assets,task)
    official=Path(official_root).resolve()
    config,location=read_train_cfg(ckpt)
    trained=config.get("tasks") or [config.get("task")]
    if isinstance(trained,str):
        trained=[trained]
    bank=official/"roboreel_env/init_states/without_distraction"/f"{task}.pkl"
    checks={
        "pinned_official_author_source":pinned_sha(official)==PIN,
        "original_pi05_runner_present":
            (official/"src/evaluation/demo_pi05_runner.py").is_file(),
        "original_evaluator_present":(official/"scripts/evaluation.py").is_file(),
        "original_siglip_extractor_present":
            (official/"scripts/compute_vision_features_for_demo.py").is_file(),
        "same_released_uniform_policy_checkpoint":
            ckpt.is_dir() and (ckpt/"params").is_dir(),
        "original_checkpoint_training_config_present":location is not None,
        "existing_policy_was_trained_for_requested_task":task in trained,
        "original_task_physics_environment_present":
            (official/"roboreel_env/src/envs").is_dir(),
        "original_task_initial_states_present":bank.is_file(),
        "one_authentic_human_train_video":video.is_file() and video.stat().st_size>0,
        "one_original_siglip_feature_bundle":
            features.is_file() and features.stat().st_size>0,
    }
    return {
      "type":"M1C_ORIGINAL_LONG_HORIZON_TRAIN_DEV_GATE",
      "task":task,
      "scope":"train_0000_only_NOT_OFFICIAL_EVAL",
      "old_m1b_custom_hard_stopped":True,
      "trained_task_list":trained,
      "preflight":checks,
      "source_commit_expected":PIN,
      "checkpoint_train_config":str(location) if location else None,
      "existing_checkpoint":str(ckpt),
      "train_source_video":str(video),
      "train_feature_bundle":str(features),
      "missing":[x for x,ok in checks.items() if not ok],
      "ready_to_develop_on_original_task":all(checks.values()),
      "physics_executed":False,
      "learned_new_method":False,
    }


def download_one(task,assets):
    """HF published TRAIN mp4 only; never download heldout eval videos."""
    from huggingface_hub import HfApi, hf_hub_download

    root,video,features,ckpt=paths(assets,task)
    if video.exists() and video.stat().st_size>512:
        print("M1C_REAL_TRAIN_VIDEO_REUSED",json.dumps({
            "path":str(video),"bytes":video.stat().st_size}),flush=True)
        return
    if video.exists():
        raise RuntimeError(f"Existing suspicious zero/short video: {video}")
    endpoint=os.environ.get("HF_ENDPOINT","https://hf-mirror.com")
    api=HfApi(endpoint=endpoint)
    # Only dataset revision metadata + the ONE requested file's metadata.
    # Never enumerate thousands of dataset files / the full ~104GB archive.
    info=api.repo_info(SOURCE_DATASET,repo_type="dataset")
    name=rel_video(task)
    if not isinstance(info.sha,str) or len(info.sha)!=40:
        raise RuntimeError("Dataset commit SHA unavailable; cannot pin source")
    entries=list(api.get_paths_info(
        SOURCE_DATASET,paths=[name],repo_type="dataset",
        revision=info.sha))
    if len(entries)!=1 or not isinstance(entries[0].size,int):
        raise RuntimeError(f"Exact authentic TRAIN video metadata missing: {name}")
    size=entries[0].size
    if not 512<size<=MAX_VIDEO_BYTES:
        raise RuntimeError(f"TRAIN video bytes {size} outside bounded 100MB limit")
    if not isinstance(info.sha,str) or len(info.sha)!=40:
        raise RuntimeError("Dataset commit SHA unavailable; cannot pin source")
    local=Path(hf_hub_download(
        repo_id=SOURCE_DATASET,repo_type="dataset",
        revision=info.sha,filename=name,local_dir=str(root/"official_eval_data"),
        endpoint=endpoint))
    if local.resolve()!=video.resolve() or local.stat().st_size!=size:
        raise RuntimeError("Downloaded human source path/size mismatch")
    digest=hashlib.sha256(local.read_bytes()).hexdigest()
    report={
        "type":"M1C_AUTHENTIC_TRAIN_SOURCE_ONLY",
        "dataset":SOURCE_DATASET,"revision":info.sha,
        "relative_path":name,"local_file":str(local),
        "bytes":size,"sha256":digest,
        "not_heldout_eval":True,
        "no_robot_policy_training":True,
    }
    save=Path("outputs/m1c_long_horizon")/task/"source_video_provenance.json"
    save.parent.mkdir(parents=True,exist_ok=True)
    if save.exists() and json.loads(save.read_text())!=report:
        raise FileExistsError(f"Source provenance changed: {save}")
    save.write_text(json.dumps(report,indent=2)+"\n")
    print("M1C_REAL_TRAIN_VIDEO_READY",json.dumps(report),flush=True)


def encode_one(task,official_root,assets):
    """Exactly original compute_vision_features_for_demo implementation."""
    _,video,feat,_=paths(assets,task)
    if feat.is_file() and feat.stat().st_size>0:
        print("M1C_ORIGINAL_SIGLIP_REUSED",str(feat),flush=True)
        return
    if not video.is_file():
        raise FileNotFoundError("Download authentic TRAIN 0000 video first")
    official=Path(official_root).resolve()
    if pinned_sha(official)!=PIN:
        raise RuntimeError("Original SigLIP source commit mismatch")
    entry=official/"scripts/compute_vision_features_for_demo.py"
    for item in (official,official/"src",official/"src/external_models"):
        if str(item) not in sys.path:
            sys.path.insert(0,str(item))
    spec=importlib.util.spec_from_file_location("_m1c_authors_siglip",entry)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rgb=module._read_camera_mp4(video.parent,"front")
    if rgb is None or len(rgb)==0:
        raise RuntimeError("Authentic TRAIN video cannot be read")
    encoder=module._PersistentEncoder(max_T=max(1024,len(rgb)))
    feat.parent.mkdir(parents=True,exist_ok=True)
    ret=encoder.encode_demo_camera(feat.parent,rgb,"front",overwrite=False)
    if not feat.is_file() or feat.stat().st_size<=0:
        raise RuntimeError(f"Original SigLIP did not create {feat}: {ret}")
    print("M1C_ORIGINAL_SIGLIP_READY",json.dumps({
        "task":task,"frames":len(rgb),"feature_bytes":feat.stat().st_size,
        "official_source_commit":PIN}),flush=True)


def pilot_one(task,official_root,assets):
    """One original author TRAIN-dev physically executed rollout, not EVAL."""
    report=frozen_gate(task,official_root,assets)
    if not report["ready_to_develop_on_original_task"]:
        raise RuntimeError("M1C original task gate blocked: "+str(report["missing"]))
    _,_,_,checkpoint=paths(assets,task)
    official=Path(official_root).resolve()
    out=Path("outputs/m1c_long_horizon")/task/"pilot_one_original_pi05"
    out.mkdir(parents=True,exist_ok=True)
    cfg={
      "model":"demo_pi05",
      "checkpoint_dir":str(checkpoint),
      "scope":"one",                 # published TRAIN human video 0000
      "tasks":[task],
      "human_demo_and_env_types":[["without_distraction","without_distraction"]],
      "max_rollout_steps":800,
      "seed_offset_base":0,
      "save_video":True,
      "output_dir":str(out.resolve()),
    }
    config_path=out/"one_train_dev_config.json"
    if config_path.exists() and json.loads(config_path.read_text())!=cfg:
        raise RuntimeError("Frozen pilot config differs; no overwrite")
    config_path.write_text(json.dumps(cfg,indent=2)+"\n")
    summary=out/"summary.json"
    if summary.is_file():
        print("M1C_REAL_ORIGINAL_DEV_PILOT_PRESERVED",str(summary),flush=True)
        return
    # If previous work generated raw videos without summary, never overwrite.
    if any(x.name not in ("one_train_dev_config.json",) for x in out.iterdir()):
        raise RuntimeError(f"Partial original pilot outputs exist, review before rerun: {out}")
    env=os.environ.copy()
    env.update({
      "ROBOREEL_DATA":str((Path(assets).resolve()/"official_eval_data")),
      "ROBOREEL_CKPT":str((Path(assets).resolve()/"released_checkpoints")),
      "CUDA_VISIBLE_DEVICES":"0","MUJOCO_GL":"egl",
      "MUJOCO_EGL_DEVICE_ID":"0",
      "OPENPI_DATA_HOME":str(Path(
          env.get("OPENPI_DATA_HOME",
                  "outputs/v120_baselines/openpi_cache")).resolve()),
      "XLA_PYTHON_CLIENT_PREALLOCATE":"false",
    })
    with (out/"original_pilot_console.log").open("w") as f:
        rc=subprocess.run(
            [sys.executable,"-u","scripts/evaluation.py",
             "--config",str(config_path.resolve()),"--gpu","0"],
            cwd=official,env=env,stdout=f,stderr=subprocess.STDOUT,
            timeout=1500,check=False)
    if rc.returncode:
        raise RuntimeError(
            f"Original RoboReel multi-step TRAIN pilot exit {rc.returncode}; "
            f"see {out/'original_pilot_console.log'}")
    if not summary.exists():
        raise RuntimeError("Official pilot exited zero but summary.json absent")
    data=json.loads(summary.read_text())
    print("M1C_ORIGINAL_PHYSICS_DEV_ONE",json.dumps({
        "task":task,"checkpoint":RELEASED,"train_demo":TRAIN_ID,
        "summary":str(summary),
        "official_eval":False,"original_author_runner":True,
        "new_method":False,"note":"Success must be checked from actual per_trial, not inferred from exit code."}),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=TASKS,default="toys_in_drawer")
    p.add_argument("--stage",choices=("gate","get-one-train","encode-one","pilot-one"),
                   default="gate")
    p.add_argument("--assets",default="outputs/v120_baselines")
    p.add_argument("--official-root",default="third_party/roboreel_models")
    args=p.parse_args(argv)
    if args.stage=="gate":
        result=frozen_gate(args.task,args.official_root,args.assets)
        print("M1C_ORIGINAL_TASK_GATE",json.dumps(result),flush=True)
    elif args.stage=="get-one-train":
        download_one(args.task,args.assets)
    elif args.stage=="encode-one":
        encode_one(args.task,args.official_root,args.assets)
    else:
        pilot_one(args.task,args.official_root,args.assets)


if __name__=="__main__":
    main()
