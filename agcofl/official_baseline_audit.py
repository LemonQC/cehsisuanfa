"""Inspect pinned official RoboReel baseline implementation and available data.

Read-only: do not download large checkpoints or pretend that model script
presence implies successful training or rollout.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess

OFFICIAL_REPO="https://github.com/weiweigu1998/roboreel_models.git"
OFFICIAL_COMMIT="79a0fb0099e73c5560114753a8706276f93aefc2"
METHODS={
 "B03_vid_cond_pi05_uniform":("scripts/train_vid_cond_pi05.py","scripts/evaluation.py"),
 "B04_vid_cond_pi05_keyframes":("scripts/train_vid_cond_pi05.py","scripts/evaluation.py"),
 "B05_vid2robot":("scripts/train_vid2robot.py","scripts/evaluation.py"),
 "B06_uniskill_policy":("scripts/train_uniskill_policy.py","scripts/evaluation.py"),
 "B07_vlm_pi05":("scripts/train_vlm_pi05.py","scripts/evaluation.py"),
}
DATA_ROOT="outputs/next_gate_v100/train_data"

def audit(repo_path, dataset_root, out):
    repo=Path(repo_path)
    dataset=Path(dataset_root)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    checked=repo.is_dir() and (repo/".git").exists()
    sha=None
    if checked:
        p=subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],
                         text=True,capture_output=True,check=False)
        sha=p.stdout.strip() if p.returncode==0 else None
    rows=[]
    for method,(train,ev) in METHODS.items():
        paths={key:str(repo/f) for key,f in (("train",train),("evaluate",ev))}
        present={key:Path(f).is_file() for key,f in paths.items()}
        rows.append(dict(id=method,source="OFFICIAL_ROBOREEL_MODELS",
            official_commit_expected=OFFICIAL_COMMIT,
            pinned_commit_matches=sha==OFFICIAL_COMMIT,
            training_entrypoint=train,evaluation_entrypoint=ev,
            entrypoints_present=present,
            status="OFFICIAL_CODE_PINNED_NOT_RUNTIME_VERIFIED" if
              all(present.values()) and sha==OFFICIAL_COMMIT else
              "OFFICIAL_CHECKOUT_INCOMPLETE_OR_UNPINNED",
            checkpoint_loaded=False,official_rollout_executed=False))
    idx=dataset/"downloaded_train_index.json"
    records=json.loads(idx.read_text()) if idx.is_file() else []
    genuine=[v for v in records
             if "/train/" in "/"+v.get("path","").lower()+"/"
             and "/eval/" not in "/"+v.get("path","").lower()+"/"
             and Path(v.get("local_video","")).is_file()]
    manifest=dict(
        type="ROBOREEL_OFFICIAL_BASELINE_AVAILABILITY_AUDIT",
        repository=OFFICIAL_REPO,expected_source_commit=OFFICIAL_COMMIT,
        checkout=str(repo.resolve()),observed_source_commit=sha,
        pinned=sha==OFFICIAL_COMMIT,
        status="OFFICIAL_PINNED_CODE_PRESENT" if sha==OFFICIAL_COMMIT else "OFFICIAL_CODE_NOT_PINNED",
        real_train_front_videos=len(genuine),
        real_train_video_ids=sorted([v["path"] for v in genuine]),
        robot_action_trajectory_local_count=0,
        robot_action_data_not_yet_verified=True,
        checkpoint_downloaded=False,
        inference_performed=False,
        methods=rows,
        caveat=("Official implementation presence != executable pretraining. "
                "Official pi0.5 and UniSkill have heavyweight external base weights "
                "and/or dependencies and may require separate isolated Conda envs. "
                "Do not pool performance with privileged-state diagnostic results."))
    out.write_text(json.dumps(manifest,indent=2))
    print("OFFICIAL_BASELINE_PREFLIGHT",json.dumps(dict(
        pinned=manifest["pinned"],source_commit=sha,
        train_videos=len(genuine),code_ready=sum(
            x["status"]=="OFFICIAL_CODE_PINNED_NOT_RUNTIME_VERIFIED" for x in rows),
        actual_gpu_training=False,
        report=str(out))),flush=True)
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",default="third_party/roboreel_models")
    p.add_argument("--dataset",default=DATA_ROOT)
    p.add_argument("--out",default="outputs/v110_baselines/official_baseline_audit.json")
    args=p.parse_args()
    audit(args.repo,args.dataset,args.out)

if __name__=="__main__":main()
