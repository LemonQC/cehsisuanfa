"""Metadata-first, strictly bounded OFFICIAL Vid2Robot released baseline assets.

The source of checkpoints is the authors' Hugging Face model repository.
The source of evaluation videos is the authors' HF dataset EVAL split.
No surrogate policies, fabricated checkpoints, or evaluated-on-train claims.

Only selected files are downloaded; official 104 GB TRAIN dataset and ~50GB
all-method weights are never downloaded by this program.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

DATASET="LogosRoboticsLab/roboreel"
MODELS="LogosRoboticsLab/roboreel_models"
ENDPOINT="https://hf-mirror.com"
MODEL_PREFIX="nd/vid2robot/"
EVAL_PREFIX="human_demos/eval/no_distraction/push_cube/"
EVAL_IDS=range(80,100)


def model_plan(api,max_gb=8.0):
    files=[]
    for x in api.list_repo_tree(MODELS,repo_type="model",
          path_in_repo="nd/vid2robot",recursive=True):
        if "file" not in type(x).__name__.lower():continue
        files.append(dict(path=x.path,bytes=int(getattr(x,"size",0) or 0)))
    expected={"ckpt.pt","train_config.json"}
    names={Path(x["path"]).name for x in files}
    if not expected.issubset(names):
        return dict(status="OFFICIAL_CHECKPOINT_FILES_MISSING",
          files=files,missing=sorted(expected-names))
    if any(x["bytes"]<=0 for x in files):
        return dict(status="CHECKPOINT_SIZE_METADATA_MISSING",files=files)
    needed=[x for x in files if Path(x["path"]).name in expected]
    size=sum(x["bytes"] for x in needed)
    if size>max_gb*1024**3:
        return dict(status="CHECKPOINT_EXCEEDS_DOWNLOAD_BUDGET",
           bytes=size,budget_gb=max_gb,files=needed)
    return dict(status="OFFICIAL_CHECKPOINT_DOWNLOAD_APPROVED",
       bytes=size,budget_gb=max_gb,files=needed)


def eval_plan(api,task="push_cube",max_mb=250):
    root=f"human_demos/eval/no_distraction/{task}"
    files=[]
    for x in api.list_repo_tree(DATASET,repo_type="dataset",
                                path_in_repo=root,recursive=True):
        if "file" not in type(x).__name__.lower():continue
        p=x.path
        if Path(p).name!="front.mp4":continue
        demo=Path(p).parent.name
        if not demo.isdigit() or int(demo) not in EVAL_IDS:continue
        files.append(dict(path=p,bytes=int(getattr(x,"size",0) or 0)))
    files.sort(key=lambda x:x["path"])
    if len(files)!=20:
        return dict(status="EVAL_SUBSET_INCOMPLETE",task=task,
                    expected_videos=20,found=len(files),files=files)
    if any(x["bytes"]<=0 for x in files):
        return dict(status="EVAL_FILE_SIZES_UNKNOWN",files=files)
    size=sum(x["bytes"] for x in files)
    if size>int(max_mb)*1024**2:
        return dict(status="EVAL_DOWNLOAD_OVER_BUDGET",
           bytes=size,budget_mb=max_mb,files=files)
    return dict(status="OFFICIAL_EVAL_VIDEOS_DOWNLOAD_APPROVED",
                task=task,files=files,bytes=size,budget_mb=max_mb)


def acquire(repo,files,root,endpoint=ENDPOINT):
    """Download with resume; return partial successes even on file failures.

    Reuses the verified existing 0080 file from the earlier RoboReel
    integration, provided the exact expected size matches the official
    metadata. No EVAL data are ever used as robot training examples.
    """
    from .resumable_hf_assets import get_file,sha256_file
    from pathlib import Path
    import os
    import shutil
    output=[]
    errors=[]
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    for x in files:
        # The original user repo already contains one official eval sample.
        # If bytes match, preserve disk by hard-linking it into the official
        # checkpoint evaluation layout, never relying on naming alone.
        if repo==DATASET and x["path"].endswith("/0080/front.mp4"):
            existing=(Path.cwd()/"third_party/roboreel"/x["path"])
            target=root/x["path"]
            if (not target.exists()) and existing.is_file() and (
                existing.stat().st_size==int(x["bytes"])):
                target.parent.mkdir(parents=True,exist_ok=True)
                try:
                    os.link(existing,target)
                except OSError:
                    shutil.copyfile(existing,target)
                print("HF_ASSET_REUSED_PRIOR_EVAL",json.dumps(dict(
                    path=x["path"],bytes=target.stat().st_size)),flush=True)
        try:
            got=get_file(x,repo,
                "model" if repo==MODELS else "dataset",
                root,endpoint=endpoint)
            output.append({k:got[k] for k in ("path","bytes","sha256","transport")})
            print("OFFICIAL_ASSET_READY",json.dumps(dict(
                path=x["path"],bytes=got["bytes"],source=got["transport"])),flush=True)
        except Exception as ex:
            errors.append(dict(path=x["path"],
                error=f"{type(ex).__name__}: {str(ex)[:550]}"))
            print("OFFICIAL_ASSET_NOT_READY",json.dumps(errors[-1]),flush=True)
            # A large released checkpoint is the primary priority. Don't
            # continue attempting thousands of model files on a broken link.
            if repo==MODELS:
                break
    return output,errors

def run(output,endpoint,max_checkpoint_gb=8.0,max_eval_mb=250,download=False):
    from huggingface_hub import HfApi
    os.environ["HF_ENDPOINT"]=endpoint
    os.environ.setdefault("HF_HUB_DISABLE_XET","1")
    api=HfApi(endpoint=endpoint)
    report=dict(type="OFFICIAL_RELEASED_VID2ROBOT_ASSETS",
                source_model_repo=MODELS,source_eval_dataset=DATASET,
                endpoint=endpoint,model_prefix=MODEL_PREFIX,eval_prefix=EVAL_PREFIX,
                official_eval_ids=list(EVAL_IDS),official_checkpoint_loaded=False,
                official_robot_rollout_executed=False,
                model=None,eval_videos=None,status="PLANNING")
    try:
        report["model"]=model_plan(api,max_checkpoint_gb)
    except Exception as ex:
        report["model"]=dict(status="MODEL_METADATA_BLOCKED",
                error=f"{type(ex).__name__}: {str(ex)[:550]}")
    try:
        report["eval_videos"]=eval_plan(api,max_mb=max_eval_mb)
    except Exception as ex:
        report["eval_videos"]=dict(status="EVAL_METADATA_BLOCKED",
                error=f"{type(ex).__name__}: {str(ex)[:550]}")
    report["status"]="OFFICIAL_RELEASE_ASSETS_PLANNED"
    for category in ("model","eval_videos"):
        step=report[category]
        step["downloaded"]=[]
        if not download:
            continue
        if step["status"] not in ("OFFICIAL_CHECKPOINT_DOWNLOAD_APPROVED",
                                 "OFFICIAL_EVAL_VIDEOS_DOWNLOAD_APPROVED"):
            continue
        target=(Path(output)/("released_checkpoints" if category=="model"
              else "official_eval_data"))
        name=MODELS if category=="model" else DATASET
        done,errors=acquire(name,step["files"],target,endpoint=endpoint)
        step["downloaded"]=done
        step["failures"]=errors
        step["status"]=("OFFICIAL_RELEASE_ASSETS_ACQUIRED"
                        if len(done)==len(step["files"]) else
                        "OFFICIAL_RELEASE_DOWNLOAD_PARTIAL")
        if errors:
            step["error"]=errors[0]["error"]
        # Persist stage progress BEFORE trying another category. An interrupted
        # 2.3GB checkpoint must not erase earlier verified downloads.
        outpath=Path(output)/"official_vid2robot_assets.json"
        outpath.parent.mkdir(parents=True,exist_ok=True)
        outpath.write_text(json.dumps(report,indent=2))
    modelok=report["model"]["status"]=="OFFICIAL_RELEASE_ASSETS_ACQUIRED"
    evalok=report["eval_videos"]["status"]=="OFFICIAL_RELEASE_ASSETS_ACQUIRED"
    if modelok and evalok:report["status"]="OFFICIAL_VID2ROBOT_RELEASE_AND_EVAL_DATA_READY"
    elif not download:report["status"]="DOWNLOAD_NOT_REQUESTED"
    else:report["status"]="OFFICIAL_VID2ROBOT_ASSET_SETUP_INCOMPLETE"
    report["model_checkpoint_local"]=str(Path(output)/"released_checkpoints/nd/vid2robot")
    report["official_eval_data_root"]=str(Path(output)/"official_eval_data")
    path=Path(output)/"official_vid2robot_assets.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2))
    print("VID2ROBOT_OFFICIAL_RELEASE_GATE",json.dumps(dict(
       status=report["status"],
       model_status=report["model"]["status"],
       eval_status=report["eval_videos"]["status"],
       release_not_a_measured_success_rate=True,
       report=str(path))),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="outputs/v120_baselines")
    p.add_argument("--endpoint",default=os.getenv("HF_ENDPOINT",ENDPOINT))
    p.add_argument("--max-checkpoint-gb",type=float,default=8.0)
    p.add_argument("--max-eval-mb",type=float,default=250)
    p.add_argument("--download",action="store_true")
    a=p.parse_args()
    run(a.out,a.endpoint,a.max_checkpoint_gb,a.max_eval_mb,a.download)


if __name__=="__main__":main()
