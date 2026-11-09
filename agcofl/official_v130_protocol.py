"""Strict RoboReel released Vid2Robot official ND evaluation, 20 demos × seeds.

One *official clean/clean cell* (PushCube, no distractions) using original
RoboReel protocol, not 10-task or WD/ED results. No checkpoint tuning on EVAL.
Three seed passes (offsets 0,100000,200000); each pass is resumable only when
a real original summary contains the complete 0080-0099 20 trials.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

EXPECTED_SOURCE="79a0fb0099e73c5560114753a8706276f93aefc2"
EXPECTED_DEMOS=list(range(80,100))
EXPECTED_CELL="demo_distract=False|env_distract=False"
SEEDS=(0,100000,200000)


def build_config(original_cfg,output_dir,offset):
    c=json.loads(Path(original_cfg).read_text())
    if c.get("model")!="vid2robot" or c.get("scope")!="last20":
        raise ValueError("Not the pinned original Vid2Robot last20 eval config")
    if int(c.get("max_rollout_steps",-1))!=800 or int(c.get("chunk_size",-1))!=1:
        raise ValueError("Original official action budget or controller chunk changed")
    if offset not in SEEDS:
        raise ValueError("Expected original 0,100000,200000 seed protocol")
    c.update(tasks=["push_cube"],
             human_demo_and_env_types=[["without_distraction","without_distraction"]],
             seed_offset_base=offset,output_dir=str(Path(output_dir).resolve()),
             save_video=False)
    return c


def validate_summary(summary,offset):
    """A paper cell requires all 20 genuine logged boolean outcomes."""
    if not isinstance(summary,dict):
        raise ValueError("Official summary is not a dict")
    required={"model":"vid2robot","runner":"vid2robot","scope":"last20",
        "seed_offset_base":offset,"tasks":["push_cube"],"chunk_size":1,
        "max_rollout_steps":800,
        "human_demo_and_env_types":[["without_distraction","without_distraction"]]}
    for key,expected in required.items():
        if summary.get(key)!=expected:
            raise ValueError(f"Official field {key} mismatch: {summary.get(key)!r} != {expected!r}")
    if summary.get("demo_ids_no_distract")!=EXPECTED_DEMOS:
        raise ValueError("Wrong original 0080-0099 scope")
    per=summary.get("per_trial")
    if not isinstance(per,dict) or set(per)!={EXPECTED_CELL}:
        raise ValueError("Missing / extra official ND evaluation cells")
    rows=per[EXPECTED_CELL]
    if not isinstance(rows,list) or len(rows)!=20:
        raise ValueError("Official ND cell must contain all 20 real trials")
    ids=[]
    for row in rows:
        if not isinstance(row,dict) or row.get("task")!="push_cube":
            raise ValueError("Wrong official task")
        if not isinstance(row.get("demo_id"),int):
            raise ValueError("Invalid official demo ID")
        if type(row.get("success")) is not bool:
            raise ValueError("A missing/ambiguous physics outcome is NOT a failure or success")
        ids.append(row["demo_id"])
    if sorted(ids)!=EXPECTED_DEMOS:
        raise ValueError("Missing or duplicate official 0080-0099 outcomes")
    nsuccess=sum(r["success"] for r in rows)
    table=summary.get("table",{}).get(EXPECTED_CELL)
    if not isinstance(table,dict) or table.get("total")!=20 or table.get("success")!=nsuccess:
        raise ValueError("Official evaluator table and 20 real physical outcomes disagree")
    if abs(float(table.get("rate",-1))-nsuccess/20)>1e-10:
        raise ValueError("Invalid recorded success rate")
    return dict(seed_offset_base=offset,success=nsuccess,total=20,
        success_rate=nsuccess/20,
        per_trial=[{"task":r["task"],"demo_id":r["demo_id"],
                    "success":r["success"]} for r in rows])


def aggregate_passes(results):
    ordered=[results[o] for o in SEEDS if o in results]
    rates=[r["success_rate"] for r in ordered]
    k=len(rates)
    mean=sum(rates)/k if rates else None
    sem=None
    if k>=2:
        variance=sum((x-mean)**2 for x in rates)/(k-1)
        sem=math.sqrt(variance/k)
    return dict(
        type="ROBOREEL_OFFICIAL_VID2ROBOT_ND_PUSHCUBE_FULL20",
        protocol="author_original_pinned_clean_clean_only",
        original_model_and_environment=True,
        original_source_commit=EXPECTED_SOURCE,
        checkpoint="LogosRoboticsLab/roboreel_models:nd/vid2robot/ckpt.pt",
        evaluation_video_ids=EXPECTED_DEMOS,
        task="push_cube",robot_control="pd_joint_pos",scope="last20",
        observation="official human FRONT video + robot base_camera RGB history",
        max_rollout_steps=800,chunk_size=1,seed_offsets=[x["seed_offset_base"] for x in ordered],
        evaluated_seed_passes=k,requested_seed_passes=3,
        per_pass=ordered,mean_success_rate=mean,sem_across_seed_passes=sem,
        pooled_successes=sum(x["success"] for x in ordered),
        pooled_trials=sum(x["total"] for x in ordered),
        full_3pass_complete=k==3,
        paper_scope=("Only the PushCube no-distraction clean human/clean env cell; "
                     "NOT the all-task RoboReel benchmark and NOT a "
                     "comparison of source human Pick to target Push."),
        no_eval_training_or_tuning=True)


def preflight(root,assets):
    root=Path(root).resolve()
    assets=Path(assets).resolve()
    result=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],
        capture_output=True,text=True,check=True)
    if result.stdout.strip()!=EXPECTED_SOURCE:
        raise RuntimeError("Official RoboReel source commit differs from pinned base")
    check=[root/"configs/eval/nd/vid2robot.json",
        root/"roboreel_env/src/envs",
        root/"roboreel_env/init_states/without_distraction/push_cube.pkl",
        assets/"released_checkpoints/nd/vid2robot/ckpt.pt",
        assets/"released_checkpoints/nd/vid2robot/train_config.json"]
    check += [assets/f"official_eval_data/human_demos/eval/no_distraction/push_cube/{n:04d}/front.mp4"
              for n in EXPECTED_DEMOS]
    missing=[str(p) for p in check if not p.exists()]
    if missing:
        raise FileNotFoundError("Original official eval resources missing: "+json.dumps(missing[:10]))
    asset_status=assets/"official_vid2robot_assets.json"
    if not asset_status.is_file():
        raise FileNotFoundError(asset_status)
    report=json.loads(asset_status.read_text())
    if report.get("status")!="OFFICIAL_VID2ROBOT_RELEASE_AND_EVAL_DATA_READY":
        raise RuntimeError("Do not run until both official released model & 20 eval videos are verified")
    return root,assets


def _pass_content(config):
    # Canonical config fingerprint for skipping an already completed pass.
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def evaluate(out,source,assets,offsets=SEEDS,gpu="0"):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    status_path=out/"run_status.json"
    result=dict(type="OFFICIAL_FULL20_3PASS_RUN_STATUS",
        status="PREFLIGHT",official_model="Vid2Robot",task="push_cube",
        original_source_commit=EXPECTED_SOURCE,seed_offsets_requested=list(offsets),
        no_eval_training_or_tuning=True,passes_completed=[],errors=[])
    def save():
        status_path.write_text(json.dumps(result,indent=2))
    try:
        source,assets=preflight(source,assets)
    except Exception as exc:
        result.update(status="PREFLIGHT_BLOCKED",errors=[
            f"{type(exc).__name__}: {str(exc)[:1000]}"])
        save()
        print("OFFICIAL_V130_PREFLIGHT_BLOCKED",result["errors"][0],flush=True)
        return result
    config_file=source/"configs/eval/nd/vid2robot.json"
    os_env=os.environ.copy()
    os_env["ROBOREEL_DATA"]=str(assets/"official_eval_data")
    os_env["ROBOREEL_CKPT"]=str(assets/"released_checkpoints")
    os_env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    os_env.setdefault("MUJOCO_GL","egl")
    os_env["MUJOCO_EGL_DEVICE_ID"]=str(gpu)
    os_env.setdefault("OMP_NUM_THREADS","4")
    results={}
    for offset in offsets:
        stage=out/f"seed_offset_{offset}"
        stage.mkdir(parents=True,exist_ok=True)
        config=build_config(config_file,stage,offset)
        fp=_pass_content(config)
        config_path=out/f"config_seed_offset_{offset}.json"
        # Prevent silently reinterpreting a stale result from a different
        # evaluation protocol when reusing the previous pass.
        if config_path.exists():
            previous=json.loads(config_path.read_text())
            if previous!=config:
                result["errors"].append(
                    f"seed={offset}: stored config differs; refusing to reuse or overwrite results")
                break
        else:
            config_path.write_text(json.dumps(config,indent=2))
        existing=stage/"summary.json"
        if existing.exists():
            try:
                row=validate_summary(json.loads(existing.read_text()),offset)
                results[offset]=row
                print("OFFICIAL_V130_PASS_REUSED",json.dumps(dict(
                   seed=offset,success=row["success"],total=20,
                   config_sha256=fp)),flush=True)
                result["passes_completed"]=list(results)
                save()
                continue
            except Exception as e:
                result["errors"].append(
                    f"seed={offset}: stale or incomplete official summary; "
                    f"manual inspection required: {type(e).__name__} {e}")
                break
        cmd=[sys.executable,"-u","-m","agcofl.vid2robot_official_full_eval",
             "--config",str(config_path),"--official-root",str(source),
             "--gpu",str(gpu)]
        print("OFFICIAL_V130_PASS_START",json.dumps(dict(
            seed=offset,demos=20,max_rollout_steps=800,config_sha256=fp)),flush=True)
        started=time.time()
        with (stage/"run.log").open("w") as logfile:
            proc=subprocess.Popen(cmd,env=os_env,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                logfile.write(line)
                logfile.flush()
                print(line,end="",flush=True)
            code=proc.wait()
        if code!=0:
            result["errors"].append(f"seed={offset}: official evaluator exit code {code}")
            print("OFFICIAL_V130_PASS_FAILED",result["errors"][-1],flush=True)
            break
        try:
            row=validate_summary(json.loads(existing.read_text()),offset)
        except Exception as e:
            result["errors"].append(f"seed={offset}: invalid original official evaluator summary: {e}")
            print("OFFICIAL_V130_SUMMARY_INVALID",result["errors"][-1],flush=True)
            break
        results[offset]=row
        result["passes_completed"]=list(results)
        save()
        print("OFFICIAL_V130_PASS_VERIFIED",json.dumps(dict(
            seed=offset,success=row["success"],total=20,
            rate=row["success_rate"],wall_seconds=int(time.time()-started))),flush=True)

    agg=aggregate_passes(results)
    (out/"aggregated_summary.json").write_text(json.dumps(agg,indent=2))
    result["passes_completed"]=list(results)
    result["status"]=(
       "OFFICIAL_VID2ROBOT_ND_3PASS_EVALUATED" if len(results)==3
       else "OFFICIAL_VID2ROBOT_ND_PARTIAL_OR_BLOCKED")
    result["aggregated_summary"]=str(out/"aggregated_summary.json")
    save()
    print("OFFICIAL_V130_EVALUATION_GATE",json.dumps(dict(
       status=result["status"],passes=len(results),mean=agg["mean_success_rate"],
       sem=agg["sem_across_seed_passes"],errors=result["errors"])),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="outputs/v130_official_vid2robot")
    p.add_argument("--source",default="third_party/roboreel_models")
    p.add_argument("--assets",default="outputs/v120_baselines")
    p.add_argument("--gpu",default="0")
    p.add_argument("--seed-offsets",default="0,100000,200000")
    args=p.parse_args()
    offsets=tuple(int(x) for x in args.seed_offsets.split(","))
    if len(set(offsets))!=len(offsets) or any(x not in SEEDS for x in offsets):
        raise ValueError("Only unique 0,100000,200000 original seed offsets allowed")
    evaluate(args.out,args.source,args.assets,offsets,args.gpu)


if __name__=="__main__":main()
