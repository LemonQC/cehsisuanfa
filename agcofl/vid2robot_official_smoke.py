"""First actual *released* Vid2Robot policy -> official RoboReel physics smoke.

Uses authors' checkpoint and their runner / evaluator unmodified. One held-out
episode (0080) is only DEV/INTEGRATION; never paper's official test score.
No fallback to oracle state, random actions, or synthetic policy predictions.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path


def run(official_root,assets_root,out,gpu="0",smoke_id=80):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    official=Path(official_root).resolve()
    assets=Path(assets_root).resolve()
    result=dict(type="RELEASED_VID2ROBOT_ACTUAL_MANISKILL_SMOKE",
                status="PREFLIGHT",checkpoint_is_official=True,
                official_model_source="weiweigu1998/roboreel_models",
                is_formal_paper_score=False,
                task="push_cube",human_demo_id=int(smoke_id),gpu=str(gpu),
                privileged_state_in_policy=False,
                rollout_executed=False,official_success=None,
                max_rollout_steps=200)
    def finish():
        (out/"smoke_result.json").write_text(json.dumps(result,indent=2))
        print("VID2ROBOT_ROLLOUT_GATE",json.dumps(result),flush=True)
        return result
    ckpt=assets/"released_checkpoints/nd/vid2robot/ckpt.pt"
    cfg=assets/"released_checkpoints/nd/vid2robot/train_config.json"
    human=assets/f"official_eval_data/human_demos/eval/no_distraction/push_cube/{smoke_id:04d}/front.mp4"
    origin=official/"configs/eval/nd/vid2robot.json"
    submodule=official/"roboreel_env/src/envs"
    missing=[str(p) for p in (ckpt,cfg,human,origin,submodule) if not p.exists()]
    if missing:
        result.update(status="PREFLIGHT_BLOCKED_ASSETS_OR_ROBOREEL_ENV",missing=missing)
        return finish()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    os.environ.setdefault("MUJOCO_GL","egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID","0")
    os.environ["ROBOREEL_DATA"]=str(assets/"official_eval_data")
    os.environ["ROBOREEL_CKPT"]=str(assets/"released_checkpoints")
    sys_path=[official,official/"src",official/"src/external_models",
              official/"src/external_models/lerobot/src",
              official/"src/external_models/UniSkill",
              official/"src/external_models/UniSkill-Policy",
              official/"roboreel_env/src"]
    for folder in sys_path:
        if folder.is_dir() and str(folder) not in sys.path:
            sys.path.insert(0,str(folder))
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in official runtime")
        import mani_skill.envs  # noqa: F401
        import envs  # noqa: F401
        from evaluation import utils
        from evaluation.vid2robot_runner import build_runner
        cfgobj=utils.load_eval_config(origin)
        cfgobj.tasks=["push_cube"]
        cfgobj.human_demo_and_env_types=[("without_distraction","without_distraction")]
        cfgobj.scope="last20"
        cfgobj.save_video=False
        cfgobj.max_rollout_steps=200
        cfgobj.output_dir=str(out/"official_runner_output")
        # For DEV ONLY, restrict original official scope to a fixed single
        # held-out example. Official model, runner, env and physics remain
        # unchanged. Never report this as the official full last20 result.
        utils.EVAL_DEMO_SCOPES["last20"]=[int(smoke_id)]
        utils.EVAL_DEMO_SCOPES_DISTRACT["last20"]=[int(smoke_id)]
        runner=build_runner(cfgobj)
        result["checkpoint_loaded"]=True
        summary=utils.run_eval_protocol(cfgobj,runner)
        result["rollout_executed"]=True
        result["status"]="OFFICIAL_VID2ROBOT_DEV_SMOKE_EXECUTED"
        result["official_summary_path"]=str(Path(cfgobj.output_dir)/"summary.json")
        if isinstance(summary,dict):
            per_trial=summary.get("per_trial")
            if isinstance(per_trial,list) and len(per_trial)==1:
                result["official_success"]=bool(per_trial[0].get("success"))
                result["official_step_record"]=per_trial[0]
    except Exception as e:
        result.update(status="OFFICIAL_VID2ROBOT_RUNTIME_BLOCKED",
            error=f"{type(e).__name__}: {str(e)[:900]}")
        import traceback
        (out/"runtime_error.txt").write_text(traceback.format_exc())
    return finish()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--official-root",default="third_party/roboreel_models")
    p.add_argument("--assets",default="outputs/v120_baselines")
    p.add_argument("--out",default="outputs/v120_baselines/official_vid2robot_smoke")
    p.add_argument("--gpu",default="0")
    a=p.parse_args()
    run(a.official_root,a.assets,a.out,a.gpu)

if __name__=="__main__":main()
