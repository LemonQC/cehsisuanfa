"""Thin official Vid2Robot full last20 evaluator entrypoint.

Uses authors' ORIGINAL Vid2Robot model, strict pretrained checkpoint and
run_eval_protocol. The ONLY adaptation skips redundant timm ImageNet
initialization (all tensors are strictly replaced by released policy weights).
Unlike the single-0080 DEV smoke, NEVER patches EVAL_DEMO_SCOPES.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

def run(config, official_root, gpu="0"):
    root=Path(official_root).resolve()
    cfg_path=Path(config).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    config_json=json.loads(cfg_path.read_text())
    if config_json.get("model")!="vid2robot":
        raise ValueError("Only original Vid2Robot official runner is allowed")
    if config_json.get("scope")!="last20":
        raise ValueError("Official last20 EVAL protocol required")
    if config_json.get("tasks")!=["push_cube"]:
        raise ValueError("Only published PushCube ND cell is supported by this data subset")
    if config_json.get("human_demo_and_env_types")!=[["without_distraction","without_distraction"]]:
        raise ValueError("Requires original clean human × clean env condition")
    if config_json.get("max_rollout_steps")!=800 or config_json.get("chunk_size")!=1:
        raise ValueError("Do not silently change official max steps or action chunk")
    if (root/"roboreel_env/src/envs").is_dir() is False:
        raise FileNotFoundError("Pinned official RoboReel environment submodule missing")

    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    os.environ.setdefault("MUJOCO_GL","egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID","0")
    # Match official scripts/evaluation.py import paths.
    for folder in (
        root,root/"src",root/"src/external_models",
        root/"src/external_models/lerobot/src",
        root/"src/external_models/UniSkill",
        root/"src/external_models/UniSkill-Policy",
        root/"roboreel_env/src"
    ):
        if folder.is_dir() and str(folder) not in sys.path:
            sys.path.insert(0,str(folder))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Cannot evaluate official pretrained robot policy without CUDA")
    import mani_skill.envs  # noqa
    import envs  # noqa
    from evaluation import utils
    from evaluation.vid2robot_runner import build_runner
    import timm

    cfg=utils.load_eval_config(cfg_path)
    if utils.demo_ids_for_scope("last20",demo_distract=False)!=list(range(80,100)):
        raise RuntimeError("Official dataset eval scope is not 0080-0099")
    original=timm.create_model
    modified=[0]
    def create_without_redundant_initializer(name,*args,**kwargs):
        if kwargs.get("pretrained"):
            modified[0]+=1
            kwargs["pretrained"]=False
        return original(name,*args,**kwargs)
    try:
        timm.create_model=create_without_redundant_initializer
        runner=build_runner(cfg)  # original strict load_state_dict()
    finally:
        timm.create_model=original
    print("OFFICIAL_VID2ROBOT_MODEL_STRICTLY_LOADED",json.dumps({
       "checkpoint":str(runner.checkpoint_path),
       "cuda":torch.cuda.get_device_name(0),
       "redundant_timm_initializer_calls_skipped":modified[0],
       "original_model_and_eval_runner":True}),flush=True)
    # Observability ONLY: original load_demo_static and rollout are called once
    # with exactly their original arguments, and their returns are unchanged.
    # This prints per-demo progress and atomically refreshes progress.json.
    from agcofl.official_live_progress import install_official_progress
    install_official_progress(
        runner, cfg.output_dir, cfg.seed_offset_base, total=20)
    return utils.run_eval_protocol(cfg,runner)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--official-root",default="third_party/roboreel_models")
    p.add_argument("--gpu",default="0")
    a=p.parse_args()
    run(a.config,a.official_root,a.gpu)

if __name__=="__main__":main()
