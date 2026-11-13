#!/usr/bin/env python3
"""Headless scene-only smoke gate for the original RoboReel ObjectInDrawer.

No policy checkpoint load, no human videos, no policy env.step() and no output
manifest. Instantiate/reconfigure official scene once to validate that original
PartNet URDF, its meshes, SAPIEN loader and ManiSkill ASSET_DIR agree.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.prepare_roboreel_drawer_partnet import configure_for_robot_dev, PROJECT


def smoke(project: Path) -> dict:
    project=project.expanduser().resolve()
    proof=configure_for_robot_dev(project,'toys_in_drawer')
    if proof['status']!='READY_FOR_SCENE_INIT':
        raise RuntimeError('PINNED_DRAWER_ASSET_GATE_NOT_READY')
    os.environ.pop('DISPLAY',None)
    os.environ.pop('WAYLAND_DISPLAY',None)
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')
    official=project/'third_party/roboreel_models'
    from scripts.roboreel_video_dependency_dev import inject_upstream
    inject_upstream(official)
    import gymnasium as gym  # noqa: F401
    import mani_skill.envs  # noqa: F401
    import envs  # noqa: F401
    from mani_skill import ASSET_DIR
    expected=project/'outputs/roboreel_partnet_single_44962/data'
    if Path(ASSET_DIR).resolve()!=expected.resolve():
        raise RuntimeError('MS_ASSET_DIR_IMPORTED_WITH_INCORRECT_ROOT')
    from evaluation.utils import make_eval_env
    environment=None
    try:
        environment,_ = make_eval_env('toys_in_drawer',False,max_episode_steps=400)
        environment.reset(seed=310000)
        # A real scene state (not inference) confirms the asset is parsed.
        state=environment.unwrapped.get_state_dict()
        if not isinstance(state,dict) or not state:
            raise RuntimeError('EMPTY_OFFICIAL_DRAWER_SCENE_STATE')
        return {'status':'DRAWER_SCENE_READY_FOR_BOUNDED_DEV',
                'task':'toys_in_drawer','asset_model':'44962',
                'asset_dir':str(ASSET_DIR),'reset_seed':310000,
                'state_keys':sorted(map(str,state.keys())),
                'policy_actions':0,'new_checkpoint_load':False,
                'new_human_video_features':False}
    finally:
        if environment is not None:environment.close()


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=PROJECT)
    args=p.parse_args(argv)
    try:
        result=smoke(args.project)
    except Exception as exc:
        # No HF token is used here. Never discard the original env exception
        # category but limit logs to a short diagnostic.
        result={'status':'DRAWER_SCENE_SMOKE_BLOCKED',
                'error_class':type(exc).__name__,
                'error':str(exc)[:400], 'policy_actions':0}
        print('ROBOREEL_DRAWER_SCENE_SMOKE',json.dumps(result,ensure_ascii=False),flush=True)
        return 3
    print('ROBOREEL_DRAWER_SCENE_SMOKE',json.dumps(result,ensure_ascii=False),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
