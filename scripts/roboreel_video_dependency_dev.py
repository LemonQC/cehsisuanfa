#!/usr/bin/env python3
"""Small, bounded RoboReel TRAIN-video sensitivity pilot with frozen official pi0.5.

Preflight never imports simulator/model. --run uses the *existing* official
DemoPi05Runner, official task/env and TRAIN human video features only.  It
compares two distinct same-task videos with identical robot initial states
and policy RNG, not fabricated opposite labels or human-vs-robot task pairs.

This is DEV only: it DOES NOT establish video necessity or T-RO novelty.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

PINNED_UPSTREAM = '79a0fb0099e73c5560114753a8706276f93aefc2'
DEFAULT_TASK = 'toys_in_drawer'
ALLOWED_TASKS = {'toys_in_drawer': 'ObjectInDrawer-v1', 'push_cube': 'PushCube-v2'}
DEFAULT_CANDIDATES = ('outputs/v140_pi05_uniform/config_seed_offset_0.json',
                      'outputs/v140_pi05_uniform/seed_offset_0/config.json')
TRAIN_IDS = range(0, 80)
EVAL_IDS = range(80, 100)
REQUIRED_STATIC = ('static_image_emb', 'static_pos_emb', 'static_state_emb', 'static_mask')


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


# The existing M0 executor exported these directory roots. The saved
# evaluator JSON uses ${ROBOREEL_CKPT}; do not rely on a prior shell export.
VARIABLE = re.compile(r'\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))')


def official_path_vars(project: Path) -> dict[str, str]:
    base = project.resolve() / 'outputs/v120_baselines'
    defaults = {'ROBOREEL_CKPT': base / 'released_checkpoints',
                'ROBOREEL_DATA': base / 'official_eval_data'}
    # An explicit environment override takes precedence over project defaults.
    return {key: os.path.expanduser(os.environ.get(key) or str(path))
            for key, path in defaults.items()}


def project_path(project: Path, value: str, path_vars: dict | None = None) -> Path:
    variables = dict(os.environ)
    if path_vars is not None:
        variables.update(path_vars)
    def replace_var(match):
        key = match.group(1) or match.group(2)
        if not variables.get(key):
            raise ValueError(f'UNEXPANDED_PATH_VARIABLE: {value}')
        return str(variables[key])
    s = os.path.expanduser(VARIABLE.sub(replace_var, str(value)))
    if '$' in s:
        raise ValueError(f'UNEXPANDED_PATH_VARIABLE: {value}')
    p = Path(s)
    return (p if p.is_absolute() else project / p).resolve()


def read_json(path:Path):
    return json.loads(path.read_text(encoding='utf-8'))


def select_config(project:Path, config:Path|None):
    if config is not None:
        return project_path(project,str(config))
    for rel in DEFAULT_CANDIDATES:
        p=project/rel
        if p.is_file():
            return p.resolve()
    return (project/DEFAULT_CANDIDATES[0]).resolve()


def check_features(dir_:Path, cameras:list[str]) -> dict:
    found=[]; missing=[]
    for camera in cameras:
        bundle=dir_/f'{camera}.npy'
        # Official loader supports either bundled images or per-timestep token files.
        old=sorted(dir_.glob(f'token_emb_*_{camera}.npy')) if dir_.is_dir() else []
        if bundle.is_file() and bundle.stat().st_size>512:
            found.append(str(bundle))
        elif old and all(p.is_file() and p.stat().st_size>512 for p in old):
            found.extend(map(str,old))
        else:
            missing.append(str(bundle))
    return {'present':len(missing)==0,'paths':found,'missing':missing}


def video_inventory(data:Path,task:str,cameras:list[str]):
    """File identities only, never assumes a video was decoded or semantics proven."""
    items=[]
    for demo_id in TRAIN_IDS:
        d=f'{demo_id:04d}'
        # RoboReel media path uses no_distraction; features use without_distraction.
        vroot=data/'human_demos'/'train'/'no_distraction'/task/d
        froot=data/'video_demonstration_features'/'without_distraction'/task/d
        videos=[vroot/f'{cam}.mp4' for cam in cameras]
        video_ok=all(p.is_file() and p.stat().st_size>1024 for p in videos)
        feats=check_features(froot,cameras)
        if video_ok or feats['present']:
            items.append({'id':demo_id, 'video_present':video_ok,
                          'feature_present':feats['present'],
                          'video_paths':[str(x) for x in videos],
                          'feature_paths':feats['paths']})
    return items


def preflight(project:Path, config:Path|None, task:str, data_override:Path|None=None) -> dict:
    project=project.resolve()
    cfg_path=select_config(project,config)
    path_vars=official_path_vars(project)
    result={'type':'ROBOREEL_DEV_VIDEO_DEPENDENCY_PREFLIGHT',
            'task':task,'task_env':ALLOWED_TASKS[task],
            'project':str(project),'eval_config':str(cfg_path),
            'resolved_official_path_vars':path_vars,
            'train_demo_ids_only':True,'heldout_eval_ids_excluded':True,
            'new_physx_actions':0,'new_training_runs':0,'status':'BLOCKED'}
    if not cfg_path.is_file():
        result.update(reason='MISSING_EVAL_CONFIG',expected=str(cfg_path))
        return result
    try:
        cfg=read_json(cfg_path)
        if cfg.get('model')!='demo_pi05':
            raise ValueError('Config must be original demo_pi05')
        if 'checkpoint_dir' not in cfg:
            raise ValueError('MISSING_CHECKPOINT_DIR_IN_CONFIG')
        checkpoint=project_path(project,cfg['checkpoint_dir'],path_vars)
        for cand in (checkpoint/'train_config.json',checkpoint.parent/'train_config.json'):
            if cand.is_file():
                train_file=cand;break
        else:
            result.update(reason='MISSING_TRAIN_CONFIG',expected=[str(checkpoint/'train_config.json'),str(checkpoint.parent/'train_config.json')]);return result
        tc=read_json(train_file)
        tasks=[t for t in (tc.get('tasks') or [tc.get('task')]) if t]
        cameras=list(tc.get('demo_cameras') or ([tc['demo_camera']] if tc.get('demo_camera') else ['front']))
        data=project_path(project,str(data_override) if data_override else tc['data_dir'],path_vars)
        root=project/'third_party'/'roboreel_models'
        source=root/'src'/'evaluation'/'demo_pi05_runner.py'
        if not source.is_file():
            result.update(reason='MISSING_OFFICIAL_RUNNER',expected=str(source));return result
        rev=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],text=True,capture_output=True)
        revision=rev.stdout.strip() if rev.returncode==0 else 'UNVERIFIED'
        result.update(checkpoint_dir=str(checkpoint),train_config=str(train_file),
                      checkpoint_params_present=(checkpoint/'params').exists(),
                      official_source_revision=revision,pinned_source_revision=PINNED_UPSTREAM,
                      official_runner_path=str(source),official_runner_sha256=digest(source),
                      eval_config_sha256=digest(cfg_path),train_config_sha256=digest(train_file),
                      data_dir=str(data),checkpoint_tasks=tasks,demo_cameras=cameras,
                      demonstration_config=tc.get('demonstration_config'))
        if task not in tasks:
            result.update(reason='CHECKPOINT_NOT_TRAINED_FOR_TASK');return result
        if str(tc.get('demonstration_config','none')).lower() in ('none',''):
            result.update(reason='CHECKPOINT_NOT_VIDEO_CONDITIONED');return result
        if not (checkpoint/'params').exists():
            result.update(reason='MISSING_CHECKPOINT_PARAMS');return result
        items=video_inventory(data,task,cameras)
        ready=[x for x in items if x['video_present'] and x['feature_present']]
        result.update(train_video_demos_with_media=sum(x['video_present'] for x in items),
                      train_feature_demos=sum(x['feature_present'] for x in items),
                      train_demos_usable=len(ready),usable_train_demo_ids=[x['id'] for x in ready],
                      seen_train_assets=items[:20],
                      missing_example={'train_media':str(data/'human_demos'/'train'/'no_distraction'/task/'0000'/'front.mp4'),
                                       'train_feature':str(data/'video_demonstration_features'/'without_distraction'/task/'0000'/'front.npy')})
        if len(ready)<2:
            result.update(reason='NEED_TWO_GENUINE_TRAIN_VIDEOS_AND_FEATURES');return result
        if revision not in (PINNED_UPSTREAM,'UNVERIFIED'):
            # Revision mismatch is explicitly blocked, not silently patched.
            result.update(reason='UNEXPECTED_OFFICIAL_REPO_REVISION');return result
        result.update(status='READY_FOR_BOUNDED_DEV',reason=None)
        return result
    except (ValueError,KeyError,OSError) as exc:
        result.update(reason=f'PREFLIGHT_ERROR: {type(exc).__name__}: {exc}')
        return result


def require_plan(plan:dict, episodes:int, steps:int):
    if plan['status']!='READY_FOR_BOUNDED_DEV':
        raise RuntimeError('DEV_RUN_BLOCKED: '+str(plan.get('reason')))
    if not 1<=episodes<=4:
        raise ValueError('Only 1..4 paired episodes allowed (2..8 rollouts)')
    if not 1<=steps<=800:
        raise ValueError('Only 1..800 physics steps per rollout')
    ids=plan['usable_train_demo_ids']
    if len(ids)<2 or any(not isinstance(i,int) or not 0 <= i < 80 for i in ids):
        raise ValueError('Require >=2 real TRAIN demo ids 0..79, no heldout EVAL')
    return ids[:2]


class AuditedEnv:
    """Records real env.step calls without fabricating success or using GT for model."""
    def __init__(self,base):
        self.base=base
        self.n_steps=0
        self.terminations=0
        self.truncations=0
        self.last_success=False
        self.action_trace=[]
        self.success_trace=[]
    def __getattr__(self,name):
        return getattr(self.base,name)
    def step(self, action):
        import numpy as np
        if hasattr(action, 'detach'):
            a=action.detach().cpu().numpy()
        else:
            a=np.asarray(action)
        a=np.asarray(a,dtype=np.float32).reshape(-1)
        if a.size != 8 or not np.isfinite(a).all():
            raise ValueError('Real Panda policy actions must be finite 8D')
        r=self.base.step(action)
        self.action_trace.append(a.copy())
        self.n_steps+=1
        _,_,term,trunc,info=r
        def one(x):
            try:
                import torch
                if isinstance(x,torch.Tensor):return bool(x.reshape(-1)[0].item())
            except ImportError:
                pass
            try:return bool(np.asarray(x).reshape(-1)[0])
            except Exception:return bool(x)
        self.last_success=one(info.get('success',False))
        self.success_trace.append(self.last_success)
        self.terminations+=int(one(term))
        self.truncations+=int(one(trunc))
        return r


def action_displacement(a, b):
    """Mean aligned 8D action L2; descriptive not proof of useful video."""
    import numpy as np
    if not a or not b:
        return None
    x=np.asarray(a[:min(len(a),len(b))],dtype=np.float32)
    y=np.asarray(b[:min(len(a),len(b))],dtype=np.float32)
    return float(np.linalg.norm(x-y,axis=1).mean())


def inject_upstream(root:Path):
    # The pinned official evaluation.py uses the same set of paths.
    for p in (root,root/'src',root/'src'/'external_models',
              root/'src'/'external_models'/'lerobot'/'src',
              root/'src'/'external_models'/'UniSkill',
              root/'src'/'external_models'/'UniSkill-Policy',
              root/'roboreel_env'/'src'):
        if str(p) not in sys.path:sys.path.insert(0,str(p))


def save_json_once(path:Path,obj):
    payload=json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True)+'\n'
    if path.exists():
        if path.read_text()!=payload:raise FileExistsError('Never overwrite changed DEV evidence: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(payload)


def run_dev(plan:dict,out:Path, episodes:int,steps:int,save_video:bool=True):
    first,other=require_plan(plan,episodes,steps)
    out=out.resolve()
    if out.exists() and any(x.name!='preflight.json' for x in out.iterdir()):
        raise FileExistsError('Never rerun/overwrite existing DEV: '+str(out))
    if (out/'preflight.json').is_file() and read_json(out/'preflight.json')!=plan:
        raise ValueError('PREFLIGHT_INPUTS_CHANGED: preserve older evidence, use new --out')
    out.mkdir(parents=True,exist_ok=True)
    # Freeze the manifest before ANY simulator initialization.
    manifest={'type':'ROBOREEL_TRAIN_VIDEO_DEV_V1', 'task':plan['task'],
              'train_video_ids':[first,other], 'episodes':episodes,
              'max_steps':steps,'output_dir':str(out), 'preflight':plan,
              'no_official_eval_demo':True, 'new_training_runs':0,
              'no_new_model_checkpoint':True,'is_tro_novelty_evidence':False}
    save_json_once(out/'manifest.json',manifest)
    os.environ.pop('DISPLAY',None)
    os.environ.pop('WAYLAND_DISPLAY',None)
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')
    # Reuse the same roots when the official runner loads train_config.json.
    for key, value in plan['resolved_official_path_vars'].items():
        current = os.environ.get(key)
        if current and os.path.expanduser(current) != value:
            raise RuntimeError('OFFICIAL_PATH_VARIABLE_CHANGED_SINCE_PREFLIGHT: ' + key)
        os.environ[key] = value
    official=Path(plan['project'])/'third_party'/'roboreel_models'
    inject_upstream(official)
    import gymnasium as gym  # noqa: F401
    import mani_skill.envs  # noqa: F401
    import envs  # noqa: F401 -- RoboReel task registration
    import jax
    import imageio.v2 as imageio
    import numpy as np
    from evaluation.demo_pi05_runner import DemoPi05Runner
    from evaluation.utils import make_eval_env, load_train_config

    original_cfg=load_train_config(plan['checkpoint_dir'])
    original_cfg['data_dir']=plan['data_dir']
    runner=DemoPi05Runner(plan['checkpoint_dir'],original_cfg)
    if plan['task'] not in runner.tasks:
        raise RuntimeError('OFFICIAL_RUNNER_TASK_MISMATCH')
    sources={i:runner.load_demo_static(False,plan['task'],i) for i in (first,other)}
    if any(not set(REQUIRED_STATIC).issubset(x) for x in sources.values()):
        raise RuntimeError('NO_VIDEO_CONDITIONING_TOKENS')
    if all(np.allclose(sources[first][k],sources[other][k]) for k in ['static_image_emb']):
        raise ValueError('TWO_TRAIN_SOURCE_VIDEOS_HAVE_IDENTICAL_TOKENS')
    # Same env per episode, same reset state, same model RNG. Nothing from
    # privileged env state goes to the policy; it is used only to RESET scenes.
    trials=[]
    for episode in range(episodes):
        seed=310000+episode
        env,_=make_eval_env(plan['task'],False,max_episode_steps=steps)
        try:
            env.reset(seed=seed)
            snapshot=copy.deepcopy(env.unwrapped.get_state_dict())
            # A/B order is deterministic and swapped across scenes to avoid
            # order artifacts. Within a pair, the baseline physics state and
            # all policy stochastic keys are exactly the same.
            order=[first,other] if episode%2==0 else [other,first]
            for order_idx,source_id in enumerate(order):
                sub=out/f'episode_{episode:02d}'/f'source_{source_id:04d}'
                sub.mkdir(parents=True,exist_ok=False)
                env.reset(seed=seed)
                env.unwrapped.set_state_dict(copy.deepcopy(snapshot))
                obs=env.unwrapped.get_obs()
                rng=jax.random.key(560000+episode)
                runner._rng=rng
                frames=[] if save_video else None
                logged=AuditedEnv(env)
                t0=time.monotonic()
                error=None
                success=False
                try:
                    success=bool(runner.rollout(logged,obs,sources[source_id],steps,runner.action_horizon,frames))
                except Exception as exc:
                    error=f'{type(exc).__name__}: {str(exc)[:700]}'
                np.save(sub/'actions.npy',np.asarray(logged.action_trace,dtype=np.float32).reshape(-1,8))
                (sub/'success_trace.json').write_text(json.dumps(logged.success_trace)+'\n')
                video_error=None
                if frames:
                    try:
                        imageio.mimwrite(str(sub/'rollout.mp4'),frames,fps=20,quality=7)
                    except Exception as exc:
                        video_error=f'{type(exc).__name__}: {str(exc)[:500]}'
                tr={'episode':episode, 'reset_seed':seed,'condition_order':order_idx,
                    'source_train_demo_id':source_id, 'source_identity':'TRAIN_HUMAN_VIDEO_FEATURES',
                    'success':success,'steps':logged.n_steps,
                    'terminated_events':logged.terminations,'truncated_events':logged.truncations,
                    'last_success':logged.last_success, 'error':error,
                    'video_error':video_error,
                    'elapsed_seconds':round(time.monotonic()-t0,2),
                    'video_path':str(sub/'rollout.mp4') if (frames and (sub/'rollout.mp4').is_file()) else None,
                    'actions_path':str(sub/'actions.npy'),
                    'actions_sha256':digest(sub/'actions.npy'),
                    'policy_rng_seed':560000+episode,
                    'source_video_paths':[str(Path(plan['data_dir'])/'human_demos'/'train'/'no_distraction'/plan['task']/f'{source_id:04d}'/f'{cam}.mp4') for cam in plan['demo_cameras']],
                    'source_video_sha256':{cam:digest(Path(plan['data_dir'])/'human_demos'/'train'/'no_distraction'/plan['task']/f'{source_id:04d}'/f'{cam}.mp4') for cam in plan['demo_cameras']},
                    'source_static_image_emb_sha256':hashlib.sha256(np.asarray(sources[source_id]['static_image_emb']).tobytes()).hexdigest(),
                    'no_privileged_state_as_model_input':True}
                trials.append(tr)
                save_json_once(sub/'status.json',tr)
                # Partial results preserved after every rollout.
                (out/'per_trial.partial.json').write_text(json.dumps(trials,indent=2,ensure_ascii=False)+'\n')
                print('ROBOREEL_VIDEO_DEV_TRIAL',json.dumps({k:tr[k] for k in ['episode','source_train_demo_id','success','steps','error']}),flush=True)
                if error:
                    raise RuntimeError('REAL_ROLLOUT_ERROR_STOP_HERE: '+error)
        finally:
            env.close()
    if len(trials)!=episodes*2:
        raise RuntimeError('Unexpected incomplete paired data')
    with (out/'per_trial.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['episode','reset_seed','source_train_demo_id','success','steps','error','elapsed_seconds'])
        w.writeheader()
        for t in trials:w.writerow({k:t[k] for k in w.fieldnames})
    pairs=[]
    for episode in range(episodes):
        p=[x for x in trials if x['episode']==episode]
        pairs.append({'episode':episode,'same_reset_seed':p[0]['reset_seed']==p[1]['reset_seed'],
                      'same_rng_seed':p[0]['policy_rng_seed']==p[1]['policy_rng_seed'],
                      'success_disagree':p[0]['success']!=p[1]['success'],
                      'source_ids':[x['source_train_demo_id'] for x in p],
                'mean_action_l2_first_common_steps':action_displacement(
                    np.load(p[0]['actions_path']).tolist(),
                    np.load(p[1]['actions_path']).tolist()),
                'successes':[x['success'] for x in p]})
    result={'type':'ROBOREEL_TRAIN_VIDEO_DEV_V1','status':'DEV_COMPLETE_NOT_FORMAL',
            'task':plan['task'],'paired_episodes':episodes,
            'real_rollouts':len(trials),'new_training_runs':0,
            'any_pair_success_difference':any(x['success_disagree'] for x in pairs),
            'pairs':pairs, 'trials':trials,
            'video_contributes_novel_information_proven':False,
            'effect_stage_sync_method_evaluated':False,
            'fair_strong_baseline_evaluated':False,
            'paper_ready':False}
    save_json_once(out/'results.json',result)
    summary=['# RoboReel official TRAIN-video sensitivity DEV','',
             f'- Task: `{plan["task"]}`; frozen pretrained `demo_pi05`; real PhysX rollouts **{len(trials)}**.',
             f'- Same-env-state + same-policy-RNG paired episodes: **{episodes}**.',
             f'- Different successful outcomes within pair: **{sum(p["success_disagree"] for p in pairs)} / {episodes}**.',
             '- Per-pair mean 8D action L2 in `results.json`: observation-only sensitivity metric, not success guarantee.',
             '- Videos are same-task TRAIN demonstrations; there is NO guaranteed semantic correctness distinction.',
             '- No train/test leakage from EVAL 0080–0099; no training; no model weights changed.',
             '- This is a sensitivity smoke, not causal proof of video necessity or T-RO-level novelty.',
             '- See `results.json`, `per_trial.csv`, and `episode_*/source_*/actions.npy`, `success_trace.json`, `rollout.mp4`.','']
    (out/'SUMMARY.md').write_text('\n'.join(summary))
    print('ROBOREEL_VIDEO_DEV_RESULT',json.dumps({'status':result['status'],
        'task':plan['task'],'pairs':episodes,'rollouts':len(trials),
        'any_pair_success_difference':result['any_pair_success_difference'],
        'report':str(out/'SUMMARY.md')}),flush=True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--config',type=Path,default=None)
    p.add_argument('--task',choices=sorted(ALLOWED_TASKS),default=DEFAULT_TASK)
    p.add_argument('--data-dir',type=Path,default=None)
    p.add_argument('--run',action='store_true',help='Explicit opt-in: 2..8 genuine PhysX rollouts')
    p.add_argument('--episodes',type=int,default=2)
    p.add_argument('--max-steps',type=int,default=800)
    p.add_argument('--no-video',action='store_true',help='DEV without mp4 output')
    p.add_argument('--out',type=Path,default=Path('outputs/roboreel_video_dependency_dev/v1'))
    args=p.parse_args(argv)
    plan=preflight(args.project,args.config,args.task,args.data_dir)
    if not args.run:
        print('ROBOREEL_VIDEO_DEV_PREFLIGHT',json.dumps({k:plan.get(k) for k in (
            'status','task','reason','checkpoint_dir','data_dir','checkpoint_tasks',
            'train_video_demos_with_media','train_feature_demos','train_demos_usable',
            'usable_train_demo_ids','missing_example','eval_config')},ensure_ascii=False),flush=True)
        # Never overwrite existing reporting on a different machine/dataset.
        out=args.out/'preflight.json'
        save_json_once(out,plan)
        return 0
    require_plan(plan,args.episodes,args.max_steps)
    run_dev(plan,args.out,args.episodes,args.max_steps,not args.no_video)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
