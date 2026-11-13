#!/usr/bin/env python3
"""Frozen ObjectInDrawer: read-only 400 vs 800-step success-trace audit.

This script does not import a simulator, JAX, pretrained model or any network
client. It reads four existing authenticated robot rollout records and records
exact first-success step from original per-step `success_trace.json`. It does
NOT run a new experiment or claim formal 400-step benchmark accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TASK = 'toys_in_drawer'
HORIZON = 400
EXTENDED = 800
MAX_JSON_BYTES = 4 * 1024 * 1024


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_json_file(p: Path, expected_type):
    if p.is_symlink() or not p.is_file() or p.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('ABSENT_UNSAFE_OR_OVERSIZED_JSON:' + p.name)
    obj = json.loads(p.read_text(encoding='utf-8'))
    if not isinstance(obj, expected_type):
        raise ValueError('INVALID_JSON_RECORD_TYPE:' + p.name)
    return obj


def audit(root: Path) -> dict:
    root = root.resolve()
    records = read_json_file(root/'results.json', dict)
    if records.get('status') != 'DEV_COMPLETE_NOT_FORMAL' or records.get('task') != TASK:
        raise ValueError('NOT_FROZEN_DRAWER_DEV_V1')
    if records.get('paired_episodes') != 2 or records.get('real_rollouts') != 4:
        raise ValueError('EXPECTED_TWO_SCENES_FOUR_PHYSX_ROLLOUTS')
    trials = records.get('trials')
    if not isinstance(trials, list) or len(trials) != 4:
        raise ValueError('EXPECTED_FOUR_TRIAL_RECORDS')
    seen, output = set(), []
    for t in trials:
        if not isinstance(t, dict):
            raise ValueError('INVALID_TRIAL')
        episode, source = t.get('episode'), t.get('source_train_demo_id')
        if type(episode) is not int or episode not in (0, 1) or type(source) is not int or source not in (0, 1):
            raise ValueError('UNEXPECTED_EPISODE_OR_SOURCE')
        if (episode, source) in seen:
            raise ValueError('DUPLICATED_EPISODE_SOURCE')
        seen.add((episode, source))
        steps = t.get('steps')
        if type(steps) is not int or not (1 <= steps <= EXTENDED):
            raise ValueError('UNEXPECTED_TRIAL_LENGTH')
        folder = root/f'episode_{episode:02d}'/f'source_{source:04d}'
        # Bind trial to real frozen actions, not an accidentally substituted trace.
        action = folder/'actions.npy'
        if action.is_symlink() or not action.is_file() or sha256_file(action) != t.get('actions_sha256'):
            raise ValueError('FROZEN_ACTION_HASH_MISMATCH')
        expected_action = Path(str(t.get('actions_path',''))).resolve()
        if expected_action != action.resolve():
            raise ValueError('ACTION_PATH_NOT_EQUAL_ORIGINAL')
        trace_path = folder/'success_trace.json'
        trace = read_json_file(trace_path, list)
        if len(trace) != steps or not all(type(x) is bool for x in trace):
            raise ValueError('NON_BOOLEAN_OR_WRONG_LENGTH_SUCCESS_TRACE')
        if type(t.get('last_success')) is not bool or t['last_success'] != trace[-1]:
            raise ValueError('LAST_SUCCESS_TRACE_CONTRADICTION')
        if type(t.get('success')) is not bool:
            raise ValueError('INVALID_ORIGINAL_SUCCESS_FLAG')
        # A True in the trace is proof of a successful env.info step, not a claim
        # about why it succeeded or formal protocol success.
        indices = [i+1 for i,x in enumerate(trace) if x]
        if t['success'] and not indices:
            raise ValueError('SUCCESS_CLAIM_WITHOUT_ANY_ENV_SUCCESS')
        if (not t['success']) and t.get('terminated_events') and indices:
            raise ValueError('TERMINATED_TRUE_WITH_FALSE_RESULT')
        output.append({
            'episode':episode,'source_train_demo_id':source,'steps':steps,
            'first_success_step':indices[0] if indices else None,
            'success_by_400':any(trace[:HORIZON]),
            'success_by_800':bool(indices),
            'end_success':trace[-1],
            'reported_run_success':t['success'],
            'success_trace_sha256':sha256_file(trace_path),
            'actions_sha256':sha256_file(action),
        })
    if seen != {(e,s) for e in (0,1) for s in (0,1)}:
        raise ValueError('MISSING_PAIRED_SOURCE')
    for ep in (0,1):
        x=[t for t in trials if t['episode']==ep]
        if len({t.get('reset_seed') for t in x}) != 1 or len({t.get('policy_rng_seed') for t in x}) != 1:
            raise ValueError('UNMATCHED_PAIRED_SCENE_OR_POLICY_RNG')
    output.sort(key=lambda t:(t['episode'],t['source_train_demo_id']))
    return {
        'status':'READ_ONLY_DRAWER_HORIZON_AUDIT','task':TASK,
        'episode_count':2,'rollout_count':4,
        'default_registered_limit':HORIZON,'experimental_limit':EXTENDED,
        'success_by_400':sum(t['success_by_400'] for t in output),
        'success_by_800':sum(t['success_by_800'] for t in output),
        'run_success_reported':sum(t['reported_run_success'] for t in output),
        'rollouts_with_no_success_by_400':sum(not t['success_by_400'] for t in output),
        'paired_success_reversals':sum(
            len({t['reported_run_success'] for t in output if t['episode']==ep})>1
            for ep in (0,1)),
        'trials':output,
        'original_results_sha256':sha256_file(root/'results.json'),
        'scientific_decision':'NOT_TRO_NOVELTY_EVIDENCE',
        'limitation':'400-step prefix is an offline metric on 800-step DEV; NOT a fresh official evaluation on heldout states/demos.',
        'new_physx_rollouts':0,'new_training_runs':0,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/roboreel_video_dependency_dev/drawer_hf_v1'))
    p.add_argument('--out',type=Path,default=Path('outputs/roboreel_video_dependency_diagnosis/drawer_horizon_v1'))
    args=p.parse_args(argv)
    root,out=args.root.resolve(),args.out.resolve()
    if root==out or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('OUTPUT_NOT_SEPARATE_FROM_SOURCE')
    if out.exists():
        raise FileExistsError('DO_NOT_OVERWRITE_FROZEN_HORIZON_REPORT')
    result=audit(root)
    out.mkdir(parents=True,exist_ok=False)
    (out/'horizon_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf-8')
    lines=['# Read-only ObjectInDrawer 400/800-step horizon audit','',
           f'- Four PhysX traces already frozen: {result["rollout_count"]}',
           f'- Actual successes within first 400 steps: **{result["success_by_400"]}/4**',
           f'- Actual successes within first 800 steps: **{result["success_by_800"]}/4**',
           '- This is a 400-step *prefix* of a DEV rollout, not an official heldout 400-step eval.',
           '- No physical rollout, inference, training or simulator import was performed.', '',
           '| Episode | TRAIN source | Steps executed | First success step | Success by 400 | Success by 800 |',
           '|---:|---:|---:|---:|---|---|']
    for t in result['trials']:
        lines.append(f"| {t['episode']} | {t['source_train_demo_id']} | {t['steps']} | "
                     f"{t['first_success_step'] or '—'} | {t['success_by_400']} | {t['success_by_800']} |")
    lines+=['','- Source hashes are in `horizon_audit.json`; original files untouched.','']
    (out/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    print('TRO1_DRAWER_HORIZON_AUDIT',json.dumps({
        'status':result['status'],'success_by_400':result['success_by_400'],
        'success_by_800':result['success_by_800'],
        'first_success_steps':[t['first_success_step'] for t in result['trials']],
        'output':str(out),'new_physx_rollouts':0},ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
