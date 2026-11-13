#!/usr/bin/env python3
"""Publish only minimal reproducible TRO1 experiment evidence to a separate GitHub branch.

Project outputs are read-only. Never uploads original MP4, datasets, checkpoints,
SigLIP bundles, logs, credential files, raw full configs, or raw status JSON.
The GitHub CLI (`gh`) supplies authentication locally; no tokens are accessed.
Supports dry-run for offline audit and safe re-publication without new PhysX.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from urllib.parse import quote

REPO = 'LemonQC/cehsisuanfa'
BRANCH = 'tro1-experiment-evidence'
ROOT = 'evidence/runs'
ALLOWED_ROOTS = (
    Path('outputs/roboreel_video_dependency_dev'),
    Path('outputs/roboreel_video_dependency_diagnosis'),
    Path('outputs/g0_goal_pair_audit'),
    Path('outputs/research_runs'),
)
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_PREVIEW_BYTES = 850_000  # GitHub Contents GET embeds base64 for files <= 1 MiB
SAFE_KEY_FIELDS = ('episode', 'source_train_demo_id', 'success', 'steps',
                   'last_success', 'terminated_events', 'truncated_events',
                   'reset_seed', 'policy_rng_seed', 'condition_order')
PAIR_KEYS = ('episode', 'same_reset_seed', 'same_rng_seed', 'success_disagree',
             'source_ids', 'successes', 'mean_action_l2_first_common_steps')


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def bounded_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    if path.is_symlink() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('UNSAFE_OR_OVERSIZED_EVIDENCE_JSON:' + path.name)
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, (list, dict)):
        raise ValueError('NOT_A_JSON_OBJECT_OR_LIST:' + path.name)
    return result


def project_source(project: Path, source: Path) -> tuple[Path, str]:
    project = project.expanduser().resolve()
    source = (project / source).resolve() if not source.is_absolute() else source.resolve()
    if not source.is_dir():
        raise FileNotFoundError('SOURCE_EXPERIMENT_DIR_MISSING')
    rel = source.relative_to(project)
    if not any(rel.is_relative_to(root) and rel != root for root in ALLOWED_ROOTS):
        raise ValueError('SOURCE_NOT_IN_ALLOWLISTED_EXPERIMENT_OUTPUTS')
    if any(p.is_symlink() for p in [source, *list(source.parents)[:3]]):
        raise ValueError('SYMLINK_SOURCE_NOT_ALLOWED')
    return source, rel.as_posix()


def scalar(x: Any) -> Any:
    if isinstance(x, bool):
        return x
    if isinstance(x, int) and abs(x) < 10**12:
        return x
    if isinstance(x, float) and math.isfinite(x) and abs(x) < 10**12:
        return round(x, 9)
    return None


def safe_trial(t: dict) -> dict:
    if not isinstance(t, dict):
        raise ValueError('INVALID_TRIAL_RECORD')
    out = {k:scalar(t[k]) for k in SAFE_KEY_FIELDS if k in t}
    # Avoid serializing uncontrolled exception strings, URLs, paths or tokens.
    for key in ('error', 'video_error'):
        err = t.get(key)
        typ = str(err).split(':', 1)[0] if isinstance(err, str) else ''
        out[key + '_type'] = (typ if re.fullmatch(r'[A-Z][A-Za-z]{2,40}(?:Error|Exception|Warning)', typ)
                             else ('NONE' if err is None else 'REDACTED'))
    return out


def safe_pair(p: dict) -> dict:
    if not isinstance(p, dict):
        raise ValueError('INVALID_PAIR_RECORD')
    out = {}
    for k in PAIR_KEYS:
        if k not in p:
            continue
        value = p[k]
        if isinstance(value, (list, tuple)) and len(value) <= 8:
            out[k] = [scalar(x) for x in value]
        else:
            out[k] = scalar(value)
    return out


def read_result(source: Path, relative_source: str, status_file: Path | None) -> dict:
    result_path = source / 'results.json'
    partial_path = source / 'per_trial.partial.json'
    result = bounded_json(result_path)
    partial = bounded_json(partial_path)
    if result is not None and not isinstance(result, dict):
        raise ValueError('RESULTS_MUST_BE_JSON_OBJECT')
    # Legacy audits used a bare list; B06's append-only writer saves
    # {"trials": [...]} so it can atomically record progress. Accept both,
    # but never silently treat an unknown schema as an empty experiment.
    if isinstance(partial, dict):
        if set(partial) != {'trials'} or not isinstance(partial['trials'], list):
            raise ValueError('PARTIAL_TRIALS_INVALID_OBJECT')
        partial = partial['trials']
    elif partial is not None and not isinstance(partial, list):
        raise ValueError('PARTIAL_TRIALS_MUST_BE_ARRAY_OR_TRIALS_OBJECT')
    origin = result or {}
    diagnosis = bounded_json(source / 'diagnosis.json')
    if diagnosis is not None and not isinstance(diagnosis, dict):
        raise ValueError('DIAGNOSIS_MUST_BE_OBJECT')
    horizon = bounded_json(source / 'horizon_audit.json')
    if horizon is not None and (not isinstance(horizon, dict) or
            horizon.get('status') != 'READ_ONLY_DRAWER_HORIZON_AUDIT' or
            horizon.get('task') != 'toys_in_drawer' or
            horizon.get('default_registered_limit') != 400 or
            horizon.get('experimental_limit') != 800 or
            horizon.get('rollout_count') != 4):
        raise ValueError('INVALID_FROZEN_DRAWER_HORIZON_AUDIT')
    trials = origin.get('trials') if isinstance(origin.get('trials'), list) else partial
    trials = trials if isinstance(trials, list) else []
    pairs = origin.get('pairs') if isinstance(origin.get('pairs'), list) else []
    task = str(origin.get('task', (diagnosis or horizon or {}).get('task','UNKNOWN')))
    if task not in ('toys_in_drawer', 'stack_cups', 'push_cube'):
        task = 'UNKNOWN'
    pipeline = None
    if status_file is not None:
        pipeline = bounded_json(status_file)
        if not isinstance(pipeline, dict):
            raise ValueError('PIPELINE_STATUS_MUST_BE_JSON_OBJECT')
        pipeline = {
          'phase': pipeline.get('phase') if pipeline.get('phase') in (
              'started','asset_inspect','scene_smoke','preflight',
              'preflight_snapshot','real_rollouts','complete') else 'UNKNOWN',
          'exit_code': scalar(pipeline.get('exit_code')),
          'stage_result': pipeline.get('stage_result') if pipeline.get('stage_result') in (
              'SUCCESS','FAILED') else 'UNKNOWN',
        }
    status = str(origin.get('status',
                 diagnosis.get('status') if isinstance(diagnosis, dict) else (
                 'PARTIAL' if trials else 'NO_RESULTS')))
    if status not in ('DEV_COMPLETE_NOT_FORMAL', 'READ_ONLY_DIAGNOSIS_NOT_NEW_ROLLOUT',
                      'PARTIAL', 'NO_RESULTS', 'READY_FOR_BOUNDED_DEV'):
        status = 'UNKNOWN'
    audit_files = sorted(source.glob('g0_audit_*.json'))
    g0 = None
    if audit_files and not result:
        # never publish raw corpus metadata, paths, or license assertions
        x = bounded_json(audit_files[-1])
        if isinstance(x, dict):
            a = x.get('human_goal_pair_audit') or {}
            gd=x.get('decision'); gs=a.get('status') if isinstance(a,dict) else None
            g0 = {'decision': gd if gd in ('G0_NOT_PASSED','G0_MANUAL_REVIEW_PENDING') else 'UNKNOWN',
                  'human_goal_pair_status':gs if gs in ('G0_BLOCKED','G0_CANDIDATE_REQUIRES_MANUAL_VISUAL_REVIEW') else 'UNKNOWN'}
    sources = {}
    for item, path in (('results', result_path), ('partial', partial_path),
                       ('diagnosis', source / 'diagnosis.json'),
                       ('horizon', source / 'horizon_audit.json')):
        if path.is_file():
            sources[item + '_sha256'] = sha256(path.read_bytes())
    diag_compact = None
    if isinstance(diagnosis, dict):
        diag_compact = {'successes':scalar(diagnosis.get('successes')),
                        'pairs':[]}
        for pair in (diagnosis.get('pairs') or [])[:10]:
            if not isinstance(pair, dict):
                continue
            prefix = pair.get('prefix_metrics') or {}
            if not isinstance(prefix, dict):
                continue
            diag_compact['pairs'].append({
                'episode':scalar(pair.get('episode')),
                'first_action_8d_l2':scalar((prefix.get('1') or {}).get('first_8d_action_l2')),
                'first20_mean_8d_l2':scalar((prefix.get('20') or {}).get('mean_8d_action_l2')),
                'full_mean_8d_l2':scalar(pair.get('mean_8d_action_l2_original'))})
    if g0 and not result and not diagnosis and not horizon:
        status = 'G0_NOT_PASSED' if g0['decision']=='G0_NOT_PASSED' else 'G0_REVIEW_REQUIRED'
    horizon_compact = None
    if horizon is not None:
        source_trials = horizon.get('trials')
        if not isinstance(source_trials, list) or len(source_trials) != 4:
            raise ValueError('HORIZON_TRIALS_NOT_FOUR')
        filtered = []
        seen = set()
        for t in source_trials:
            if not isinstance(t,dict):
                raise ValueError('INVALID_HORIZON_TRIAL')
            ep,src = t.get('episode'),t.get('source_train_demo_id')
            by400,by800 = t.get('success_by_400'),t.get('success_by_800')
            if (type(ep) is not int or ep not in (0,1) or
                type(src) is not int or src not in (0,1) or
                (ep,src) in seen or type(by400) is not bool or type(by800) is not bool):
                raise ValueError('INVALID_HORIZON_PAIR_OR_BOOLEAN')
            seen.add((ep,src))
            steps=t.get('steps')
            first=t.get('first_success_step')
            if type(steps) is not int or not 1<=steps<=800 or (
                    first is not None and (type(first) is not int or not 1<=first<=steps)):
                raise ValueError('INVALID_HORIZON_STEPS')
            if by400 != (first is not None and first<=400) or by800!=(first is not None):
                raise ValueError('INCONSISTENT_HORIZON_METRICS')
            filtered.append({'episode':ep,'source_train_demo_id':src,
                             'steps':steps,'first_success_step':first,
                             'success_by_400':by400,'success_by_800':by800})
        count400=sum(t['success_by_400'] for t in filtered)
        count800=sum(t['success_by_800'] for t in filtered)
        if horizon.get('success_by_400')!=count400 or horizon.get('success_by_800')!=count800:
            raise ValueError('INCONSISTENT_HORIZON_COUNTS')
        horizon_compact={'default_registered_limit':400,'experimental_limit':800,
                         'success_by_400':count400,'success_by_800':count800,
                         'trials':sorted(filtered,key=lambda x:(x['episode'],x['source_train_demo_id'])),
                         'not_official_heldout_eval':True}
        status='READ_ONLY_DRAWER_HORIZON_AUDIT'
    out = {
        'schema': 'TRO1_RESEARCH_EVIDENCE_V1',
        'source_relative': relative_source,
        'status':status,
        'task':task,
        'physical_rollouts_reported':scalar(origin.get('real_rollouts', origin.get('rollouts'))),
        'paired_episodes_reported':scalar(origin.get('paired_episodes', origin.get('episodes'))),
        'any_pair_success_difference': scalar(origin.get('any_pair_success_difference')),
        'paper_ready': False,
        'g0_audit':g0,
        'diagnosis':diag_compact,
        'horizon_audit':horizon_compact,
        'pipeline':pipeline,
        'source_file_hashes':sources,
        'pairs':[safe_pair(x) for x in pairs[:10]],
        'trials':[safe_trial(x) for x in trials[:24]],
    }
    if len(trials)>24 or len(pairs)>10:
        out['truncated_entries'] = {'trials':len(trials),'pairs':len(pairs)}
    return out


def _mp4_files(source: Path) -> list[tuple[str, Path]]:
    # ONLY generated robot-policy videos, never TRAIN/EVAL demos.
    found=[]
    for p in sorted(source.glob('episode_*/source_*/rollout.mp4')):
        if p.is_symlink() or not p.is_file():
            continue
        if len(found)>=8:
            break
        found.append(('/'.join(p.relative_to(source).parts[:2]), p))
    return found


def robot_preview(source: Path, outfile: Path) -> bytes | None:
    """Create a small contact sheet using only ROBOT rollout videos (not human source).

    Failure to decode a video never blocks uploading the JSON metrics.
    """
    videos=_mp4_files(source)
    existing = source / 'rollout_montage.jpg'
    if not videos and existing.is_file() and not existing.is_symlink() and 0 < existing.stat().st_size <= 12*1024*1024:
        try:
            if existing.stat().st_size <= MAX_PREVIEW_BYTES:
                return existing.read_bytes()  # generated montage, still output-only
            from PIL import Image
            with Image.open(existing) as original:
                preview = original.convert('RGB')
                preview.thumbnail((1600,900))
                buf=io.BytesIO()
                preview.save(buf,format='JPEG',quality=56,optimize=True)
                if len(buf.getvalue()) <= MAX_PREVIEW_BYTES:
                    return buf.getvalue()
        except Exception:
            pass
    if not videos:
        return None
    try:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
        total_cols=6
        tile_w, tile_h=160, 120
        label_height=18
        canvas=Image.new('RGB', (total_cols*tile_w, len(videos)*(tile_h+label_height)), (238,238,238))
        draw=ImageDraw.Draw(canvas)
        for row,(label,video) in enumerate(videos):
            # Decode only sampled frames, without extracting MP4s to disk.
            with imageio.get_reader(str(video),'ffmpeg') as reader:
                n=reader.count_frames()
                if not isinstance(n,int) or n < 1:
                    continue
                positions=[min(n-1, round(i*(n-1)/(total_cols-1))) for i in range(total_cols)]
                for col, idx in enumerate(positions):
                    f=reader.get_data(idx)
                    pic=Image.fromarray(f).convert('RGB')
                    pic.thumbnail((tile_w,tile_h))
                    ox=col*tile_w+(tile_w-pic.width)//2
                    oy=row*(tile_h+label_height)+label_height+(tile_h-pic.height)//2
                    canvas.paste(pic,(ox,oy))
                    draw.text((col*tile_w+3,row*(tile_h+label_height)+2),f'{label} step~{idx}',fill=(0,0,0))
        buf=io.BytesIO()
        canvas.save(buf,format='JPEG',quality=64,optimize=True)
        data=buf.getvalue()
        if len(data)>MAX_PREVIEW_BYTES:
            return None
        outfile.parent.mkdir(parents=True,exist_ok=True)
        outfile.write_bytes(data)
        return data
    except Exception:
        return None


def render_readme(obj: dict, preview: bool) -> str:
    rows=[]
    for t in obj['trials']:
        rows.append(f"| {t.get('episode','?')} | {t.get('source_train_demo_id','?')} | {t.get('steps','?')} | {t.get('success','?')} | {t.get('error_type','?')} |")
    pairs=[]
    for p in obj['pairs']:
        pairs.append(f"| {p.get('episode','?')} | {p.get('successes','?')} | {p.get('mean_action_l2_first_common_steps','?')} |")
    lines=[
        '# TRO1 automated, sanitized experiment evidence', '',
        f"- Task: `{obj['task']}`",
        f"- Status: `{obj['status']}` (may be DEV; not paper-ready)",
        f"- Source in project: `{obj['source_relative']}`",
        f"- Reported PhysX rollouts: `{obj['physical_rollouts_reported']}`",
        f"- Paired episodes: `{obj['paired_episodes_reported']}`",
        f"- Any pair differs in success: `{obj['any_pair_success_difference']}`",
        '- These are *copied numeric findings*, not new runs; raw video, model and gated datasets are excluded.',
        '', '## Per-trial results', '',
        '| Episode | TRAIN demo ID | Steps | Success | Runtime error type |',
        '|---:|---:|---:|---|---|', *(rows or ['| — | — | — | — | — |']),
        '', '## Matched pair action sensitivity', '',
        '| Episode | Success outcomes | Full mean 8D action L2 |',
        '|---:|---|---:|',*(pairs or ['| — | — | — |']),
        '', '## Artifact verification', '',
        '- `evidence.json` contains source SHA256 hashes and the detailed sanitized metrics.',
        '- All uploads are snapshots: existing changed evidence files will never be overwritten.',
        '- This branch is dedicated to measurements; the repository `main` branch is unaffected.',
    ]
    if obj.get('diagnosis'):
        lines += ['', '## Read-only action-difference diagnosis', '']
        for d in obj['diagnosis']['pairs']:
            lines.append(f"- Episode {d['episode']}: first action L2={d['first_action_8d_l2']}, first20 mean={d['first20_mean_8d_l2']}, full mean={d['full_mean_8d_l2']}")
    if obj.get('pipeline'):
        lines += ['',f"- Pipeline termination: `{obj['pipeline']}`"]
    if obj.get('horizon_audit'):
        h=obj['horizon_audit']
        lines += ['', '## Read-only 400-step vs 800-step success audit', '',
                  f"- Recorded success within first 400 steps: **{h['success_by_400']}/4**",
                  f"- Recorded success within first 800 steps: **{h['success_by_800']}/4**",
                  '- Prefix audit of frozen DEV trajectories; NOT official heldout 400-step evaluation.',
                  '',
                  '| Episode | TRAIN source | Steps | First success step | By 400 | By 800 |',
                  '|---:|---:|---:|---:|---|---|']
        for trial in h['trials']:
            lines.append(f"| {trial['episode']} | {trial['source_train_demo_id']} | "
                         f"{trial['steps']} | {trial['first_success_step'] or '—'} | "
                         f"{trial['success_by_400']} | {trial['success_by_800']} |")
    if preview:
        lines += ['', '## ROBOT policy rollout preview (not original human demonstrations)', '',
                  '![Robot execution contact sheet](robot_rollout_preview.jpg)']
    return '\n'.join(lines)+'\n'


def _run_gh(args: list[str], *, payload: dict | None = None) -> tuple[bool, dict | None]:
    command=['gh','api','-H','Accept: application/vnd.github+json', *args]
    try:
        cp=subprocess.run(command,input=json.dumps(payload) if payload is not None else None,
                          capture_output=True,text=True,timeout=70,check=False)
    except (OSError,subprocess.TimeoutExpired) as e:
        raise RuntimeError('GITHUB_CLI_UNAVAILABLE_OR_TIMEOUT:'+type(e).__name__) from None
    if cp.returncode != 0:
        # Don't print gh stderr/stdout, which may include server-returned details.
        if 'HTTP 404' in cp.stderr:
            return False,None
        raise RuntimeError('GITHUB_API_REQUEST_FAILED_EXIT_CODE_'+str(cp.returncode))
    try:
        return True,json.loads(cp.stdout) if cp.stdout.strip() else None
    except json.JSONDecodeError:
        raise RuntimeError('GITHUB_API_RESPONSE_NOT_JSON') from None


def _gh_path(path: str) -> str:
    return f'repos/{REPO}/contents/'+quote(path,safe='/')


def write_immutable_report(relative_path: str, data: bytes) -> str:
    route=_gh_path(relative_path)
    found,old=_run_gh(['-X','GET',route,'-f',f'ref={BRANCH}'])
    if found:
        if not isinstance(old,dict) or old.get('type')!='file':
            raise FileExistsError('EXISTING_REMOTE_IS_NOT_FILE')
        original=base64.b64decode((old.get('content') or '').replace('\n',''))
        if original != data:
            raise FileExistsError('IMMUTABLE_REPORT_PATH_HAS_DIFFERENT_BYTES')
        return 'EXISTS_SAME_BYTES'
    payload={'message':'evidence: '+relative_path.split('/')[-2]+' '+relative_path.split('/')[-1],
             'branch':BRANCH,'content':base64.b64encode(data).decode('ascii')}
    ok,_=_run_gh(['-X','PUT',route,'--input','-'],payload=payload)
    if not ok:
        raise RuntimeError('GITHUB_REPORT_UPLOAD_FAILED')
    return 'PUBLISHED'


def assert_evidence_branch_exists():
    ok,res=_run_gh([f'repos/{REPO}/branches/{BRANCH}'])
    if not ok or not res:
        raise RuntimeError('DEDICATED_EVIDENCE_BRANCH_NOT_FOUND')


def build_evidence(project:Path,source:Path,status_file:Path|None,with_preview:bool=True):
    source,relative=project_source(project,source)
    if status_file is not None:
        s=status_file.resolve()
        allow=(project.resolve()/'outputs/experiment_publisher').resolve()
        if not s.is_relative_to(allow) or s.is_symlink():
            raise ValueError('STATUS_FILE_MUST_BE_INSIDE_EXPERIMENT_PUBLISHER')
        status_file=s
    obj=read_result(source,relative,status_file)
    raw=json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode('utf-8')+b'\n'
    run_name=re.sub('[^a-zA-Z0-9_-]','_',source.name)[:40]
    uid=sha256(raw)[:16]
    destination=f'{ROOT}/{run_name}_{uid}'
    preview_data=robot_preview(source,project/'outputs/experiment_publisher'/'preview_cache'/f'{run_name}_{uid}.jpg') if with_preview else None
    files={
        'evidence.json':raw,
        'README.md':render_readme(obj,preview_data is not None).encode('utf-8'),
    }
    if preview_data:
        files['robot_rollout_preview.jpg']=preview_data
    return destination,files,obj


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source',type=Path,required=True,help='Existing completed experiment under an allowed outputs/ root')
    parser.add_argument('--pipeline-status',type=Path,default=None)
    parser.add_argument('--no-preview',action='store_true')
    parser.add_argument('--dry-run',action='store_true',help='Create local sanitized report only; never use GitHub network')
    args=parser.parse_args(argv)
    project=args.project.resolve()
    try:
        destination,files,obj=build_evidence(project,args.source,args.pipeline_status,not args.no_preview)
        local=project/'outputs/experiment_publisher'/'snapshots'/destination.removeprefix(ROOT+'/')
        local.mkdir(parents=True,exist_ok=True)
        for name,data in files.items():
            target=local/name
            if target.exists() and target.read_bytes()!=data:
                raise FileExistsError('LOCAL_EVIDENCE_SNAPSHOT_CHANGED')
            if not target.exists():
                target.write_bytes(data)
        if args.dry_run:
            print('TRO1_EVIDENCE_DRY_RUN',json.dumps({'snapshot':str(local),'status':obj['status'],
                'file_names':list(files),'branch':BRANCH,'remote_dir':destination},ensure_ascii=False),flush=True)
            return 0
        assert_evidence_branch_exists()
        upload={name:write_immutable_report(destination+'/'+name,data) for name,data in files.items()}
        url=f'https://github.com/{REPO}/tree/{BRANCH}/{destination}'
        print('TRO1_EVIDENCE_PUBLISHED',json.dumps({'status':obj['status'],'branch':BRANCH,
              'remote_dir':destination,'url':url,'files':upload},ensure_ascii=False),flush=True)
        return 0
    except Exception as e:
        # No raw exceptions, paths, secrets, JSON payloads or auth data printed.
        msg=str(e)
        code=(msg.split(':',1)[0] if re.fullmatch('[A-Z0-9_]{4,80}(:.*)?',msg) else type(e).__name__)
        print('TRO1_EVIDENCE_BLOCKED',json.dumps({'reason':code,'new_physx_actions':0,
              'raw_datasets_or_tokens_uploaded':False}),flush=True)
        return 3


if __name__=='__main__':
    raise SystemExit(main())
