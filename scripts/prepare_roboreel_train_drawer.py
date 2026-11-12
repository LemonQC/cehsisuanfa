#!/usr/bin/env python3
"""Prepare TWO genuine RoboReel toys_in_drawer human TRAIN videos + official SigLIP tokens.

Default = local-only inventory; --download = bounded HF remote metadata + two MP4s;
--features = genuine, pinned official SigLIP encoding in isolated subprocesses.
No EVAL IDs, no simulator, no policy model download, no new rollouts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
DATASET = 'LogosRoboticsLab/roboreel'
TASK = 'toys_in_drawer'
ALLOWED_ENDPOINTS = {'https://hf-mirror.com', 'https://huggingface.co'}
REMOTE = re.compile(r'^human_demos/train/no_distraction/toys_in_drawer/(\d{4})/front\.mp4$')
MAX_FILES = 2
MAX_BYTES = 300 * 1024 * 1024  # hard budget for *both* human videos


def media_rel(demo_id: int) -> str:
    if type(demo_id) is not int or demo_id not in range(80):
        raise ValueError('Only genuine RoboReel TRAIN IDs 0000..0079 allowed')
    return f'human_demos/train/no_distraction/{TASK}/{demo_id:04d}/front.mp4'


def feature_rel(demo_id: int) -> str:
    media_rel(demo_id)  # enforce the same split
    return f'video_demonstration_features/without_distraction/{TASK}/{demo_id:04d}/front.npy'


def select_two(rows: list[dict], budget: int = MAX_BYTES) -> dict:
    """Select two lowest TRAIN demo IDs with exact upstream front.mp4 metadata."""
    if budget <= 0 or budget > MAX_BYTES:
        raise ValueError('Unbounded or invalid total download budget')
    mapping = {}
    for row in rows:
        path = str(row.get('path', ''))
        match = REMOTE.fullmatch(path)
        if not match:
            continue
        demo_id = int(match.group(1))
        if demo_id not in range(80):
            continue
        size = row.get('bytes')
        if type(size) is not int or size <= 1024:
            raise ValueError(f'Upstream TRAIN video size unavailable/invalid: {path}')
        if demo_id in mapping and mapping[demo_id]['bytes'] != size:
            raise ValueError('Conflicting upstream metadata for demo ' + str(demo_id))
        mapping[demo_id] = {'id': demo_id, 'path': path, 'bytes': size}
    choices = [mapping[i] for i in sorted(mapping)[:MAX_FILES]]
    total = sum(row['bytes'] for row in choices)
    if len(choices) != MAX_FILES:
        raise RuntimeError('NEED_TWO_OFFICIAL_TRAIN_FRONT_VIDEOS_IN_METADATA')
    if total > budget:
        raise RuntimeError(f'REMOTE_TWO_VIDEO_SIZE_EXCEEDS_BUDGET: {total} bytes > {budget}')
    return {'type': 'ROBOREEL_TRAIN_DRAWER_TWO_VIDEO_SELECTION', 'dataset': DATASET,
            'task': TASK, 'file_count': 2, 'video_files': choices,
            'total_download_bytes': total, 'budget_bytes': budget,
            'no_eval_data_or_robot_actions': True, 'download_executed': False}


def list_remote(endpoint: str) -> tuple[list[dict], str | None]:
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError('Only Hugging Face official/mirror endpoints approved')
    # Must set BEFORE importing huggingface_hub.
    os.environ['HF_ENDPOINT'] = endpoint
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '15')
    from huggingface_hub import HfApi
    api = HfApi(endpoint=endpoint)
    dirname = f'human_demos/train/no_distraction/{TASK}'
    rows = []
    for obj in api.list_repo_tree(DATASET, repo_type='dataset',
                                  path_in_repo=dirname, recursive=True):
        name = obj.__class__.__name__.lower()
        if 'file' in name:
            path = str(obj.path)
            match = REMOTE.fullmatch(path)
            if match and int(match.group(1)) < 80:
                rows.append({'path': path, 'bytes': int(getattr(obj, 'size', 0) or 0)})
        if len(rows) > 500:
            raise RuntimeError('Too many metadata entries; refuse unexpected layout')
    return rows, endpoint


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def video_decodes(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 1024:
        return False
    with path.open('rb') as f:
        if b'ftyp' not in f.read(24):
            return False
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return False
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count < 2:
                return False
            first, _ = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
            last, _ = cap.read()
            return bool(first and last)
        finally:
            cap.release()
    except ImportError as exc:
        raise RuntimeError('cv2 required to validate genuine official MP4') from exc


def existing_local(data: Path) -> list[dict]:
    result = []
    for demo_id in range(80):
        v = data / media_rel(demo_id)
        f = data / feature_rel(demo_id)
        if v.is_file() or f.is_file():
            result.append({'id': demo_id,
                           'video_bytes': v.stat().st_size if v.is_file() else None,
                           'feature_bytes': f.stat().st_size if f.is_file() else None})
    return result


def cache_home(project: Path) -> Path | None:
    """Only select locally validated SigLIP pickle or complete cached Orbax mirror.

    Never ask OpenPI to download the ~12GB base parameters automatically.
    """
    from scripts.check_pi05_siglip_cache import trusted_mirror_install
    candidates = []
    for p in (os.environ.get('OPENPI_DATA_HOME'),
              str(project / 'outputs/v120_baselines/openpi_cache'),
              str(Path('~/.cache/openpi').expanduser())):
        if p:
            resolved = Path(p).expanduser().resolve()
            if resolved not in candidates:
                candidates.append(resolved)
    for home in candidates:
        pickle = home / 'pi05_vision_encoder/siglip_params.pkl'
        if pickle.is_file() and pickle.stat().st_size > 1024:
            return home
        if trusted_mirror_install(project, home):
            return home
    return None


def assert_video_sources(selection: dict, data: Path) -> None:
    for row in selection['video_files']:
        path = data / row['path']
        if not path.is_file() or path.stat().st_size != row['bytes']:
            raise RuntimeError('MISSING_OR_WRONG_SIZE_DOWNLOADED_TRAIN_MEDIA: ' + str(path))
        if not video_decodes(path):
            raise RuntimeError('INVALID_OFFICIAL_TRAIN_MP4: ' + str(path))


def download_two(selection: dict, data: Path, endpoint: str) -> list[dict]:
    from agcofl.resumable_hf_assets import get_file
    # Transport's allowlist is intentionally extended for *this* exact
    # two-file, train-only task; no EVAL video can enter this selection.
    info = []
    for row in selection['video_files']:
        item = {'path': row['path'], 'bytes': row['bytes']}
        dst = data / row['path']
        if dst.exists() and dst.stat().st_size != row['bytes']:
            raise RuntimeError('EXISTING_FILE_SIZE_MISMATCH_PRESERVE_EVIDENCE: ' + str(dst))
        record = get_file(item, DATASET, 'dataset', data, endpoint=endpoint)
        if not video_decodes(dst):
            raise RuntimeError('DOWNLOADED_MP4_FAILED_DECODING: ' + str(dst))
        info.append(record)
        print('ROBOREEL_TRAIN_DRAWER_VIDEO_OK', json.dumps({'id': row['id'],
            'bytes': row['bytes'], 'sha256': record['sha256'],
            'transport': record['transport']}), flush=True)
    return info


def encode_one(project: Path, data: Path, demo_id: int, home: Path, frames_per_batch: int = 2) -> None:
    from scripts.roboreel_video_dependency_dev import PINNED_UPSTREAM
    media = data / media_rel(demo_id)
    target = data / feature_rel(demo_id)
    if target.is_file() and target.stat().st_size > 512:
        print('ROBOREEL_TRAIN_DRAWER_FEATURE_EXISTING', str(target), flush=True)
        return
    if target.exists():
        raise RuntimeError('ZERO_OR_PARTIAL_FEATURE_PRESERVE_FOR_AUDIT: ' + str(target))
    if not video_decodes(media):
        raise RuntimeError('Cannot decode authentic TRAIN video ' + str(media))
    validated = cache_home(project)
    if validated is None or validated.resolve() != home.resolve():
        raise RuntimeError('SIGLIP_CACHE_NOT_VALIDATED_OFFLINE_NO_GCS_DOWNLOAD')
    official = project / 'third_party/roboreel_models'
    git = subprocess.run(['git', '-C', str(official), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True)
    if git.returncode or git.stdout.strip() != PINNED_UPSTREAM:
        raise RuntimeError('OFFICIAL_ENCODER_COMMIT_NOT_PINNED')
    script = official / 'scripts/compute_vision_features_for_demo.py'
    if not script.is_file():
        raise FileNotFoundError(script)
    os.environ['OPENPI_DATA_HOME'] = str(home)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    for folder in (official, official / 'src', official / 'src/external_models'):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    module_spec = importlib.util.spec_from_file_location('_official_siglip_drawer', script)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError('Cannot import pinned official encoder')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    rgb = module._read_camera_mp4(media.parent, 'front')
    if rgb is None or len(rgb) < 2:
        raise RuntimeError('Official encoder failed to decode '+str(media))
    from agcofl.official_siglip_batched import encode_camera_batched, validate_batch_frames
    validate_batch_frames(frames_per_batch)
    encoder = module._PersistentEncoder(max_T=max(1024, len(rgb)))
    target.parent.mkdir(parents=True, exist_ok=True)
    def log_progress(done, total):
        if done == total or done == frames_per_batch or done % 32 == 0:
            print('ROBOREEL_TRAIN_DRAWER_SIGLIP_FRAMES', json.dumps({
                'id': demo_id, 'done': done, 'total': total,
                'frames_per_batch': frames_per_batch}), flush=True)
    status = encode_camera_batched(encoder, target.parent, rgb, 'front',
                                   batch_frames=frames_per_batch, progress=log_progress)
    if not target.is_file() or target.stat().st_size <= 512:
        raise RuntimeError('OFFICIAL_ENCODER_FAILED: '+str(status))
    print('ROBOREEL_TRAIN_DRAWER_SIGLIP_READY', json.dumps({
        'id': demo_id, 'feature': str(target), 'bytes': target.stat().st_size,
        'sha256': file_sha256(target)}), flush=True)


def save_once(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text() != payload:
            raise FileExistsError('Frozen selection changed; select fresh --out: ' + str(path))
    else:
        path.write_text(payload)


def run(args) -> int:
    project = args.project.expanduser().resolve()
    data = args.data_root.expanduser().resolve()
    if not data.is_relative_to(project):
        raise ValueError('Keep TRAIN downloads inside project path, not home caches')
    output = args.out.expanduser().resolve()
    if not output.is_relative_to(project):
        raise ValueError('Report must remain inside project path')
    if args.endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError('Only approved official/mirror Hugging Face endpoints')
    if args.max_mb <= 0 or args.max_mb > 300:
        raise ValueError('Hard limit is 300 MB for exactly 2 TRAIN videos')
    print('ROBOREEL_TRAIN_DRAWER_LOCAL', json.dumps({
        'existing': existing_local(data)[:10], 'siglip_cache_home':
        str(cache_home(project)) if args.features else 'NOT_CHECKED',
        'new_physx_actions': 0, 'new_policy_training': 0}), flush=True)
    if not args.download and not args.features and not args.discover:
        return 0
    if args.discover or args.download:
        try:
            rows, endpoint = list_remote(args.endpoint)
            selection = select_two(rows, budget=int(args.max_mb*1024*1024))
        except Exception as exc:
            print('ROBOREEL_TRAIN_DRAWER_METADATA_BLOCKED', json.dumps({
                'type': type(exc).__name__, 'error': str(exc)[:600],
                'endpoint': args.endpoint}), flush=True)
            return 2
        selection['endpoint'] = endpoint
        selection['data_root'] = str(data)
        print('ROBOREEL_TRAIN_DRAWER_PLAN', json.dumps(selection, ensure_ascii=False), flush=True)
        if args.discover and not args.download:
            return 0
        save_once(output / 'selection.json', selection)
        records = download_two(selection, data, args.endpoint)
        stable = [{'path': x['path'], 'bytes': x['bytes'], 'sha256': x['sha256']}
                  for x in records]
        save_once(output / 'downloaded.json', {'selection': selection,
                                               'downloaded': stable})
    if args.features:
        manifest = output / 'selection.json'
        if not manifest.is_file():
            raise FileNotFoundError('Need previous --download selection: ' + str(manifest))
        selection = json.loads(manifest.read_text())
        assert_video_sources(selection, data)
        home = cache_home(project)
        if home is None:
            print('ROBOREEL_TRAIN_DRAWER_SIGLIP_BLOCKED', json.dumps({
                'reason': 'NO_VERIFIED_LOCAL_OPENPI_SIGLIP_CACHE',
                'suggested_audit': 'python -m scripts.check_pi05_siglip_cache',
                'no_base_weight_download_attempted': True}), flush=True)
            return 3
        for item in selection['video_files']:
            demo_id = int(item['id'])
            target = data / feature_rel(demo_id)
            if target.is_file() and target.stat().st_size > 512:
                continue
            print('ROBOREEL_TRAIN_DRAWER_SIGLIP_ISOLATED_START', demo_id, flush=True)
            child_env = os.environ.copy()
            child_env['OPENPI_DATA_HOME'] = str(home)
            child_env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
            result = subprocess.run([sys.executable, '-u', '-m',
                'scripts.prepare_roboreel_train_drawer', '--encode-one', str(demo_id),
                '--project', str(project), '--data-root', str(data),
                '--out', str(output), '--endpoint', args.endpoint,
                '--frames-per-batch', str(args.frames_per_batch)], cwd=project,
                env=child_env, check=False)
            if result.returncode != 0:
                raise RuntimeError('OFFICIAL_SIGLIP_WORKER_FAILED id='+str(demo_id)+
                                   ' exit='+str(result.returncode))
        print('ROBOREEL_TRAIN_DRAWER_ASSETS_READY', json.dumps({
            'ids': [r['id'] for r in selection['video_files']],
            'feature_files': [str(data / feature_rel(r['id']))
                              for r in selection['video_files']],
            'data_root': str(data)}), flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=PROJECT)
    p.add_argument('--data-root', type=Path,
                   default=PROJECT / 'outputs/v120_baselines/official_eval_data')
    p.add_argument('--out', type=Path,
                   default=PROJECT / 'outputs/roboreel_train_drawer_assets/v1')
    p.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com'))
    p.add_argument('--max-mb', type=int, default=300)
    p.add_argument('--frames-per-batch', type=int, default=2,
                   help='Microbatch of frames for pinned SigLIP, 1..8; default 2 on 4090')
    p.add_argument('--discover', action='store_true', help='Remote metadata only, no download')
    p.add_argument('--download', action='store_true', help='Explicitly download exactly 2 verified TRAIN videos')
    p.add_argument('--features', action='store_true', help='Extract real official SigLIP features, isolated per video')
    p.add_argument('--encode-one', type=int, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.encode_one is not None:
        project = args.project.resolve()
        data = args.data_root.resolve()
        home = cache_home(project)
        if home is None:
            raise RuntimeError('SIGLIP_LOCAL_CACHE_MISSING_NO_GCS')
        encode_one(project, data, args.encode_one, home, args.frames_per_batch)
        return 0
    from agcofl.official_siglip_batched import validate_batch_frames
    validate_batch_frames(args.frames_per_batch)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
