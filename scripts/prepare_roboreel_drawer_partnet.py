#!/usr/bin/env python3
"""Acquire/inspect only RoboReel's PartNet-Mobility drawer 44962.

Authoritative scene source: gupta-anmol99/roboreel, commit
 afaf4a03df0be1e2347becde96da4f34e0de83fc.
The environment's asset_utilities reads mani_skill.ASSET_DIR/{configs,dataset},
not the standard ManiSkill partnet_mobility/dataset location.

By default this performs read-only inspection, with zero network or physics.
--import-dir accepts previously authorized files; --download explicitly uses
SAPIEN's registered-user mechanism and suppresses any token-bearing URL.
"""
from __future__ import annotations

import argparse
import contextlib
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

MODEL_ID = '44962'
PROJECT = Path(__file__).resolve().parents[1]
ASSET_ROOT_REL = Path('outputs/roboreel_partnet_single_44962')
UPSTREAM_INFO_REL = Path('third_party/roboreel_models/roboreel_env/assets/data/configs/info_drawer.json')
MAX_MODEL_BYTES = 512 * 1024 * 1024  # physical unpacked disk cap, one authorized model only


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda: f.read(1024 * 1024), b''):
            h.update(x)
    return h.hexdigest()


def locations(project: Path):
    base = project.resolve()
    root = base / ASSET_ROOT_REL
    return {'home': root, 'data': root / 'data', 'model': root / 'data/dataset' / MODEL_ID,
            'metadata': root / 'data/configs/info_drawer.json',
            'upstream_metadata': base / UPSTREAM_INFO_REL}


def verified_upstream_metadata(upstream_path: Path) -> dict:
    if not upstream_path.is_file():
        raise FileNotFoundError('PINNED_ROBOREEL_DRAWER_METADATA_MISSING: ' + str(upstream_path))
    d = json.loads(upstream_path.read_text(encoding='utf-8'))
    item = d.get(MODEL_ID)
    if not isinstance(item, dict) or str(item.get('partnet_mobility_id')) != MODEL_ID:
        raise ValueError('ROBOREEL_METADATA_DOES_NOT_REFERENCE_44962')
    scale = float(item.get('scale', 0))
    if not 0 < scale <= 3:
        raise ValueError('INVALID_PINNED_DRAWER_SCALE')
    return d


def validate_model(model_dir: Path) -> dict:
    """Conservative file-only gate; NOT proof that the scene parses in SAPIEN."""
    result = {'model_path': str(model_dir), 'valid': False, 'urdf': None,
              'file_count': 0, 'total_bytes': 0, 'reason': None}
    if not model_dir.is_dir():
        result['reason'] = 'MISSING_MODEL_DIRECTORY';return result
    files = [p for p in model_dir.rglob('*') if p.is_file()]
    if not files:
        result['reason'] = 'EMPTY_MODEL_DIRECTORY';return result
    if any(p.is_symlink() for p in model_dir.rglob('*')):
        result['reason'] = 'SYMLINK_IN_MODEL_DIRECTORY';return result
    n = sum(p.stat().st_size for p in files)
    result.update(file_count=len(files), total_bytes=n)
    if n > MAX_MODEL_BYTES:
        result['reason'] = 'MODEL_DISK_SIZE_EXCEEDS_512MIB';return result
    candidates = [model_dir / s for s in ('mobility_cvx.urdf', 'mobility_fixed.urdf','mobility.urdf')]
    urn = next((p for p in candidates if p.is_file() and p.stat().st_size > 128), None)
    if urn is None:
        result['reason'] = 'MISSING_NONEMPTY_MOBILITY_URDF';return result
    try:
        parsed = ET.parse(urn).getroot()
        if parsed.tag != 'robot' or not parsed.findall('.//link'):
            raise ValueError('not a robot URDF with links')
    except (ValueError, ET.ParseError, OSError) as exc:
        result['reason'] = 'INVALID_MOBILITY_URDF:' + type(exc).__name__;return result
    result.update(urdf=str(urn), valid=True, reason=None,
                  urdf_sha256=sha256(urn),
                  referenced_meshes=len(parsed.findall('.//mesh')))
    return result


def inspect(project: Path) -> dict:
    loc = locations(project)
    out = {'type': 'ROBOREEL_PARTNET_44962_ASSET_GATE',
           'expected_model_id': MODEL_ID, 'task': 'toys_in_drawer',
           'ms_asset_dir': str(loc['home']), 'mani_skill_asset_dir': str(loc['data']),
           'official_auth_required_for_download': True, 'new_physx_actions': 0,
           'new_training_runs': 0}
    try:
        metadata = verified_upstream_metadata(loc['upstream_metadata'])
        out['pinned_metadata_sha256'] = sha256(loc['upstream_metadata'])
        out['pinned_metadata_entry'] = metadata[MODEL_ID]
        out['metadata_installed'] = (loc['metadata'].is_file() and
            loc['metadata'].read_bytes() == loc['upstream_metadata'].read_bytes())
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc:
        out.update(status='BLOCKED',reason='UPSTREAM_METADATA_UNAVAILABLE:'+type(exc).__name__)
        return out
    model = validate_model(loc['model'])
    out['model'] = model
    out['status'] = ('READY_FOR_SCENE_INIT' if model['valid'] and out['metadata_installed']
                     else 'NEED_OFFICIAL_PARTNET_44962')
    out['reason'] = None if out['status']=='READY_FOR_SCENE_INIT' else (
        'PINNED_METADATA_NOT_INSTALLED' if not out['metadata_installed'] else model['reason'])
    return out


def ensure_metadata(loc: dict):
    verified_upstream_metadata(loc['upstream_metadata'])
    dst = loc['metadata']
    if dst.is_file():
        if dst.read_bytes() != loc['upstream_metadata'].read_bytes():
            raise FileExistsError('DIFFERENT_EXISTING_DRAWER_METADATA_DO_NOT_OVERWRITE: ' + str(dst))
        return
    dst.parent.mkdir(parents=True,exist_ok=True)
    # Audit against pinned author data and do not change existing metadata.
    with tempfile.NamedTemporaryFile(dir=dst.parent, delete=False) as f:
        tmp = Path(f.name)
        f.write(loc['upstream_metadata'].read_bytes())
    try:
        if dst.exists():
            raise FileExistsError('CONCURRENT_METADATA_WRITE: '+str(dst))
        tmp.rename(dst)
    finally:
        if tmp.exists():tmp.unlink()


def _source_model_folder(source: Path) -> Path:
    source = source.expanduser().resolve()
    candidate = source / MODEL_ID if (source / MODEL_ID).is_dir() else source
    if candidate.name != MODEL_ID:
        raise ValueError('IMPORT_SOURCE_MUST_BE_44962_DIRECTORY_OR_ITS_PARENT')
    verified = validate_model(candidate)
    if not verified['valid']:
        raise ValueError('AUTHORIZED_SOURCE_MODEL_INCOMPLETE: ' + str(verified['reason']))
    return candidate


def install_from_authorized_source(project: Path, source: Path) -> dict:
    loc = locations(project)
    upstream = _source_model_folder(source)
    dest = loc['model']
    if dest.exists():
        if not validate_model(dest)['valid']:
            raise FileExistsError('EXISTING_PARTIAL_MODEL_PRESERVE: '+str(dest))
        ensure_metadata(loc)
        return inspect(project)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.44962-staging-', dir=dest.parent) as staging:
        staged = Path(staging) / MODEL_ID
        shutil.copytree(upstream, staged, symlinks=False)
        if not validate_model(staged)['valid']:
            raise RuntimeError('STAGED_MODEL_INCOMPLETE')
        # No overwrite, source remains untouched. Same-volume rename.
        if dest.exists():
            raise FileExistsError('CONCURRENT_PARTNET_INSTALL: '+str(dest))
        staged.rename(dest)
    ensure_metadata(loc)
    return inspect(project)


def download_authorized(project: Path, token: str) -> dict:
    """Use the original official SAPIEN helper, suppressing its TOKEN-bearing URL.

    It is a SINGLE explicit download of model 44962. No bulk cabinet dataset.
    The installed artifact is validated in a fresh local staging directory.
    """
    if not token:
        raise ValueError('MISSING_SAPIEN_ACCESS_TOKEN')
    loc = locations(project)
    if loc['model'].exists():
        raise FileExistsError('MODEL_DIRECTORY_ALREADY_EXISTS_REFUSE_REDOWNLOAD')
    try:
        import sapien.asset
    except (ImportError,AttributeError) as exc:
        raise RuntimeError('OFFICIAL_SAPIEN_ASSET_MODULE_UNAVAILABLE') from exc
    loc['model'].parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.official-44962-',dir=loc['model'].parent) as tmp:
        # Official library prints full URL *including secret token* by default.
        # All its console output must be discarded, including error paths.
        try:
            with open(os.devnull, 'w') as sink:
                with contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
                    sapien.asset.download_partnet_mobility(
                        int(MODEL_ID),token=token,directory=str(tmp))
        except Exception as exc:
            # NEVER interpolate exception text: requests URLs may contain token.
            raise RuntimeError('AUTHORIZED_SAPIEN_44962_DOWNLOAD_FAILED:'+type(exc).__name__) from None
        staged=Path(tmp)/MODEL_ID
        verified=validate_model(staged)
        if not verified['valid']:
            raise RuntimeError('OFFICIAL_DOWNLOAD_NOT_A_VALID_44962_URDF:'+str(verified['reason']))
        if loc['model'].exists():
            raise FileExistsError('CONCURRENT_PARTNET_INSTALL')
        staged.rename(loc['model'])
    ensure_metadata(loc)
    return inspect(project)



def configure_for_robot_dev(project: Path, task: str) -> dict:
    """Fail before pi0.5/ManiSkill startup when drawer mesh not authorized.

    Setting MS_ASSET_DIR before importing mani_skill is essential: ManiSkill's
    ASSET_DIR is computed from environment variables at module import time.
    """
    if task != 'toys_in_drawer':
        return {'status': 'NOT_NEEDED_FOR_TASK', 'task': task}
    result = inspect(project)
    if result['status'] != 'READY_FOR_SCENE_INIT':
        raise RuntimeError('ROBOREEL_DRAWER_PARTNET_44962_NOT_READY: '
                           + str(result['reason'])
                           + '; inspect: python -m scripts.prepare_roboreel_drawer_partnet')
    home = str(result['ms_asset_dir'])
    existing = os.environ.get('MS_ASSET_DIR')
    if existing and Path(existing).expanduser().resolve() != Path(home).resolve():
        raise RuntimeError('MS_ASSET_DIR_POINTS_TO_DIFFERENT_LOCATION: '
                           'use project-scoped 44962 home or unset variable')
    os.environ['MS_ASSET_DIR'] = home
    return result


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=PROJECT)
    actions=p.add_mutually_exclusive_group()
    actions.add_argument('--import-dir',type=Path,default=None,
                         help='Existing legitimately acquired 44962 model dir or parent')
    actions.add_argument('--download',action='store_true',help='Explicit SAPIEN authenticated download')
    args=p.parse_args(argv)
    project=args.project.resolve()
    pre=inspect(project)
    print('ROBOREEL_PARTNET_44962_PREFLIGHT',json.dumps(pre,ensure_ascii=False),flush=True)
    if pre['status']=='READY_FOR_SCENE_INIT':
        return 0
    if not args.import_dir and not args.download:
        return 2
    try:
        if args.import_dir:
            post=install_from_authorized_source(project,args.import_dir)
        else:
            token=os.environ.get('SAPIEN_ACCESS_TOKEN','')
            if not token and sys.stdin.isatty():
                token=getpass.getpass('SAPIEN registered-user token (input hidden): ')
            if not token:
                raise ValueError('SAPIEN_ACCESS_TOKEN_REQUIRED_USE_REGISTERED_ACCOUNT')
            post=download_authorized(project,token)
        print('ROBOREEL_PARTNET_44962_READY',json.dumps(post,ensure_ascii=False),flush=True)
        return 0 if post['status']=='READY_FOR_SCENE_INIT' else 3
    except Exception as exc:
        print('ROBOREEL_PARTNET_44962_BLOCKED',json.dumps({
            'reason':str(exc)[:240],'class':type(exc).__name__,
            'no_physx_actions':True,'no_bulk_download':True},ensure_ascii=False),flush=True)
        return 3


if __name__=='__main__':
    raise SystemExit(main())
