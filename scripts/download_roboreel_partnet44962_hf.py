#!/usr/bin/env python3
"""Authorized, minimal and bounded Hugging Face acquisition of ONE PartNet 44962.

Only repo sapien-sim/PartNetMobility, 44962.zip, and the requesting user's
existing Hugging Face read token are used. Never downloads ShapeNet or a
whole snapshot. The exact RoboReel ASSET_DIR layout is installed atomically.

Priority: hf-mirror.com -> huggingface.co; if gated auth is unsupported at the
mirror, retry on the official service with the SAME existing credentials.
The download ZIP, HF cache, Xet cache, and temporary workspaces live within
project outputs/. HF_HOME is respected, not overwritten, to preserve login.
No passwords/tokens are printed, transmitted to GitHub, or passed in URLs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import stat
import tempfile
import zipfile
from types import SimpleNamespace
from typing import Callable

from scripts.prepare_roboreel_drawer_partnet import (
    MODEL_ID, MAX_MODEL_BYTES, PROJECT, locations, inspect,
    ensure_metadata, validate_model,
)

REPO_ID = "sapien-sim/PartNetMobility"
ARCHIVE = f"{MODEL_ID}.zip"
OFFICIAL = "https://huggingface.co"
MIRROR = "https://hf-mirror.com"
ALLOWED_ENDPOINTS = (MIRROR, OFFICIAL)
MAX_ZIP_BYTES = 512 * 1024 * 1024
MAX_FILES = 4096
URDF_NAMES = {"mobility_cvx.urdf", "mobility_fixed.urdf", "mobility.urdf"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as src:
        for buf in iter(lambda: src.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def safe_archive_entries(zf: zipfile.ZipFile):
    """Reject unsafe/ambiguous archives before ANY extraction.

    Accept a flat model ZIP or a ZIP containing a single /44962/ model root.
    This is a deliberately strict installer; inspect blocked ZIPs rather than
    silently extracting unrelated files or alternate models.
    """
    items = []
    total = 0
    for info in zf.infolist():
        raw = info.filename
        if '\\' in raw or '\x00' in raw or raw.startswith('/'):
            raise ValueError('UNSAFE_ZIP_FILENAME')
        parts = PurePosixPath(raw).parts
        if not parts or any(p in ('..', '') or ':' in p for p in parts):
            raise ValueError('UNSAFE_ZIP_FILENAME')
        if info.flag_bits & 1:
            raise ValueError('ENCRYPTED_ZIP_MEMBER')
        if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
            raise ValueError('SYMLINK_ZIP_MEMBER')
        if info.is_dir():
            continue
        if not info.file_size or info.file_size > MAX_MODEL_BYTES:
            raise ValueError('EMPTY_OR_OVERSIZED_ARCHIVE_MEMBER')
        total += info.file_size
        if total > MAX_MODEL_BYTES or len(items) >= MAX_FILES:
            raise ValueError('ARCHIVE_UNPACK_BUDGET_EXCEEDED')
        items.append((info,parts))
    if not items:
        raise ValueError('ARCHIVE_HAS_NO_MODEL_FILES')
    model_urdfs = [parts for _,parts in items if parts[-1] in URDF_NAMES]
    if not model_urdfs:
        raise ValueError('ARCHIVE_HAS_NO_MOBILITY_URDF')
    # Find the URDF closest to the archive root; a ZIP may contain different
    # valid URDF variants. Its parent is the common extraction prefix.
    urdf = min(model_urdfs, key=len)
    prefix = urdf[:-1]
    if prefix not in ((),(MODEL_ID,)) and (not prefix or prefix[-1]!=MODEL_ID):
        raise ValueError('UNRECOGNIZED_MODEL_ROOT')
    normalized=[]
    seen=set()
    for info,parts in items:
        if tuple(parts[:len(prefix)]) != tuple(prefix):
            raise ValueError('ZIP_CONTAINS_OTHER_MODEL_OR_ROOT_FILES')
        relative=parts[len(prefix):]
        if not relative or relative in seen:
            raise ValueError('DUPLICATE_OR_EMPTY_ZIP_PATH')
        seen.add(relative)
        normalized.append((info,relative))
    if not any(len(p)==1 and p[0] in URDF_NAMES for _,p in normalized):
        raise ValueError('URDF_MISSING_AT_MODEL_ROOT')
    return normalized


def extract_single_model(zip_path: Path, directory: Path) -> dict:
    """Extract into a NEW staging dir, no zipfile.extractall, no overwrite."""
    if not zip_path.is_file() or not 0 < zip_path.stat().st_size <= MAX_ZIP_BYTES:
        raise ValueError('ZIP_BYTES_NOT_WITHIN_BUDGET')
    model = directory / MODEL_ID
    if model.exists():
        raise FileExistsError('STAGING_MODEL_ALREADY_EXISTS')
    with zipfile.ZipFile(zip_path) as zf:
        normalized = safe_archive_entries(zf)
        model.mkdir(parents=True)
        count = 0
        total = 0
        for entry,rel in normalized:
            file = model.joinpath(*rel)
            file.parent.mkdir(parents=True,exist_ok=True)
            # All names are validated and no symlinks are extracted.
            with zf.open(entry) as inp, file.open('xb') as out:
                while True:
                    buf = inp.read(1024*1024)
                    if not buf:
                        break
                    count += len(buf)
                    if count > MAX_MODEL_BYTES:
                        raise ValueError('ACTUAL_UNPACK_BUDGET_EXCEEDED')
                    out.write(buf)
            total += 1
    v = validate_model(model)
    if not v['valid']:
        raise RuntimeError('OFFICIAL_HF_MODEL_INCOMPLETE:'+str(v['reason']))
    return {'model':str(model),'file_count':total,'bytes':count,
            'validated_urdf':str(v['urdf']),'urdf_sha256':v['urdf_sha256']}


def _dataset_zip_info(api, token: str) -> dict:
    info = api.dataset_info(REPO_ID,files_metadata=True,token=token)
    sha = str(getattr(info,'sha',''))
    if len(sha)!=40 or any(c not in '0123456789abcdef' for c in sha.lower()):
        raise ValueError('UNPINNABLE_HF_REPO_REVISION')
    matches=[x for x in (getattr(info,'siblings',None) or [])
             if getattr(x,'rfilename',None)==ARCHIVE]
    if len(matches)!=1:
        raise FileNotFoundError('MODEL_44962_ZIP_NOT_IN_AUTHORIZED_REPO')
    size=getattr(matches[0],'size',None)
    if not isinstance(size,int) or not 0<size<=MAX_ZIP_BYTES:
        raise ValueError('REMOTE_ZIP_SIZE_MISSING_OR_OVER_512MIB')
    return {'revision':sha,'archive':ARCHIVE,'declared_bytes':size}


def configure_project_transfer_storage(project: Path) -> dict[str, str]:
    """Keep bulky HF/Xet caches and temporary download data on project disk.

    Do not override HF_HOME or HF_TOKEN_PATH: those identify the operator's
    already authenticated account. An operator may independently set HF_HOME
    to a protected location inside outputs/ *before* running `hf auth login`.
    The explicit `cache_dir` download arg remains project-local too.
    """
    loc=locations(project.expanduser().resolve())
    root=loc['home']
    layout={
        'HF_HUB_CACHE':root/'hf_cache',
        'HF_XET_CACHE':root/'hf_xet_cache',
        'HF_ASSETS_CACHE':root/'hf_assets_cache',
        'TMPDIR':root/'tmp',
    }
    for key,folder in layout.items():
        folder.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(folder)
    return {key:str(folder) for key,folder in layout.items()}


def authorized_hf_single(project: Path, *, endpoints=ALLOWED_ENDPOINTS,
                         api_cls: Callable|None=None,
                         file_downloader: Callable|None=None,
                         token: str|None=None) -> dict:
    """Acquire only 44962.zip with caller's authorized HF identity.

    No app performs login or needs the secret on ChatGPT. HF uses locally
    saved login. ``token`` may be injected only in CPU tests.
    """
    project=project.expanduser().resolve()
    # Configure BEFORE any huggingface_hub import; HF reads cache environment
    # at module import time. Xet otherwise defaults to ~/.cache/huggingface.
    configure_project_transfer_storage(project)
    loc=locations(project)
    if loc['model'].exists():
        if not validate_model(loc['model'])['valid']:
            raise FileExistsError('EXISTING_PARTIAL_MODEL_PRESERVED')
        ensure_metadata(loc)
        return inspect(project)
    if api_cls is None or file_downloader is None or token is None:
        from huggingface_hub import HfApi, hf_hub_download, get_token
        api_cls = api_cls or HfApi
        file_downloader = file_downloader or hf_hub_download
        token = token or get_token()
    if not isinstance(token,str) or not token.startswith('hf_'):
        raise RuntimeError('HF_AUTH_REQUIRED: run hf auth login using approved account')
    if not endpoints or any(x not in ALLOWED_ENDPOINTS for x in endpoints):
        raise ValueError('UNAPPROVED_HF_ENDPOINT')
    loc['model'].parent.mkdir(parents=True,exist_ok=True)
    cache = loc['home']/'hf_cache'
    cache.mkdir(parents=True,exist_ok=True)
    errors=[]
    for endpoint in endpoints:
        try:
            api=api_cls(endpoint=endpoint,token=token)
            meta=_dataset_zip_info(api,token)
            downloaded=Path(file_downloader(
                repo_id=REPO_ID, filename=ARCHIVE, repo_type='dataset',
                revision=meta['revision'],endpoint=endpoint,token=token,
                cache_dir=str(cache),force_download=False))
            if not downloaded.is_file() or downloaded.stat().st_size != meta['declared_bytes']:
                raise ValueError('ARCHIVE_LENGTH_DOES_NOT_MATCH_PINNED_METADATA')
            zip_hash=sha256(downloaded)
            with tempfile.TemporaryDirectory(prefix='.official-hf-44962-',dir=loc['model'].parent) as temp:
                extracted=extract_single_model(downloaded,Path(temp))
                if loc['model'].exists():
                    raise FileExistsError('CONCURRENT_MODEL_INSTALL_REFUSED')
                (Path(temp)/MODEL_ID).rename(loc['model'])
            ensure_metadata(loc)
            outcome=inspect(project)
            if outcome['status']!='READY_FOR_SCENE_INIT':
                raise RuntimeError('MODEL_NOT_READY_AFTER_INSTALL')
            outcome['hf_provenance']={
                'dataset':REPO_ID,'filename':ARCHIVE,'endpoint':endpoint,
                'revision':meta['revision'],'archive_bytes':meta['declared_bytes'],
                'archive_sha256':zip_hash,'unpacked_file_count':extracted['file_count'],
                'model_urdf_sha256':extracted['urdf_sha256']}
            return outcome
        except Exception as e:
            # Do not interpolate errors or request URLs: some transports may
            # include sensitive auth parameters. Only coarse error codes.
            if loc['model'].exists():
                # An installed model cannot be silently overwritten/removed.
                raise RuntimeError('HF_MODEL_INSTALLED_BUT_METADATA_OR_POSTCHECK_FAILED') from None
            errors.append({'endpoint':endpoint,'class':type(e).__name__})
    raise RuntimeError('AUTHENTICATED_HF_44962_FETCH_FAILED:'+json.dumps(errors))


def main(argv=None) -> int:
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=PROJECT)
    p.add_argument('--official-only',action='store_true',help='Skip mirror, use Hugging Face official only')
    args=p.parse_args(argv)
    current=inspect(args.project)
    print('ROBOREEL_HF_PARTNET44962_PREFLIGHT',json.dumps({
        'status':current['status'],'model_id':MODEL_ID,'model_present':current.get('model',{}).get('valid',False),
        'dataset':REPO_ID,'source_archive':ARCHIVE,'download_entire_dataset':False},ensure_ascii=False),flush=True)
    if current['status']=='READY_FOR_SCENE_INIT':
        return 0
    try:
        result=authorized_hf_single(args.project,
            endpoints=(OFFICIAL,) if args.official_only else ALLOWED_ENDPOINTS)
    except Exception as exc:
        print('ROBOREEL_HF_PARTNET44962_BLOCKED',json.dumps({'reason':str(exc)[:400],
            'no_training':True,'no_physx':True,'no_bulk_download':True}),flush=True)
        return 3
    print('ROBOREEL_HF_PARTNET44962_READY',json.dumps({
        'status':result['status'],'dataset':REPO_ID,'metadata_installed':result['metadata_installed'],
        'model_id':MODEL_ID,'model_files':result['model']['file_count'],
        'hf_provenance':result.get('hf_provenance')},ensure_ascii=False),flush=True)
    return 0 if result['status']=='READY_FOR_SCENE_INIT' else 4

if __name__=='__main__':
    raise SystemExit(main())
