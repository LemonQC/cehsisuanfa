"""Resumable official Hugging Face artifact transport (RoboReel only).

Why this exists: HF metadata listing succeeded but hf_hub_download later
returned LocalEntryNotFoundError. Use explicitly pinned endpoints, retry
transient hub failures, and (only if needed) stream from official /resolve/
URLs with HTTP Range resume. Never fetch arbitrary URLs, trust partial data,
silently replace 2GiB checkpoints or copy user secrets.

Expected bytes MUST come from official HfApi metadata; SHA256 is calculated
locally after the complete file arrives. Downloaded files stay private/local.
"""
from __future__ import annotations

import hashlib
import re
import inspect
import json
import os
import shutil
import time
from pathlib import Path
from urllib.parse import quote

APPROVED_REPOS={
    ("LogosRoboticsLab/roboreel","dataset"),
    ("LogosRoboticsLab/roboreel_models","model"),
}
APPROVED_ENDPOINTS=("https://hf-mirror.com","https://huggingface.co")


def sha256_file(path:Path):
    hash_=hashlib.sha256()
    with Path(path).open("rb") as f:
        for piece in iter(lambda:f.read(8*1024*1024),b""):
            hash_.update(piece)
    return hash_.hexdigest()


def _valid(item,repo,repo_type,endpoint):
    if (repo,repo_type) not in APPROVED_REPOS:
        raise ValueError("Download repo is not on RoboReel allowlist")
    if endpoint.rstrip("/") not in APPROVED_ENDPOINTS:
        raise ValueError("Only the official HF and verified mirror endpoints are accepted")
    path=item.get("path","")
    parts=Path(path).parts
    if not path or "\\" in path or path.startswith("/") or any(p in ("..",".") for p in parts):
        raise ValueError("Unsafe HF path")
    n=int(item.get("bytes") or 0)
    if n<=0 or n>10*1024**3:
        raise ValueError("Artifact expected bytes absent or above 10GiB hard limit")
    if repo_type=="model" and not path.startswith("nd/vid2robot/"):
        raise ValueError("Only pinned official Vid2Robot ND weights are allowed")
    if repo_type=="dataset":
        # Preserve prior official PushCube EVAL transport behavior. Narrowly
        # permit only genuine drawer-task HUMAN TRAIN front MP4s, IDs 0000-0079.
        # Callers still impose a separate two-video/300MiB aggregate budget.
        drawer_train=re.fullmatch(
            r"human_demos/train/no_distraction/toys_in_drawer/(\\d{4})/front\\.mp4",
            path)
        allow_drawer_train=bool(drawer_train and int(drawer_train.group(1))<80)
        if not (path.startswith("human_demos/eval/no_distraction/push_cube/")
                or allow_drawer_train):
            raise ValueError("Only pinned PushCube EVAL videos or drawer TRAIN front videos are allowed")
    return n,path


def _safe_local_target(root,rel):
    root=Path(root).resolve()
    target=root/rel
    if not target.resolve().is_relative_to(root):
        raise ValueError("HF target escaped output root")
    target.parent.mkdir(parents=True,exist_ok=True)
    return target


def _direct_resolve(repo,repo_type,endpoint,path):
    rtype="datasets" if repo_type=="dataset" else ""
    # Path pieces are individually URL-encoded; no full-string quote of '/'.
    quoted="/".join(quote(part,safe="") for part in path.split("/"))
    prefix=f"/{rtype}/{repo}" if rtype else f"/{repo}"
    return endpoint.rstrip("/")+prefix+"/resolve/main/"+quoted


def _request_download(url,target,expected,attempts=3):
    """Bounded HTTP Range-resumable stream. Keep only complete target visible."""
    import requests
    tmp=target.with_name(target.name+".part")
    target.parent.mkdir(parents=True,exist_ok=True)
    errors=[]
    for attempt in range(1,attempts+1):
        resume=tmp.stat().st_size if tmp.exists() else 0
        if resume>expected:
            tmp.unlink()
            resume=0
        if resume==expected:
            os.replace(tmp,target)
            return target
        header={"Range":f"bytes={resume}-"} if resume else {}
        try:
            with requests.get(url,headers=header,timeout=(15,90),
                              stream=True,allow_redirects=True) as res:
                if res.status_code not in (200,206):
                    raise IOError(f"HTTP_{res.status_code} from official HF resolve endpoint")
                # Some reverse proxies ignore Range, so restart cleanly
                # instead of concatenating two complete files.
                if resume and res.status_code==200:
                    resume=0
                if res.status_code==206:
                    content_range=res.headers.get("Content-Range","")
                    if not content_range.startswith(f"bytes {resume}-"):
                        raise IOError(f"Unexpected partial Content-Range {content_range[:130]}")
                mode="ab" if resume and res.status_code==206 else "wb"
                written=resume
                with tmp.open(mode) as f:
                    for chunk in res.iter_content(chunk_size=4*1024*1024):
                        if not chunk:continue
                        written+=len(chunk)
                        if written>expected:
                            raise IOError("Server returned more bytes than published expected size")
                        f.write(chunk)
                if written!=expected:
                    raise IOError(f"Transferred {written}/{expected} bytes; remaining will resume")
                os.replace(tmp,target)
                return target
        except Exception as e:
            errors.append(f"attempt={attempt}:{type(e).__name__}:{str(e)[:260]}")
            print("HF_TRANSFER_RETRY",json.dumps(dict(
                file=target.name,attempt=attempt,resume_bytes=(tmp.stat().st_size
                if tmp.exists() else 0),error=errors[-1])),flush=True)
            if attempt<attempts:time.sleep(min(2*attempt,6))
    raise RuntimeError("Official HTTP resume failed: "+" | ".join(errors[-3:]))


def get_file(item,repo,repo_type,root,endpoint="https://hf-mirror.com"):
    expected,rel=_valid(item,repo,repo_type,endpoint)
    dst=_safe_local_target(root,rel)
    if dst.is_file():
        if dst.stat().st_size==expected:
            return dict(path=rel,bytes=expected,sha256=sha256_file(dst),
                        local_path=str(dst),transport="existing_verified_size_sha256")
        # Any old wrong-sized file is NEVER interpreted as a valid checkpoint.
        bad=dst.with_name(dst.name+".invalid_size")
        if bad.exists():bad.unlink()
        dst.rename(bad)
    errors=[]
    # First use upstream hub cache manager. It may reuse a previously
    # fetched 2.37GB blob, which direct requests cannot know about.
    try:
        from huggingface_hub import hf_hub_download
        kwargs=dict(repo_id=repo,repo_type=repo_type,filename=rel,
                    local_dir=str(Path(root).resolve()),etag_timeout=15)
        if "endpoint" in inspect.signature(hf_hub_download).parameters:
            kwargs["endpoint"]=endpoint
        resolved=Path(hf_hub_download(**kwargs))
        if resolved.stat().st_size!=expected:
            raise IOError(f"Hub download size mismatch {resolved.stat().st_size} != {expected}")
        if resolved.resolve()!=dst.resolve():
            shutil.copyfile(resolved,dst)
        return dict(path=rel,bytes=expected,sha256=sha256_file(dst),
                    local_path=str(dst),transport="hf_hub_download")
    except Exception as e:
        errors.append(f"hub:{type(e).__name__}:{str(e)[:220]}")
        print("HF_HUB_FALLBACK",json.dumps(dict(path=rel,error=errors[-1])),flush=True)
    # HTTPS resolve fallback first tries the configured mirror, then
    # official Hugging Face. Either must reproduce the PUBLISHED byte count.
    for host in dict.fromkeys([endpoint.rstrip("/"),"https://huggingface.co"]):
        try:
            _request_download(_direct_resolve(repo,repo_type,host,rel),dst,expected)
            return dict(path=rel,bytes=expected,sha256=sha256_file(dst),
                        local_path=str(dst),transport=f"resumable_https:{host}")
        except Exception as e:
            errors.append(f"{host}:{type(e).__name__}:{str(e)[:250]}")
    raise RuntimeError("No authenticated/public HF route delivered expected bytes: "+
                       " | ".join(errors[-3:]))
