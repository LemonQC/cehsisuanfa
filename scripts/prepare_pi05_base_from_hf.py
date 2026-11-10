#!/usr/bin/env python3
"""Prepare original OpenPI pi05_base/params from a pinned public HF *mirror*.

The pinned RoboReel SigLipTokenizer needs the original pi05_base
PaliGemma.img subtree, not the RoboReel ND Uniform fine-tuned checkpoint.
This command downloads only that base params subtree (roughly 12.4 GB),
checks the complete Hugging Face manifest against local files, and
symlinks the complete directory to OpenPI's normal GCS cache pathname.

Default: metadata-only, NO download. --download: explicit large transfer.
No GCS credentials, no paid model inference API, no modification of
original model code, no modification to pretrained Uniform weights.
WARNING: community HF mirror provenance is asserted by the uploader;
HF hashes prove transport integrity but do not establish byte-identity
with the primary GCS bucket. Keep this distinction in research reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import quote, urlsplit

import requests

REPO="robotgeneralist/openpi_checkpoint_mirrors2"
REVISION="1fda6b04e83d625eb1a55d68db32050e248cac2b"
PREFIX="pi05_base/params/"
REQUIRED=("_METADATA","_CHECKPOINT_METADATA","manifest.ocdbt")
PROJECT=Path(__file__).resolve().parents[1]
CHUNK_SIZE=2*1024*1024
GOOD_STATES=("sha256-verified","git-blob-verified","size-only")


def _digest_file(file: Path, algo: str) -> str:
    h=hashlib.new(algo)
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def file_state(file: Path, sibling):
    """Verify size and published HF LFS SHA256 / Git blob SHA1 if present."""
    if not file.is_file():
        return "missing"
    size=getattr(sibling,"size",None)
    if size is None:
        raise ValueError("Published file size is required for safe resume")
    if file.stat().st_size!=size:
        return "wrong-size"
    lfs=getattr(sibling,"lfs",None) or {}
    if isinstance(lfs,dict) and lfs.get("sha256"):
        return ("sha256-verified" if _digest_file(file,"sha256")==lfs["sha256"].lower()
                else "bad-sha256")
    blob=getattr(sibling,"blob_id",None)
    if blob and len(blob)==40:
        h=hashlib.sha1()
        h.update(f"blob {size}".encode() + bytes([0]))
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(8*1024*1024),b""):
                h.update(block)
        return "git-blob-verified" if h.hexdigest()==blob.lower() else "bad-blob-sha1"
    return "size-only"


def download_one(sibling, root: Path, blob_endpoint: str, retries=4):
    """Resumable HTTPS streamed GET, bypassing hf_hub_download HEAD headers.

    Mirrors may redirect from /resolve to huggingface.co (308) without
    x-repo-commit and x-linked-etag. Requests follows redirects while
    refusing a final HTTP downgrade; its ordinary proxy configuration
    is supplied by the operator's environment.
    """
    relative=sibling.rfilename
    if not relative.startswith(PREFIX) or ".." in Path(relative).parts:
        raise ValueError(f"Refusing unexpected manifest path: {relative}")
    final=root/relative
    final.parent.mkdir(parents=True,exist_ok=True)
    state=file_state(final,sibling)
    if state in GOOD_STATES:
        return state
    if state not in ("missing",):
        raise RuntimeError(f"Existing finished file {relative} is {state}; "
                           "preserve and inspect instead of overwriting")
    part=final.with_name(final.name+".part")
    size=sibling.size
    if size is None:
        raise RuntimeError(f"Unknown metadata size for {relative}")
    url=blob_endpoint.rstrip("/")+"/"+REPO+"/resolve/"+REVISION+"/"+quote(relative,safe="/")
    last=None
    for attempt in range(retries+1):
        offset=part.stat().st_size if part.is_file() else 0
        if offset>size:
            raise RuntimeError(f"Overlong .part file {relative}, refuse overwrite")
        if offset==size:
            state=file_state(part,sibling)
            if state in GOOD_STATES:
                part.replace(final)
                return state
            raise RuntimeError(f"Finished .part has bad hash: {relative} {state}")
        headers={"Accept-Encoding":"identity"}
        if offset:
            headers["Range"]=f"bytes={offset}-"
        try:
            with requests.get(url,headers=headers,stream=True,
                              timeout=(20,120),allow_redirects=True) as response:
                if urlsplit(response.url).scheme!="https":
                    raise RuntimeError(f"Non-HTTPS redirect for {relative}")
                response.raise_for_status()
                if response.status_code==206:
                    match=re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+|[*])",
                                       response.headers.get("Content-Range",""))
                    if not match or int(match.group(1))!=offset:
                        raise RuntimeError(f"Invalid Range response for {relative}")
                    mode="ab" if offset else "wb"
                elif response.status_code==200:
                    mode="wb"  # Server ignored Range: restart .part only.
                else:
                    raise RuntimeError(f"HTTP {response.status_code} for {relative}")
                next_notice=32*1024*1024
                t0=time.monotonic()
                total=offset if mode=="ab" else 0
                with part.open(mode) as handle:
                    for block in response.iter_content(chunk_size=CHUNK_SIZE):
                        if block:
                            handle.write(block)
                            total+=len(block)
                            if total>=next_notice or time.monotonic()-t0>20:
                                print("OPENPI_BASE_PROGRESS",relative,
                                      total,"/",size,"bytes",flush=True)
                                next_notice=total+32*1024*1024
                                t0=time.monotonic()
            if part.stat().st_size!=size:
                raise IOError(f"Partial {relative}: {part.stat().st_size}/{size} bytes")
            state=file_state(part,sibling)
            if state not in GOOD_STATES:
                raise RuntimeError(f"Hash/size mismatch for {relative}: {state}")
            part.replace(final)
            return state
        except (requests.RequestException,IOError) as exc:
            last=type(exc).__name__+": "+str(exc)[:130]
            if attempt<retries:
                print("OPENPI_BASE_RETRY",relative,"offset",part.stat().st_size
                      if part.exists() else 0,"reason",last,flush=True)
                time.sleep(min(2**attempt,15))
                continue
    raise RuntimeError(f"File {relative} download failed: {last}")



def selection(siblings):
    selected=sorted(
        (s for s in siblings if s.rfilename.startswith(PREFIX)),
        key=lambda x:x.rfilename)
    paths={s.rfilename for s in selected}
    if not selected or not all(PREFIX+p in paths for p in REQUIRED):
        raise RuntimeError("Pinned mirror listing missing critical original Orbax metadata")
    return selected


def local_manifest_check(selected, root):
    missing=[]
    bad_size=[]
    for s in selected:
        local=root/s.rfilename
        state=file_state(local,s)
        if state=="missing":
            missing.append(s.rfilename)
        elif state not in GOOD_STATES:
            bad_size.append(s.rfilename)
    params=root/PREFIX
    structural={
        "_METADATA":(params/"_METADATA").is_file(),
        "_CHECKPOINT_METADATA":(params/"_CHECKPOINT_METADATA").is_file(),
        "manifest.ocdbt":(params/"manifest.ocdbt").is_file(),
        "d":(params/"d").is_dir() and any((params/"d").iterdir()),
        "ocdbt.process_0":(params/"ocdbt.process_0").is_dir() and
                               any((params/"ocdbt.process_0").iterdir()),
    }
    verified=sum(file_state(root/s.rfilename,s) in
                 ("sha256-verified","git-blob-verified") for s in selected)
    return {"files_expected":len(selected),
            "verified_by_published_hash":verified,
            "files_present_matching_manifest":len(selected)-len(missing)-len(bad_size),
            "missing":missing[:15], "wrong_size":bad_size[:15],
            "missing_total":len(missing), "wrong_size_total":len(bad_size),
            "structural":structural,
            "complete":not missing and not bad_size and all(structural.values())}


def link_original_cache(params, cache):
    expected=cache/"openpi-assets/checkpoints/pi05_base/params"
    expected.parent.mkdir(parents=True,exist_ok=True)
    if expected.is_symlink():
        if expected.resolve()!=params.resolve():
            raise RuntimeError(f"Existing OpenPI cache link points elsewhere: {expected}")
    elif expected.exists():
        if expected.resolve()!=params.resolve():
            raise RuntimeError(f"Existing OpenPI cache is a different directory: {expected}")
    else:
        expected.symlink_to(params.resolve(),target_is_directory=True)
    return str(expected)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint",default=os.environ.get("HF_ENDPOINT","https://hf-mirror.com"))
    p.add_argument("--download",action="store_true",
                   help="EXPLICIT: download approximately 12.4GB pinned pi05_base Orbax params")
    p.add_argument("--workers",type=int,default=4,
                   help="Reserved for compatible CLI; downloads are sequential for safe .part resume")
    p.add_argument("--probe-only",action="store_true",
                   help="GET only pinned _METADATA via HTTPS, no large files")
    p.add_argument("--blob-endpoint",default="https://huggingface.co",
                   help="Blob host; use official huggingface.co via trusted HTTPS_PROXY")
    p.add_argument("--retries",type=int,default=4)
    p.add_argument("--dest",type=Path,
        default=PROJECT/"outputs/v120_baselines/openpi_base_hf_mirror")
    p.add_argument("--cache",type=Path,
        default=PROJECT/"outputs/v120_baselines/openpi_cache")
    args=p.parse_args(argv)
    url=urlsplit(args.endpoint)
    blob_url=urlsplit(args.blob_endpoint)
    if (any(u.scheme!="https" or not u.hostname or u.username or u.password
            for u in (url,blob_url))
        or not 1<=args.workers<=16 or not 0<=args.retries<=12):
        p.error("HTTPS metadata/blob endpoints, 1..16 workers, and 0..12 retries required")
    # Import after setting HF_ENDPOINT so a single pinned model snapshot is used.
    os.environ["HF_ENDPOINT"]=args.endpoint
    from huggingface_hub import HfApi
    api=HfApi(endpoint=args.endpoint,token=False)
    info=api.model_info(REPO,revision=REVISION,files_metadata=True,token=False)
    if info.sha!=REVISION:
        raise RuntimeError(f"Unexpected mirror revision {info.sha}, expected {REVISION}")
    selected=selection(info.siblings)
    root=args.dest.expanduser().resolve()
    cache=args.cache.expanduser().resolve()
    nominal=sum(s.size or 0 for s in selected)
    print("OPENPI_BASE_MIRROR_MANIFEST",json.dumps({
        "repo":REPO,"revision":REVISION,"endpoint":args.endpoint,
        "file_count":len(selected),"nominal_bytes":nominal,
        "destination":str(root),"download_requested":args.download,
        "provenance":"THIRD_PARTY_MIRROR_OF_PUBLIC_OPENPI_GCS_NOT_YET_BYTE_VERIFIED",
    }),flush=True)
    if not args.download and not args.probe_only:
        print("PLAN_ONLY_NO_CHECKPOINT_DOWNLOAD")
        return 0

    # The mirror's /resolve often replies 308 without the headers required
    # by hf_hub_download. Use a direct GET on the official hub through a
    # user-configured HTTPS_PROXY; first confirm tiny metadata transfer.
    by_path={s.rfilename:s for s in selected}
    probe=by_path[PREFIX+"_METADATA"]
    state=download_one(probe,root,args.blob_endpoint,args.retries)
    print("OPENPI_BASE_SMALL_GET_OK",probe.rfilename,state,flush=True)
    if args.probe_only:
        print("SMALL_PROBE_ONLY_NO_LARGE_DOWNLOAD",flush=True)
        return 0
    for index,sibling in enumerate(selected,1):
        if sibling.rfilename==probe.rfilename:
            continue
        print("OPENPI_BASE_FILE",index,"/",len(selected),
              sibling.size,sibling.rfilename,flush=True)
        state=download_one(sibling,root,args.blob_endpoint,args.retries)
        print("OPENPI_BASE_FILE_OK",sibling.rfilename,state,flush=True)
    check=local_manifest_check(selected,root)
    if not check["complete"]:
        print("OPENPI_BASE_MIRROR_INCOMPLETE",json.dumps(check),flush=True)
        raise RuntimeError("Missing/wrong-size Orbax shards; no OpenPI cache link created")
    cache_link=link_original_cache((root/PREFIX).resolve(),cache)
    out=PROJECT/"outputs/v120_baselines/openpi_base_hf_manifest.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    record={
        "source":REPO,"revision":REVISION,"endpoint":args.endpoint,
        "local_dir":str(root),"openpi_data_home":str(cache),
        "openpi_base_cache_link":cache_link,
        "manifest_verification":check,
        "hash_note":"Manifest SHA256/Git blob IDs verified when published; otherwise size-only. Original GCS identity is NOT independently established",
        "siglip_pickle_exported":False,"official_pi05_rollouts":0,
    }
    out.write_text(json.dumps(record,indent=2)+"\n")
    print("OPENPI_BASE_MIRROR_STAGED",json.dumps(record),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
