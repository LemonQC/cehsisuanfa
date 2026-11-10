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
import json
import os
from pathlib import Path
import sys

REPO="robotgeneralist/openpi_checkpoint_mirrors2"
REVISION="1fda6b04e83d625eb1a55d68db32050e248cac2b"
PREFIX="pi05_base/params/"
REQUIRED=("_METADATA","_CHECKPOINT_METADATA","manifest.ocdbt")
PROJECT=Path(__file__).resolve().parents[1]


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
        if not local.is_file():
            missing.append(s.rfilename)
        elif s.size is not None and local.stat().st_size!=s.size:
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
    return {"files_expected":len(selected),
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
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--dest",type=Path,
        default=PROJECT/"outputs/v120_baselines/openpi_base_hf_mirror")
    p.add_argument("--cache",type=Path,
        default=PROJECT/"outputs/v120_baselines/openpi_cache")
    args=p.parse_args(argv)
    from urllib.parse import urlsplit
    url=urlsplit(args.endpoint)
    if url.scheme!="https" or not url.hostname or url.username or url.password or args.workers<1 or args.workers>16:
        p.error("HTTPS endpoint and 1..16 workers required")
    # Import after setting HF_ENDPOINT so a single pinned model snapshot is used.
    os.environ["HF_ENDPOINT"]=args.endpoint
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
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
    if not args.download:
        print("PLAN_ONLY_NO_CHECKPOINT_DOWNLOAD")
        return 0

    # First test a tiny real blob with this endpoint. If the HF mirror
    # redirects to unreachable huggingface.co, stop BEFORE attempting 12GB.
    probe=hf_hub_download(repo_id=REPO,filename=PREFIX+"_METADATA",
                          revision=REVISION,endpoint=args.endpoint,
                          token=False,local_dir=str(root))
    if not Path(probe).is_file():
        raise RuntimeError("Failed tiny upstream Orbax metadata probe")
    print("OPENPI_BASE_MIRROR_SMALL_PROBE_OK",flush=True)

    # Hugging Face handles retries and caching for large blob shards.
    # Never download any other pi05 variant or other repo directories.
    snapshot_download(repo_id=REPO,revision=REVISION,repo_type="model",
                      allow_patterns=[PREFIX+"*"],endpoint=args.endpoint,
                      token=False,local_dir=str(root),max_workers=args.workers)
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
        "hash_note":"HF content-addressed integrity of downloads; GCS original byte identity not separately established",
        "siglip_pickle_exported":False,"official_pi05_rollouts":0,
    }
    out.write_text(json.dumps(record,indent=2)+"\n")
    print("OPENPI_BASE_MIRROR_STAGED",json.dumps(record),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
