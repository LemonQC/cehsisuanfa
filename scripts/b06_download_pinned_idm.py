#!/usr/bin/env python3
"""Download the one pinned official UniSkill IDM weight with safe mirror fallback.

The original B06 setup downloaded the IDM with a single mirror call. Use the
same verified per-file HF downloader as the official policy. This is transport
only: no model code, tensors, architectures, or experiment protocol is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from scripts.b06_download_pinned_policy import endpoints, fetch_one, is_complete

IDM_REPO = "HanjungKim/UniSkill"
IDM_FILE = "UniSkill_final_weight/idm.pth"
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def verify_lock(data: object) -> tuple[str, int]:
    if not isinstance(data, dict) or data.get("schema") != "B06_HF_SOURCE_LOCK_V1":
        raise ValueError("INVALID_B06_IDM_LOCK")
    if data.get("idm_repo") != IDM_REPO or data.get("idm_file") != IDM_FILE:
        raise ValueError("UNEXPECTED_B06_IDM_SOURCE")
    revision = data.get("idm_revision")
    if not isinstance(revision, str) or not SHA40.fullmatch(revision):
        raise ValueError("INVALID_B06_IDM_REVISION")
    size = data.get("idm_bytes")
    if type(size) is not int or not 10 * 1024 * 1024 <= size <= 1024 * 1024 * 1024:
        raise ValueError("INVALID_B06_IDM_SIZE")
    return revision, size


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def run(*, lock: Path, dest: Path, preferred: str | None,
        downloader=None, sleep=None) -> str:
    revision, size = verify_lock(json.loads(lock.read_text(encoding="utf-8")))
    if dest.is_symlink():
        raise ValueError("B06_IDM_DEST_SYMLINK_FORBIDDEN")
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "idm.pth"
    if target.exists() or target.is_symlink():
        if not is_complete(target, size):
            raise FileExistsError("B06_IDM_EXISTING_TARGET_UNVERIFIED")
        print("B06_IDM_REUSED bytes=" + str(size) + " sha256=" + _hash(target), flush=True)
        return "reused"
    sources = endpoints(preferred)
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    kwargs = {"root": dest, "record": {"filename": IDM_FILE, "bytes": size},
              "revision": revision, "download": downloader,
              "sources": sources, "repo_id": IDM_REPO}
    if sleep is not None:
        kwargs["sleep"] = sleep
    fetch_one(**kwargs)
    source = dest / IDM_FILE
    if not is_complete(source, size):
        raise RuntimeError("B06_IDM_SOURCE_UNVERIFIED")
    # Atomic same-drive copy to the path used by our strict runner; preserves
    # cache and original nested file. Never overwrite a prior target.
    partial = dest / "idm.pth.b06-copying"
    if partial.exists() or partial.is_symlink():
        raise FileExistsError("B06_IDM_COPY_IN_PROGRESS_INSPECT_BEFORE_RETRY")
    with source.open("rb") as src, partial.open("xb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
    if not is_complete(partial, size):
        raise RuntimeError("B06_IDM_ATOMIC_COPY_INCOMPLETE")
    os.replace(partial, target)
    if not is_complete(target, size):
        raise RuntimeError("B06_IDM_FINAL_VERIFICATION_FAILED")
    print("B06_IDM_DOWNLOAD_COMPLETE bytes=" + str(size) +
          " sha256=" + _hash(target) + " revision=" + revision, flush=True)
    return "downloaded"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--preferred-endpoint",
                   default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    a = p.parse_args(argv)
    try:
        run(lock=a.lock, dest=a.dest, preferred=a.preferred_endpoint)
    except Exception as exc:
        print("B06_IDM_DOWNLOAD_FAILED type=" + type(exc).__name__ +
              " reason=" + str(exc)[:240], flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
