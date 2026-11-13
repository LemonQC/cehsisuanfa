#!/usr/bin/env python3
"""Fetch only B06 official policy files, pinned to an immutable HF commit.

The official RoboReel downloader calls snapshot_download with concurrent
workers; a transient 503 for tiny config.json aborts the entire snapshot.
This limited wrapper retries files sequentially, reuses completed target files
and HF .cache/huggingface partial downloads, and falls back from hf-mirror.com
to the canonical Hugging Face endpoint. It never deletes or overwrites an
incomplete final file, never changes the frozen model revision, and does not
import robot/torch modules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time

REPO = "LogosRoboticsLab/roboreel_models"
PREFIX = "nd/uniskill_policy/"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
KNOWN_ENDPOINTS = ("https://hf-mirror.com", "https://huggingface.co")


def check_manifest(obj: object) -> tuple[str, list[dict]]:
    if not isinstance(obj, dict) or obj.get("schema") != "B06_HF_SOURCE_LOCK_V1":
        raise ValueError("INVALID_B06_LOCK_SCHEMA")
    if obj.get("policy_repo") != REPO:
        raise ValueError("UNEXPECTED_B06_POLICY_REPO")
    revision = obj.get("policy_revision")
    if not isinstance(revision, str) or not _SHA40.fullmatch(revision):
        raise ValueError("INVALID_B06_POLICY_COMMIT")
    records = obj.get("policy_files")
    if not isinstance(records, list) or not records or len(records) > 20:
        raise ValueError("INVALID_B06_POLICY_FILES")
    seen = set()
    for entry in records:
        if not isinstance(entry, dict):
            raise ValueError("INVALID_B06_POLICY_ENTRY")
        name = entry.get("filename")
        size = entry.get("bytes")
        if not isinstance(name, str) or not name.startswith(PREFIX):
            raise ValueError("UNEXPECTED_B06_POLICY_PATH")
        part = PurePosixPath(name)
        if part.is_absolute() or ".." in part.parts or "\\" in name or name.endswith("/"):
            raise ValueError("UNSAFE_B06_POLICY_PATH")
        if len(part.parts) != 3 or part.name.startswith(".") or len(name) > 200:
            raise ValueError("UNSAFE_B06_POLICY_FILE")
        if type(size) is not int or size <= 0 or size > 30 * 1024**3:
            raise ValueError("INVALID_B06_POLICY_SIZE")
        if name in seen:
            raise ValueError("DUPLICATE_B06_POLICY_FILE")
        seen.add(name)
    # RoboReel's official UniSkillRunner explicitly says UniSkill has NO
    # sibling train_config.json: unlike pi0.5 it loads the policy architecture
    # from src/external_models/UniSkill-Policy/configs/uniskill_policy.json,
    # and shape/normalization metadata from ckpt.pth itself.
    # Requiring a non-existent train_config.json blocked a complete download.
    if PREFIX + "ckpt.pth" not in seen:
        raise ValueError("B06_POLICY_CHECKPOINT_NOT_IN_LOCK: " +
                         ",".join(sorted(seen)))
    # Small control/config files first; the 740MB+ weight last.
    return revision, sorted(records, key=lambda x: (x["bytes"], x["filename"]))


def is_complete(path: Path, size: int) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    if path.stat().st_size != size:
        return False
    if path.suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as f:
                if not isinstance(json.load(f), dict):
                    return False
        except (OSError, UnicodeError, ValueError):
            return False
    return True


def endpoints(preferred: str | None) -> tuple[str, ...]:
    result = []
    for name in (preferred, *KNOWN_ENDPOINTS):
        if name is None:
            continue
        cleaned = name.strip().rstrip("/")
        if cleaned not in KNOWN_ENDPOINTS:
            raise ValueError("B06_UNTRUSTED_HF_ENDPOINT")
        if cleaned not in result:
            result.append(cleaned)
    return tuple(result)


def fetch_one(
    *, root: Path, record: dict, revision: str, download,
    sources: tuple[str, ...], attempts_per_source: int = 2,
    sleep=lambda _: None, repo_id: str = REPO,
) -> str:
    name = record["filename"]
    expected_size = record["bytes"]
    target = root / name
    # An existing, wrong-size final file could be partial or user-owned.
    # Don't silently overwrite it; HF keeps partial transfers under .cache.
    if target.exists() or target.is_symlink():
        if is_complete(target, expected_size):
            print(f"B06_POLICY_FILE_REUSED file={name} bytes={expected_size}", flush=True)
            return "reused"
        raise FileExistsError("B06_POLICY_EXISTING_FILE_UNVERIFIED: " + name)
    target.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for source in sources:
        for attempt in range(1, attempts_per_source + 1):
            print(f"B06_POLICY_FILE_FETCH file={name} endpoint={source} attempt={attempt}", flush=True)
            try:
                # local_dir retains the completed target and its HF .cache
                # transfer bookkeeping; incomplete files are not removed.
                download(
                    repo_id=repo_id,
                    filename=name,
                    repo_type="model",
                    revision=revision,
                    local_dir=str(root),
                    endpoint=source,
                    force_download=False,
                    etag_timeout=30,
                )
                if is_complete(target, expected_size):
                    print(f"B06_POLICY_FILE_VERIFIED file={name} bytes={expected_size}", flush=True)
                    return "downloaded"
                # If we got a final invalid file, fail closed: do not let
                # another attempt overwrite or replace it.
                if target.exists() or target.is_symlink():
                    raise FileExistsError("B06_POLICY_DOWNLOADED_FILE_UNVERIFIED: " + name)
                errors.append((source, attempt, "NO_VERIFIED_OUTPUT"))
            except FileExistsError:
                raise
            except Exception as exc:
                errors.append((source, attempt, type(exc).__name__))
                print(f"B06_POLICY_FILE_RETRY file={name} endpoint={source} reason={type(exc).__name__}", flush=True)
            if attempt < attempts_per_source:
                sleep(2)
    print("B06_POLICY_DOWNLOAD_BLOCKED file=" + name + " attempts=" + str(errors), flush=True)
    raise RuntimeError("B06_POLICY_ALL_ENDPOINTS_FAILED:" + name)


def run(*, lock: Path, dest: Path, preferred: str | None, downloader=None, sleep=time.sleep):
    revision, records = check_manifest(json.loads(lock.read_text(encoding="utf-8")))
    if dest.is_symlink():
        raise ValueError("B06_POLICY_DEST_SYMLINK_FORBIDDEN")
    dest.mkdir(parents=True, exist_ok=True)
    sources = endpoints(preferred)
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    results = {}
    for record in records:
        results[record["filename"]] = fetch_one(
            root=dest, record=record, revision=revision,
            download=downloader, sources=sources, sleep=sleep,
        )
    print("B06_POLICY_DOWNLOAD_COMPLETE verified=" + str(len(results)) +
          " reused=" + str(sum(v == "reused" for v in results.values())) +
          " revision=" + revision, flush=True)
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--preferred-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    args = p.parse_args(argv)
    try:
        run(lock=args.lock, dest=args.dest, preferred=args.preferred_endpoint)
    except Exception as exc:
        print("B06_POLICY_DOWNLOAD_FAILED type=" + type(exc).__name__ +
              " reason=" + str(exc)[:240], flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
