#!/usr/bin/env python3
"""Plan/repair *only missing* RoboReel ND Uniform official files.

Uses hf-mirror.com ONLY for public model metadata (no blobs), and the
original huggingface.co CDN route for bytes through the operator's approved
HTTPS_PROXY/ALL_PROXY. Supports HTTP Range resume to .part files.
Does not modify released checkpoints, third_party authors' code or v1.3.
No tokens, no TLS bypass, no paid model inference APIs.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import time
from urllib.parse import quote, urlsplit

from huggingface_hub import HfApi
from huggingface_hub.utils import filter_repo_objects
import requests

REPO = "LogosRoboticsLab/roboreel_models"
MODEL_SHA = "ef21150c02b2685347fe3e7c8ba5a2e313ff1b7e"
PATTERNS = ["README.md", "nd/vid_cond_pi05_uniform/*"]
CHUNK = 2 * 1024 * 1024


def digest(path: Path, name: str) -> str:
    h = hashlib.new(name)
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_digest(sibling):
    lfs = getattr(sibling, "lfs", None) or {}
    if isinstance(lfs, dict) and lfs.get("sha256"):
        return ("sha256", lfs["sha256"].lower())
    return None


def check(path: Path, sibling):
    if not path.is_file():
        return "missing"
    size = getattr(sibling, "size", None)
    if size is not None and path.stat().st_size != size:
        return "wrong-size"
    expected = expected_digest(sibling)
    if expected:
        algo, value = expected
        return "verified" if digest(path, algo) == value else "bad-checksum"
    # Non-LFS files expose Git blob SHA1 in metadata. Do not incorrectly
    # compare plain SHA1(content) with SHA1('blob {len}\0'+content).
    blob_id = getattr(sibling, "blob_id", None)
    if blob_id and len(blob_id) == 40:
        h = hashlib.sha1()
        h.update(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as fd:
            for chunk in iter(lambda: fd.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return "verified" if h.hexdigest() == blob_id.lower() else "bad-checksum"
    return "size-only"


def get_manifest(endpoint: str):
    api = HfApi(endpoint=endpoint, token=False)
    info = api.model_info(REPO, files_metadata=True, token=False)
    if info.sha != MODEL_SHA:
        raise RuntimeError(f"Unexpected official model revision {info.sha}; expected {MODEL_SHA}")
    selected = sorted(filter_repo_objects(
        [s.rfilename for s in info.siblings], allow_patterns=PATTERNS))
    if len(selected) != 25:
        raise RuntimeError(f"Official manifest changed unexpectedly: {len(selected)} files, expected 25")
    by_path = {s.rfilename: s for s in info.siblings}
    if any(by_path[name].size is None for name in selected):
        raise RuntimeError("Model file sizes unavailable; cannot safely resume")
    return [(name, by_path[name]) for name in selected]


def _download_one(dest: Path, name, sibling, retry_count: int):
    final = dest / name
    final.parent.mkdir(parents=True, exist_ok=True)
    if check(final, sibling) in ("verified", "size-only"):
        return (name, "skip-existing")
    part = final.with_name(final.name + ".part")
    expected_size = sibling.size
    url = ("https://huggingface.co/" + REPO +
           "/resolve/" + MODEL_SHA + "/" + quote(name, safe="/"))
    last = None
    for attempt in range(retry_count + 1):
        offset = part.stat().st_size if part.is_file() else 0
        if offset > expected_size:
            raise RuntimeError(f"{name}: .part exceeds metadata size; inspect it manually")
        if offset == expected_size:
            state = check(part, sibling)
            if state in ("verified", "size-only"):
                part.replace(final)
                return (name, "resumed-verified" if state == "verified" else "resumed-size-only")
            raise RuntimeError(f"{name}: .part has correct size but bad digest")
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            # requests follows HTTPS redirects to the official CDN.
            # The proxy, if needed, comes only from the user's environment.
            with requests.get(url, headers=headers, stream=True,
                              timeout=(20, 180), allow_redirects=True) as response:
                if urlsplit(response.url).scheme != "https":
                    raise RuntimeError(f"{name}: refused non-HTTPS redirect")
                response.raise_for_status()
                if response.status_code not in (200, 206):
                    raise RuntimeError(f"{name}: unexpected HTTP {response.status_code}")
                if response.status_code == 206:
                    cr = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", cr)
                    if not match or int(match.group(1)) != offset:
                        raise RuntimeError(f"{name}: invalid Range response")
                    mode = "ab" if offset else "wb"
                else:
                    # Server does not support resume; restart only .part,
                    # never overwrite the original checkpoint file.
                    mode = "wb"
                with part.open(mode) as fh:
                    for block in response.iter_content(chunk_size=CHUNK):
                        if block:
                            fh.write(block)
            if part.stat().st_size != expected_size:
                raise IOError(f"size {part.stat().st_size} != {expected_size}")
            state = check(part, sibling)
            if state == "bad-checksum":
                raise RuntimeError(f"{name}: checkpoint hash mismatch; check proxy/source")
            if state not in ("verified", "size-only"):
                raise IOError(f"unexpected final state {state}")
            part.replace(final)
            return (name, state)
        except (requests.RequestException, IOError) as error:
            last = type(error).__name__
            if attempt < retry_count:
                time.sleep(min(2 ** attempt, 15))
                continue
    raise RuntimeError(f"{name}: exhausted download retries ({last})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--dest", default="outputs/v120_baselines/released_checkpoints")
    parser.add_argument("--download-missing", action="store_true",
                        help="Explicit opt-in, uses huggingface.co via HTTPS proxy")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=None,
                        help="Optional small budget for testing one missing file via proxy")
    parser.add_argument("--routed-vpn", action="store_true",
                        help="Allow explicit trusted system-level VPN/TUN routing without proxy env")
    args = parser.parse_args(argv)
    endpoint = urlsplit(args.metadata_endpoint)
    if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username:
        parser.error("Metadata endpoint must be HTTPS and contain no credentials")
    if args.retries < 0 or args.retries > 15:
        parser.error("--retries must be between 0 and 15")
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be positive")
    manifest = get_manifest(args.metadata_endpoint)
    dest = Path(args.dest).resolve()
    remaining = []
    bytes_missing = 0
    tally = {}
    for name, s in manifest:
        state = check(dest / name, s)
        tally[state] = tally.get(state, 0) + 1
        print("UNIFORM_FILE", state, s.size, name, flush=True)
        if state not in ("verified", "size-only"):
            remaining.append((name, s))
            bytes_missing += s.size
    print("UNIFORM_REPAIR_PLAN", "revision", MODEL_SHA,
          "states", tally, "remaining", len(remaining),
          "nominal_bytes", bytes_missing, flush=True)
    if not args.download_missing:
        print("PLAN_ONLY_NO_DOWNLOAD", flush=True)
        return 0
    if not any(os.getenv(n) for n in ("HTTPS_PROXY", "https_proxy",
                                      "ALL_PROXY", "all_proxy")) and not args.routed_vpn:
        parser.error("No proxy environment detected. Configure an approved HTTPS_PROXY "
                     "or explicitly use --routed-vpn if your trusted VPN/TUN handles traffic.")
    selected = remaining[:args.max_files] if args.max_files is not None else remaining
    failures = []
    for index, (name, s) in enumerate(selected, 1):
        print("RECOVER", index, "/", len(selected), name, flush=True)
        try:
            _, result = _download_one(dest, name, s, args.retries)
            print("RECOVERED", name, result, flush=True)
        except Exception as error:
            print("RECOVER_FAILED", name, type(error).__name__,
                  str(error)[:200], flush=True)
            failures.append(name)
    if failures:
        print("PARTIAL_RECOVERY_ONLY", len(failures), "failed download attempts",
              flush=True)
        return 2
    if len(selected) < len(remaining):
        print("PARTIAL_RECOVERY_MAX_FILES", len(selected), "of", len(remaining),
              "attempted; run without --max-files to finish", flush=True)
        return 0
    print("RECOVERY_COMPLETE_LOCAL_FILE_SET; evaluation still requires "
          "official preflight and SigLIP features", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
