#!/usr/bin/env python3
"""Inspect all 25 official RoboReel ND Uniform mirror file redirects.

Metadata + HEAD requests ONLY: no model downloads and no Hugging Face token.
Requires huggingface_hub and requests in the selected Conda environment.
This is a network diagnostic, not an alternative checkpoint producer.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import filter_repo_objects

REPO = "LogosRoboticsLab/roboreel_models"
VARIANT = "nd/vid_cond_pi05_uniform/"
DEFAULT_MIRROR = "https://hf-mirror.com"
DEFAULT_DEST = "outputs/v120_baselines/released_checkpoints"


def inspected_redirect(status: int, location: str) -> tuple[str, bool]:
    """Print destination *hostname*, never a presigned URL or query."""
    if status not in (301, 302, 303, 307, 308):
        return "", False
    parsed = urlsplit(location)
    host = (parsed.hostname or "(relative)").lower()
    blocked = host in ("huggingface.co", "hf.co", "www.huggingface.co")
    return host, blocked


def probe_file(endpoint: str, filename: str, dest: Path, expected_size: int | None):
    url = endpoint.rstrip("/") + "/" + REPO + "/resolve/main/" + quote(filename, safe="/")
    file = dest / filename
    actual_size = file.stat().st_size if file.is_file() else None
    local = (
        "matches-metadata-size" if expected_size is not None and actual_size == expected_size
        else ("present-unverified" if actual_size else "missing")
    )
    result = dict(path=filename, local=local, size=actual_size, expected=expected_size,
                  code=None, redirect="", missing_commit=False, points_to_blocked_hub=False)
    try:
        # Do not follow redirects; a mirror that redirects back to the blocked
        # official endpoint is not a functional mirror for this network.
        with requests.head(url, allow_redirects=False, timeout=(7, 15),
                           headers={"Accept-Encoding": "identity"}) as response:
            result["code"] = response.status_code
            result["redirect"], result["points_to_blocked_hub"] = inspected_redirect(
                response.status_code, response.headers.get("Location", ""))
            result["missing_commit"] = not bool(response.headers.get("x-repo-commit"))
            result["has_x_linked_etag"] = bool(response.headers.get("x-linked-etag"))
    except requests.RequestException as error:
        result["error"] = type(error).__name__
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default=DEFAULT_MIRROR)
    p.add_argument("--dest", default=DEFAULT_DEST)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    parsed = urlsplit(a.endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        p.error("--endpoint must be HTTPS without credentials")
    if a.workers < 1 or a.workers > 8:
        p.error("--workers must be between 1 and 8")
    endpoint = a.endpoint.rstrip("/")
    api = HfApi(endpoint=endpoint, token=False)
    try:
        info = api.model_info(REPO, files_metadata=True, token=False)
    except Exception as error:
        p.exit(2, f"MIRROR_MODEL_INFO_FAILED {type(error).__name__}: {error}\n")
    paths = sorted(filter_repo_objects(
        [s.rfilename for s in info.siblings],
        allow_patterns=["README.md", "nd/vid_cond_pi05_uniform/*"]))
    if not paths:
        p.exit(2, "MIRROR_NO_UNIFORM_FILES: unexpected model structure\n")
    sizes = {s.rfilename: s.size for s in info.siblings}
    dest = Path(a.dest).resolve()
    print(f"MIRROR_ND_UNIFORM_HEAD_ONLY count={len(paths)} repo_revision={info.sha}",
          flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(
            lambda name: probe_file(endpoint, name, dest, sizes.get(name)), paths))
    bad = []
    for row in rows:
        extras = []
        if row["redirect"]:
            extras.append("redirect_host=" + row["redirect"])
        if row["missing_commit"]:
            extras.append("missing_x_repo_commit=true")
        if row["points_to_blocked_hub"]:
            extras.append("BACK_TO_BLOCKED_HUB=true")
        if row.get("error"):
            extras.append("error=" + row["error"])
        print(f"  HTTP={row['code']} {row['local']} {row['path']} " +
              " ".join(extras), flush=True)
        if row["points_to_blocked_hub"] or row["code"] is None or (
                row["code"] is not None and row["code"] >= 400):
            bad.append(row["path"])
    if bad:
        print(f"MIRROR_UNUSABLE_FOR_SOME_FILES count={len(bad)}", flush=True)
        print("Stop before 6.2GB download. Options: an actually hosting mirror "
              "or official checkpoint transfer from an internet-capable machine.",
              flush=True)
        return 2
    if any(row["missing_commit"] for row in rows):
        print("MIRROR_METADATA_WARNING: Some HEAD replies lack x-repo-commit; "
              "huggingface_hub may reject these even when HTTP succeeds.",
              flush=True)
        return 3
    print("MIRROR_RESOLVE_HEAD_OK: Only metadata tested, not large blobs.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
