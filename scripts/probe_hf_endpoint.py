#!/usr/bin/env python3
"""No-download probe for an optional Hugging Face mirror's public model API.

Verifies the HTTPS certificate (requests default), tests only JSON metadata,
rejects redirects and wrong repositories. No tokens are sent explicitly.
A 200 metadata response does NOT guarantee large LFS/Xet objects will work.
"""
from __future__ import annotations

import argparse
from urllib.parse import urlsplit

import requests

REPO = "LogosRoboticsLab/roboreel_models"


def probe(endpoint: str):
    ep = endpoint.rstrip("/")
    url = urlsplit(ep)
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise ValueError("Mirror endpoint must be an HTTPS URL without credentials")
    uri = ep + "/api/models/" + REPO
    print("MIRROR_METADATA_ONLY", url.hostname, flush=True)
    # Explicitly avoid Hugging Face auth in a public mirror health probe.
    with requests.Session() as s:
        # trust_env keeps the operator's approved network proxy settings.
        # Removing Authorization below prevents explicitly supplied tokens.
        s.headers.pop("Authorization", None)
        with s.get(uri, timeout=(7, 15), allow_redirects=False, stream=True,
                   headers={"Accept": "application/json"}) as resp:
            print("MIRROR_HTTP_STATUS", resp.status_code, flush=True)
            if 300 <= resp.status_code < 400:
                loc = resp.headers.get("Location", "")
                target = urlsplit(loc)
                print("MIRROR_REDIRECT_HOST", target.hostname or "relative", flush=True)
                if target.hostname == "huggingface.co":
                    raise RuntimeError("Mirror redirects back to blocked huggingface.co; not usable here")
                raise RuntimeError("Mirror redirected metadata instead of serving the requested API response")
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("id") != REPO:
                raise RuntimeError(f"Mirror returned unexpected model id {payload.get('id')!r}")
    print("MIRROR_METADATA_OK_NO_WEIGHTS_DOWNLOADED", REPO, flush=True)
    print("NOTE: large checkpoint object URLs may still redirect to blocked hosts.", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="https://hf-mirror.com")
    a = p.parse_args(argv)
    try:
        probe(a.endpoint)
    except (ValueError, RuntimeError, requests.RequestException) as e:
        p.exit(2, f"MIRROR_METADATA_FAIL {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
