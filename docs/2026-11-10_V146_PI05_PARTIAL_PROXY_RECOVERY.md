# 2026-11-10 — v1.4.6 Official π0.5 Uniform checkpoint partial repair

## Confirmed observed download state

The 4090 server originally selected 25 released files (6.20 GB) from
LogosRoboticsLab/roboreel_models ND Uniform. Its file-by-file metadata audit
found **12 local size matches** and **13 missing paths**. Every mirror file
HEAD returned HTTP 308 redirect to huggingface.co; even the mirror metadata
endpoint sometimes returned 308. Consequently, HF-Mirror is not a viable
blob mirror for this model on this network. The 12 size matches must not
be called checksum-verified until independently hashed.

The user reports previous partial downloads via a slow proxy/VPN. Stop
restarting the original 25-file snapshot_download loop; preserve all
checkpoint paths and any valid downloaded fragments. There have been **zero
new official π0.5 physical trials**. The v1.3 Vid2Robot 3/60 stays archived
unchanged, no reruns.

## New recovery code — no paid LLM/VLM API

scripts/recover_pi05_uniform_missing.py:
- requests ONLY public official model metadata via hf-mirror.com; does not
  request mirror-resolved file bodies, because they return 308;
- freezes the exact model repository revision shown in the user's audit,
  ef21150c02b2685347fe3e7c8ba5a2e313ff1b7e; refuses changed revision
  or an unexpected number of selected files;
- checks each local path against official metadata file size and, when
  available, LFS SHA256 or Git blob SHA1. Marks unverifiable files
  as size-only instead of claiming a verified checksum;
- by default prints the actual remaining file count and total remaining
  bytes without downloading;
- explicit --download-missing streams only absent/invalid model files
  through the user's own approved HTTPS_PROXY/ALL_PROXY or explicit
  --routed-vpn, from pinned HTTPS huggingface.co official URLs;
- writes *.part, sends HTTP Range to resume interruptions, validates
  Content-Range, original file size and available hash before atomic
  replacement. Does NOT edit checkpoint metadata or model weights.
- --max-files 1 limits a connectivity smoke to one missing file,
  preserving a partial transfer for a subsequent resume.

Network routing is still an operational dependency: this script cannot
accelerate a slow proxy beyond limiting duplicate bytes, and the
metadata mirror or official CDN may remain blocked. No model scores are
created by the script.

## Commands on the RTX4090 server

Confirm a clean worktree and main branch before fast-forwarding:

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git status
    git branch --show-current
    git pull --ff-only origin main
    conda activate agcofl4090

First inventory, no download:

    bash scripts/run_v141_next_gate.sh plan-checkpoint-repair

For an HTTP(S) proxy, first set environment variables to the user's
actual trusted proxy (example port below is illustrative only):

    export HTTPS_PROXY=http://127.0.0.1:<ACTUAL_PORT>
    export HTTP_PROXY="$HTTPS_PROXY"

For a system VPN/TUN route with no proxy environment, pass --routed-vpn.
Try a single missing file before committing to many gigabytes:

    python scripts/recover_pi05_uniform_missing.py --download-missing --max-files 1

Then if it works and download speed is acceptable:

    bash scripts/run_v141_next_gate.sh recover-checkpoint-proxy

Or with confirmed system-level VPN/TUN, no HTTP proxy configured:

    python scripts/recover_pi05_uniform_missing.py --download-missing --routed-vpn

The script only consumes public repository metadata and official weight
files; it does not invoke paid inference services or require HF tokens.
Do not paste proxy credentials into logs or GitHub.

After all 25 file paths are present, continue in original authors'
separate JAX/OpenPI environment:

    bash scripts/run_v141_next_gate.sh prepare-features
    bash scripts/run_v141_next_gate.sh evaluate-uniform

Evaluation is still blocked until the full preflight is ready=true.
If the VPN is prohibitively slow, copy only the missing official files
from a machine with fast unrestricted internet, preserving source paths
and checkpoint revision; use the same local plan/check to verify them.
Do not buy external model API tokens or invest more time in a defunct
mirror: advance custom-physics-dev on the 4090 independently.

## Verification

tests/test_recover_pi05_uniform_missing.py contains offline checks for
SHA256, Git blob SHA1, HTTP 206 resume and zero re-download of valid files.
These tests are committed to GitHub but have **not run in the user's GPU
environment** as of writing.
