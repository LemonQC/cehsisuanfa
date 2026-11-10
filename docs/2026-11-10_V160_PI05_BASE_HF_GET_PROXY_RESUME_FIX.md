# 2026-11-10 — v1.6.0 OpenPI π0.5 Base: fix HF mirror HEAD errors without changing M0

## Real operator error

The user ran the explicitly authorized ~12.4 GB original `pi05_base/params`
transfer after the pinned mirror manifest had been loaded. The first
tiny `pi05_base/params/_METADATA` test failed inside
`huggingface_hub.hf_hub_download` with
`FileMetadataError: Distant resource does not seem to be on huggingface.co`
followed by `LocalEntryNotFoundError`. No large blobs were downloaded
in this attempt. This is an HTTP metadata/redirect compatibility or route
failure before transfer, **not missing files, a Pi0.5 runtime issue, or
a model checksum error**. Earlier analogous mirror HEAD checks of a
DIFFERENT Hugging Face model reported 308 redirects to huggingface.co;
that specific HTTP status for this OpenPI mirror is not independently
observed. The original RoboReel Uniform weights remain downloaded.

## One clear, concrete fix

`scripts/prepare_pi05_base_from_hf.py` now retrieves ONLY metadata with
`HfApi` from the pinned public mirror, revision
`1fda6b04e83d625eb1a55d68db32050e248cac2b`, but fetches each
blob using ordinary `requests.get(stream=True)` from:
`https://huggingface.co/robotgeneralist/openpi_checkpoint_mirrors2/resolve/<pinned-sha>/<path>`.

Unlike `hf_hub_download`, this bypasses the strict Hub **HEAD metadata**
validation that rejected the mirror's resolve redirect. Standard HTTPS
redirects to public Hugging Face blob CDN are followed. The operator's
existing HTTPS_PROXY/ALL_PROXY (or a real VPN route) is respected.
This only works if that path is actually reachable; Python cannot bypass
a network firewall just by changing HTTP clients.

Each selected Orbax file is saved under `.part`, resumed using HTTP Range
on a later invocation, size-checked and SHA256/Git-object-checked when
published in the fixed Hugging Face manifest. Large files print
`OPENPI_BASE_PROGRESS` every ~32 MiB or ~20 s.
The old directory, partial data, and already verified files are preserved.
Only after every manifest item and Orbax structural file is present do we
symlink the complete base params directory to the original OpenPI
`$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params` cache.
No original source code or RoboReel Uniform weights are modified.
HF mirror/GCS byte-for-byte provenance remains not independently proved.

## Exact server commands — FIRST tiny probe

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05

    # Keep the same trusted proxy/VPN route that previously completed
    # the 6.2GB RoboReel Uniform checkpoint, if it is still needed.
    # Example only (replace with user's real existing proxy address):
    # export HTTPS_PROXY=http://<approved-proxy-host>:<port>
    # export HTTP_PROXY="$HTTPS_PROXY"

    # GET only pi05_base/params/_METADATA. No 12.4GB transfer.
    bash scripts/run_v141_next_gate.sh probe-pi05-base-hf

Success marker: `OPENPI_BASE_SMALL_GET_OK`, then
`SMALL_PROBE_ONLY_NO_LARGE_DOWNLOAD`. The pinned SHA and small file
are checked for correctness. Do NOT run full download if this fails.

If it fails due to the user's outbound route, inspect only the
transport without printing proxy credentials or signed redirect URLs:

    curl -L --fail --silent --show-error --max-time 30 -o /dev/null \
      -w '%{http_code} %{size_download} bytes from %{url_effective}\n' \
      'https://huggingface.co/robotgeneralist/openpi_checkpoint_mirrors2/resolve/1fda6b04e83d625eb1a55d68db32050e248cac2b/pi05_base/params/_METADATA'

CAUTION: `url_effective` can include a time-limited signed CDN query
token. If sharing logs, redact it; avoid posting the raw full URL.
Use `curl -w '%{http_code} %{size_download}\n'` instead if sharing.
Do not disable TLS verification or blindly supply paid cloud credentials.
Try the same approved proxy/VPN that worked for previous checkpoint
bytes, or copy the complete pinned model on another network-capable host.

## AFTER small probe passes

    bash scripts/run_v141_next_gate.sh prepare-pi05-base-hf
    bash scripts/run_v141_next_gate.sh check-siglip-cache
    bash scripts/run_v141_next_gate.sh smoke-one-feature
    bash scripts/run_v141_next_gate.sh prepare-features
    python -m agcofl.v141_pi05_assets --variant uniform --require-ready
    bash scripts/run_v141_next_gate.sh evaluate-uniform

This is M0 of the frozen `V159` plan: original source SigLIP, 20 EVAL
features, actual 60 original Pi0.5 Uniform trials (3x20). It does NOT
alter the separate M1 same-goal cross-behavior physical task or M2
capability-conditioned learned editor requirements.

## Evidence state

- Previous 6.2GB Uniform released checkpoint available: YES.
- Original pi05_base from community HF mirror: MANIFEST ACCESSIBLE,
  **tiny blob still NOT VERIFIED on operator network after this code change**.
- Original SigLIP weights extracted: NO.
- 20 EVAL SigLIP bundles: 0/20.
- Official Pi0.5 Uniform physical rollouts: 0/60.
- New mocked HTTP Range tests: committed, not executed on operator GPU.
- Frozen Vid2Robot original 3/60: unchanged.

If probe succeeds the real acceptance is observed small-file HTTP transfer;
if probe fails the task is a networking/proxy route problem, not a
scientific basis for switching research topics.
