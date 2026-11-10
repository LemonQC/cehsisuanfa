# 2026-11-10 — v1.6.1 M0 OpenPI base checkpoint: zero-byte Orbax marker bug

## Observed user-server outcome

The newly implemented HTTPS GET downloader worked beyond the previous
Hugging Face metadata-HEAD failure. Operator log confirmed:

    OPENPI_BASE_FILE_OK pi05_base/params/array_metadatas/process_0 git-blob-verified
    OPENPI_BASE_FILE 5 / 20 0 pi05_base/params/commit_success.txt

The first fatal error was now:

    RuntimeError: Finished .part has bad hash:
      pi05_base/params/commit_success.txt missing

This is NOT a network failure or checkpoint problem. The upstream
Orbax `commit_success.txt` is a valid **zero-byte file** according to
the pinned public HF metadata. In `download_one`, an absent temporary
file was treated as offset=0, and therefore offset==size for a zero-size
manifest entry, but no file existed to validate, yielding "missing".

## Fix

For zero-byte manifest entries where `.part` is absent:
- Materialize the empty temporary file (`Path.touch(exist_ok=False)`)
  with no HTTP GET.
- Verify its Git blob SHA-1 / LFS SHA-256 against the pinned manifest
  when available (Git SHA1 of empty blob is the hash of
  `b"blob 0\\x00"`).
- Atomically rename only after successful validation.
- Reuse a verified final file on subsequent invocations.
- Refuse a bad published digest instead of falsely succeeding.

Two regression tests were added: valid zero-byte marker is created and
reused without any HTTP request; invalid digest is rejected. Existing
byte-sized files remain unchanged, as does the original pretrained
pi05_base (not the RoboReel fine-tuned 6.2GB policy).

## Exact next command

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05
    python -m pytest -q tests/test_prepare_pi05_base_from_hf.py
    bash scripts/run_v141_next_gate.sh prepare-pi05-base-hf

**DO NOT delete** `outputs/v120_baselines/openpi_base_hf_mirror`
or `.part` files. All previously verified files are checked and
skipped. This is an intentional resumed checkpoint download, not a
new task or another model.

Once complete, the standard frozen M0 chain remains:

    bash scripts/run_v141_next_gate.sh check-siglip-cache
    bash scripts/run_v141_next_gate.sh smoke-one-feature
    bash scripts/run_v141_next_gate.sh prepare-features
    python -m agcofl.v141_pi05_assets --variant uniform --require-ready
    bash scripts/run_v141_next_gate.sh evaluate-uniform

No evidence yet of a full successful 20-file base restore, SigLIP
weight export or official π0.5 physical rollout; nothing should
be reported as already achieved.
