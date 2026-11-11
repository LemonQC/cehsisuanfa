# 2026-11-11 — v1.6.2 Offline hardlink fix for completed OpenPI π0.5 Base checkpoint

## Verified operator failure

Operator ran a real original-author SigLIP one-video smoke after the pinned
pi05_base full-HF download. Our own `PI05_PINNED_HF_BASE_STAGED` appeared
and the original author `SigLipTokenizer` reached:
`openpi.shared.download.maybe_download("gs://openpi-assets/checkpoints/pi05_base/params")`.

This is now **past** the 12.4GB HF transfer stage, but failed before
SigLIP tensors were restored:

    ValueError: '/.../openpi_base_hf_mirror/pi05_base/params'
    is not in the subpath of '/.../openpi_cache'

The source is pinned original RoboReel model code:
`src/external_models/openpi/shared/download.py`.

Its `maybe_download` does:

    local_path = (cache_dir / "openpi-assets/checkpoints/pi05_base/params").resolve()

Then it calls `_should_invalidate_cache(cache_dir, local_path)` which uses

    local_path.relative_to(cache_dir)

Our prior installation used a directory **symlink** from within
`OPENPI_DATA_HOME` into a sibling directory under
`openpi_base_hf_mirror`. `.resolve()` follows that symlink and
`relative_to` necessarily raises. This is our cache installation
design bug, not corrupted weights or the original JAX model code.

## Fix without another 12.4GB transfer or duplicate 12.4GB disk usage

The downloader now has:

    python scripts/prepare_pi05_base_from_hf.py --repair-cache-only \
      --cache "$OPENPI_DATA_HOME"

Runner entry point:

    bash scripts/run_v141_next_gate.sh repair-pi05-cache

This **does not contact Hugging Face or GCS**. It requires the
previously recorded complete pinned HF manifest and local Orbax
metadata. It builds a REAL physical directory at

    $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params

by hardlinking each completed Orbax file from

    outputs/v120_baselines/openpi_base_hf_mirror/pi05_base/params

On the SAME filesystem (both are beneath the same project path), a
hardlink points to the same underlying inode; it does NOT duplicate
the ~12.4GB file contents. No original data is deleted. If crossing
filesystems, the script fails safely instead of silently copying
12.4GB. File placement uses a recoverable
`.params-hardlink-staging` sibling directory and verifies every
linked source filename/inode before activating the real cache path.
Only a symlink verified to point to the original mirrored path is
removed; no unrelated cache directory is overwritten. The original
author's OpenPI code remains completely unmodified.

The updated `check_pi05_siglip_cache.py` rejects out-of-tree
directory symlinks, checks that the real target is inside
OPENPI_DATA_HOME, and verifies all cache files refer to their
original pinned mirror inodes.

## Execute exactly once, no large downloads

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05

    python -m pytest -q \
      tests/test_prepare_pi05_base_from_hf.py \
      tests/test_check_pi05_siglip_cache.py

    bash scripts/run_v141_next_gate.sh repair-pi05-cache

Expected output includes:

    OPENPI_BASE_CACHE_REPAIRED_OFFLINE
    "symlink": false
    "same_inode_metadata": true
    "no_network_used": true
    PI05_PINNED_HF_BASE_STAGED

After that, RETRY the original **one-video** feature smoke, not the
entire 20 videos / physics immediately:

    bash scripts/run_v141_next_gate.sh smoke-one-feature

If SigLIP actually restores successfully and writes
`0080/front.npy`, then continue original frozen M0 gates:

    bash scripts/run_v141_next_gate.sh prepare-features
    python -m agcofl.v141_pi05_assets --variant uniform --require-ready
    bash scripts/run_v141_next_gate.sh evaluate-uniform

Checkpoint file integrity had passed our pinned HF manifest, but
actual Orbax parameter restore and compatibility with GCS primary
source are NOT yet demonstrated; do not claim π0.5 results until they
execute. Original Vid2Robot 3/60 is unchanged. M1 and M2 scientific
workstreams remain defined in V159; this is only repair of necessary
M0 baseline infrastructure.

## Tests

New regressions reproduce old out-of-tree symlink behavior, validate
a hardlink cache (same inode, no copy), test idempotent rerun, ensure
the SigLIP cache gate refuses old symlinks, and test offline
`--repair-cache-only` without network. These tests were committed
and require execution on the operator environment; no server test
has been observed yet.
